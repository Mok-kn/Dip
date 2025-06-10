import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.fft as fft
import cv2
import numpy as np
import kornia.losses as K
import pywt 
from mmcv.cnn import constant_init, kaiming_init
from ..builder import DISTILL_LOSSES
from ...models.builder import DETECTORS, build_backbone, build_head, build_neck
from ...models.detectors import TwoStageDetector
from mmdet.core.bbox.iou_calculators import *
from mmcv.runner import  load_checkpoint, _load_checkpoint, load_state_dict
from collections import OrderedDict


@DISTILL_LOSSES.register_module()
class FeatureLoss(nn.Module):

    """PyTorch version of `Focal and Global Knowledge Distillation for Detectors`
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        temp (float, optional): Temperature coefficient. Defaults to 0.5.
        name (str): the loss name of the layer
        alpha_fgd (float, optional): Weight of fg_loss. Defaults to 0.001
        beta_fgd (float, optional): Weight of bg_loss. Defaults to 0.0005
        gamma_fgd (float, optional): Weight of mask_loss. Defaults to 0.001
        lambda_fgd (float, optional): Weight of relation_loss. Defaults to 0.000005
    """
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 name,
                 temp=0.5,
                 alpha_fgd=0.001,
                 beta_fgd=0.0005,
                 gamma_rit=0.001,
                 lambda_mask=0.5,
                 ):
        super(FeatureLoss, self).__init__()
        self.temp = temp
        self.alpha_fgd = alpha_fgd
        self.beta_fgd = beta_fgd
        self.gamma_rit = gamma_rit
        self.lambda_mask = lambda_mask

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

        self.conv_mask_s = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.conv_mask_t = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.channel_add_conv_s = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels//2, kernel_size=1),
            nn.LayerNorm([teacher_channels//2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels//2, teacher_channels, kernel_size=1))
        self.channel_add_conv_t = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels//2, kernel_size=1),
            nn.LayerNorm([teacher_channels//2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels//2, teacher_channels, kernel_size=1))
        
        ############ generation ###########
        self.generation = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1))
        ############ generation ###########
        
        ############ 2025 04 25 ###########
        self.res_conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.res_conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        ############ 2025 04 25 ###########

        self.reset_parameters()


    def forward(self,
                preds_S,
                preds_T,
                rpn_S,
                gt_bboxes,
                img_metas):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
            gt_bboxes(tuple): Bs*[nt*4], pixel decimal: (tl_x, tl_y, br_x, br_y)
            img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:],'the output dim of teacher and student differ'

        if self.align is not None:
            preds_S = self.align(preds_S)
        
        N,C,H,W = preds_S.shape

        S_attention_t, C_attention_t = self.get_attention(preds_T, self.temp)
        S_attention_s, C_attention_s = self.get_attention(preds_S, self.temp)

        Mask_fg = torch.zeros_like(S_attention_t)
        Mask_bg = torch.ones_like(S_attention_t)

        ################
        Mask_bg_mask = torch.ones_like(S_attention_t)
        ################

        wmin,wmax,hmin,hmax = [],[],[],[]
        for i in range(N):
            new_boxxes = torch.ones_like(gt_bboxes[i])
            new_boxxes[:, 0] = gt_bboxes[i][:, 0]/img_metas[i]['img_shape'][1]*W
            new_boxxes[:, 2] = gt_bboxes[i][:, 2]/img_metas[i]['img_shape'][1]*W
            new_boxxes[:, 1] = gt_bboxes[i][:, 1]/img_metas[i]['img_shape'][0]*H
            new_boxxes[:, 3] = gt_bboxes[i][:, 3]/img_metas[i]['img_shape'][0]*H

            wmin.append(torch.floor(new_boxxes[:, 0]).int())
            wmax.append(torch.ceil(new_boxxes[:, 2]).int())
            hmin.append(torch.floor(new_boxxes[:, 1]).int())
            hmax.append(torch.ceil(new_boxxes[:, 3]).int())

            area = 1.0/(hmax[i].view(1,-1)+1-hmin[i].view(1,-1))/(wmax[i].view(1,-1)+1-wmin[i].view(1,-1))

            for j in range(len(gt_bboxes[i])):
                Mask_fg[i][hmin[i][j]:hmax[i][j]+1, wmin[i][j]:wmax[i][j]+1] = \
                        torch.maximum(Mask_fg[i][hmin[i][j]:hmax[i][j]+1, wmin[i][j]:wmax[i][j]+1], area[0][j])

            Mask_bg[i] = torch.where(Mask_fg[i]>0, 0, 1)
            if torch.sum(Mask_bg[i]):
                Mask_bg[i] /= torch.sum(Mask_bg[i])

            ################## MASK
            Mask_bg_mask[i] = torch.where(Mask_fg[i]>0, 0, 1)           # Mask_bg_mask 是背景掩码的随机置零掩码

            random_mask = (torch.rand_like(Mask_bg_mask[i]) < self.lambda_mask).to(torch.long)

            # 以概率 p 将掩码中大于 0 的部分置零
            replacement_value = torch.tensor(0.0, dtype=Mask_bg[i].dtype).to(random_mask.device)

            Mask_bg_mask[i] = torch.where((Mask_bg_mask[i] > 0) & (random_mask < self.lambda_mask), replacement_value, Mask_bg[i])
            if torch.sum(Mask_bg_mask[i]):
                Mask_bg_mask[i] /= torch.sum(Mask_bg_mask[i])
            ################### MASK
       
        
        cls_score_S = rpn_S[0]
        new_preds_S = torch.zeros_like(preds_S)

        # 对每个尺度的cls_score_S进行处理
        ######### 2025 4 25 这是之前的 ##########
        # for scale_cls_score in cls_score_S:
        #     if scale_cls_score.size()[2:] == (H, W):
        #         sigmoid_cls_score = scale_cls_score.sigmoid()
        #         mask = torch.max(sigmoid_cls_score, dim=1).values
        #         new_preds_S += mask[:,None,:,:] * preds_S
        ######### 2025 4 25 这是之前的 ##########
                
        
        ############  2025 4 25 这是后来改的 RIT部分 ############
        new_preds_S = torch.zeros_like(preds_S)
        rit_loss = 0.0
        # 创建残差注意力模块（Conv + Conv + 残差 + Sigmoid）
        for scale_cls_score in cls_score_S:
            if scale_cls_score.size()[2:] == (H, W):
                # 1. 残差模块
                residual = self.res_conv2(F.relu(self.res_conv1(scale_cls_score))) + scale_cls_score
                residual = torch.clamp(residual, min=-10.0, max=10.0)
                scale_cls_score_fm = torch.sigmoid(residual)  # [N, 3, H, W]

                # 2. 获取注意力 mask（压缩通道维度）
                attention_mask = torch.max(scale_cls_score_fm, dim=1, keepdim=True).values  # [N, 1, H, W]
                attention_mask = attention_mask + 1e-6

                # 3. 加权特征
                preds_S_fm = preds_S * attention_mask  # [N, 256, H, W]

                # 4. 特征匹配损失（跨通道广播）
                # 你可以选择将 scale_cls_score_fm 展开成 256 通道再比较，或者只比较 spatial 信息
                # 简单做法（按 spatial）：
                scale_cls_score_map = attention_mask  # [N, 1, H, W]
                preds_S_map = torch.mean(preds_S_fm, dim=1, keepdim=True)  # [N, 1, H, W]

                # 5. MSE损失
                rit_loss = F.mse_loss(preds_S_map, scale_cls_score_map)
        ############  2025 4 25 这是后来改的 RIT部分 ############
        
        
        #### 2025 4 25这是后来改的 给小波变换加注意力 ####
        # C_attn_T = C_attention_t.view(N, C, 1, 1)
        # S_attn_T = S_attention_t
        # weighted_preds_T = preds_T * C_attn_T * S_attn_T.unsqueeze(1)

        # C_attn_S = C_attention_s.view(N, C, 1, 1)
        # S_attn_S = S_attention_s
        # weighted_preds_S = new_preds_S * C_attn_S * S_attn_S.unsqueeze(1)

        # wavelet_T = self.get_wavelet(weighted_preds_T)
        # wavelet_S = self.get_wavelet(weighted_preds_S)
        #### 2025 4 25这是后来改的 给小波变换加注意力 ####
        
        #### 2025 4 25这是之前的 ####
        wavelet_T = self.get_wavelet(preds_T)
        # wavelet_S = self.get_wavelet(new_preds_S)
        wavelet_S = self.get_wavelet(preds_S_fm)
        #### 2025 4 25这是之前的 ####

        wavelet_T = self.equal(wavelet_T, preds_T)
        wavelet_S = self.equal(wavelet_S, preds_S)
        
        fg_loss, bg_loss = self.get_fea_loss(wavelet_S, wavelet_T, Mask_fg, Mask_bg, Mask_bg_mask,
                          C_attention_s, C_attention_t, S_attention_s, S_attention_t)

        # fg_loss, bg_loss = self.get_fea_loss(preds_S, preds_T, Mask_fg, Mask_bg, Mask_bg_mask,
        #                   C_attention_s, C_attention_t, S_attention_s, S_attention_t)

        # loss_1 = self.get_fea_loss1(wavelet_S, wavelet_T, C_attention_s, C_attention_t, S_attention_s, S_attention_t)

        # loss = 0.0000000125 * loss_1
        # loss = self.alpha_fgd * fg_loss + self.beta_fgd * bg_loss
        loss = self.alpha_fgd * fg_loss + self.beta_fgd * bg_loss + self.gamma_rit * rit_loss
        
        return loss


    def get_attention(self, preds, temp):
        """ preds: Bs*C*W*H """
        N, C, H, W= preds.shape

        value = torch.abs(preds)
        # Bs*W*H
        fea_map = value.mean(axis=1, keepdim=True)
        S_attention = (H * W * F.softmax((fea_map/temp).view(N,-1), dim=1)).view(N, H, W)

        # Bs*C
        channel_map = value.mean(axis=2,keepdim=False).mean(axis=2,keepdim=False)
        C_attention = C * F.softmax(channel_map/temp, dim=1)

        return S_attention, C_attention


    def get_fea_loss(self, preds_S, preds_T, Mask_fg, Mask_bg, Mask_bg_mask, C_s, C_t, S_s, S_t):
        loss_mse = nn.MSELoss(reduction='sum')
        
        Mask_fg = Mask_fg.unsqueeze(dim=1)
        Mask_bg = Mask_bg.unsqueeze(dim=1)

        #############
        Mask_bg_mask = Mask_bg_mask.unsqueeze(dim=1)
        #############

        C_t = C_t.unsqueeze(dim=-1)
        C_t = C_t.unsqueeze(dim=-1)

        S_t = S_t.unsqueeze(dim=1)

        #### Rebattal
        fea_t= torch.mul(preds_T, torch.sqrt(S_t))
        fea_t = torch.mul(fea_t, torch.sqrt(C_t))
        # fea_t= torch.mul(preds_T, torch.sqrt(C_t))
        fg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_fg))
        bg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_bg))

        fea_s = torch.mul(preds_S, torch.sqrt(S_t))
        fea_s = torch.mul(fea_s, torch.sqrt(C_t))
        # fea_s = torch.mul(preds_S, torch.sqrt(C_t))
        fg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_fg))
        bg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_bg_mask))
        bg_fea_s = self.generation(bg_fea_s)
        bg_fea_s = torch.mul(bg_fea_s, Mask_bg_mask)

        fg_loss = loss_mse(fg_fea_s, fg_fea_t)/len(Mask_fg)
        bg_loss = loss_mse(bg_fea_s, bg_fea_t)/len(Mask_bg)

        return fg_loss, bg_loss
    

    def get_fea_loss1(self, preds_S, preds_T, C_s, C_t, S_s, S_t):
        loss_mse = nn.MSELoss(reduction='sum')

        C_t = C_t.unsqueeze(dim=-1)
        C_t = C_t.unsqueeze(dim=-1)

        S_t = S_t.unsqueeze(dim=1)

        fea_t= torch.mul(preds_T, torch.sqrt(S_t))
        fea_t = torch.mul(fea_t, torch.sqrt(C_t))

        fea_s = torch.mul(preds_S, torch.sqrt(S_t))
        fea_s = torch.mul(fea_s, torch.sqrt(C_t))

        loss = loss_mse(fea_s, fea_t)/len(fea_s)

        return loss


    def min_max_normalization(self, input_tensor):
        if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
            input_tensor[torch.isnan(input_tensor) | torch.isinf(input_tensor)] = 0.0
        
        mean = input_tensor.mean(dim=(0, 2, 3), keepdim=True)
        std = input_tensor.std(dim=(0, 2, 3), keepdim=True)
        normalized_tensor = (input_tensor - mean) / std

        return normalized_tensor


    def get_fft(self, preds):
        result = fft.fft2(preds, dim=(2, 3), norm='ortho').real

        if torch.isnan(result).any() or torch.isinf(result).any():
            result[torch.isnan(result) | torch.isinf(result)] = 0.0

        # fft_result_abs = torch.abs(result.abs()).float()

        return result


    def get_fft_loss(self, fft_s, fft_t):
        loss_mse = nn.MSELoss(reduction='sum')

        N, C, H, W = fft_t.shape

        # C_t = C_t.unsqueeze(dim=-1)
        # C_t = C_t.unsqueeze(dim=-1)

        # S_t = S_t.unsqueeze(dim=1)

        # fea_t= torch.mul(fft_t, torch.sqrt(S_t))
        # fea_t = torch.mul(fea_t, torch.sqrt(C_t))

        # fea_s = torch.mul(fft_s, torch.sqrt(S_t))
        # fea_s = torch.mul(fea_s, torch.sqrt(C_t))

        device = fft_s.device
        mat = torch.rand((N,1,H,W)).to(device)
        mat = torch.where(mat>1-self.lambda_mask, 0, 1).to(device)

        fft_s = torch.mul(fft_s, mat)
        
        fft_s = self.generation(fft_s)

        fft_loss = loss_mse(fft_s, fft_t)/N

        return fft_loss
    

    def get_wavelet(self, feature_tensor, wavelet='db1', level=3):
        
        feature_numpy = feature_tensor.detach().cpu().numpy()

        denoised_channels = []
        for channel in feature_numpy:
            coeffs = pywt.wavedec2(channel, wavelet, level=level)

            processed_coeffs = [coeffs[0]] 

            for i in range(1, level + 1):
                cH, cV, cD = coeffs[i]
                threshold = np.std(cH) * 0.5  
                cH_soft = pywt.threshold(cH, threshold, mode='soft')
                cV_soft = pywt.threshold(cV, threshold, mode='soft')
                cD_soft = pywt.threshold(cD, threshold, mode='soft')
                processed_coeffs.append((cH_soft, cV_soft, cD_soft))

            denoised_channel = pywt.waverec2(processed_coeffs, wavelet)

            denoised_channels.append(denoised_channel)

        denoised_feature_numpy = np.stack(denoised_channels, axis=0)

        denoised_feature_tensor = torch.from_numpy(denoised_feature_numpy).to(feature_tensor.device)

        return denoised_feature_tensor


    def get_wavelet_loss(self, wavelet_T, wavelet_S, Mask_fg, C_s, C_t, S_s, S_t):
        # loss_mse = nn.MSELoss(reduction='sum')
        # N, C, H, W = wavelet_T.shape
        # return loss_mse(wavelet_T, wavelet_S)/N

        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = wavelet_T.shape

        Mask_fg = Mask_fg.unsqueeze(dim=1)

        C_t = C_t.unsqueeze(dim=-1)
        C_t = C_t.unsqueeze(dim=-1)

        S_t = S_t.unsqueeze(dim=1)

        fea_t = torch.mul(wavelet_T, torch.sqrt(S_t))
        fea_t = torch.mul(fea_t, torch.sqrt(C_t))
        fg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_fg))

        fea_s = torch.mul(wavelet_S, torch.sqrt(S_t))
        fea_s = torch.mul(fea_s, torch.sqrt(C_t))
        fg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_fg))

        return loss_mse(fg_fea_t, fg_fea_s)/N


    def get_wavelet_loss1(self, wavelet_T, wavelet_S):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = wavelet_T.shape
        return loss_mse(wavelet_T, wavelet_S)/N


    def equal(self, feat1, feat2):
        h2, w2 = feat2.size(2), feat2.size(3)
        A_resized = F.interpolate(feat1, size=(h2, w2), mode='bilinear', align_corners=False)
        return A_resized


    def last_zero_init(self, m):
        if isinstance(m, nn.Sequential):
            constant_init(m[-1], val=0)
        else:
            constant_init(m, val=0)

    
    def reset_parameters(self):
        kaiming_init(self.conv_mask_s, mode='fan_in')
        kaiming_init(self.conv_mask_t, mode='fan_in')
        self.conv_mask_s.inited = True
        self.conv_mask_t.inited = True

        self.last_zero_init(self.channel_add_conv_s)
        self.last_zero_init(self.channel_add_conv_t)

############ mgd ###########
    def get_dis_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        device = preds_S.device
        mat = torch.rand((N,1,H,W)).to(device)
        mat = torch.where(mat>1-self.lambda_mgd, 0, 1).to(device)

        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation(masked_fea)

        dis_loss = loss_mse(new_fea, preds_T)/N

        return dis_loss
############ mgd ###########

