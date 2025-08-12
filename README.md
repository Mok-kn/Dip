
## **Dual-Information-Purification-for-Lightweight-SAR-Object-Detection**

![image](https://github.com/Mok-kn/Dip/blob/main/figure.png)

### Introduction

Our codes are based on MMDetection. Please follow the installation of MMDetection and make sure you can run it successfully.

### Add and Replace the codes

- Add the configs/. in our codes to the configs/ in mmdetectin's codes.

- Add the mmdet/distillation/. in our codes to the mmdet/ in mmdetectin's codes.

- Unzip COCO dataset into data/

### Train
```
python tools/train.py configs/faster_rcnn_r50_distill_r18_fpn_HRSID.py.py
```

### Test
```
python tools/test.py configs/faster_rcnn_r50_distill_r18_fpn_HRSID.py.py ***.pth
```

### Citation
```
@inproceedings{
  title={Dual Information Purification for Lightweight SAR Object Detection},
  author={Yang, X., Sun, J., Duan, S., & Cheng, D},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  pages={9274-9282},
  year={2025},
  url={https://doi.org/10.1609/aaai.v39i9.33004}
}
```
https://account.aliyun.com/login/login.htm?oauth_callback=https%3A%2F%2Faccount-devops.aliyun.com%2Flogin%3Fnext_url%3Dhttp%3A%2F%2Faccount-devops.aliyun.com%2Flogin%3Fnext_url%3Dhttps%253A%252F%252Fthoughts.aliyun.com%252Fworkspaces%252F6655879cf459b7001ba42f1b%252Ffiles%252F66ab8caa4c60500001cbb864%26referrer%3Dthoughts
