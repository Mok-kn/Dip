
## **Dual-Information-Purification-for-Lightweight-SAR-Object-Detection**

![image]([https://github.com/ZhiliangMa/MPU6500-HMC5983-AK8975-BMP280-MS5611-10DOF-IMU-PCB/blob/main/img/IMU-V5-TOP.jpg](https://github.com/Mok-kn/Dip/blob/main/figure2.pdf))

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
