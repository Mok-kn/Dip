## Introduction

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

