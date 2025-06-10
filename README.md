## Introduction

Our codes are based on MMDetection. Please follow the installation of MMDetection and make sure you can run it successfully.

### Add and Replace the codes

- Add the configs/. in our codes to the configs/ in mmdetectin's codes.

- Add the mmdet/distillation/. in our codes to the mmdet/ in mmdetectin's codes.

- Unzip COCO dataset into data/

### Train
```
#single GPU
python tools/train.py configs/distillers/fgd/fgd_retina_rx101_64x4d_distill_retina_r50_fpn_2x_coco.py
```

  All basic bbox and mask operations run on GPUs. The training speed is faster than or comparable to other codebases, including [Detectron2](https://github.com/facebookresearch/detectron2), [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and [SimpleDet](https://github.com/TuSimple/simpledet).

