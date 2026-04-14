# YOLO-NFA Small Target Detection

This repository is an implementation of the paper **[A Contrario Paradigm for YOLO-Based Infrared Small Target Detection](https://arxiv.org/abs/2402.02288)** on the **[NUAA-SIRST](https://github.com/YimianDai/sirst)** dataset.

The YOLOv7-tiny box regression pipeline is kept, while the objectness branch is replaced with an **OL-NFA** head. In addition to the paper, this implementation adds:

- a score-adapter convolution before the NFA computation
- an ECA-style scale attention block across the three detection scales

## Setup

```bash
pip install -r requirements.txt
```

Log in to Weights & Biases:

```bash
wandb login
```

## Dataset

We train and evaluate on **NUAA-SIRST** with **1 class**: `target`.


To prepare the dataset:

```bash
unzip datasets.zip
```

This creates:

- `datasets/sirst/images/{train,val,test}`
- `datasets/sirst/labels/{train,val,test}`
- `datasets/sirst/images/fs{15,25}_fold{1,2,3}`
- `datasets/sirst/labels/fs{15,25}_fold{1,2,3}`

It reuses the dataset configs in `data/datasets/`, including:

- `data/datasets/sirst.yaml`
- `data/datasets/sirst_fs15_fold{1,2,3}.yaml`
- `data/datasets/sirst_fs25_fold{1,2,3}.yaml`

## Run Locally

### Train the baseline YOLOv7-tiny model on full SIRST

```bash
python train.py \
  --seed 42 \
  --device 0 \
  --workers 2 \
  --batch-size 16 \
  --img-size 640 640 \
  --iou-thres 0.05 \
  --epochs 600 \
  --data data/datasets/sirst.yaml \
  --cfg cfg/training/yolov7-tiny.yaml \
  --weights '' \
  --hyp data/hyperparameters/hyp.sirst.tiny.yolo.yaml \
  --adam \
  --name yolov7_tiny_sirst
```
### Train the baseline YOLOv7-tiny model on the 25-shot split
```bash
python train.py \
  --seed 42 \
  --device 0 \
  --workers 2 \
  --seed 42 \
  --batch-size 16 \
  --img-size 640 640 \
  --iou-thres 0.05 \
  --epochs 600 \
  --data data/datasets/sirst_fs25_fold1.yaml \
  --cfg cfg/training/yolov7-tiny.yaml \
  --weights '' \
  --hyp data/hyperparameters/hyp.sirst.tiny.yolo.yaml \
  --adam \
  --name yolov7_tiny_sirst_f251
```
### Train the OL-NFA model on full SIRST

```bash
python train.py \
  --seed 42 \
  --device 0 \
  --workers 2 \
  --batch-size 16 \
  --img-size 640 640 \
  --iou-thres 0.05 \
  --epochs 600 \
  --data data/datasets/sirst.yaml \
  --cfg cfg/training/yolov7-tiny-olnfa.yaml \
  --weights '' \
  --hyp data/hyperparameters/hyp.sirst.tiny.yolo.olnfa.yaml \
  --adam \
  --name yolov7_tiny_olnfa_sirst
```

### Train the OL-NFA model on the 25-shot split

```bash
python train.py \
  --seed 42 \
  --device 0 \
  --workers 2 \
  --seed 42 \
  --batch-size 16 \
  --img-size 640 640 \
  --iou-thres 0.05 \
  --epochs 600 \
  --data data/datasets/sirst_fs25_fold1.yaml \
  --cfg cfg/training/yolov7-tiny-olnfa.yaml \
  --weights '' \
  --hyp data/hyperparameters/hyp.sirst.tiny.yolo.olnfa.fs25.yaml \
  --adam \
  --name yolov7_tiny_olnfa_sirst_f251
```


### Evaluate a trained checkpoint

Example on the full SIRST test split:

```bash
python test.py \
  --weights runs/train/yolov7_tiny_olnfa_sirst/weights/best.pt \
  --data data/datasets/sirst.yaml \
  --img-size 640 \
  --batch-size 16 \
  --task test \
  --device 0 \
  --iou-thres 0.05 \
  --match-iou-thres 0.05 \
```

