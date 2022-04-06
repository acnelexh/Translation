#!/bin/bash
#lr=0.0026
#   --data-augmentation retinanet\
#   --data-path /data/datasets/coco\
#   --output-dir model_resnetssd_coco_unfreeze\
#   --model ssd_resnet_baseline_unfreeze
python train.py\
    --batch-size 32\
    --dataset mnist\
    --epochs 65\
    --lr-steps 43 54\
    --lr 0.001\
    --weight-decay 0.0005\
    --momentum 0.9\
    --data-augmentation mnist\
    --workers 10\
    --model ResNet50
    

