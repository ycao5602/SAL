#!/bin/bash

# abort entire script on error
set -e

# train model
cd ..
python train.py \
       -s PETA \
       --root dataset \
       --optim adam \
       --label-smooth \
       --lr-pt 0.00003 \
       --lr-jt 0.001 \
       --lr-al 0.0001 \
       --lr-sc 0.01 \
       --max-epoch-jt 100 \
       --max-epoch-pt 100 \
       --max-epoch-al 60 \
       --stepsize 20 40 60 80 100 120 140 160 180 \
       --stepsize-sal 20 30 40 50 \
       --train-batch-size 128 \
       --test-batch-size 100 \
       -a resnet50 \
       --save-dir log/peta-results \
       --eval-freq 10 \
       --save-pt 20 \
       --gpu-devices 0,1
