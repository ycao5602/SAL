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
       --max-epoch-jt 200 \
       --max-epoch-pt 200 \
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
