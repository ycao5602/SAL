#!/bin/bash

# abort entire script on error
set -e

# train model
cd ..
python train.py \
       -s Market-1501 \
       --root dataset \
       --optim adam \
       --label-smooth \
       --lr-al 5e-5 \
       --lr-gf 5e-4 \
       --lr-df 5e-4 \
       --max-epoch-pt 100 \
       --max-epoch-jt 100 \
       --max-epoch-al 60 \
       --stepsize 20 40 60 80  \
       --stepsize-sal 20 40 \
       --train-batch-size 128 \
       --test-batch-size 100 \
       -a resnet50 \
       --save-dir log/market-results \
       --eval-freq 10 \
       --save-pt 20 \
       --gpu-devices 0,1
