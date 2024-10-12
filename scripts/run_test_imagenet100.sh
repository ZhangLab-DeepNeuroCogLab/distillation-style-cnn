#!/bin/bash

python test_imagenet.py  \
        --new-classes 10 \
        --start-classes 50 \
        --epochs1 70 \
        --epochs2 40 \
        --K 20 \
        --batch-size 128 \
        --save-freq 10 \
        --dataset imagenet100 \
        --exp-name in_style1_aug_k20_cnn_t5_bkd01_kd_replay
