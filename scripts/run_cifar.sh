#!/bin/bash

# python main_cifar.py  \
#         --new_classes 10 \
#         --start_classes 50 \
#         --cosine \
#         --kd \
#         --w-kd 1 \
#         --epochs 120 \
#         --save \
#         --num-sd 0

for i in {1..4}
do
  echo "Running experiment $i"
  python main_cifar.py  \
        --new_classes 10 \
        --start_classes 50 \
        --cosine \
        --kd \
        --w-kd 1 \
        --epochs 120 \
        --save \
        --num-sd 0
done