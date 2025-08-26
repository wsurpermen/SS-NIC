#!/bin/bash

# single gpu
python3 CWD.py --exp CWD_test --config config/animal10N/CWD.cfg --gpu 0 --batch_size 128 --lr 0.0001 --path data/Animal_10N/raw_image_ver

# multi gpu
# python3 CWD.py --exp CWD_test --config config/animal10N/CWD.cfg --gpu 0,1,2,3 --batch_size 128 --lr 0.0001 --path data/Animal10/raw_image_ver