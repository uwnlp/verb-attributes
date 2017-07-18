#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
mkdir ../checkpoints/imsitu_devise
python ../models/imsitu_train.py -lr 1e-5 -l2_weight 1e-3 -save_dir imsitu_devise -imsitu_model devise -use_emb > ../checkpoints/imsitu_devise/log.txt