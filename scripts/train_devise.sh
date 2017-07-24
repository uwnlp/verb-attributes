#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
mkdir ../checkpoints/imsitu_devise
mkdir ../checkpoints/imsitu_ours
python ../models/imsitu_train.py -lr 1e-5 -l2_weight 1e-4 -eps 1e-1 -save_dir imsitu_ours/embatt -imsitu_model ours -use_att -use_emb > ../checkpoints/imsitu_ours/att_emb_log.txt
python ../models/imsitu_train.py -lr 1e-5 -l2_weight 1e-4 -save_dir imsitu_devise/d1 -imsitu_model devise -use_emb > ../checkpoints/imsitu_devise/log.txt