#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2
mkdir ../checkpoints/imsitu_ours
python ../models/imsitu_train.py -lr 1e-5 -l2_weight 1e-4 -eps 1e-1 -save_dir imsitu_ours/emb -imsitu_model ours -use_emb > ../checkpoints/imsitu_ours/emb_log.txt
python ../models/imsitu_train.py -lr 1e-5 -l2_weight 1e-4 -eps 1e-1 -save_dir imsitu_ours/att -imsitu_model ours -use_att > ../checkpoints/imsitu_ours/att_log.txt