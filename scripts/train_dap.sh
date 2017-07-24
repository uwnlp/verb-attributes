#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
mkdir ../checkpoints/imsitu_dap
#python ../models/imsitu_train.py -lr 1e-5 -l2_weight 1e-4 -save_dir imsitu_dap/emb -imsitu_model dap -use_emb > ../checkpoints/imsitu_dap/emb_log2.txt
python ../models/imsitu_train.py -lr 1e-5 -l2_weight 1e-4 -save_dir imsitu_dap/att -imsitu_model dap -use_att > ../checkpoints/imsitu_dap/att_log2.txt