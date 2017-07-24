#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
mkdir ../checkpoints/def2atts_train
python ../models/def_to_atts_train.py -lr 1e-3 -l2_weight 1e-2 -use_emb -save_dir def2atts_train/emb1 > ../checkpoints/def2atts_train/emb1.txt
python ../models/def_to_atts_train.py -lr 1e-4 -l2_weight 1e-2 -use_emb -save_dir def2atts_train/emb2 > ../checkpoints/def2atts_train/emb2.txt
python ../models/def_to_atts_train.py -lr 1e-4 -l2_weight 1e-2 -use_emb -eps 1e-8 -save_dir def2atts_train/emb3 > ../checkpoints/def2atts_train/emb3.txt
python ../models/def_to_atts_train.py -lr 5e-4 -l2_weight 1e-2 -use_emb -eps 1e-8 -save_dir def2atts_train/emb4 > ../checkpoints/def2atts_train/emb4.txt
python ../models/def_to_atts_train.py -lr 1e-3 -l2_weight 1e-2 -use_emb -dropout 0.2 -save_dir def2atts_train/emb5 > ../checkpoints/def2atts_train/emb5.txt
python ../models/def_to_atts_train.py -lr 1e-4 -l2_weight 1e-2 -use_emb -dropout 0.2 -save_dir def2atts_train/emb6 > ../checkpoints/def2atts_train/emb6.txt
python ../models/def_to_atts_train.py -lr 1e-4 -l2_weight 1e-2 -use_emb -eps 1e-8 -dropout 0.2 -save_dir def2atts_train/emb7 > ../checkpoints/def2atts_train/emb7.txt
python ../models/def_to_atts_train.py -lr 5e-4 -l2_weight 1e-2 -use_emb -eps 1e-8 -dropout 0.2 -save_dir def2atts_train/emb8 > ../checkpoints/def2atts_train/emb8.txt
python ../models/def_to_atts_train.py -lr 1e-3 -l2_weight 1e-2 -use_emb -eps 1e-8 -save_dir def2atts_train/emb9 > ../checkpoints/def2atts_train/emb9.txt

python ../models/def_to_atts_train.py -lr 1e-3 -l2_weight 1e-2 -save_dir def2atts_train/att1 > ../checkpoints/def2atts_train/att1.txt
python ../models/def_to_atts_train.py -lr 1e-4 -l2_weight 1e-2 -save_dir def2atts_train/att2 > ../checkpoints/def2atts_train/att2.txt
python ../models/def_to_atts_train.py -lr 1e-4 -l2_weight 1e-2 -eps 1e-8 -save_dir def2atts_train/att3 > ../checkpoints/def2atts_train/att3.txt
python ../models/def_to_atts_train.py -lr 5e-4 -l2_weight 1e-2 -eps 1e-8 -save_dir def2atts_train/att4 > ../checkpoints/def2atts_train/att4.txt
python ../models/def_to_atts_train.py -lr 1e-3 -l2_weight 1e-2 -dropout 0.2 -save_dir def2atts_train/att5 > ../checkpoints/def2atts_train/att5.txt
python ../models/def_to_atts_train.py -lr 1e-4 -l2_weight 1e-2 -dropout 0.2 -save_dir def2atts_train/att6 > ../checkpoints/def2atts_train/att6.txt
python ../models/def_to_atts_train.py -lr 1e-4 -l2_weight 1e-2 -eps 1e-8 -dropout 0.2 -save_dir def2atts_train/att7 > ../checkpoints/def2atts_train/att7.txt
python ../models/def_to_atts_train.py -lr 5e-4 -l2_weight 1e-2 -eps 1e-8 -dropout 0.2 -save_dir def2atts_train/att8 > ../checkpoints/def2atts_train/att8.txt


