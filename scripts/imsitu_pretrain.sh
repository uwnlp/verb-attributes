#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2
mkdir ../checkpoints/imsitu_pretrain
nohup python ../models/imsitu_pretrain.py > ../checkpoints/imsitu_pretrain/log.txt