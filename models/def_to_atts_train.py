"""
Script to pretrain the LSTM -> linear model for definition to attributes
"""

from data.dictionary_dataset import DictionaryChallengeDataset, PackedBucketIterator
from config import ModelConfig
from lib.rnn_encoder import DictionaryModel
from torch import optim
import os
import torch
from lib.misc import CosineRankingLoss, optimize, cosine_ranking_loss
import numpy as np
import time

# Recommended hyperparameters
args = ModelConfig(margin=0.1, lr=2e-4, batch_size=64, eps=1e-8,
                   ckpt='def2atts_pretrain/ckpt_50.tar', save_dir='def2atts_train')

train_data, val_data = DictionaryChallengeDataset.splits()
train_iter = PackedBucketIterator(train_data, batch_size=args.batch_size)
val_iter = PackedBucketIterator(val_data, batch_size=args.batch_size, shuffle=False)


m = DictionaryModel(train_data.fields['text'].vocab)
optimizer = optim.Adam(m.parameters(), lr=args.lr, eps=args.eps, betas=(args.beta1, args.beta2))

if len(args.ckpt) > 0 and os.path.exists(args.ckpt):
    print("loading checkpoint from {}".format(args.ckpt))
    ckpt = torch.load(args.ckpt)
    m.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])

crit = CosineRankingLoss(size_average=True, margin=args.margin)

if torch.cuda.is_available():
    m.cuda()
    crit.cuda()


@optimize
def train_batch(batch, optimizers=None):
    pred = m(batch)
    gt = batch.words
    return crit(pred, gt)


def deploy(batch):
    pred = m(batch)
    gt = batch.words
    cost, correct_contrib, inc_contrib = cosine_ranking_loss(pred, gt, margin=args.margin)
    return torch.mean(cost), torch.mean(correct_contrib)

last_best_epoch = 1
prev_best = 0.0
for epoch in range(1, 51):
    val_l = []
    val_l_correct = []
    train_l = []

    m.eval()
    for val_b, val_batch in enumerate(val_iter):
        cost, correct_contrib = deploy(val_batch)
        val_l.append(cost.data[0])
        val_l_correct.append(correct_contrib.data[0])

    print("--- \n E{:2d} (VAL) Cost {:.3f} Correct score {:.3f} \n --- \n".format(
        epoch,
        np.mean(val_l),
        np.mean(val_l_correct),
    ), flush=True)

    if np.mean(val_l_correct) > prev_best:
        prev_best = np.mean(val_l_correct)
        last_best_epoch = epoch
    else:
        if last_best_epoch < (epoch - 3):
            print("Early stopping at epoch {}".format(epoch))
            break

    m.train()
    start_epoch = time.time()
    for b, batch in enumerate(train_iter):
        start = time.time()
        l = train_batch(batch, optimizers=[optimizer])

        train_l.append(l.data[0])

        dur = time.time() - start
        if b % 1000 == 0 and b >= 100:
            print("e{:2d}b{:5d} Cost {:.3f} , {:.3f} s/batch".format(
                epoch, b,
                np.mean(train_l),
                dur,
            ), flush=True)
    dur_epoch = time.time() - start_epoch
    print("Duration of epoch was {:.3f}/batch, overall loss was {:.3f}".format(
        dur_epoch/b,
        np.mean(train_l),
    ))
    torch.save({
                'args': args.args,
                'epoch': epoch,
                'm_state_dict': m.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.save_dir, 'ckpt_{}.tar'.format(epoch)))