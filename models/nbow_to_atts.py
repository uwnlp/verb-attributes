"""
Script to pretrain the LSTM -> linear model for definition to attributes
"""

from data.dictionary_dataset import DictionaryChallengeDataset, PackedBucketIterator
from config import ModelConfig
from torch import optim
import os
import torch
from lib.misc import CosineRankingLoss, optimize, cosine_ranking_loss
import numpy as np
import time
from data.attribute_loader import Attributes
from lib.att_prediction import FeedForwardModel
from lib.attribute_loss import AttributeLoss, evaluate_accuracy

# Recommended hyperparameters
args = ModelConfig(lr=5e-4, batch_size=16, eps=1e-8, save_dir='nbow2atts')
train_data, val_data, test_data = Attributes.splits(use_defns=False,
                                                    cuda=torch.cuda.is_available())

crit = AttributeLoss(train_data.domains, size_average=True)
m = FeedForwardModel(input_size=300, output_size=crit.input_size, init_dropout=0.05)
optimizer = optim.Adam(m.parameters(), lr=args.lr, eps=args.eps, betas=(args.beta1, args.beta2))

if len(args.ckpt) > 0 and os.path.exists(args.ckpt):
    print("loading checkpoint from {}".format(args.ckpt))
    ckpt = torch.load(args.ckpt)
    m.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])

if torch.cuda.is_available():
    m.cuda()
    crit.cuda()

@optimize
def train_batch(inds, optimizers=None):
    atts = train_data.atts_matrix[inds]
    logits = m(train_data.embeds[inds])
    return torch.sum(crit(logits, atts))


def val_batch():
    atts = val_data.atts_matrix
    logits = m(val_data.embeds)
    val_loss = torch.sum(crit(logits, atts))
    preds = crit.predict(logits)
    acc_table = evaluate_accuracy(preds, atts.cpu().data.numpy())
    acc_table['loss'] = val_loss.cpu().data.numpy()[None,:]
    return acc_table


last_best_epoch = 1
prev_best = 100000
for epoch in range(1, 101):
    train_l = []

    m.eval()
    acc_table = val_batch()
    print("--- \n E{:2d} (VAL) \n {} \n --- \n".format(
        epoch,
        acc_table,
    ), flush=True)

    if acc_table['loss'].values[0] < prev_best:
        prev_best = acc_table['loss'].values[0]
        last_best_epoch = epoch
    else:
        if last_best_epoch < (epoch - 5):
            print("Early stopping at epoch {}".format(epoch))
            break

    m.train()
    start_epoch = time.time()
    inds = torch.randperm(len(train_data)).cuda()
    for b in range(inds.size(0) // args.batch_size):
        b_inds = inds[b*args.batch_size:(b+1)*args.batch_size]

        start = time.time()
        l = train_batch(b_inds, optimizers=[optimizer])
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

# Now get the test results
acc_table = evaluate_accuracy(crit.predict(m(test_data.embeds)),
                              test_data.atts_matrix.cpu().data.numpy())
acc_table.to_csv(os.path.join(args.save_dir, 'results.csv'))

torch.save({
            'args': args.args,
            'epoch': epoch,
            'm_state_dict': m.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.save_dir, 'ckpt_{}.tar'.format(epoch)))