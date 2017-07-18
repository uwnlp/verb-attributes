"""
Script to pretrain the LSTM -> linear model for definition to attributes
"""

from config import ModelConfig
from torch import optim
import os
import torch
from lib.misc import CosineRankingLoss, optimize, cosine_ranking_loss
import numpy as np
import time
from data.attribute_loader import Attributes, COLUMNS, _load_vectors
from lib.att_prediction import FeedForwardModel
from lib.attribute_loss import AttributeLoss, evaluate_accuracy
from torch import nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import math
import pandas as pd
from lib.bce_loss import binary_cross_entropy_with_logits
# Recommended hyperparameters
PREINIT = True

args = ModelConfig(lr=5e-4, batch_size=16, eps=1e-8, save_dir='rtml')
train_data, val_data, test_data = Attributes.splits(use_defns=False,
                                                    cuda=False)
for d in [train_data, val_data, test_data]:
    trans = d.atts_df[['intrans', 'trans_pers', 'trans_obj']]
    trans.columns = ['intransitive', 'transitive', 'transitive']

    atomicity = pd.get_dummies(d.atts_df.atomicity)
    atomicity.columns = ['accomplishment', 'none', 'activity', 'achievement', 'state']

    energy = pd.get_dummies(d.atts_df.energy, prefix='energy')
    energy.columns = ['none', 'no', 'low', 'medium', 'high']

    temporal = pd.get_dummies(d.atts_df.time, prefix='temporal')
    temporal.columns = ['none', 'seconds', 'minutes', 'hours', 'days']

    solitary = pd.get_dummies(d.atts_df.solitary, prefix='solitary')
    solitary.columns = ['solitary', 'solitary', 'either', 'social', 'social']

    bp = d.atts_df[['bodyparts_Arms', 'bodyparts_Head', 'bodyparts_Legs',
                     'bodyparts_Torso', 'bodyparts_other']]
    bp.columns = ['arms', 'head', 'legs', 'torso', 'other']

    effect = d.atts_df[['intrans_effect_0', 'intrans_effect_1',
                         'intrans_effect_2', 'intrans_effect_3',
                         'trans_obj_effect_0', 'trans_obj_effect_1',
                         'trans_obj_effect_2', 'trans_obj_effect_3',
                         'trans_pers_effect_0', 'trans_pers_effect_1',
                         'trans_pers_effect_2', 'trans_pers_effect_3',
                         ]]

    effect.columns = ['motion', 'world', 'state', 'nothing',
                      'motion', 'world', 'state', 'nothing',
                      'motion', 'world', 'state', 'nothing',
                      ]
    att_names = pd.concat((trans, atomicity, energy, temporal, solitary, bp, effect), 1).columns
    dom_sizes = [x.shape[1] for x in (trans, atomicity, energy, temporal, solitary, bp, effect)]
    d.atts_list = [Variable(torch.LongTensor(df.as_matrix().astype(np.int64)).cuda())
                     for df in (trans, atomicity, energy, temporal, solitary, bp, effect)]

class RTML(nn.Module):
    def __init__(self, L=3, lamb=5):
        super(RTML, self).__init__()
        self.L = L
        self.N = len(att_names)
        self.lamb = lamb
        self.theta = Parameter(torch.Tensor(self.L, 300, 300))
        self.alpha = Parameter(torch.Tensor(self.N, self.L+1)) # L+1 is so to parameterize
                                                               # being smaller than norm lamb
        self.reset_parameters()

        self.att_emb = nn.Embedding(self.N, 300)
        if PREINIT:
            self.att_emb.weight.data = _load_vectors(att_names).cuda()
        else:
            _np_emb = np.random.randn(self.N, 300)
            _np_emb = _np_emb / np.square(_np_emb).sum(1)[:, None]
            self.att_emb.weight.data = torch.FloatTensor(_np_emb).cuda()

    def reset_parameters(self):
        for weight in [self.theta, self.alpha]:
            stdv = 1. / math.sqrt(weight.size(1))
            weight.data.uniform_(-stdv, stdv)

    def forward(self, word_embs):
        alpha_norm = self.alpha.abs().sum(1)
        alpha_constrained = self.lamb * self.alpha / alpha_norm.expand_as(self.alpha)

        R_flat = alpha_constrained[:, :-1] @ self.theta.view(self.L, -1)
        R = R_flat.view(self.N, 300, 300)

        s = 0
        preds = []
        for i, att_size in enumerate(dom_sizes):
            e = s + att_size
            att_embs = self.att_emb.weight[s:e].t()
            s = e

            p1 = word_embs @ R[i]
            p2 = p1 @ att_embs
            preds.append(p2)
        preds = torch.cat(preds, 1)
        return preds

m = RTML()
optimizer = optim.Adam(m.parameters(), lr=args.lr, eps=args.eps, betas=(args.beta1, args.beta2))
crit = AttributeLoss(train_data.domains)

if torch.cuda.is_available():
    m.cuda()
    train_data.embeds = train_data.embeds.cuda()
    val_data.embeds = val_data.embeds.cuda()
    test_data.embeds = test_data.embeds.cuda()


@optimize
def train_batch(inds, optimizers=None):
    embs = train_data.embeds[inds]
    atts_list = [a[inds] for a in train_data.atts_list]
    preds = m(embs)
    gt_atts = torch.cat(atts_list, 1)
    loss = binary_cross_entropy_with_logits(preds, gt_atts, size_average=True)
    return loss


def deploy(data):
    embs = data.embeds
    preds = m(embs)
    gt_atts = torch.cat(data.atts_list, 1)
    loss = binary_cross_entropy_with_logits(preds, gt_atts, size_average=True)

    # Now get the test results
    acc_table = evaluate_accuracy(crit.predict(m(data.embeds)),
                                  data.atts_matrix.cpu().data.numpy())
    acc_table['loss'] = loss.cpu().data.numpy()[None,:]
    return acc_table


last_best_epoch = 1
prev_best = 100000
for epoch in range(6):
    train_l = []

    m.eval()
    acc_table = deploy(val_data)
    print("--- \n E{:2d} (VAL) \n {} \n --- \n".format(
        epoch,
        acc_table,
    ), flush=True)

    if acc_table['loss'].values[0] < prev_best:
        prev_best = acc_table['loss'].values[0]
        last_best_epoch = epoch
    else:
        #if last_best_epoch < (epoch - 5):
        print("Early stopping at epoch {}".format(epoch))
        break

    m.train()
    start_epoch = time.time()
    inds = torch.randperm(len(train_data)).cuda()
    for b in range(inds.size(0) // args.batch_size):
        b_inds = inds[b*args.batch_size:(b+1)*args.batch_size]

        start = time.time()
        train_l.append(train_batch(b_inds, optimizers=[optimizer]))

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
acc_table = deploy(test_data)
acc_table.to_csv('rtml-{}.csv'.format('preinit' if PREINIT else 'rand'),
                 float_format='%.2f')