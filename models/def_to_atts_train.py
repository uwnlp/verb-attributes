"""
Script to pretrain the LSTM -> linear model for definition to attributes
"""

from data.dictionary_dataset import load_vocab
from lib.attribute_loss import AttributeLoss, evaluate_accuracy
from lib.bucket_iterator import DictionaryAttributesIter
from config import ModelConfig
from lib.att_prediction import DictionaryModel
from torch import optim
import os
import torch
from lib.misc import optimize
import numpy as np
import time
from data.attribute_loader import Attributes
from lib.misc import print_para
import pandas as pd

# Recommended hyperparameters
args = ModelConfig(margin=0.1, lr=1e-4, batch_size=16, eps=0.1, l2_weight=0.001, dropout=0.5,
                   ckpt='def2atts_pretrain/ckpt_28.tar', save_dir='def2atts_train')
args.use_emb = True

train_data, val_data, test_data = Attributes.splits(use_defns=True, cuda=True)
dict_field, _ = load_vocab()

train_iter = DictionaryAttributesIter(dict_field, train_data, batch_size=args.batch_size)
val_iter = DictionaryAttributesIter(dict_field, val_data, batch_size=args.batch_size*10,
                                    shuffle=False, train=False)
test_iter = DictionaryAttributesIter(dict_field, test_data, batch_size=args.batch_size*10,
                                     shuffle=False, train=False)

crit = AttributeLoss(train_data.domains, size_average=True)
m = DictionaryModel(dict_field.vocab, output_size=crit.input_size, embed_input=args.use_emb,
                    dropout_rate=args.dropout)
m.load_pretrained(args.ckpt)

for name, p in m.named_parameters():
    if name.startswith('embed'):
        p.requires_grad = False

print(print_para(m))
optimizer = optim.Adam([p for p in m.parameters() if p.requires_grad],
                       lr=args.lr, eps=args.eps, betas=(args.beta1, args.beta2))

# if len(args.ckpt) > 0 and os.path.exists(args.ckpt):
#     print("loading checkpoint from {}".format(args.ckpt))
#     ckpt = torch.load(args.ckpt)
#     m.load_state_dict(ckpt['state_dict'])
#     optimizer.load_state_dict(ckpt['optimizer'])

if torch.cuda.is_available():
    m.cuda()
    crit.cuda()


@optimize
def train_batch(atts, words, defns, weights, optimizers=None):
    logits = m(defns, words)
    loss = torch.mean(crit(logits, atts, weights=weights))

    for name, p in m.named_parameters():
        if name.startswith('fc.weight'):
            loss += args.l2_weight*torch.pow(p,2).sum()/2
    return loss


def val_batch(atts, words, defns, weights):
    logits = m(defns, words)
    val_loss = torch.mean(crit(logits, atts))
    preds = crit.predict(logits)
    acc_table = evaluate_accuracy(preds, atts.cpu().data.numpy())
    acc_table['loss'] = val_loss.cpu().data.numpy()[None,:]
    return acc_table.T.squeeze()

last_best_epoch = 0
prev_best = 100000
for epoch in range(50):
    val_info = []
    train_l = []

    m.eval()
    for b, (atts, words, defns, weights) in enumerate(val_iter):
        val_info.append(val_batch(atts, words, defns, weights))

    val_info = pd.DataFrame(val_info).mean()
    print("--- \n E{:2d} (VAL) \n {} \n --- \n".format(
        epoch,
        val_info,
    ), flush=True)

    if val_info['loss'] < prev_best:
        prev_best = val_info['loss']
        last_best_epoch = epoch
        torch.save({
            'args': args.args,
            'epoch': epoch,
            'm_state_dict': m.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.save_dir, 'ckpt_{}.tar'.format(epoch)))
    else:
        if last_best_epoch < (epoch - 5):
            print("Early stopping at epoch {}".format(epoch))
            break

    m.train()
    start_epoch = time.time()
    for b, (atts, words, defns, weights) in enumerate(train_iter):
        start = time.time()
        train_l.append(train_batch(atts, words, defns, weights, optimizers=[optimizer]))

        dur = time.time() - start
        if b % 100 == 0 and b >= 100:
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