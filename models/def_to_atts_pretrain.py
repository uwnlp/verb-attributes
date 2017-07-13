"""
Script to pretrain the LSTM -> linear model for definition to attributes
"""

from data.dictionary_dataset import DictionaryChallengeDataset
from lib.bucket_iterator import DictionaryChallengeIter
from config import ModelConfig
from lib.att_prediction import DictionaryModel
from torch import optim
import os
import torch
from lib.misc import CosineRankingLoss, optimize, cosine_ranking_loss, get_cosine_ranking
import numpy as np
import time
from torch.nn.utils.rnn import pad_packed_sequence

# Recommended hyperparameters
args = ModelConfig(margin=0.1, lr=1e-4, batch_size=64, eps=1e-8,
                   save_dir='def2atts_pretrain', dropout=0.2)

train_data, val_data = DictionaryChallengeDataset.splits()
train_iter = DictionaryChallengeIter(train_data, batch_size=args.batch_size)
val_iter = DictionaryChallengeIter(val_data, batch_size=args.batch_size * 10, sort=False,
                                   shuffle=False)

m = DictionaryModel(train_data.fields['text'].vocab, 300)
optimizer = optim.Adam(m.parameters(), lr=args.lr, eps=args.eps, betas=(args.beta1, args.beta2))

crit = CosineRankingLoss(size_average=True, margin=args.margin)

if torch.cuda.is_available():
    m.cuda()
    crit.cuda()
    train_data.embeds = train_data.embeds.cuda()
    val_data.embeds = val_data.embeds.cuda()


@optimize
def train_batch(word_inds, defns, optimizers=None):
    pred_embs = m(defns)
    return crit(pred_embs, train_data.embeds[word_inds])


def deploy(word_inds, defns):
    pred_embs = m(defns)
    cost, correct_contrib, inc_contrib = cosine_ranking_loss(pred_embs,
                                                             val_data.embeds[word_inds],
                                                             margin=args.margin)
    cost = cost.data.cpu().numpy()
    correct_contrib = correct_contrib.data.cpu().numpy()
    rank, ranking = get_cosine_ranking(pred_embs, val_data.embeds, word_inds)
    return cost, correct_contrib, rank, ranking


def log_val(word_inds, defns, cost, rank, ranking, num_ex=10):
    print("mean rank {:.1f}------------------------------------------".format(np.mean(rank)))

    engl_defns, ls = pad_packed_sequence(defns, batch_first=True)
    spacing = np.linspace(len(ls) // num_ex, len(ls), endpoint=False, num=num_ex, dtype=np.int64)

    engl_defns = [' '.join([val_data.fields['text'].vocab.itos[x] for x in d[1:(l - 1)]])
                  for d, l in zip(engl_defns.cpu().data.numpy()[spacing], [ls[s] for s in spacing])]
    top_scorers = [[val_data.fields['label'].vocab.itos[x] for x in t]
                   for t in ranking[spacing, :3]]
    words = [val_data.fields['label'].vocab.itos[wi] for wi in word_inds.cpu().numpy()[spacing]]

    for w, (word, rank_, top3, l, defn) in enumerate(
            zip(words, rank, top_scorers, cost, engl_defns)):
        print("w{:2d}/{:2d}, R{:5d} {:>30} ({:.3f}){:>13}: {}".format(
            w, 64, rank_, ' '.join(top3), l, word, defn))
    print("------------------------------------------------------------")


last_best_epoch = 1
prev_best = 0.0
for epoch in range(1, 101):
    val_l = []
    val_l_correct = []
    train_l = []

    m.eval()
    for val_b, (word_inds, defns) in enumerate(val_iter):
        cost, correct_contrib, rank, ranking = deploy(word_inds, defns)

        val_l.append(np.mean(cost))
        val_l_correct.append(np.mean(correct_contrib))

        if val_b == 0:
            log_val(word_inds, defns, cost, rank, ranking, num_ex=10)

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
    for b, (word_inds, defns) in enumerate(train_iter):
        start = time.time()
        l = train_batch(word_inds, defns, optimizers=[optimizer])

        train_l.append(l)

        dur = time.time() - start
        if b % 1000 == 0 and b >= 100:
            print("e{:2d}b{:5d} Cost {:.3f} , {:.3f} s/batch".format(
                epoch, b,
                np.mean(train_l),
                dur,
            ), flush=True)
    dur_epoch = time.time() - start_epoch
    print("Duration of epoch was {:.3f}/batch, overall loss was {:.3f}".format(
        dur_epoch / b,
        np.mean(train_l),
    ))
    torch.save({
        'args': args.args,
        'epoch': epoch,
        'm_state_dict': m.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, os.path.join(args.save_dir, 'ckpt_{}.tar'.format(epoch)))
