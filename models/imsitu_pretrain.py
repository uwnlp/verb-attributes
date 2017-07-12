"""
Pretrains the imsitu model.
"""
from data.imsitu_loader import ImSitu, CudaDataLoader
from config import ModelConfig
from torch import optim
import os
import torch
from lib.misc import CosineRankingLoss, optimize, cosine_ranking_loss, get_ranking
import numpy as np
import time
from torch.nn.utils.rnn import pad_packed_sequence
from lib.imsitu_model import ImsituModel
import pandas as pd
import random
from tqdm import tqdm

# Recommended hyperparameters
args = ModelConfig(lr=1e-4, batch_size=32, eps=1e-8, save_dir='imsitu_pretrain')

train_data, val_data, test_data = ImSitu.splits(zeroshot=False)

train_iter, val_iter, test_iter = CudaDataLoader.splits(
    train_data, val_data, test_data, batch_size=args.batch_size, num_workers=2)

m = ImsituModel(
    zeroshot=False,
    num_train_classes=train_data.attributes.atts_matrix.size(0),
)
optimizer = optim.Adam(m.parameters(), lr=args.lr, eps=args.eps, betas=(args.beta1, args.beta2))

crit = torch.nn.CrossEntropyLoss(size_average=True)
if torch.cuda.is_available():
    m.cuda()
    crit.cuda()


@optimize
def train_batch(x, labels, optimizers=None):
    return crit(m(x), labels)


def deploy(x, labels):
    pred = m(x)
    loss = crit(pred, labels)

    values, bests = pred.topk(pred.size(1), dim=1)
    _, ranking = bests.topk(bests.size(1), dim=1, largest=False)   # [batch_size, dict_size]
    rank = torch.gather(ranking.data, 1, labels.data[:, None]).cpu().numpy().squeeze()

    top5_preds = bests[:, :5].cpu().data.numpy()

    top1_acc = np.mean(rank==1)
    top5_acc = np.mean(rank<5)
    return loss.data[0], top1_acc, top5_acc

last_best_epoch = 1
prev_best = 0.0
for epoch in range(1, 50):
    train_l = []
    val_info = []
    m.eval()
    start_epoch = time.time()
    for val_b, (img_batch, label_batch) in enumerate(val_iter):
        val_info.append(deploy(img_batch, label_batch))

    val_info = pd.DataFrame(np.stack(val_info,0),
                            columns=['loss', 'top1_acc', 'top5_acc']).mean(0)

    print("--- E{:2d} VAL ({:.3f} s/batch) \n {}".format(
        epoch,
        (time.time() - start_epoch)/(len(val_iter) * val_iter.batch_size / train_iter.batch_size),
        val_info), flush=True)

    if val_info['top5_acc'] > prev_best:
        prev_best = val_info['top5_acc']
        last_best_epoch = epoch
    else:
        if last_best_epoch < (epoch - 3):
            print("Early stopping at epoch {}".format(epoch))
            break
    m.train()
    start_epoch = time.time()
    for b, (img_batch, label_batch) in enumerate(train_iter):
        l = train_batch(img_batch, label_batch, optimizers=[optimizer])
        train_l.append(l)
        if b % 100 == 0 and b >= 100:
            print("e{:2d}b{:5d} Cost {:.3f} , {:.3f} s/batch".format(
                epoch, b,
                np.mean(train_l),
                (time.time() - start_epoch) / (b+1),
            ), flush=True)
    print("overall loss was {:.3f}".format(np.mean(train_l)))

torch.save({
    'args': args.args,
    'epoch': epoch,
    'm_state_dict': m.state_dict(),
    'optimizer': optimizer.state_dict(),
}, os.path.join(args.save_dir, 'ckpt_{}.tar'.format(epoch)))
