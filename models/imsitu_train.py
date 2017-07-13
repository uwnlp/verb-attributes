"""
Trains the imsitu model for ZSL
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
from lib.imsitu_model import ImsituModel, dap_train, dap_deploy, ours_train, ours_deploy, \
    devise_train, devise_deploy
import pandas as pd
import random
from tqdm import tqdm
from lib.attribute_loss import AttributeLoss

# Recommended hyperparameters
args = ModelConfig(lr=5e-5, batch_size=32, eps=1e-8, save_dir='imsitu_trainDEVISE',
                   imsitu_model='ours', use_att=True, use_emb=True)

args.imsitu_model = 'devise'
args.use_att = False

train_data, val_data, test_data = ImSitu.splits(zeroshot=True)

train_iter, val_iter, test_iter = CudaDataLoader.splits(
    train_data, val_data, test_data, batch_size=args.batch_size, num_workers=2)

att_crit = AttributeLoss(train_data.attributes.domains, size_average=True)
m = ImsituModel(
    zeroshot=True,
    embed_dim=300 if args.use_emb else None,
    att_domains=att_crit.domains_per_att if args.use_att else None,
)
m.load_pretrained(
    '/home/rowan/code/verb-attributes/checkpoints/imsitu_pretrain/pretrained_ckpt.tar')
optimizer = optim.Adam(m.parameters(), lr=args.lr, eps=args.eps, betas=(args.beta1, args.beta2))


# Call the function
train_fn = locals()[args.imsitu_model + '_train']
deploy_fn = locals()[args.imsitu_model + '_deploy']

train_update = lambda ib, lb: train_fn(
    m, ib, lb, data=train_data, att_crit=att_crit, optimizers=[optimizer])
val_update = lambda ib, lb: deploy_fn(
    m, ib, lb, data=val_data, att_crit=att_crit)
test_update = lambda ib, lb: deploy_fn(
    m, ib, lb, data=test_data, att_crit=att_crit)

if torch.cuda.is_available():
    m.cuda()
    att_crit.cuda()
    train_data.attributes.cuda()
    val_data.attributes.cuda()
    test_data.attributes.cuda()

last_best_epoch = 1
prev_best = 0.0
for epoch in range(1, 50):
    train_l = []
    val_info = []
    m.eval()
    start_epoch = time.time()
    for val_b, (img_batch, label_batch) in enumerate(tqdm(val_iter)):
        val_info.append(val_update(img_batch, label_batch))
    val_info = pd.DataFrame(val_info).mean()

    print("--- E{:2d} VAL ({:.3f} s/batch) \n {}".format(
        epoch,
        (time.time() - start_epoch) / (len(val_iter) * val_iter.batch_size / train_iter.batch_size),
        val_info), flush=True)

    assert epoch < 10

    if val_info['top5_acc'] > prev_best:
        prev_best = val_info['top5_acc']
        last_best_epoch = epoch
    else:
        if last_best_epoch < (epoch - 2):
            print("Early stopping at epoch {}".format(epoch))
            break
    m.train()
    start_epoch = time.time()
    for b, (img_batch, label_batch) in enumerate(train_iter):
        l = train_update(img_batch, label_batch)
        train_l.append(l)
        if b % 100 == 0 and b >= 100:
            print("e{:2d}b{:5d} Cost {:.3f} , {:.3f} s/batch".format(
                epoch, b,
                np.mean(train_l),
                (time.time() - start_epoch) / (b + 1),
            ), flush=True)
    print("overall loss was {:.3f}".format(np.mean(train_l)))
    torch.save({
        'args': args.args,
        'epoch': epoch,
        'm_state_dict': m.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, os.path.join(args.save_dir, 'ckpt_{}.tar'.format(epoch)))

# test_info = []
# for img_batch, label_batch in test_iter:
#     test_info.append(val_update(img_batch, label_batch))
# test_info = pd.DataFrame(test_info).mean()
