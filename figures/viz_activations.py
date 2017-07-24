"""
Visualization for fig4

warning. terrible monkeypatched code!
"""
import matplotlib
from data.imsitu_loader import ImSitu, CudaDataLoader
from config import ModelConfig
import torch
from lib.misc import get_ranking
import numpy as np
from lib.imsitu_model import ImsituModel
import pandas as pd
from tqdm import tqdm
from lib.attribute_loss import AttributeLoss
from copy import deepcopy
from torch.autograd import Variable
from data.attribute_loader import Attributes
from config import ATTRIBUTES_SPLIT, ATTRIBUTES_PATH
from scipy.misc import imsave
import matplotlib.pyplot as plt
from PIL import Image
from subprocess import call
from torch.nn import functional as F
from lib.imsitu_model import dap_deploy, ours_deploy, devise_deploy, ours_logits
import pickle as pkl

def _gi(self, index):
    fn, ind = self.examples[index]
    img = self.transform(Image.open(fn).convert('RGB'))
    return img, ind, fn

ImSitu.__getitem__ = _gi

train_data, val_data, test_data = ImSitu.splits(zeroshot=True, test_full=True)

def _load(self, item):
    img = Variable(item[0], volatile=self.volatile)
    label = Variable(item[1], volatile=self.volatile)
    if torch.cuda.is_available():
        img = img.cuda()
        label = label.cuda()
    return img, label, item[2]

CudaDataLoader._load = _load

def collate_fn(data):
    imgs, labels, fns = zip(*data)
    imgs = torch.stack(imgs, 0)
    labels = torch.LongTensor(labels)
    return imgs, labels, fns

test_iter = CudaDataLoader(
            dataset=test_data,
            batch_size=16,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn,
            volatile=True,
        )
att_crit = AttributeLoss(train_data.attributes.domains, size_average=True)

if torch.cuda.is_available():
    test_data.attributes.cuda()
    att_crit.cuda()
# Recommended hyperparameters
args = ModelConfig(lr=2e-5, batch_size=32, eps=1e-8,
                   imsitu_model='ours', l2_weight=1e-2,
                   use_att=True, use_emb=True, ckpt='imsitu_ours/embatt/ckpt_7.tar',
                   )
m = ImsituModel(
    zeroshot=True,
    embed_dim=300 if args.use_emb else None,
    att_domains=att_crit.domains_per_att if args.use_att else None,
    l2_weight=args.l2_weight,
)
m.load_state_dict(torch.load(args.ckpt)['m_state_dict'])
m.eval()
if torch.cuda.is_available():
    test_data.attributes.cuda()
    att_crit.cuda()
    m.cuda()

def ours_logits(m, x):
    res = m(x)

    if (res.att_pred is not None) and (att_crit is None):
        raise ValueError("Attribute learning incomplete")
    embed_logits = res.embed_pred @ test_data.attributes.embeds.t()

    # We need a matrix of [att_dim, labels] now with everything +1 or -1
    att_mat = (-1) * torch.ones(att_crit.input_size, test_data.attributes.atts_matrix.size(0)).cuda()
    start_col = 0
    for gt_col, d_size in enumerate(att_crit.domains_per_att):
        if d_size == 1:
            att_mat[start_col] = test_data.attributes.atts_matrix[:, gt_col].data.float() * 2 - 1.0
        else:
            att_mat[start_col:(start_col + d_size)].scatter_(
                0, test_data.attributes.atts_matrix[:, gt_col].data[None, :], 1.0)
        start_col += d_size
    att_mat = Variable(att_mat, volatile=True)

    # For each attribute dot product between att + example and att + label matrix affinitys
    att_aff = res.att_pred.t()[:,:,None]
    affinities = torch.bmm(att_aff, att_mat[:,None,:]) # number of atts, batch size, labels

    return embed_logits, affinities

# Don't take the mean until the end
datoms = []
for img_batch, label_batch, fns in tqdm(test_iter):
    batch_inds = np.arange(img_batch.size(0))
    gt_label = label_batch.data.cpu().numpy()

    embed_contrib, att_contrib = ours_logits(m, img_batch)

    preds = att_contrib.sum(0).squeeze() + embed_contrib
    sm = F.softmax(preds).data.cpu().numpy()
    pred_label = sm.max(1)

    for i, (gt, pred, fn, s) in enumerate(zip(gt_label, pred_label, fns, sm)):
        if i % 10 == 0:
            print("labels gt {} fn {}".format(test_data.attributes.atts_df.index[gt], fn))

        if gt == pred:
            datoms.append((fn, gt, s))
        elif np.random.rand() < 0.1:
            datoms.append((fn, gt, s))
with open('cache.pkl', 'wb') as f:
    pkl.dump((datoms, test_data.attributes.atts_df.index), f)

