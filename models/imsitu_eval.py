"""
Trains the imsitu model for ZSL
"""
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

train_data, val_data, test_data = ImSitu.splits(zeroshot=True, test_full=True)
_, _, test_iter = CudaDataLoader.splits(
    train_data, val_data, test_data, batch_size=32, num_workers=2)
att_crit = AttributeLoss(train_data.attributes.domains, size_average=True)

stacked_data = deepcopy(test_data)
stacked_data.attributes.atts_matrix = torch.cat((
    train_data.attributes.atts_matrix,
    val_data.attributes.atts_matrix,
    test_data.attributes.atts_matrix,
), 0)
stacked_data.attributes.embeds = torch.cat((
    train_data.attributes.embeds,
    val_data.attributes.embeds,
    test_data.attributes.embeds,
), 0)

if torch.cuda.is_available():
    test_data.attributes.cuda()
    stacked_data.attributes.cuda()
    att_crit.cuda()


def eval(model, ckpt, use_att=False, use_emb=False, fullvocab=False):
    # Recommended hyperparameters
    args = ModelConfig(lr=2e-5, batch_size=32, eps=1e-8,
                       imsitu_model=model, l2_weight=1e-2,
                       use_att=use_att, use_emb=use_emb, ckpt=ckpt,
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
        m.cuda()
    offset = train_data.attributes.atts_matrix.size(0) + val_data.attributes.atts_matrix.size(0)

    from lib.imsitu_model import dap_deploy, ours_deploy, devise_deploy
    deploy_fn = locals()[args.imsitu_model + '_deploy']
    test_update = lambda ib, lb: deploy_fn(m, ib, lb, data=test_data, att_crit=att_crit)
    full_update = lambda ib, lb: deploy_fn(m, ib, lb, data=stacked_data, att_crit=att_crit)

    # Don't take the mean until the end
    probs_ = []
    labels_ = []
    probs_full_ = []
    labels_full_ = []
    for img_batch, label_batch in tqdm(test_iter):
        probs_.append(test_update(img_batch, label_batch))
        labels_.append(label_batch.data)
        probs_full_.append(test_update(img_batch, label_batch))
        labels_full_.append(label_batch.data + offset)

    ranks_np = get_ranking(torch.cat(probs_,0), torch.cat(labels_, 0))[0].cpu().numpy()
    top1_acc = np.mean(ranks_np==0)*100
    top5_acc = np.mean(ranks_np<5)*100

    ranks_np = get_ranking(torch.cat(probs_full_,0), torch.cat(labels_full_, 0))[0].cpu().numpy()
    top1_acc_full = np.mean(ranks_np==0)*100
    top5_acc_full = np.mean(ranks_np<5)*100

    return pd.Series(data=[top1_acc, top5_acc, top1_acc_full, top5_acc_full],
                     index=['top1_acc', 'top5_acc', 'top1_acc_full', 'top5_acc_full'])

if __name__ == '__main__':
    dap_gold = eval('dap', ckpt='imsitu_dap/att/ckpt_11.tar', use_att=True)
    ours_gold = eval('ours', ckpt='imsitu_ours/att/ckpt_2.tar', use_att=True)
    ours_embatt = eval('ours', ckpt='imsitu_ours/embatt/ckpt_3.tar', use_att=True, use_emb=True)
    devise = eval('devise', ckpt='imsitu_devise/ckpt_2.tar', use_emb=True)

    rez = pd.DataFrame([dap_gold, ours_gold, ours_embatt, devise],
                       index=['dap_gold', 'ours_gold', 'ours_embatt', 'devise'])