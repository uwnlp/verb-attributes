"""
Builds an RNN encoder that can go to attributes or word embeddings
"""

import torch.nn as nn
from torchvision import models
import math
import torch
from lib.misc import optimize, _normalize
from torch.nn import functional as F
from lib.misc import get_ranking
from torch.autograd import Variable
import numpy as np
import pandas as pd
from lib.bce_loss import binary_cross_entropy_with_logits
from operator import xor
from collections import namedtuple
ENCODING_SIZE = 2048

ZSResult = namedtuple('ZSResult', ['att_pred', 'embed_pred'])

def _init_fc(fc):
    # This seems to be how resnet linear layers are initialized.
    n = fc.in_features * fc.out_features
    fc.weight.data.normal_(0, math.sqrt(2. / n))
    fc.bias.data.zero_()


def get_pretrained_resnet(new_fc_dim=None):
    """
    Fetches a pretrained resnet model (downloading if necessary) and chops off the top linear
    layer. If new_fc_dim isn't None, then a new linear layer is added.
    :param new_fc_dim: 
    :return: 
    """
    resnet152 = models.resnet152(pretrained=True)
    del resnet152.fc

    if new_fc_dim is not None:
        resnet152.fc = nn.Linear(ENCODING_SIZE, new_fc_dim)

        _init_fc(resnet152.fc)
    else:
        resnet152.fc = lambda x: x

    return resnet152


class ImsituModel(nn.Module):
    def __init__(self, zeroshot, embed_dim=None, att_domains=None, num_train_classes=None, l2_weight=None):
        """
        :param zeroshot: Whether we're running in zeroshot mode (
            can be true or False).
        :param embed_dim: Dimension of embeddings (probably 300)
        :param att_dims: List of domain sizes per attribute.
        :param num_train_classes: If we're doing pretraining, number of classes to use
        """
        super(ImsituModel, self).__init__()

        self.l2_weight = l2_weight
        if zeroshot:
            if (embed_dim is not None) and (att_domains is not None):
                print("Using embeddings and attributes for zeroshot")
            elif embed_dim is not None:
                print("Using embeddings for zeroshot")
            elif att_domains is not None:
                print("using attributes for zeroshot")
            else:
                raise ValueError("Must supply embeddings or attributes for zeroshot")
            self.fc_dim = None
            self.att_domains = att_domains if att_domains is not None else []
            self.embed_dim = embed_dim
        else:
            if num_train_classes is None:
                raise ValueError("Must supply a # of training classes")
            self.fc_dim = num_train_classes
            self.att_domains = []
            self.embed_dim = None

        self.resnet152 = get_pretrained_resnet(self.fc_dim)

        if self.embed_dim is not None:
            self.embed_linear = nn.Linear(ENCODING_SIZE, self.embed_dim)
            _init_fc(self.embed_linear)

        if self.att_dim is not None:
            self.att_linear = nn.Linear(ENCODING_SIZE, self.att_dim)
            _init_fc(self.att_linear)

    @property
    def l2_penalty(self):
        if self.l2_weight is None or self.l2_weight == 0:
            return 0.0
        l2_p = 0.0
        for n, p in self.named_parameters():
            if p.requires_grad:
                l2_p += self.l2_weight*torch.pow(p,2).sum()/2
        return l2_p
    

    @property
    def att_dim(self):
        ad = sum(self.att_domains)
        if ad == 0:
            ad = None
        return ad

    @property
    def is_zeroshot(self):
        return self.fc_dim is None

    def __call__(self, imgs):
        img_feats = self.resnet152(imgs)

        if not self.is_zeroshot:
            return img_feats

        att_res = self.att_linear(img_feats) if self.att_dim is not None else None
        embed_res = self.embed_linear(img_feats) if self.embed_dim is not None else None
        return ZSResult(att_res, embed_res)

    def load_pretrained(self, ckpt_file):
        ckpt = torch.load(ckpt_file)['m_state_dict']
        for k, v in list(self.state_dict().items()):
            if k in ckpt:
                if v.size() == ckpt[k].size():
                    v.copy_(ckpt[k])
                else:
                    print("Size mismatch for {}".format(k))
            else:
                print("{} not found".format(k))

###########################################################################################
# Functions for training the model
###########################################################################################

@optimize
def dap_train(m, x, labels, data, att_crit=None, optimizers=None):
    """
    Train the direct attribute prediction model
    :param m: Model we're using
    :param x: [batch_size, 3, 224, 224] Image input
    :param labels: [batch_size] variable with indices of the right verbs
    :param embeds: [vocab_size, 300] Variables with embeddings of all of the verbs
    :param atts_matrix: [vocab_size, att_dim] matrix with GT attributes of the verbs
    :param att_crit: AttributeLoss module that computes the loss
    :param optimizers: the decorator will use these to update parameters
    :return: 
    """
    res = m(x)

    if xor(att_crit is None, res.att_pred is None):
        raise ValueError("Attribute learning incomplete")

    loss = m.l2_penalty
    if res.embed_pred is not None:
        embed_logits = res.embed_pred @ data.attributes.embeds.t()
        full_labels = torch.zeros(embed_logits.size(0), embed_logits.size(1)).cuda()
        full_labels.scatter_(1, labels.data[:, None], 1.0)
        full_labels = Variable(full_labels)

        loss += binary_cross_entropy_with_logits(embed_logits, full_labels, size_average=True)

    if res.att_pred is not None:
        loss += att_crit(res.att_pred, data.attributes.atts_matrix[labels.data]).sum()
    return loss


def dap_deploy(m, x, labels, data, att_crit=None):
    """
    Deploy DAP
    :param m: 
    :param x: 
    :param labels: 
    :param data: 
    :param att_crit: 
    :return: Pandas series 
    """
    res = m(x)
    if res.embed_pred is not None:
        embed_logits = res.embed_pred @ data.attributes.embeds.t()
        att_probs = [torch.sigmoid(embed_logits)]
    else:
        att_probs = []

    # Start off with the embedding probabilities
    if res.att_pred is None:
        domains = []
    else:
        domains = att_crit.domains_per_att

    start_col = 0
    for gt_col, d_size in enumerate(domains):

        # Get the attributes per verb
        atts_by_verb = data.attributes.atts_matrix[:, gt_col]
        if d_size == 1:

            # Get the right indexing by taking the outer product between the
            # [batch_size] attributes \in {+1, -1} and the logits
            # This gives us a [batch_size x num_labels] matrix.
            raw_ap = torch.ger(
                res.att_pred[:, start_col],
                2*(atts_by_verb.float() - 0.5),
            )
            att_probs.append(torch.sigmoid(raw_ap))
        else:
            # [batch_size x attribute domain_size] matrix
            ap = F.softmax(res.att_pred[:, start_col:(start_col+d_size)])

            #[batch_size x num_labels]
            prob_contrib_by_label = torch.index_select(ap, 1, atts_by_verb)
            att_probs.append(prob_contrib_by_label)

        start_col += d_size

    #[batch_size x num labels x num attributes]
    probs_by_att = torch.stack(att_probs, 2)
    # [batch_size, range size]
    probs_prod = torch.prod(probs_by_att + 1e-12, 2)
    denom = probs_prod.sum(1)[:,None]  # [batch_size, 1]
    probs = probs_prod / denom
    return probs

###

def ours_logits(m, x, data, att_crit=None):
    res = m(x)

    if (res.att_pred is not None) and (att_crit is None):
        raise ValueError("Attribute learning incomplete")

    logits = []
    if res.embed_pred is not None:
        embed_logits = res.embed_pred @ data.attributes.embeds.t()
        logits.append(embed_logits)

    if res.att_pred is not None:
        # We need a matrix of [att_dim, labels] now with everything +1 or -1
        att_mat = (-1) * torch.ones(att_crit.input_size, data.attributes.atts_matrix.size(0)).cuda()
        start_col = 0
        for gt_col, d_size in enumerate(att_crit.domains_per_att):
            if d_size == 1:
                att_mat[start_col] = data.attributes.atts_matrix[:, gt_col].data.float() * 2 - 1.0
            else:
                att_mat[start_col:(start_col + d_size)].scatter_(
                    0, data.attributes.atts_matrix[:, gt_col].data[None, :], 1.0)
            start_col += d_size

        att_mat = Variable(att_mat)
        att_logits = res.att_pred @ att_mat
        logits.append(att_logits)
    return logits

@optimize
def ours_train(m, x, labels, data, att_crit=None, optimizers=None):
    """
    Train the direct attribute prediction model
    :param m: Model we're using
    :param x: [batch_size, 3, 224, 224] Image input
    :param labels: [batch_size] variable with indices of the right verbs
    :param embeds: [vocab_size, 300] Variables with embeddings of all of the verbs
    :param atts_matrix: [vocab_size, att_dim] matrix with GT attributes of the verbs
    :param att_crit: AttributeLoss module that computes the loss
    :param optimizers: the decorator will use these to update parameters
    :return: 
    """
    logits = ours_logits(m, x, data, att_crit=att_crit)
    loss = m.l2_penalty
    if len(logits) == 1:
        loss += F.cross_entropy(logits[0], labels, size_average=True)
    else:
        sum_logits = sum(logits)
        for l in logits:
            loss += F.cross_entropy(l, labels, size_average=True)/(len(logits)+1)
        loss += F.cross_entropy(sum_logits, labels, size_average=True)/(len(logits)+1)
    return loss


def ours_deploy(m, x, labels, data, att_crit=None):
    """
    Deploy DAP
    :param m: 
    :param x: 
    :param labels: 
    :param data: 
    :param att_crit: 
    :return: Pandas series 
    """
    logits = sum(ours_logits(m, x, data, att_crit=att_crit))
    probs = F.softmax(logits)
    return probs


# ---------------

@optimize
def devise_train(m, x, labels, data, att_crit=None, optimizers=None):
    """
    Train the direct attribute prediction model
    :param m: Model we're using
    :param x: [batch_size, 3, 224, 224] Image input
    :param labels: [batch_size] variable with indices of the right verbs
    :param embeds: [vocab_size, 300] Variables with embeddings of all of the verbs
    :param atts_matrix: [vocab_size, att_dim] matrix with GT attributes of the verbs
    :param att_crit: AttributeLoss module that computes the loss
    :param optimizers: the decorator will use these to update parameters
    :return: 
    """
    # Make embed unit normed
    embed_normed = _normalize(data.attributes.embeds)
    mv_image = m(x).embed_pred
    tmv_image = mv_image @ embed_normed.t()

    # Use a random label from the same batch
    correct_contrib = torch.gather(tmv_image, 1, labels[:,None])

    # Should be fine to ignore where the correct contrib intersects because the gradient
    # wrt input is 0
    losses = (0.1 + tmv_image - correct_contrib.expand_as(tmv_image)).clamp(min=0.0)
    # losses.scatter_(1, labels[:, None], 0.0)
    loss = m.l2_penalty + losses.sum(1).mean()
    return loss


def devise_deploy(m, x, labels, data, att_crit=None):
    """
    Deploy DAP
    :param m: 
    :param x: 
    :param labels: 
    :param data: 
    :param att_crit: 
    :return: Pandas series 
    """
    # Make embed unit normed
    embed_normed = _normalize(data.attributes.embeds)
    mv_image = m(x).embed_pred
    probs = mv_image @ embed_normed.t()
    return probs
