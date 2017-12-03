"""
Contains losses for attribute prediction, along with evaluation.
"""

import torch
from lib.bce_loss import binary_cross_entropy_with_logits
import numpy as np
from torch import nn
import pandas as pd
from data.attribute_loader import COLUMNS
from torch.nn import functional as F

class AttributeLoss(nn.Module):
    def __init__(self, att_domains, size_average=True):
        super(AttributeLoss, self).__init__()
        self.att_domains = att_domains
        self.size_average = size_average

    @property
    def domains_per_att(self):
        return [1 if x[1] == 2 else x[1] for x in self.att_domains]

    @property
    def input_size(self):
        return sum(self.domains_per_att)

    def forward(self, input_data, gt_atts):
        """
        Computes the prediction loss
        :param input_data: Predicted attribute logits of size (batch_size, sum(domains))
        :param gt_atts: Ground truth attributes of size (batch_size, len(domains))
        :return: A len(attributes) vector of losses.
        """
        if not input_data.size(1) == self.input_size:
            raise ValueError("Input is of wrong size {} vs (batch_size, {})".format(
                input_data.size(), self.input_size))

        loss_per_att = []
        start_col = 0
        for gt_col, d_size in enumerate(self.domains_per_att):
            if d_size == 1:
                loss_per_att.append(binary_cross_entropy_with_logits(
                    input_data[:, start_col],
                    gt_atts[:, gt_col],
                    size_average=self.size_average,
                ))
            else:
                loss_per_att.append(F.cross_entropy(
                    input_data[:, start_col:(start_col + d_size)],
                    gt_atts[:, gt_col],
                    size_average=self.size_average,
                ))
            start_col += d_size
        assert start_col == input_data.size(1)
        return torch.cat(loss_per_att)

    def predict(self, input_data):
        """ 
        Computes prediction from logits
        :param input_data: Predicted logits of size (batch_size, sum(domains))
        :return: Predicted attributes, of size (batch_size, len(domains))
        """
        if not input_data.size(1) == self.input_size:
            raise ValueError("Input is of wrong size {} vs (batch_size, {})".format(
                input_data.size(), self.input_size))

        input_data_np = input_data.cpu().data.numpy()
        predictions = []
        start_col = 0
        for gt_col, d_size in enumerate(self.domains_per_att):
            if d_size == 1:
                predictions.append(
                    np.array(input_data_np[:, start_col] > 0, dtype=np.int64)
                )
            else:
                predictions.append(
                    input_data_np[:, start_col:(start_col + d_size)].argmax(1)
                )
            start_col += d_size
        return np.column_stack(predictions)


def evaluate_accuracy(preds, gt_atts):
    """
    Evaluates the accuracy of 
    :param preds: [num_templates, len(columns)] array of predicted attributes
    :param atts: [num_templates, len(columns)] array of gt attributes 
    :return: 
    """
    accs = (preds == gt_atts).mean(0)*100
    acc_df = pd.DataFrame(data=accs[None, :], columns=COLUMNS)
    acc_micro = acc_df.mean(1).values[0]

    body = acc_df[['bodyparts_Arms', 'bodyparts_Head', 'bodyparts_Legs', 'bodyparts_Torso',
                   'bodyparts_other']].mean(1).values[0]
    duration = acc_df['time'].values[0]
    aspect = acc_df['atomicity'].values[0]
    motion = acc_df['energy'].values[0]
    social = acc_df['solitary'].values[0]
    effect = acc_df[['intrans_effect_0', 'intrans_effect_1', 'intrans_effect_2',
                     'intrans_effect_3', 'trans_obj_effect_0', 'trans_obj_effect_1',
                     'trans_obj_effect_2', 'trans_obj_effect_3', 'trans_pers_effect_0',
                     'trans_pers_effect_1', 'trans_pers_effect_2', 'trans_pers_effect_3']].mean(
        1).values[0]
    trans = acc_df[['intrans', 'trans_pers', 'trans_obj']].mean(1).values[0]

    acc_macro = np.mean([body, duration, aspect, motion, social, effect, trans])

    results_table = pd.DataFrame(
        data=np.array([acc_macro, acc_micro, body, duration,
                       aspect, motion, social, effect, trans])[None,:],
        columns=['acc-macro','acc-micro','body',
                 'duration','aspect','motion','social','effect','trans'],
        index=[0],
    )
    return results_table
