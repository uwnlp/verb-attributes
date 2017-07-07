"""
Data loader for everything.  Word vectors are loaded later.

1. Words -> Attributes (use AttributesDataLoader)
2. Imsitu + Attributes -> imsitu labels (use ImsituAttributesDataLoader
4. Defintiions dataset -> counts (use DefinitionsDataLoader
"""

import os
import random

import numpy as np
import pandas as pd

from config import ATTRIBUTES_PATH, ATTRIBUTES_SPLIT, IMSITU_VERBS, GLOVE, DEFNS_PATH
import torch
from text.torchtext.vocab import load_word_vectors

np.random.seed(123456)
random.seed(123)

COLUMNS = ['intrans', 'trans_pers', 'trans_obj', 'atomicity', 'energy',
       'time', 'solitary', 'bodyparts_Arms', 'bodyparts_Head',
       'bodyparts_Legs', 'bodyparts_Torso', 'bodyparts_other',
       'intrans_effect_0', 'intrans_effect_1', 'intrans_effect_2',
       'intrans_effect_3', 'trans_obj_effect_0', 'trans_obj_effect_1',
       'trans_obj_effect_2', 'trans_obj_effect_3', 'trans_pers_effect_0',
       'trans_pers_effect_1', 'trans_pers_effect_2', 'trans_pers_effect_3']

def invert_permutation(p):
    '''The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1. 
    Returns an array s, where s[i] gives the index of i in p.
    '''
    s = {ind:i for i, ind in enumerate(p)}
    return s


def _load_imsitu_verbs():
    """
    :return: a list of imsitu verbs
    """
    imsitu_verbs = []
    with open(IMSITU_VERBS, 'r') as f:
        for v in f.read().splitlines():
            imsitu_verbs.append(v)
    assert len(imsitu_verbs) == 504
    assert imsitu_verbs == sorted(imsitu_verbs)
    return imsitu_verbs


def _load_attributes(imsitu_only=False):
    """
    :param imsitu_only: if true, only return data for verbs in imsitu
    :return: a pandas dataframe containing attributes along with split info  
    """
    attributes_df = pd.read_csv(ATTRIBUTES_PATH)
    split_df = pd.read_csv(ATTRIBUTES_SPLIT)
    merged_df = pd.merge(split_df, attributes_df, on='verb', how='inner')

    if imsitu_only:
        imsitu_part = merged_df[merged_df['in_imsitu'] & ~merged_df['template'].str.contains(' ')]
        merged_df = imsitu_part.reset_index(drop=True)
        assert len(merged_df.index) == 504

    # Remove the in_imsitu and verb information (only templates are relevant)
    merged_df = merged_df.drop(['in_imsitu', 'verb'], 1)
    return merged_df


def attributes_split(imsitu_only=False):
    """
    :param imsitu_only: if true, only return data for verbs in imsitu
    :return: train, test, val dataframes 
    """
    df = _load_attributes(imsitu_only).reset_index()
    train_atts = df[df['train']].drop(['train','val','test'],1).set_index('template')
    val_atts = df[df['val']].drop(['train','val','test'],1).set_index('template')
    test_atts = df[df['test']].drop(['train','val','test'],1).set_index('template')
    return train_atts, val_atts, test_atts


class Attributes(object):
    def __init__(self, use_train=True, use_val=False, use_test=False, imsitu_only=False,
                       use_defns=False):
        """
        Use this class to represent a chunk of attributes for each of the test labels. 
        This is needed because at test time we'll need to compare against all of the attributes
                
        :param imsitu_only: 
        """
        assert use_train or use_val or use_test

        train_atts, val_atts, test_atts = attributes_split(imsitu_only)
        cat_atts = [a for a, use_a in zip([train_atts, val_atts, test_atts],
                                          [use_train, use_val, use_test]) if use_a]
        self.atts_df = pd.concat(cat_atts)

        # Get the definitions. At test time, we'll get the first definition.
        if use_defns:
            verb_defns = pd.read_csv(DEFNS_PATH)
            if use_test:
                verb_defns = verb_defns.groupby('template').first()





        # perm is a permutation from the normal index to the new one.
        # This can be used for getting the attributes for Imsitu
        self.ind_perm = invert_permutation(self.atts_df['index'].as_matrix())

        self.domains = [(c, len(self.atts_df[c].unique())) for c in COLUMNS]

        self.atts_matrix = torch.LongTensor(self.atts_df[COLUMNS].as_matrix())

        wv_dict, wv_arr, _ = load_word_vectors(GLOVE, 'glove.6B', 300)

        self.embeds = torch.Tensor(self.atts_matrix.size(0), 300).zero_()

        # self.embeds.normal_(0, 1)
        for i, token in enumerate(self.atts_df.index.values):
            wv_index = wv_dict.get(token, None)
            if wv_index is None and len(token.split(' ')) > 1:
                wv_index = wv_dict.get(token.split(' ')[0], None)

            if wv_index is not None:
                self.embeds[i] = wv_arr[wv_index]
            else:
                print("CANT FIND {}".format(token))
