"""
Data loader for everything.  Word vectors are loaded later.

1. Words -> Attributes (use AttributesDataLoader)
2. Imsitu + Attributes -> imsitu labels (use ImsituAttributesDataLoader
4. Defintiions dataset -> counts (use DefinitionsDataLoader
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import numpy as np
import pandas as pd

from config import ATTRIBUTES_PATH, ATTRIBUTES_SPLIT, IMSITU_VERBS


np.random.seed(123456)
random.seed(123)


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
    def __init__(self, use_train=True, use_val=False, use_test=False, imsitu_only=False):
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

        # perm is a permutation from the normal index to the new one.
        # This can be used for getting the attributes for Imsitu
        self.ind_perm = invert_permutation(self.atts_df['index'].as_matrix())





# class AttributesDataLoader(object):
#     def __init__(self, batch_size=64, filter_imsitu=False):
#         """
#         things_to_load is a list or tuple of strings with datasets
#         """
#         self.nogroup = nogroup
#         self.batch_size = batch_size
#         self.max_val_batches = max_val_batches
#         # Keys:
#         self.skip_columns = ('verb', 'train', 'test', 'val', 'template')
#
#         # Attributes is a pandas dataframe with columns
#         # ('verb','train','test','val', 'in_imsitu', 'template')
#         # perm is a permutation from the normal index to the new one (useful for imsitu maybe).
#         self._att_sorted, self.ind_perm = argsort_pandas(
#             load_attribute_split(just_imsitu=filter_imsitu, nogroup=nogroup),
#             ['test', 'val', 'train'],
#         )
#
#         self.all_templates, self.train_templates, self.val_templates, self.test_templates, \
#             self.attributes = get_template_split(self._att_sorted, nogroup=nogroup)
#
#         self.domains = [(c, len(self.attributes[c].unique())) for c in self.attributes.columns]
#         self.epochs_seen = -1
#
#         # For continuous loading
#         self._cur_train_state = None
#         self.full_templates = list(self.attributes.index)
#
#     def train_reader(self):
#         """
#         Continual updating of train
#         :return:
#         """
#         if self._cur_train_state is None:
#             self._cur_train_state = self.train_epoch()
#         elem = next(self._cur_train_state, None)
#         if elem is None:
#             self._cur_train_state = self.train_epoch()
#             elem = next(self._cur_train_state, None)
#             assert elem is not None, "We just checked for none"
#         return elem
#
#     @property
#     def val_size(self):
#         return self.batch_size * self.num_val_batches
#
#     @property
#     def _max_val_size(self):
#         return self.batch_size * self.max_val_batches
#
#     def num_batches(self, train_mode):
#         m_to_batches = {
#             'train': len(self.train_templates) // self.batch_size,
#             'val': len(self.val_templates) // self.batch_size,
#             'test': len(self.test_templates) // self.batch_size,
#         }
#         return m_to_batches[train_mode]
#
#     @property
#     def num_train_batches(self):
#         return self.num_batches('train')
#
#     @property
#     def num_val_batches(self):
#         return self.num_batches('val')
#
#     @property
#     def num_test_batches(self):
#         return self.num_batches('test')
#
#     def epoch(self, templates, skip_end=False):
#         """ Return the templates, data"""
#         for b_start, b_end in batch_index_iterator(len(templates), self.batch_size, skip_end=skip_end):
#             inds, ts = zip(*templates[b_start:b_end])
#             inds = np.array(inds, dtype=np.float32)
#             #data = self.attributes.loc[temps].as_matrix().astype(np.int32)
#             yield inds, ts
#
#     def test_epoch(self):
#         """
#         :return: A generator providing all needed for a train epoch
#         """
#         offset = len(self.train_templates) + len(self.val_templates)
#         return self.epoch([(i+offset, x) for i,x in enumerate(self.test_templates)])
#
#     def val_epoch(self):
#         """
#         :return: A generator providing all needed for a train epoch
#         """
#         offset = len(self.train_templates)
#         return self.epoch([(i + offset,x) for i,x in enumerate(self.val_templates)])
#
#     def train_epoch(self):
#         """
#         :return: A generator providing all needed for a train epoch
#         """
#         self.epochs_seen += 1
#         keys = [(i, self.train_templates[i]) for i in np.random.permutation(len(self.train_templates))]
#         return self.epoch(keys, skip_end=True)
#
#     @property
#     def num_verbs(self):
#         return len(self.all_templates)
#
#     @property
#     def num_train(self):
#         if self.nogroup:
#             return len(self.train_templates)*3
#         return len(self.train_templates)
#     @property
#     def num_val(self):
#         return len(self.val_templates)
#
#     @property
#     def num_test(self):
#         return len(self.test_templates)
#
#
#
# class DefnsDataLoader(AttributesDataLoader):
#     def __init__(self, vocab, config, correct_bias=True):
#         """
#         Loads syntactic n-grams
#         """
#         super(DefnsDataLoader, self).__init__(batch_size=config.batch_size,
#                                               max_val_batches=config.val_batches,
#                                               filter_imsitu=False)
#         self.correct_bias = correct_bias
#         self.vocab = vocab
#         self.train_dps, self.val_dps, self.test_dps = self.load()
#         self.max_len = 20
#
#
#     def num_batches(self, train_mode):
#         m_to_batches = {
#             'train': len(self.train_dps) // self.batch_size,
#             'val': (len(self.val_dps)+ self.batch_size - 1) // self.batch_size,
#             'test': (len(self.test_dps) + self.batch_size - 1) // self.batch_size,
#         }
#         return m_to_batches[train_mode]
#
#     def epoch(self, all_obs, skip_end=False, augment=False):
#         """ Return the all_obs, data"""
#         for b_start, b_end in batch_index_iterator(len(all_obs), self.batch_size, skip_end):
#             raw_verbs, verb_inds, defns = zip(*all_obs[b_start:b_end])
#
#             verbs = self.vocab.apply_list(raw_verbs, max_len=1)
#             defns = self.vocab.apply_list(defns, max_len=self.max_len)
#
#             if augment:
#                 defns = augment_defns(defns)
#
#             verb_inds = np.array(verb_inds, dtype=np.int32)
#             yield raw_verbs, verb_inds, verbs, defns
#
#     def test_epoch(self):
#         """
#         :return: A generator providing all needed for a train epoch
#         """
#         return self.epoch(self.test_dps)
#
#     def val_epoch(self):
#         """
#         :return: A generator providing all needed for a train epoch
#         """
#         return self.epoch(self.val_dps)
#
#     def train_epoch(self):
#         """
#         :return: A generator providing all needed for a train epoch
#         """
#         self.epochs_seen += 1
#         keys = [self.train_dps[i] for i in np.random.permutation(len(self.train_dps))]
#         return self.epoch(keys, skip_end=True, augment=True)
#
#     def load(self):
#         word_to_defns = load_word_to_defns(self.all_templates)
#
#         train_dps = data_to_list(self.train_templates, word_to_defns, oversample=self.correct_bias)
#
#         val_dps = data_to_list(self.val_templates, word_to_defns, self.num_train, oversample=self.correct_bias)
#         test_dps = data_to_list(self.test_templates, word_to_defns, self.num_train + self.num_val,
#                                 just_one=self.correct_bias)
#
#         random.shuffle(val_dps)
#         random.shuffle(test_dps)
#         val_dps = val_dps[:self._max_val_size]
#
#         if not self.vocab.load():
#             print("Fitting vocabulary, this could take a while")
#             text_to_fit = [unicode(x) for def_block in word_to_defns.itervalues()
#                            for x in def_block]
#             self.vocab.fit(text_to_fit, {x: u'VB' for x in word_to_defns.iterkeys()})
#             self.vocab.save()
#             print("done fitting vocab")
#         else:
#             print("Vocabulary loaded")
#
#         return train_dps, val_dps, test_dps
#
#
# class ImsituAttributesDataLoader(AttributesDataLoader):
#     def __init__(self, config, use_pred_test_attributes = False):
#         """
#         Loads imsitu attributes. Mode has to be "normal", "zeroshot_transfer", or "zeroshot_train"
#         """
#         super(ImsituAttributesDataLoader, self).__init__(batch_size=config.batch_size,
#                                                          max_val_batches=config.val_batches,
#                                                          filter_imsitu=True)
#         assert config.mode in ('normal', 'zeroshot_transfer', 'zeroshot_train')
#         self.mode = config.mode
#         print("Mode is {}".format(self.mode))
#
#         if use_pred_test_attributes:
#             self._load_preds()
#
#
#         self.train_dps, self.val_dps, self.test_dps = self.load()
#         self.val_images, self.val_labels = self.preload_val()
#
#     def _load_preds(self):
#         """ Use the predicted from text attributes for testing."""
#
#         # Escape templates!
#         pred_attributes = pd.read_csv(PRED_ATTRIBUTES_PATH, index_col=0)
#
#         if 'defns' in pred_attributes.columns:
#             pred_attributes = pred_attributes.drop('defns',1)
#         # Le replace of attributes
#         for att in pred_attributes.index:
#             if att in self.attributes.index:
#
#                 # this will be the case if there are multiple templates per verb and if these
#                 # multiple templates didn't have separate definitions
#                 # Since the current classifier pretty much ignores templates anyways we're fine here
#                 if pred_attributes.loc[att].ndim == 2:
#                     self.attributes.loc[att] = pred_attributes.loc[att].iloc[0]
#                 else:
#                     self.attributes.loc[att] = pred_attributes.loc[att]
#
#
#     def num_batches(self, train_mode):
#         m_to_batches = {
#             'train': len(self.train_dps) // self.batch_size,
#             'val': (len(self.val_labels)+self.batch_size-1) // self.batch_size,
#             'test': (len(self.test_dps)+self.batch_size-1) // self.batch_size,
#         }
#         return m_to_batches[train_mode]
#
#     def train_reader(self):
#         return _distorted(self.train_dps, self.batch_size)
#
#     def val_epoch(self):
#         for b_start, b_end in batch_index_iterator(len(self.val_images), self.batch_size):
#             yield self.val_images[b_start:b_end], self.val_labels[b_start:b_end]
#
#     def test_epoch(self):
#         for b_start, b_end in batch_index_iterator(len(self.test_dps), self.batch_size, skip_end=False):
#             fns, labels = zip(*self.test_dps[b_start:b_end])
#             yield load_fns(fns), np.array(labels, dtype=np.int32)
#
#     def _full_train_epoch(self):
#         for b_start, b_end in batch_index_iterator(len(self.train_dps), self.batch_size, skip_end=False):
#             fns, labels = zip(*self.train_dps[b_start:b_end])
#             yield load_fns(fns), np.array(labels, dtype=np.int32)
#
#     def preload_val(self):
#         # Preload validation
#         fns, labels = zip(*self.val_dps[:self._max_val_size])
#         val_images = load_fns(fns)
#         val_labels = np.array(labels, dtype=np.int32)
#         return val_images, val_labels
#
#     def load(self):
#         train_raw = _load_imsitu_file(IMSITU_TRAIN_LIST)
#         test_raw = _load_imsitu_file(IMSITU_TEST_LIST)
#
#         # Do all the random stuff here
#         random.shuffle(train_raw)
#         random.shuffle(test_raw)
#
#         if self.mode == 'normal':  # Ignore attribute splits
#             test_dps = test_raw
#             val_dps = train_raw[:self._max_val_size]
#             train_dps = train_raw[self._max_val_size:]
#         elif self.mode == 'zeroshot_train':
#             train_dps = [(x, self.ind_perm[ind]) for x, ind in train_raw if
#                          self._att_sorted.iloc[self.ind_perm[ind]]['train']]
#             test_dps = [(x, self.ind_perm[ind]) for x, ind in test_raw if
#                         self._att_sorted.iloc[self.ind_perm[ind]]['train']]
#             val_dps = test_dps[:self._max_val_size]
#             test_dps = test_dps[self._max_val_size:]
#         else:  # self.mode == 'zeroshot_transfer':
#             train_dps = [(x, self.ind_perm[ind]) for x, ind in train_raw + test_raw if
#                          self._att_sorted.iloc[self.ind_perm[ind]]['train']]
#             val_dps = [(x, self.ind_perm[ind]) for x, ind in train_raw + test_raw if
#                        self._att_sorted.iloc[self.ind_perm[ind]]['val']][:self._max_val_size]
#             test_dps = [(x, self.ind_perm[ind]) for x, ind in test_raw if
#                         self._att_sorted.iloc[self.ind_perm[ind]]['test']]
#         return train_dps, val_dps, test_dps
