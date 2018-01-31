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

from config import ATTRIBUTES_PATH, ATTRIBUTES_SPLIT, IMSITU_VERBS, GLOVE_PATH, GLOVE_TYPE, \
    DEFNS_PATH, IMSITU_VAL_LIST, DATA_PATH
import torch
from torch.autograd import Variable
import zipfile
from six.moves.urllib.request import urlretrieve
from tqdm import trange, tqdm

### From older torchtext
URL = {
    'glove.42B': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
    'glove.840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
    'glove.twitter.27B': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
    'glove.6B': 'http://nlp.stanford.edu/data/glove.6B.zip'
}

def reporthook(t):
    """https://github.com/tqdm/tqdm"""
    last_b = [0]
    def inner(b=1, bsize=1, tsize=None):
        """
        b: int, optionala
        Number of blocks just transferred [default: 1].
        bsize: int, optional
        Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
        Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return inner

def load_word_vectors(root, wv_type, dim):
    """Load word vectors from a path, trying .pt, .txt, and .zip extensions."""
    if isinstance(dim, int):
        dim = str(dim) + 'd'
    fname = os.path.join(root, wv_type + '.' + dim)
    if os.path.isfile(fname + '.pt'):
        fname_pt = fname + '.pt'
        print('loading word vectors from', fname_pt)
        return torch.load(fname_pt)
    if os.path.isfile(fname + '.txt'):
        fname_txt = fname + '.txt'
        cm = open(fname_txt, 'rb')
        cm = [line for line in cm]
    elif os.path.basename(wv_type) in URL:
        url = URL[wv_type]
        print('downloading word vectors from {}'.format(url))
        filename = os.path.basename(fname)
        if not os.path.exists(root):
            os.makedirs(root)
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            fname, _ = urlretrieve(url, fname, reporthook=reporthook(t))
            with zipfile.ZipFile(fname, "r") as zf:
                print('extracting word vectors into {}'.format(root))
                zf.extractall(root)
        if not os.path.isfile(fname + '.txt'):
            raise RuntimeError('no word vectors of requested dimension found')
        return load_word_vectors(root, wv_type, dim)
    else:
        raise RuntimeError('unable to load word vectors')
#######

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
    s = {ind: i for i, ind in enumerate(p)}
    return s

def get_lemma_to_infinitive():
    with open(IMSITU_VERBS, 'r') as f:
        imsitu_verbs_lemmatized = f.read().splitlines()
    with open(IMSITU_VAL_LIST, 'r') as f:
        imsitu_verbs_nonlemmatized = {int(x.split(' ')[1]):x.split('_')[0] for x in f.read().splitlines()}
    imsitu_verbs_nonlemmatized = [imsitu_verbs_nonlemmatized[x] for x in range(504)]

    l2f = {lem:inf for inf, lem in zip(imsitu_verbs_nonlemmatized, imsitu_verbs_lemmatized)}
    assert len(l2f) == 504
    return l2f

def _load_imsitu_verbs():
    """
    :return: a list of imsitu verbs
    """
    imsitu_verbs = []
    with open(IMSITU_VERBS, 'r') as f:
        for v in f.read().splitlines():
            imsitu_verbs.append(v)
    assert len(imsitu_verbs) == 504
    # assert imsitu_verbs == sorted(imsitu_verbs)
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

        # permute with imsitu verbs
        imsitu_verbs = _load_imsitu_verbs()
        for v in imsitu_verbs:
            if v not in list(merged_df.template):
                print("NO {}".format(v))
        merged_df = pd.DataFrame([
            merged_df.iloc[merged_df[merged_df.template == v].index[0]].T.rename(idx) for idx, v in enumerate(imsitu_verbs)
        ])
    # Remove the in_imsitu and verb information (only templates are relevant)
    merged_df = merged_df.drop(['in_imsitu', 'verb'], 1)
    return merged_df


def attributes_split(imsitu_only=False):
    """
    :param imsitu_only: if true, only return data for verbs in imsitu
    :return: train, test, val dataframes 
    """
    df = _load_attributes(imsitu_only).reset_index()
    train_atts = df[df['train']].drop(['train', 'val', 'test'], 1).set_index('template')
    val_atts = df[df['val']].drop(['train', 'val', 'test'], 1).set_index('template')
    test_atts = df[df['test']].drop(['train', 'val', 'test'], 1).set_index('template')
    return train_atts, val_atts, test_atts


def _load_defns(atts_df, is_test=False):
    """
    Loads a dataframe with the definition joined to the attributes.
    
    At test time, we only use the first definition per template.
    
    Importantly, some of the templates might lack definitions. To avoid this problem, we
    drop the particle if this occurs. This works for all of the verbs in the test set :)
    However, it means that some training verbs (such as "Unpocket") can't be used because they
    don't have definitions.
    
    :param atts_df: Dataframe with attributes
    :param is_test: If true, we'll drop everything except the first definition.
                            
    :return: A dataframe with the definitions.
    """
    verb_defns = pd.read_csv(DEFNS_PATH)
    if is_test:
        verb_defns = verb_defns.groupby('template').first().reset_index()

    # Some phrases aren't going to have definitions. The fix is to drop off the
    # particle...
    verbs_with_defns = set(verb_defns['template'])
    verbs_we_want = set(atts_df.index)
    for v in (verbs_we_want - verbs_with_defns):
        if len(v.split(' ')) == 1 and is_test:
            raise ValueError("{} has no definition".format(v))
        else:

            append_defns = verb_defns[verb_defns['template'] == v.split(' ')[0]].copy()
            append_defns['template'] = v

            verb_defns = pd.concat((verb_defns, append_defns), ignore_index=True)

    missing_verbs = verbs_we_want - set(verb_defns['template'])
    if len(missing_verbs) != 0:
        if is_test:
            raise ValueError("Some verbs are missing: {}".format(missing_verbs))
        else:
            print("Some verbs are missing definitions: {}".format(missing_verbs))

    joined_df = verb_defns.join(atts_df, 'template', how='inner')
    joined_df = joined_df.drop(['POS'], 1).set_index('template')
    return joined_df


def _get_template_emb(template, wv_dict, wv_arr):
    """
    Ideally, we'll get the word embedding directly. Otherwise, presumably it's a multiword
    expression, and we'll get the average of the expressions. If these don't work, and it starts
    with "un", we'll split on that.
    
    :param template: Possibly a multiword template
    :param wv_dict: dictionary mapping tokens -> indices
    :param wv_arr: Array of word embeddings
    :return: The embedding for template
    """
    wv_index = wv_dict.get(template, None)
    if wv_index is not None:
        return wv_arr[wv_index]

    if len(template.split(' ')) > 1:
        t0, t1 = template.split(' ')
        ind0 = wv_dict.get(t0, None)
        ind1 = wv_dict.get(t1, None)
        if (ind0 is None) or (ind1 is None):
            raise ValueError("Error on {}".format(template))
        return (wv_arr[ind0] + wv_arr[ind1]) / 2.0

    if template.startswith('un'):
        print("un-ning {}".format(template))
        ind0 = wv_dict.get('un', None)
        ind1 = wv_dict.get(template[2:], None)
        if (ind0 is None) or (ind1 is None):
            raise ValueError("Error on {}".format(template))
        return (wv_arr[ind0] + wv_arr[ind1]) / 2.0

    if template == 'cheerlead':
        return (wv_arr[wv_dict.get('cheer', None)] + wv_arr[wv_dict.get('lead', None)]) / 2.0
    if template == 'intermingle':
        return (wv_arr[wv_dict.get('inter', None)] + wv_arr[wv_dict.get('mingle', None)]) / 2.0
    if template == 'moisturize':
        return wv_arr[wv_dict.get('moisture', None)]
    else:
        print("Problem with {}".format(template))
        return torch.FloatTensor(np.random.randn(300))

    raise ValueError("Problem with {}".format(template))


def _load_glove(words):
    """
    Loads word vectors of a list of words
    :param words: 
    :return: 
    """
    wv_dict, wv_arr, _ = load_word_vectors(GLOVE_PATH, GLOVE_TYPE, 300)
    embeds = torch.Tensor(len(words), 300).zero_()
    for i, token in enumerate(words):
        embeds[i] = _get_template_emb(token, wv_dict, wv_arr)
    return embeds


def _load_counterfit(words):
    """
    Loads word vectors of a list of words
    :param words:
    :return:
    """
    wv_dict, wv_arr, _ = load_word_vectors(DATA_PATH, 'ffdnglove', 300)
    # wv_dict, wv_arr, _ = load_word_vectors(DATA_PATH, 'cfv', 300)
    embeds = torch.Tensor(len(words), 300).zero_()
    for i, token in enumerate(words):
        embeds[i] = _get_template_emb(token, wv_dict, wv_arr)
    return embeds

class Attributes(object):
    def __init__(self, vector_type='glove', word_type='lemma', use_train=False, use_val=False, use_test=False, imsitu_only=False,
                 use_defns=False, first_defn_at_test=True):
        """
        Use this class to represent a chunk of attributes for each of the test labels. 
        This is needed because at test time we'll need to compare against all of the attributes
        """
        assert use_train or use_val or use_test
        self.atts_df = pd.concat([a for a, use_a in zip(attributes_split(imsitu_only),
                                                        (use_train, use_val, use_test)) if use_a])
        self.use_defns = use_defns
        if self.use_defns:
            self.atts_df = _load_defns(self.atts_df,
                                       is_test=(use_val or use_test) and first_defn_at_test)

        # perm is a permutation from the normal index to the new one.
        # This can be used for getting the attributes for Imsitu
        self.ind_perm = invert_permutation(self.atts_df['index'].as_matrix())

        self.domains = [(c, len(self.atts_df[c].unique())) for c in COLUMNS]

        self.atts_matrix = Variable(torch.LongTensor(self.atts_df[COLUMNS].as_matrix()),
                                    volatile=not use_train)

        # LOAD THE VECTORS
        assert word_type in ('lemma', 'infinitive')
        if word_type == 'lemma':
            all_words = self.atts_df.index.values
        else:
            l2i = get_lemma_to_infinitive()
            all_words = [l2i[w] for w in self.atts_df.index.values]

        if vector_type == 'glove':
            matrix = _load_glove(all_words)
        elif vector_type == 'counterfit':
            matrix = _load_counterfit(all_words)
        else:
            raise ValueError("unspecified vector type")


        self.embeds = Variable(matrix, volatile=not use_train)

    @property
    def _balanced_inds(self):
        # Returns the inds that balance the dataset
        counts = self.atts_df.groupby('template').defn.nunique()
        max_count = max(counts)
        all_inds = []
        for template, inds in self.atts_df.groupby('template').indices.items():
            all_inds.append(inds)
            all_inds.append(np.random.choice(inds, size=max_count-len(inds)))
        all_inds = np.concatenate(all_inds, 0)
        np.random.shuffle(all_inds)
        return all_inds

    def __len__(self):
        return self.atts_df.shape[0]

    def __getitem__(self, index):
        if self.use_defns:
            return self.atts_matrix[index], self.embeds[index], self.atts_df.defn.iloc[index]
        return self.atts_matrix[index], self.embeds[index]

    def cuda(self, device_id=None):
        self.atts_matrix = self.atts_matrix.cuda(device_id)
        self.embeds = self.embeds.cuda(device_id)

    @classmethod
    def splits(cls, cuda=True, **kwargs):
        train = cls(use_train=True, use_val=False, use_test=False, **kwargs)
        val = cls(use_train=False, use_val=True, use_test=False, **kwargs)
        test = cls(use_train=False, use_val=False, use_test=True, **kwargs)

        if cuda:
            train.cuda()
            val.cuda()
            test.cuda()

        return train, val, test
