"""
This file contains methods for loading the dictionary challenge.
"""
import spacy
from config import ModelConfig, CHECKPOINT_PATH, DICTIONARY_PATH, GLOVE
import os
from text.torchtext.data import Field, Dataset
import dill as pkl
from data.load_verb_templates import _load_attributes
import random


class DictionaryChallengeDataset(Dataset):
    """ Dataset for the dictionary challenge, where the goal is to predict a word embedding
        from dictionary data"""
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, examples):
        """
        TODO: FIX THIS
        :param examples: 
        :param defns_field: 
        """
        # fields = [('text', text_field), ('label', label_field)]
        #
        # def get_label_str(label):
        #     pre = 'very ' if fine_grained else ''
        #     return {'0': pre + 'negative', '1': 'negative', '2': 'neutral',
        #             '3': 'positive', '4': pre + 'positive', None: None}[label]
        # label_field.preprocessing = data.Pipeline(get_label_str)
        #
        fields = [load_vocab()]
        super(DictionaryChallengeDataset, self).__init__(examples, fields)

    @classmethod
    def splits(cls, num_val=10000):
        """
        :return: Gets training and validation splits for this.
        """
        with open(DICTIONARY_PATH, 'rb') as f:
            _words, _defns = pkl.load(f)
        examples = [(x,y) for x,y in zip(_words, _defns)]
        random.seed(123456)
        random.shuffle(examples)

        val_data = cls(examples[:num_val])
        train_data = cls(examples[num_val:])
        return train_data, val_data


def load_vocab(vocab_size=30000,
               vocab_path=os.path.join(CHECKPOINT_PATH, 'vocab_pretrained.pkl')):
    """
    Loads the vocab. If the vocab doesn't exist we'll reconstruct it
    :param vocab_size: # words in vocab
    :param vocab_path: Where to cache the vocab. Importantly, it's really
    expensive to build the dictionary from scratch, so we'll cache the dictionary.
    :return: Vocab file
    """

    defns_field = Field(
        tokenize='spacy',
        init_token='<bos>',
        eos_token='<eos>',
        lower=True,
        include_lengths=True,
    )
    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            vocab = pkl.load(f)
            defns_field.vocab = vocab
            return defns_field

    def dict_gen():
        with open(DICTIONARY_PATH, 'rb') as f:
            _words, _defns = pkl.load(f)
        for w, d in zip(_words, _defns):
            yield defns_field.preprocess(' '.join([w] + d))

        # Hack to make sure that all the verb templates are included.
        # We'll make them appear 100 times...
        verbs = _load_attributes()['template'].tolist()
        for v in verbs:
            for i in range(100):
                yield defns_field.preprocess(v)

    defns_field.build_vocab(dict_gen(), max_size=vocab_size)
    defns_field.vocab.load_vectors(wv_dir=GLOVE, wv_type='glove.6B')

    print("saving to {}".format(vocab_path))
    with open(vocab_path, 'wb') as f:
        pkl.dump(defns_field.vocab, f)
    return defns_field

if __name__ == '__main__':
    hi = load_vocab()
