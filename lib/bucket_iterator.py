"""
Contains BucketIterators for text to attributes
"""

from text.torchtext.data import BucketIterator, batch
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
import torch
from torch.autograd import Variable
import numpy as np

class DictionaryAttributesIter(BucketIterator):
    def __init__(self, field, dataset, batch_size, **kwargs):
        """
        Dictionary defn to attribute dataloader
        :param field: Field for transforming text examples to a vector format
        :param dataset: Dataset object. Must contain __len__ and __getitem__(ind) methods.
                        Each item is a tuple (attributes, word embedding, string definition)
        :param batch_size: Batch size to use
        """
        self.field = field
        super(DictionaryAttributesIter, self).__init__(
            dataset,
            batch_size,
            sort_key=lambda x: -len(self.field.preprocess(x[2])),  # Must tokenize first
            repeat=False,
            **kwargs,
        )

    def __iter__(self):
        if self.shuffle:
            # Oversample and shuffle (balanced inds does this for us)
            self.batches = batch((self.dataset[i] for i in self.dataset._balanced_inds),
                                 self.batch_size)
        else:
            self.batches = batch((self.dataset[i] for i in range(len(self.dataset))), self.batch_size)
            #self.init_epoch()
        for minibatch in self.batches:
            self.iterations += 1

            lens = [self.sort_key(x) for x in minibatch]
            sorted_inds = np.argsort(lens)
            reverse_inds = np.argsort(sorted_inds)

            atts, words, defns = zip(*[minibatch[i] for i in sorted_inds])
            atts_var = torch.stack(atts, 0)
            words_var = torch.stack(words, 0)

            defns_packed = _defns_to_packed_seq(defns, self.field, volatile=not self.train)
            yield atts_var, words_var, defns_packed, reverse_inds


class DictionaryChallengeIter(BucketIterator):
    def __init__(self, dataset, batch_size, **kwargs):
        """
        Dictionary defn to word embedding dataloader
        :param dataset: Dataset object. Must contain 'label' and 'text' fields; 'label' is for
                        the word embeddings and 'text' is for the definitions.
                        Each item from __getitem__ is a tuple (word INDEX, tokenized definition).
        :param batch_size: Batch size to use
        """
        super(DictionaryChallengeIter, self).__init__(dataset, batch_size,
                                                      sort_key=lambda x: -len(x[1]),
                                                      repeat=False,
                                                      **kwargs)

    def __iter__(self):
        if self.shuffle:
            self.init_epoch()
        else:
            self.batches = batch(sorted(self.data(), key=self.sort_key),
                                 self.batch_size, self.batch_size_fn)

        for minibatch in self.batches:
            self.iterations += 1

            words, defns = zip(*minibatch)
            # words_tensor = torch.stack([
            #     self.dataset.fields['label'].vocab.vectors[
            #         self.dataset.fields['label'].vocab.stoi[x]
            #     ] for x in words
            # ]).cuda()
            # words_var = Variable(words_tensor, volatile=not self.train)
            words_var = torch.LongTensor([
                self.dataset.fields['label'].vocab.stoi[x] for x in words
            ]).cuda()
            # words_var = torch.stack(words, 0)
            defns_packed = _defns_to_packed_seq(defns,
                                                self.dataset.fields['text'],
                                                volatile=not self.train)
            yield words_var, defns_packed


def _defns_to_packed_seq(defns, field, cuda=torch.cuda.is_available(), volatile=False):
    """
    Pads a list of definitions (in sorted order!)
    :param tokenized_defns: List of lists containing tokenized definitions OR
                            List of string containind definitions
    :param field: Contains padding and vocab functions.
    :param cuda: if true, we'll cudaize it
    :param volatile:
    :return: PackedSequence with a Variable.
    """
    tokenized_defns = [field.preprocess(x) for x in defns]
    defns_padded, lengths = field.pad(tokenized_defns)
    if not all(lengths[i] >= lengths[i + 1] for i in range(len(lengths) - 1)):
        raise ValueError("Sequences must be in decreasing order")

    defns_tensor = torch.LongTensor([
        [field.vocab.stoi[x] for x in ex] for ex in defns_padded
    ])

    defns_packed_ = pack_padded_sequence(defns_tensor, lengths, batch_first=True)
    packed_data = Variable(defns_packed_.data, volatile=volatile)
    if cuda:
        packed_data = packed_data.cuda()
    return PackedSequence(packed_data, defns_packed_.batch_sizes)
