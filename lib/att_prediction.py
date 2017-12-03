"""
Builds an RNN encoder that can go to attributes or word embeddings
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
from lib.bce_loss import binary_cross_entropy_with_logits
from lib.selu import selu, AlphaDropout
from config import INIT_SCALE

class DictionaryModel(nn.Module):
    def __init__(self, vocab, output_size, embed_size=300, hidden_size=300, embed_input=False,
                 dropout_rate=0.2):
        """
        :param vocab: 
        :param output_size
        :param input_size: Dimensionality of the input tokens
        :param hidden_size:  
        :param embed_input: Whether to augment the hidden representation with the embedding
        """
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embed_size = embed_size
        self.vocab_size = len(vocab)
        self.embed_input = embed_input
        super(DictionaryModel, self).__init__()

        # GRU encoder
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.load_wv(vocab.vectors)
        self.gru = nn.GRU(self.embed_size, self.hidden_size, bidirectional=True)

        self.d = nn.Dropout(p=dropout_rate)

        # hidden layer
        emb_inp_feats = self.embed_size if self.embed_input else 0
        self.fc = nn.Linear(self.hidden_size*2 + emb_inp_feats, self.output_size)
        self.fc.weight.data.normal_(0.0, INIT_SCALE)
        self.fc.bias.data.fill_(0)

    def load_wv(self, vocab_vecs):
        self.embed.state_dict()['weight'] = vocab_vecs

    def forward(self, defns, word_embeds=None):
        """
        Forward pass
        :param defns: PackedSequence with definitions 
        :param word_embeds: [batch_size, array] of word embeddings
        :return: 
        """
        batch_embed = PackedSequence(self.embed(defns.data), defns.batch_sizes)
        output, h_n = self.gru(batch_embed)
        h_rep = h_n.transpose(0, 1).contiguous().view(-1, self.hidden_size * 2)
        h_rep = self.d(h_rep)

        if self.embed_input and (word_embeds is None):
            raise ValueError("Must supply word embedding")
        elif self.embed_input:
            h_rep = torch.cat((h_rep, word_embeds),1)

        return self.fc(h_rep)

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


class FeedForwardModel(nn.Module):
    def __init__(self, input_size, output_size, init_dropout=0.1, hidden_size=None,
                 hidden_dropout=0.2):
        """
        A simple feedforward neural network with 1 or 2 layers that tries to predict attributes
        directly from
        1) an embedding
        2) embeddings averaged from the definition
        3) some combination
        4) etc.
        :param input_size: dimensionality of input features
        :param output_size: Dimensionality of output feature (use AttributePrediction.input_size)
        :param hidden_size: Dimensionality of hidden layer. If None, don't use a hidden layer.
        :param dropout_rate: Dropout rate to use. We'll use AlphaDropout and selu activation
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.hidden_dropout = hidden_dropout
        self.init_dropout = init_dropout
        super(FeedForwardModel, self).__init__()

        self.fc0 = nn.Linear(self.input_size,
                             self.output_size if self.hidden_size is None else self.hidden_size)
        self.fc0.weight.data.normal_(0.0, INIT_SCALE)
        self.fc0.bias.data.fill_(0)

        self.d_init = AlphaDropout(self.init_dropout)
        if self.hidden_size is not None:
            self.fc1 = nn.Linear(self.hidden_size, self.output_size)
            self.fc1.weight.data.normal_(0.0, INIT_SCALE)
            self.fc1.bias.data.fill_(0)
            self.d0 = AlphaDropout(self.hidden_dropout)
        else:
            self.fc1 = None
            self.d0 = None

    def forward(self, input_data):
        pred = self.fc0(self.d_init(input_data))
        if self.hidden_size is not None:
            pred = self.fc1(self.d0(selu(pred)))
        return pred

