"""
Script to train and eval the nbow to attribute model
"""

from data.dictionary_dataset import load_vocab
from sklearn.linear_model import LogisticRegression
from lib.attribute_loss import evaluate_accuracy
from data.attribute_loader import Attributes
import numpy as np
import pandas as pd
from collections import defaultdict

VOCAB_SIZE = 5000

train_data, val_data, test_data = Attributes.splits(use_defns=True, cuda=False)
dict_field, _ = load_vocab()

def bowize(text_defn, emb=None):
    """
    Gets the embedding.
    :param text_defn: 
    :param emb: 
    :return: 
    """
    inds = [dict_field.vocab.stoi[x] for x in dict_field.preprocess(text_defn)]
    bow = np.zeros(VOCAB_SIZE, dtype=np.float64)
    bow[[i for i in inds if i < VOCAB_SIZE]] = 1.0
    return bow

def nbowize(text_defn, emb=None):
    """
    Gets the embedding.
    :param text_defn: 
    :param emb: 
    :return: 
    """
    inds = [dict_field.vocab.stoi[x] for x in dict_field.preprocess(text_defn)]
    nbow = np.stack([dict_field.vocab.vectors[i].squeeze().numpy() for i in inds]).mean(0)

    if emb is not None:
        feats = np.concatenate((nbow, emb))
        return feats
    return nbow

def get_x(data, enc_fn, use_emb=False):
    if use_emb:
        return np.stack([enc_fn(d,e) for d, e in zip(list(data.atts_df['defn']),
                                                 data.embeds.data.numpy())])
    else:
        return np.stack([enc_fn(d) for d in list(data.atts_df['defn'])])

def defn_to_atts(defn_type, use_emb=False):

    enc_fn = {'bow':bowize, 'nbow':nbowize}[defn_type]
    X_train = get_x(train_data, enc_fn, use_emb=use_emb)
    X_val = get_x(val_data, enc_fn, use_emb=use_emb)
    X_test = get_x(test_data, enc_fn, use_emb=use_emb)

    Y_train = train_data.atts_matrix.data.numpy()
    Y_val = val_data.atts_matrix.data.numpy()
    Y_test = test_data.atts_matrix.data.numpy()

    # cross validate
    cs = np.power(10., [-3,-2,-1,0,1])
    accs = defaultdict(list)
    for c in cs:
        for d, (dom_name, dom_size) in enumerate(train_data.domains):
            M = LogisticRegression(C=c)
            print("fitting {}".format(d))
            M.fit(X_train, Y_train[:, d])
            s = M.score(X_val, Y_val[:, d])
            accs[d].append(s)

    c_to_use = {d:cs[np.argmax(scores)] for d, scores in accs.items()}
    print("Using c={}, acc of {:.3f} on val".format(
        '\n'.join('{:2d}:{:.3f}'.format(d,c) for d,c in c_to_use.items()),
        np.mean([max(accs[d]) for d in c_to_use.keys()])
    ))

    # -----------------------------------------------
    preds = []
    for d, (dom_name, dom_size) in enumerate(train_data.domains):
        M = LogisticRegression(C=c_to_use[d])
        print("fitting {}".format(d))
        M.fit(X_train, Y_train[:,d])
        s = M.score(X_test, Y_test[:,d])
        print("Score for {} is {}".format(dom_name, s))

        preds.append(M.predict(X_test))

    preds_full = np.array(preds).T
    acc_table = evaluate_accuracy(preds_full, Y_test)

    acc_table.index = ['{}{}'.format(defn_type, '+ GloVe' if use_emb else '')]
    return acc_table

if __name__ == '__main__':
    results = pd.concat([defn_to_atts(t, e) for t in ('bow', 'nbow') for e in (True, False)], axis=0)
    results.to_csv('defn.csv', float_format='%.2f')
