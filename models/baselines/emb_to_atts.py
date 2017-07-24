"""
GLOVE embeddings -> attributes
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from lib.attribute_loss import evaluate_accuracy
from data.attribute_loader import Attributes
from config import ROOT_PATH
import os
import pandas as pd
from collections import defaultdict

retrofit_dir = os.path.join(ROOT_PATH, 'models','baselines','retrofitting')
TYPES = ('glove', 'framenet', 'ppdb', 'wordnet')

train_data, val_data, test_data = Attributes.splits(use_defns=False, cuda=False)
embeds_train = train_data.embeds.data.numpy()
embeds_val = val_data.embeds.data.numpy()
embeds_test = test_data.embeds.data.numpy()

Y_train = train_data.atts_matrix.data.numpy()
Y_val = val_data.atts_matrix.data.numpy()
Y_test = test_data.atts_matrix.data.numpy()

def emb_to_atts(emb_type):
    assert emb_type in TYPES

    if emb_type != 'glove':
        # Replace the matrices
        vecs = np.load(os.path.join(retrofit_dir, emb_type + '.pkl'))
        X_train = np.stack([vecs[v] for v in train_data.atts_df.index])
        X_val = np.stack([vecs[v] for v in val_data.atts_df.index])
        X_test = np.stack([vecs[v] for v in test_data.atts_df.index])
    else:
        X_train = embeds_train
        X_val = embeds_val
        X_test = embeds_test
    print("train {} val {} test {}".format(X_train.shape, X_val.shape, X_test.shape))


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

    c_to_use = {d: cs[np.argmax(scores)] for d, scores in accs.items()}
    print("Using c={}, acc of {:.3f} on val".format(
        '\n'.join('{:2d}:{:.3f}'.format(d, c) for d, c in c_to_use.items()),
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
    accs = evaluate_accuracy(preds_full, Y_test)
    accs.index = [emb_type]
    return accs, preds_full

if __name__ == '__main__':
    #results = pd.concat([emb_to_atts(t)[0] for t in TYPES], axis=0)
    #results.to_csv('emb.csv', float_format='%.2f')

    attpreds = emb_to_atts('glove')[1]
    np.save('att_preds_embed', attpreds)
