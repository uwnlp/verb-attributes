"""
Script to train and eval the nbow to attribute model
"""

from data.dictionary_dataset import load_vocab
from sklearn.linear_model import LogisticRegression
from lib.attribute_loss import evaluate_accuracy
from data.attribute_loader import Attributes
import torch
import numpy as np

# -------------------------------------------------------------------------
# Whether we should train the ensemble dictionary + embedding model
USE_EMB_ALSO = False
# --------------------------------------------------------------------------


train_data, val_data, test_data = Attributes.splits(use_defns=True, cuda=False)
dict_field, _ = load_vocab()


def nbowize(text_defn, emb):
    """
    Gets the embedding.
    :param text_defn: 
    :param emb: 
    :return: 
    """
    inds = [dict_field.vocab.stoi[x] for x in dict_field.preprocess(text_defn)]
    nbow = torch.stack([dict_field.vocab.vectors[i] for i in inds]).mean(0).squeeze().numpy()

    if USE_EMB_ALSO:
        feats = np.concatenate((nbow, emb))
        return feats
    return nbow


def get_x(data):
    return np.stack([nbowize(d,e) for d, e in zip(list(data.atts_df['defn']),
                                                  data.embeds.data.numpy())])

X_train = get_x(train_data)
X_val = get_x(val_data)
X_test = get_x(test_data)

Y_train = train_data.atts_matrix.data.numpy()
Y_val = val_data.atts_matrix.data.numpy()
Y_test = test_data.atts_matrix.data.numpy()

# cross validate
scores_by_c = []
cs = np.power(10., [-3,-2,-1,0,1])
for c in cs:
    accs = []
    for d, (dom_name, dom_size) in enumerate(train_data.domains):
        M = LogisticRegression(C=c)
        print("fitting {}".format(d))
        M.fit(X_train, Y_train[:, d])
        s = M.score(X_val, Y_val[:, d])
        accs.append(s)
    scores_by_c.append(np.mean(accs))

c = cs[np.argmax(scores_by_c)]
print("Using c={:.3f}, acc of {:.3f} on val".format(c, np.max(scores_by_c)))


# -----------------------------------------------
accs = []
preds = []
for d, (dom_name, dom_size) in enumerate(train_data.domains):
    M = LogisticRegression(C=c)
    print("fitting {}".format(d))
    M.fit(X_train, Y_train[:,d])
    s = M.score(X_test, Y_test[:,d])
    print("Score for {} is {}".format(dom_name, s))
    accs.append(s)

    preds.append(M.predict(X_test))

preds_full = np.array(preds).T
acc_table = evaluate_accuracy(preds_full, Y_test)

name = 'NBoW + GLOVE' if USE_EMB_ALSO else 'NBoW'

acc_table.index = [name]
acc_table.to_csv(name)