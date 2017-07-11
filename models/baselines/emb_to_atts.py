"""
GLOVE embeddings -> attributes
"""


import numpy as np
from sklearn.linear_model import LogisticRegression
from lib.attribute_loss import evaluate_accuracy
from data.attribute_loader import Attributes

train_data, val_data, test_data = Attributes.splits(use_defns=False, cuda=False)

X_train = train_data.embeds.data.numpy()
X_val = val_data.embeds.data.numpy()
X_test = test_data.embeds.data.numpy()

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
acc_table.index = ['w2v']
acc_table.to_csv('GLOVE.csv')