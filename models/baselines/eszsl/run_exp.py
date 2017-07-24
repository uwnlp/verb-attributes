import numpy as np
import pandas as pd
from data.imsitu_loader import ImSitu
from data.attribute_loader import Attributes, COLUMNS

train_data, val_data, test_data = ImSitu.splits(zeroshot=True)

# X \in \R^{d x m} where D is dimensionality and m is # examples
train_feats = np.load('train_feats.npy').T
val_feats = np.load('val_feats.npy').T
test_feats = np.load('test_feats.npy').T
train_labels = np.load('train_labels.npy')
val_labels = np.load('val_labels.npy')
test_labels = np.load('test_labels.npy')

# Predicted atts
old_index = Attributes(use_defns=True, use_test=True).atts_df.index
inds = np.array([np.where(old_index==i)[0][0] for i in test_data.attributes.atts_df.index])
pred_atts = pd.DataFrame(
    np.load('/home/rowan/code/verb-attributes/data/att_preds_ensemble.npy')[inds],
    columns=COLUMNS,
    index=test_data.attributes.atts_df.index,
)

def dummies(atts_df, col, dom_size):
    if dom_size > 2:
        d = pd.get_dummies(pd.concat((atts_df['time'], pd.Series(np.arange(dom_size), np.arange(dom_size))), 0), prefix=col)[:-dom_size]
    else:
        d = atts_df[col]
    return d


# Binarize
offsets = [0, train_labels.max() + 1, train_labels.max() + val_labels.max() + 2]
for data, labels, offset in zip(
        (train_data, val_data, test_data),
        (train_labels, val_labels, test_labels),
        offsets
    ):
    full_dummies = [dummies(data.attributes.atts_df, col, dom_size) for col, dom_size in data.attributes.domains]

    #number of attributes by labels  [a x z]
    data.S = pd.concat(full_dummies, axis=1).as_matrix().astype(np.float32).T

    #number of examples * labels [z x m]
    data.Y = -np.ones((labels.shape[0], labels.max() + 1), dtype=np.float32)
    data.Y[np.arange(labels.shape[0]), labels] = 1

    data.Y_full = -np.ones((labels.shape[0], 504), dtype=np.float32)
    data.Y_full[np.arange(labels.shape[0]), labels+offset] = 1

full_dummies = [dummies(pred_atts, col, dom_size) for col, dom_size in data.attributes.domains]
S_pred = pd.concat(full_dummies, axis=1).as_matrix().astype(np.float32).T
assert np.allclose(S_pred.shape, test_data.S.shape)
S_full = np.concatenate((train_data.S, val_data.S, test_data.S), 1)
S_full_pred = np.concatenate((train_data.S, val_data.S, S_pred), 1)


def soln(gamma=1, l=1):
    first_term = train_feats.dot(train_feats.T)
    first_term += gamma * np.eye(first_term.shape[0])

    #first_term = np.linalg.inv(first_term)

    middle_term = train_feats.dot(train_data.Y).dot(train_data.S.T)

    final_term = train_data.S.dot(train_data.S.T)
    final_term += np.eye(final_term.shape[0])*l
    # final_term = np.linalg.inv(final_term)

    # We want to compute BC^{-1}
    # C^TX = B^T -> X = (C^{-1})^TB^T  -> X = (BC^{-1})^T
    BCinv = np.linalg.solve(final_term.T, middle_term.T).T
    # BCinv = middle_term.dot(np.linalg.inv(final_term))

    # Ax = BC^-1 is equiv to A^-1BC^-1
    V = np.linalg.solve(first_term, BCinv)
    return V

def test(V, X, S):
    return X.T.dot(V).dot(S)


def test_deploy(V, pred=False):
    if pred:
        preds_part = test(V, test_feats, S_pred)
        preds_full = test(V, test_feats, S_full_pred)
    else:
        preds_part = test(V, test_feats, test_data.S)
        preds_full = test(V, test_feats, S_full)

    ranking_part = (-preds_part).argsort(1).argsort(1)
    ranking_full = (-preds_full).argsort(1).argsort(1)

    rank_part = ranking_part[np.arange(ranking_part.shape[0]), test_labels]
    rank_full = ranking_full[np.arange(ranking_part.shape[0]), (test_labels + offsets[2])]

    top1_acc = (rank_part == 0).mean()*100
    top5_acc = (rank_part < 5).mean()*100
    top1_acc_full = (rank_full == 0).mean()*100
    top5_acc_full = (rank_full < 5).mean()*100

    return pd.Series(data=[top1_acc, top5_acc, top1_acc_full, top5_acc_full],
              index=['top1_acc', 'top5_acc', 'top1_acc_full', 'top5_acc_full'])

def val_deploy(V):
    preds_part = test(V, val_feats, val_data.S).argmax(1)
    part_acc = (preds_part == val_labels).mean()*100
    return part_acc

def log_sample(minv, maxv):
    miv = np.log(minv)
    mav = np.log(maxv)
    sa = np.random.random() * (mav-miv) + miv
    return np.exp(sa)

params = []
ax = []
for i in range(20):
    g = log_sample(2, 1000)
    l = log_sample(0.001, 0.1)
    params.append((g,l))
    V = soln(g,l)
    acc = val_deploy(V)
    ax.append(acc)
    print("Accuracy for g={:.3f}, l={:.3f} is {:.2f}".format(g,l,acc))

best_params = params[np.argmax(ax)]
V = soln(*best_params)
res = test_deploy(V)
res_pred = test_deploy(V, pred=True)
res = pd.DataFrame([res, res_pred], index=['eszsl', 'eszsl-pred'])
res.to_csv('eszsl.csv', float_format='%.2f')
print(res)
