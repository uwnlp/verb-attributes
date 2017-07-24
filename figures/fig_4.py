from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import random
import pickle as pkl
import numpy as np
from subprocess import call
import seaborn as sns
import pandas as pd
NUM_TO_SHOW = 6
from copy import deepcopy
from PIL import Image

with open('cache.pkl', 'rb') as f:
    data, labels = pkl.load(f)

random.shuffle(data)


def att_plot(top_labels, gt_ind, probs, fn):
    # plt.figure(figsize=(5, 5))
    #
    # color_dict = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    # colors = [color_dict[c] for c in
    #           ['lightcoral', 'steelblue', 'forestgreen', 'darkviolet', 'sienna', 'dimgrey',
    #            'darkorange', 'gold']]
    # colors[gt_ind] = color_dict['crimson']
    # w = 0.9
    # plt.bar(np.arange(len(top_labels)), probs, w, color=colors, alpha=.9, label='data')
    # plt.axhline(0, color='black')
    # plt.ylim([0, 1])
    # plt.xticks(np.arange(len(top_labels)), top_labels, fontsize=6)
    # plt.subplots_adjust(bottom=.15)
    # plt.tight_layout()
    # plt.savefig(fn)
    lab = deepcopy(top_labels)
    lab[gt_ind] += ' (gt)'
    d = pd.DataFrame(data={'probs': probs, 'labels':lab})
    fig, ax = plt.subplots(figsize=(4,5))
    ax.tick_params(labelsize=15)

    sns.barplot(y='labels', x='probs', ax=ax, data=d, orient='h', ci=None)
    ax.set(xlim=(0,1))

    for rect, label in zip(ax.patches,lab):
        w = rect.get_width()
        ax.text(w+.02, rect.get_y() + rect.get_height()*4/5, label, ha='left', va='bottom',
                fontsize=25)

    # ax.yaxis.set_label_coords(0.5, 0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().label.set_visible(False)
    fig.savefig(fn, bbox_inches='tight', transparent=True)
    plt.close('all')

def crop_and_move(fn, ext='good'):
    img = Image.open(fn).convert('RGB')

    w, h = img.size
    if w < h:
        ow = 256
        oh = int(256 * h / w)
        img = img.resize((ow, oh), Image.BILINEAR)
    else:
        oh = 256
        ow = int(256 * w / h)
        img = img.resize((ow, oh), Image.BILINEAR)

    w, h = img.size
    th = 224
    tw = 224
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    img = img.crop((x1, y1, x1 + tw, y1 + th))
    img.save(ext+ '/' + fn.split('/')[-1].split('.')[0] + '.jpg')


for (fn, gt, s) in data:
    order = np.argsort(-s)
    gt_ind = order.argsort()[gt]
    if gt_ind != 0:
        print("labels gt {} fn {}".format(labels[gt], fn))

    if gt_ind == 0:
        crop_and_move(fn, 'good')
        plotfn = 'good/' + fn.split('/')[-1].split('.')[0] + '.pdf'
    else:
        crop_and_move(fn, 'bad')
        plotfn = 'bad/' + fn.split('/')[-1].split('.')[0] + '.pdf'

    if gt_ind < NUM_TO_SHOW:
        top_labels = [labels[l] for l in order[:NUM_TO_SHOW]]
        gt_ind = gt_ind
        probs = s[order[:NUM_TO_SHOW]]
    else:
        top_labels = [labels[l] for l in order[:(NUM_TO_SHOW-1)]] + [labels[gt]]
        gt_ind=(NUM_TO_SHOW-1)
        probs = s[order[:(NUM_TO_SHOW-1)].tolist() + [gt]]
    att_plot(top_labels, gt_ind, probs, plotfn)