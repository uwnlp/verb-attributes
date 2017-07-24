# Just reads stuff on imsitu.
from data.attribute_loader import Attributes, COLUMNS
import numpy as np
import pandas as pd
train_data, val_data, test_data = Attributes.splits(use_defns=True, cuda=False)
_, _, test_data2 = Attributes.splits(use_defns=False, cuda=False)

pred_atts_bgrup = np.load('../data/att_preds_ensemble.npy')
pred_atts_bgru = np.load('../data/att_preds_bgru.npy')
pred_atts_embed = np.load('../data/att_preds_embed.npy')

gt_df = test_data.atts_df
bgrup_df = pd.DataFrame(pred_atts_bgrup, index=test_data.atts_df.index, columns=COLUMNS)
bgru_df = pd.DataFrame(pred_atts_bgru, index=test_data.atts_df.index, columns=COLUMNS)
embed_df = pd.DataFrame(
    [pred_atts_embed[np.where(test_data2.atts_df.index == v)[0][0]] for v in test_data.atts_df.index],
    index=test_data.atts_df.index,
    columns=COLUMNS)

# Must switch around the embed df

table = {
    'intrans': ['', 'intransitive'],
    'trans_pers': ['', 'transitive for people'],
    'trans_obj': ['', 'transitive for objects'],
    'atomicity': ['Accomplishment', 'unclear without context',
                  'Activity', 'Achievement', 'State'],
    'energy': ['unclear without context', 'No motion', 'Low motion',
               'Medium motion', 'High motion'],
    'time': ['time n/a', 'On the order of seconds',
             'On the order of minutes', 'On the order of hours',
             'On the order of days'],
    'solitary': ['solitary','likely solitary','solitary or social','likely social','social'],
    'bodyparts_Arms': ['', 'uses arms'],
    'bodyparts_Head': ['', 'uses head'],
    'bodyparts_Legs': ['', 'uses legs'],
    'bodyparts_Torso': ['', 'uses torso'],
    'bodyparts_other': ['', 'uses a different bodypart'],
    'intrans_effect_0':  ['', 'the subject moves somewhere'],
    'intrans_effect_1': ['', 'the external world changes'],
    'intrans_effect_2': ['', 'the subject\'s state changes'],
   #'intrans_effect_3': ['', 'nothing changes'],
    'trans_obj_effect_0': ['', 'the object (a thing) moves somewhere'],
    'trans_obj_effect_1': ['', 'the external world changes'],
    'trans_obj_effect_2': ['', 'the object\'s state changes'],
   # 'trans_obj_effect_3': ['', 'nothing changes'],
    'trans_pers_effect_0': ['', 'the object (a person) moves somewhere'],
    'trans_pers_effect_1': ['', 'the external world changes'],
    'trans_pers_effect_2': ['', 'the object\'s state changes'],
  #  'trans_pers_effect_3': ['', 'nothing changes'],
}

def _bp(row):
    r = []
    for col in ('bodyparts_Arms', 'bodyparts_Head',
                'bodyparts_Legs', 'bodyparts_Torso', 'bodyparts_other'):
        if row[col] == 1:
            r.append(col.split('_')[1].lower())
    return '&' + ','.join(r)

ls = []
COLS = ['solitary', 'atomicity', 'energy', 'time']
for i in range(gt_df.shape[0]):
    l = '\\multirow{4}{*}{\\rotatebox[origin = c]{90}{' + gt_df.iloc[i].name
    l += '}} &GT&  \\multirow{4}{*}{\\parbox{30mm}{' + gt_df.iloc[i].defn + '}} &'
    l += '&'.join([table[k][gt_df.iloc[i][k]] for k in COLS])
    l += _bp(gt_df.iloc[i]) + '\\\\'
    ls.append(l)

    for m_name, df in zip(['embed', 'BGRU', 'BGRU+'], [embed_df, bgru_df, bgrup_df]):
        l = '&{}& &'.format(m_name) + '&'.join([table[k][df.iloc[i][k]] for k in COLS])
        l += _bp(df.iloc[i]) + '\\\\'
        ls.append(l)
    ls.append('\\hline')

with open('ex.txt', 'w') as f:
    for l in ls:
        f.write(l + '\n')
