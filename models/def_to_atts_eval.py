"""
Trains the imsitu model for ZSL
"""
from data.dictionary_dataset import load_vocab
from data.attribute_loader import Attributes, COLUMNS
from config import ModelConfig
import torch
import numpy as np
from tqdm import tqdm
from lib.attribute_loss import AttributeLoss, evaluate_accuracy
from lib.bucket_iterator import DictionaryAttributesIter
from lib.att_prediction import DictionaryModel
import pandas as pd

train_data, val_data, test_data = Attributes.splits(use_defns=True, cuda=True)
dict_field, _ = load_vocab()
test_iter = DictionaryAttributesIter(dict_field, test_data, batch_size=64 * 10,
                                     shuffle=False, train=False)
att_crit = AttributeLoss(train_data.domains, size_average=True)


def eval(ckpt, use_emb=False):

    # Recommended hyperparameters
    args = ModelConfig(batch_size=64, ckpt=ckpt, dropout=0.5,
                       use_emb=use_emb)

    m = DictionaryModel(dict_field.vocab, output_size=att_crit.input_size, embed_input=args.use_emb,
                        dropout_rate=args.dropout)
    m.load_state_dict(torch.load(args.ckpt)['m_state_dict'])

    if torch.cuda.is_available():
        m.cuda()
        att_crit.cuda()
        train_data.atts_matrix.cuda()
        val_data.atts_matrix.cuda()
        test_data.atts_matrix.cuda()
    m.eval()

    # Don't take the mean until the end
    preds_ = []
    labels_ = []
    for val_b, (atts, words, defns, perm) in enumerate(tqdm(test_iter)):
        preds_.append(att_crit.predict(m(defns, words))[perm])
        labels_.append(atts.data.cpu().numpy()[perm])
    preds = np.concatenate(preds_, 0)
    labels = np.concatenate(labels_, 0)

    acc_table = evaluate_accuracy(preds, labels).T.squeeze()
    return acc_table, preds

if __name__ == '__main__':
    ens, ens_preds = eval('def2atts_train/emb2/ckpt_48.tar', use_emb=True)
    bgru, bgru_preds = eval('def2atts_train/att2/ckpt_33.tar', use_emb=False)

    np.save('att_preds_ensemble', ens_preds)
    np.save('att_preds_bgru', bgru_preds)

    # # For visualization ------------------------------------------------
    # pred_df = pd.DataFrame(bgru_preds, columns=[c+'_pred' for c in COLUMNS],
    #                        index=test_data.atts_df.index)
    # full_df = pd.concat([test_data.atts_df.defn] + [pd.concat((test_data.atts_df[c], pred_df[c + '_pred']),1) for c in COLUMNS], 1)
    # full_df.to_csv('attribute_preds.csv')
    rez = pd.DataFrame([ens, bgru],
                       index=['ensemble', 'bgru'])
    rez.to_csv('dict_results.csv', float_format='%.2f')