"""
Trains the imsitu model for ZSL
"""
from data.dictionary_dataset import load_vocab
from data.attribute_loader import Attributes
from config import ModelConfig
import torch
import numpy as np
from tqdm import tqdm
from lib.attribute_loss import AttributeLoss, evaluate_accuracy
from lib.bucket_iterator import DictionaryAttributesIter
from lib.att_prediction import DictionaryModel
import os

# Recommended hyperparameters
args = ModelConfig(batch_size=64, ckpt='def2atts_train/ckpt_10.tar', dropout=0.2)

train_data, val_data, test_data = Attributes.splits(use_defns=True, cuda=True)
dict_field, _ = load_vocab()

test_iter = DictionaryAttributesIter(dict_field, test_data, batch_size=args.batch_size*10,
                                     shuffle=False, train=False)
att_crit = AttributeLoss(train_data.domains, size_average=True)
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
for val_b, (atts, words, defns, _) in enumerate(tqdm(test_iter)):
    preds_.append(att_crit.predict(m(defns, words)))
    labels_.append(atts.data)
preds = np.concatenate(preds_, 0)
labels = torch.cat(labels_, 0).cpu().numpy()

acc_table = evaluate_accuracy(preds, labels)
acc_table.to_csv(os.path.join(os.path.dirname(args.ckpt), 'eval.csv'),
                 float_format='%.2f')
print(acc_table)
