"""
This model tries to predict the word from definition
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from data.attribute_loader import Attributes
from scipy.stats import mode
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from lib.attribute_loss import evaluate_accuracy

train_data, val_data, test_data = Attributes.splits(use_defns=False, cuda=False)
preds = mode(train_data.atts_matrix.data.numpy()).mode

print("Always predicting {}".format(preds))

Y_test = test_data.atts_matrix.data.numpy()
acc_table = evaluate_accuracy(
    np.repeat(preds, Y_test.shape[0], axis=0),
    test_data.atts_matrix.data.numpy(),
)
acc_table.to_csv('mfc.csv', float_format='%.2f')