import gc
import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn import linear_model as lm

import sys
import wordbatch

from wordbatch.extractors import WordBag, WordHash
from wordbatch.models import FTRL, FM_FTRL
from nltk.stem.porter import *

from nltk.corpus import stopwords
import re

sparse_merge = pd.read_pickle('sparse_merge.pkl')

print(sparse_merge.shape)
mask = np.array(np.clip(sparse_merge.getnnz(axis=0) - 1, 0, 1), dtype=bool)
sparse_merge = sparse_merge[:, mask]
X = sparse_merge[:nrow_train]
X_test = sparse_merge[nrow_test:]
print(sparse_merge.shape)

gc.collect()
train_X, train_y = X, y

fm_ftrl = FM_FTRL(alpha=0.01, beta=0.01, L1=0.00001, L2=0.1, D=sparse_merge.shape[1], alpha_fm=0.01, L2_fm=0.0, init_fm=0.01,
                     D_fm=200, e_noise=0.0001, iters=15, inv_link="identity", threads=4)
fm_ftrl.fit(train_X, train_y)
print('[{}] Train FM_FTRL completed'.format(time.time() - start_time))

predsFm_ftrl = fm_ftrl.predict(X_test)
print('[{}] Predict FM_FTRL completed'.format(time.time() - start_time))

# ridge = lm.Ridge(solver='auto',fit_intercept=True,alpha=1,max_iter=500,normalize=False,tol=0.05).fit(X=X,y=y) #0.4647
# ridge.fit(train_X, train_y)
# print('[{}] Train Ridge completed'.format(time.time() - start_time))

# predsRidge = ridge.predict(X_test)
# print('[{}] Predict Ridge completed'.format(time.time() - start_time))

# ftrl = FTRL(alpha=0.01, beta=0.1, L1=0.00001, L2=1.0, D=sparse_merge.shape[1], iters=50, inv_link="identity", threads=1)
# ftrl.fit(train_X, train_y)
# print('[{}] Train FTRL completed'.format(time.time() - start_time))

# predsFTRL = ftrl.predict(X_test)
# print('[{}] Predict FTRL completed'.format(time.time() - start_time))

params = {
        'learning_rate': 0.6,
        'application': 'regression',
        'max_depth': 4,
        'num_leaves': 31,
        'verbosity': -1,
        'metric': 'RMSE',
        'data_random_seed': 1,
        'bagging_fraction': 0.75,
        'bagging_freq': 5,
        'feature_fraction': 0.8,
        'nthread': 4,
        'min_data_in_leaf': 100,
        'max_bin': 31
    }

# Remove features with document frequency <=100
print(sparse_merge.shape)
mask = np.array(np.clip(sparse_merge.getnnz(axis=0) - 100, 0, 1), dtype=bool)
sparse_merge = sparse_merge[:, mask]
X = sparse_merge[:nrow_train]
X_test = sparse_merge[nrow_test:]
print(sparse_merge.shape)

train_X, train_y = X, y
d_train = lgb.Dataset(train_X, label=train_y)
watchlist = [d_train]
gb = lgb.train(params, train_set=d_train, num_boost_round=5000, valid_sets=watchlist, early_stopping_rounds=1000, verbose_eval=1000)
predLGB = gb.predict(X_test)

preds = (0.7*predsFm_ftrl + 0.3*predLGB)

submission['price'] = np.expm1(preds)
submission.to_csv("submission.csv", index=False)
