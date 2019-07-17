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

def normalize_text(text):
    return u" ".join(
        [x for x in [y for y in non_alphanums.sub(' ', text).lower().strip().split(" ")] \
         if len(x) > 1 and x not in stopwords])

merge = pd.read_pickle('merge.pkl')

wb = wordbatch.WordBatch(extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.5, 1.0],
                                                            "hash_size": 2 ** 29, "norm": None, "tf": 'binary',"idf": None}), procs=0)

merge['name'] = merge['name'].map(lambda x: normalize_text(x))

wb.dictionary_freeze= True
X_name = wb.fit_transform(merge['name'])

del(wb)
X_name = X_name[:, np.array(np.clip(X_name.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
print('[{}] Vectorize `name` completed.'.format(time.time() - start_time))

wb = CountVectorizer()
X_category1 = wb.fit_transform(merge['general_cat'])
X_category2 = wb.fit_transform(merge['subcat_1'])
X_category3 = wb.fit_transform(merge['subcat_2'])
print('[{}] Count vectorize `categories` completed.'.format(time.time() - start_time))

wb = wordbatch.WordBatch(extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.0, 1.0],
                                                                  "hash_size": 2 ** 28, "norm": "l2", "tf": 1.0,
                                                                  "idf": None}), procs=0)

name_desc = ['Condition'+ str(c) + ' ' + str(a) + ' ' + str(b) for c, a, b in merge[['item_condition_id','name','item_description']].values]
merge['name_desc'] = name_desc
merge['name_desc'] = merge['name_desc'].map(lambda x: normalize_text(x))

wb.dictionary_freeze= True
X_description = wb.fit_transform(merge['name_desc'])

del(wb)
X_description = X_description[:, np.array(np.clip(X_description.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
print('[{}] Vectorize `item_description` completed.'.format(time.time() - start_time))

wb_condsub2 = wordbatch.WordBatch(extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [2, 1.0],
                                                                  "hash_size": 2 ** 17, "norm": "l2", "tf": 1.0,
                                                                  "idf": None}), procs=0)

merge['condsub2'] = [str(x) + ' ' + str(y) for x,y in merge[['brand_name','subcat_2']].values]
X_condsub2 = wb_condsub2.fit_transform(merge['condsub2'])
X_condsub2 = X_condsub2[:, np.array(np.clip(X_condsub2.getnnz(axis=0) - 1, 0, 1), dtype=bool)]
print('[{}] Vectorize `coname` completed.'.format(time.time() - start_time))



lb = LabelBinarizer(sparse_output=True)
X_brand = lb.fit_transform(merge['brand_name'])
print('[{}] Label binarize `brand_name` completed.'.format(time.time() - start_time))


X_dummies = csr_matrix(pd.get_dummies(merge[['item_condition_id', 'shipping']],
                                          sparse=True).values)
print('[{}] Get dummies on `item_condition_id` and `shipping` completed.'.format(time.time() - start_time))
print(X_dummies.shape, X_description.shape, X_brand.shape, X_category1.shape, X_category2.shape, X_category3.shape,
          X_name.shape)


sparse_merge = hstack((X_dummies, X_description, X_brand, X_category1, X_category2, X_category3, X_name,X_condsub2)).tocsr()
print('[{}] Create sparse merge completed'.format(time.time() - start_time))

pd.to_pickle('sparse_merge.pkl')
