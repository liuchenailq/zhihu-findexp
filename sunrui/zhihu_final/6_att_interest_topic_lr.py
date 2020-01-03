# @time    : 2019/11/11 10:17 上午
# @author  : srtianxia

import numpy as np
import pickle
from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import pickle

with open('./pkl/question_info.pkl', 'rb') as file:
    question_info = pickle.load(file)

with open('./pkl/member_info.pkl', 'rb') as file:
    member_info = pickle.load(file)

member_info['topic_interest_keys'] = member_info['topic_interest'].map(lambda x: list(x.keys()))

member_info['topic_interest_keys'] = member_info['topic_interest_keys'].map(lambda x: ' '.join([str(i) for i in x]))

member_info['topic_attent'] = member_info['topic_attent'].map(lambda x: ' '.join([str(i) for i in x]))

question_info['topic'] = question_info['topic'].map(lambda x: ' '.join([str(i) for i in x]))

with open('./pkl/topic.pkl', 'rb') as file:
    topic = pickle.load(file)

topic_id_list = topic['id'].tolist()

cv = CountVectorizer(vocabulary=[str(i) for i in topic_id_list])

invite_info_evaluate = pd.read_hdf('./my_feat/convert_test_b.h5', key='data')

invite_info_evaluate = pd.merge(invite_info_evaluate, member_info, on='author_id', how='left')
invite_info_evaluate = pd.merge(invite_info_evaluate, question_info, on='question_id', how='left')

invite_info_evaluate = invite_info_evaluate[['topic', 'topic_attent', 'topic_interest_keys']]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)

topic_vec_df_list = []

user_interest_topic_test = cv.fit_transform(invite_info_evaluate['topic_interest_keys'])
question_topic_test = cv.fit_transform(invite_info_evaluate['topic'])
user_topic_attent_test = cv.fit_transform(invite_info_evaluate['topic_attent'])

test_data = sparse.hstack([user_interest_topic_test, question_topic_test, user_topic_attent_test])

topic_test_probs = []
for k in range(5):
    lr = pickle.load(open(f'./model/topic_cv_lr_fold_{k + 1}.pkl', 'rb'))

    test_probs = lr.predict_proba(test_data)
    test_probs = 1 - np.max(test_probs, axis=1)

    topic_test_probs.append(test_probs)

test_probs_np = np.zeros((topic_test_probs[0].shape))
for test_prob in topic_test_probs:
    test_probs_np += test_prob
test_probs_np /= 5

test_probs_df = pd.DataFrame()
test_probs_df['topic_vec_probs'] = test_probs_np
test_probs_df.to_hdf('./my_feat/topic_cv_probs_test_df_b.h5', key='data')
