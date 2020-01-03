import pickle

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

tqdm.pandas()

with open('./pkl/question_info.pkl', 'rb') as file:
    question_info = pickle.load(file)

question_info['topic'] = question_info['topic'].map(lambda x: ' '.join([str(i) for i in x]))
question_info.drop(['title_sw_series', 'title_w_series', 'desc_sw_series', 'desc_w_series'], axis=1, inplace=True)
question_info.drop(['question_day', 'question_hour'], axis=1, inplace=True)


def fun(x):
    if not isinstance(x, float):
        return ' '.join([str(i) for i in x])
    else:
        return ''


with open('./pkl/topic.pkl', 'rb') as file:
    topic = pickle.load(file)

topic_id_list = topic['id'].tolist()

cv = CountVectorizer(vocabulary=[str(i) for i in topic_id_list])

# ------------------------ 分割线 -------------------------------------
invite_info_evaluate = pd.read_hdf('./my_feat/convert_test_b.h5', key='data')
history_topic_with_current_topic_test = pd.read_hdf('./my_feat/history_topic_with_current_topic_test_b.h5', key='data')

history_topic_with_current_topic_test['topic_history'] = history_topic_with_current_topic_test['topic_history'].map(
    lambda x: list(set(x)))

invite_info_evaluate = invite_info_evaluate.merge(history_topic_with_current_topic_test, how='left',
                                                  on=['question_id', 'author_id'])

invite_info_evaluate.drop(
    ['invite_time', 'invite_day', 'invite_hour', 'author_id_convert', 'author_id_label_count', 'invite_q_topic',
     'intersect1d_topic', 'intersect1d_topic_nums'], axis=1, inplace=True)

invite_info_evaluate['topic_history'] = invite_info_evaluate['topic_history'].progress_apply(fun)

invite_info_evaluate = pd.merge(invite_info_evaluate, question_info, on='question_id', how='left')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)

topic_vec_df_list = []

user_history_topic_test = cv.fit_transform(invite_info_evaluate['topic_history'])
question_topic_test = cv.fit_transform(invite_info_evaluate['topic'])

test_data = sparse.hstack([user_history_topic_test, question_topic_test])

topic_test_probs = []

for k in range(5):
    print(f'start fold {k}')

    lr = pickle.load(open(f'./model/history_topic_cv_lr_fold_{k + 1}.pkl', 'rb'))

    test_probs = lr.predict_proba(test_data)
    test_probs = 1 - np.max(test_probs, axis=1)

    topic_test_probs.append(test_probs)

test_probs_np = np.zeros((topic_test_probs[0].shape))
for test_prob in topic_test_probs:
    test_probs_np += test_prob

test_probs_np /= 5
test_probs_df = pd.DataFrame()
test_probs_df['topic_vec_probs'] = test_probs_np
test_probs_df.to_hdf('./my_feat/history_topic_vec_cv_test_1118_test_b.h5', key='data')
