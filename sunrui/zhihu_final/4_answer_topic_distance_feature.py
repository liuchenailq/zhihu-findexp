import pandas as pd
from tqdm import tqdm
import pickle
import numpy as np

tqdm.pandas()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)

with open('./pkl/question_info.pkl', 'rb') as file:
    question_info = pickle.load(file)
question_info.drop(['title_sw_series', 'title_w_series', 'desc_sw_series', 'desc_w_series'], axis=1, inplace=True)

with open('./pkl/answer_info.pkl', 'rb') as file:
    answer_info = pickle.load(file)

invite_info_evaluate = pd.read_hdf('./my_feat/convert_test_b.h5', key='data')

invite_info_evaluate['i_time'] = invite_info_evaluate.progress_apply(
    lambda row: (row['invite_day'] * 24 + row['invite_hour']), axis=1)
answer_info['a_time'] = answer_info.progress_apply(lambda row: (row['answer_day'] * 24 + row['answer_hour']), axis=1)

merge_df = pd.merge(invite_info_evaluate, answer_info, on='author_id', how='inner')

merge_df.drop(['author_id_convert', 'author_id_label_count', 'invite_time'], axis=1, inplace=True)

question_info.drop(['question_day', 'question_hour'], axis=1, inplace=True)

question_info.columns = ['question_id_y', 'topic']  # 先生成一下历史回答过的问题的topic

answer_info_df = merge_df[['question_id_x', 'author_id', 'question_id_y']]

answer_info_df = pd.merge(answer_info_df, question_info, on='question_id_y', how='left')


def agg(x):
    result = []
    for i in x:
        result.extend(i)
    return result


temp = answer_info_df.groupby(['question_id_x', 'author_id'])['topic'].progress_apply(agg)

answer_info_df = pd.merge(answer_info_df, temp, on=['question_id_x', 'author_id'], how='left')

answer_info_df.drop(['topic_x'], axis=1, inplace=True)

question_info.columns = ['question_id_x', 'topic']

answer_info_df.columns = ['question_id_x', 'author_id', 'question_id_y', 'topic_history']
answer_info_df = pd.merge(answer_info_df, question_info, on='question_id_x', how='left')

answer_info_df.drop(['question_id_y'], axis=1, inplace=True)
answer_info_df.columns = ['question_id', 'author_id', 'topic_history', 'invite_q_topic']

answer_info_df.drop_duplicates(['question_id', 'author_id'], inplace=True)

answer_info_df['intersect1d_topic'] = answer_info_df.progress_apply(
    lambda row: list(np.intersect1d(row['topic_history'], row['invite_q_topic'])), axis=1)
answer_info_df['intersect1d_topic_nums'] = answer_info_df['intersect1d_topic'].map(len)

answer_info_df.to_hdf('./my_feat/history_topic_with_current_topic_test_b.h5', key='data', index=None)  # 生成test_b的历史回答数据

answer_info_df['topic_history_set'] = answer_info_df['topic_history'].progress_apply(lambda x: {}.fromkeys(x).keys())
answer_info_df['topic_history_set'] = answer_info_df['topic_history_set'].map(list)

with open('./pkl/topic.pkl', 'rb') as f:
    topic_info = pickle.load(f)

topic_vec_dict = {}

for id_, embed in tqdm(topic_info[['id', 'embed']].values, 'gen topic dict'):
    topic_vec_dict[id_] = embed


def gen_ave_vec(t_list, dim=64):
    vec_sum = np.zeros((dim,))
    for t in t_list:
        vec = topic_vec_dict.get(t, np.zeros(dim, ))
        vec_sum = vec_sum + vec
    vec_ave = vec_sum / len(t_list)
    return vec_ave


def cos_distance(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * (np.linalg.norm(v2)))


def gen_ave_topic_vec_distance(row):
    t_history = row['topic_history_set']  # 历史的topic list
    t_q = row['invite_q_topic']  # 当前问题的topic list
    t_history_vec = gen_ave_vec(t_history)
    t_q_vec = gen_ave_vec(t_q)
    return cos_distance(t_history_vec, t_q_vec)


answer_info_df['history_topic_vec_distance'] = answer_info_df.apply(gen_ave_topic_vec_distance, axis=1)

answer_info_df[['question_id', 'author_id', 'history_topic_vec_distance']].to_hdf(
    './my_feat/history_topic_vec_distance_test_b.h5', key='data', index=None)