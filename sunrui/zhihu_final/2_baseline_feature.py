import gc
import multiprocessing as mp
import os
import pickle

import numpy as np
import pandas as pd


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


# 记得更换路径

with open('./pkl/invite_info.pkl', 'rb') as file:
    invite_info = pickle.load(file)
with open('./pkl/invite_info_evaluate_b.pkl', 'rb') as file:
    invite_info_evaluate = pickle.load(file)

with open('./pkl/member_info.pkl', 'rb') as file:
    member_info = pickle.load(file)
with open('./pkl/question_info.pkl', 'rb') as file:
    question_info = pickle.load(file)

# 合并 author_id，question_id
invite = pd.concat([invite_info, invite_info_evaluate])
invite_id = invite[['author_id', 'question_id']]
invite_id['author_question_id'] = invite_id['author_id'] + invite_id['question_id']
invite_id.drop_duplicates(subset='author_question_id', inplace=True)
invite_id_qm = invite_id.merge(member_info[['author_id', 'topic_attent', 'topic_interest']], 'left', 'author_id').merge(
    question_info[['question_id', 'topic']], 'left', 'question_id')
invite_id_qm.head(2)


# 分割 df，方便多进程跑
def split_df(df, n):
    chunk_size = int(np.ceil(len(df) / n))
    return [df[i * chunk_size:(i + 1) * chunk_size] for i in range(n)]


def gc_mp(pool, ret, chunk_list):
    del pool
    for r in ret:
        del r
    del ret
    for cl in chunk_list:
        del cl
    del chunk_list
    gc.collect()


# 用户关注topic和问题 topic的交集
def process(df):
    return df.apply(lambda row: list(set(row['topic_attent']) & set(row['topic'])), axis=1)


pool = mp.Pool()
chunk_list = split_df(invite_id_qm, 100)
ret = pool.map(process, chunk_list)
invite_id_qm['topic_attent_intersection'] = pd.concat(ret)
gc_mp(pool, ret, chunk_list)


# 用户感兴趣topic和问题 topic的交集
def process(df):
    return df.apply(lambda row: list(set(row['topic_interest'].keys()) & set(row['topic'])), axis=1)


pool = mp.Pool()
chunk_list = split_df(invite_id_qm, 100)
ret = pool.map(process, chunk_list)
invite_id_qm['topic_interest_intersection'] = pd.concat(ret)
gc_mp(pool, ret, chunk_list)


# 用户感兴趣topic和问题 topic的交集的兴趣值
def process(df):
    return df.apply(lambda row: [row['topic_interest'][t] for t in row['topic_interest_intersection']], axis=1)


pool = mp.Pool()
chunk_list = split_df(invite_id_qm, 100)
ret = pool.map(process, chunk_list)
invite_id_qm['topic_interest_intersection_values'] = pd.concat(ret)
gc_mp(pool, ret, chunk_list)

# 交集topic计数
invite_id_qm['num_topic_attent_intersection'] = invite_id_qm['topic_attent_intersection'].apply(len)
invite_id_qm['num_topic_interest_intersection'] = invite_id_qm['topic_interest_intersection'].apply(len)

# 交集topic兴趣值统计
invite_id_qm['topic_interest_intersection_values'] = invite_id_qm['topic_interest_intersection_values'].apply(
    lambda x: [0] if len(x) == 0 else x)
invite_id_qm['min_topic_interest_intersection_values'] = invite_id_qm['topic_interest_intersection_values'].apply(
    np.min)
invite_id_qm['max_topic_interest_intersection_values'] = invite_id_qm['topic_interest_intersection_values'].apply(
    np.max)
invite_id_qm['mean_topic_interest_intersection_values'] = invite_id_qm['topic_interest_intersection_values'].apply(
    np.mean)
invite_id_qm['std_topic_interest_intersection_values'] = invite_id_qm['topic_interest_intersection_values'].apply(
    np.std)

feats = ['author_question_id', 'num_topic_attent_intersection', 'num_topic_interest_intersection',
         'min_topic_interest_intersection_values', 'max_topic_interest_intersection_values',
         'mean_topic_interest_intersection_values', 'std_topic_interest_intersection_values']
feats += []
member_question_feat = invite_id_qm[feats]
member_question_feat.head(3)

member_question_feat.to_hdf('./feats/member_question_feat_final.h5', key='data')

del invite_id_qm, member_question_feat
gc.collect()
