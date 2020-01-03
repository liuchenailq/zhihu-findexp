# 构建question的mean_vector
import pandas as pd
import pickle
import numpy as np
from joblib import Parallel, delayed
import multiprocessing

from multiprocessing.pool import Pool


'''
使用示例


def kw_distance(row):
    cos_dis_list = []
    kw_list = row['key_words_q_title']
    for kw in kw_list:
        kw_vec = w2v_dict.get(kw, np.zeros((64,)))
        cos_dis_list.append(cos_distance(kw_vec, row['topic_history_vec']))
    return cos_dis_list

def apply_fun(input_df):
    result_df = pd.DataFrame()
    result_df['kw_history_distance_list'] = input_df.apply(kw_distance, axis=1)
    return result_df

kw_cos_distance = multiprocessing_apply_data_frame(apply_fun, kw_q_title_history_topic)
'''


def multiprocessing_apply_data_frame(multiprocessing_apply_function, input_df: pd.DataFrame(), multiprocessing_nums=10):
    train_parts = np.array_split(input_df, multiprocessing_nums)

    with Pool(processes=multiprocessing_nums) as pool:
        result_parts = pool.map(multiprocessing_apply_function, train_parts)

    return pd.concat(result_parts)



qus_info = pd.read_csv('../datasets/question_info.csv',usecols=['qid','q_word_seq'])
with open('../datasets/word_vector.pkl','rb') as f:
    vector_list = pickle.load(f)
def mean_vector(df):
    word_seq = df['q_word_seq']
    if word_seq == '-1' or word_seq == '0' or word_seq == 0:
        return []
    tmp = []
    for w in word_seq.split(','):
        tmp.append(vector_list[int(w)])
    return np.around(np.mean(tmp,axis=0),decimals=4)

def tmp_func(df1):
    df1['mean_q_vector'] = df1.apply(mean_vector, axis=1)
    return df1

qus_info = multiprocessing_apply_data_frame(tmp_func,qus_info,10)

print(qus_info.head(10))

q_dict = dict(zip(qus_info['qid'].tolist(),qus_info['mean_q_vector'].tolist()))

with open('../datasets/qestion_mean_vector.pkl','wb+') as f:
    pickle.dump(q_dict,f,pickle.HIGHEST_PROTOCOL)




qus_info = pd.read_csv('../datasets/question_info.csv',usecols=['qid','topic_id'])
with open('../datasets/topic_vector.pkl','rb') as f:
    vector_list = pickle.load(f)
def mean_vector(df):
    word_seq = df['topic_id']
    if word_seq == '-1' or word_seq == '0' or word_seq == 0:
        return []
    tmp = []
    for w in word_seq.split(','):
        tmp.append(vector_list[int(w)])
    return np.around(np.mean(tmp,axis=0),decimals=4)


def tmp_func(df1):
    df1['mean_topic_vector'] = df1.apply(mean_vector, axis=1)
    return df1

qus_info = multiprocessing_apply_data_frame(tmp_func,qus_info,10)

print(qus_info.head(10))

q_dict = dict(zip(qus_info['qid'].tolist(),qus_info['mean_topic_vector'].tolist()))

with open('../datasets/topic_mean_vector.pkl','wb+') as f:
    pickle.dump(q_dict,f,pickle.HIGHEST_PROTOCOL)