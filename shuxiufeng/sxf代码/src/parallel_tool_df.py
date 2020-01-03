from multiprocessing.pool import Pool

import pandas as pd
import numpy as np

'''
使用示例
from parallel_tool_df import multiprocessing_apply_data_frame

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
