import pandas as pd
import pickle

with open('./pkl/question_info.pkl', 'rb') as f:
    q_info = pickle.load(f)

q_info = q_info[['question_id']]

invite_info = pd.read_hdf('./my_feat/convert_train.h5', key='data').rename(
    columns={'author_id_label_count': 'author_id_click_count'})
invite_info_evaluate = pd.read_hdf('./my_feat/convert_test_b.h5', key='data').rename(
    columns={'author_id_label_count': 'author_id_click_count'})

drop_cols = ['author_id_convert', 'author_id_click_count', 'invite_time', 'invite_day', 'invite_hour', 'invite_time']
invite_info.drop(drop_cols, axis=1, inplace=True)
invite_info_evaluate.drop(drop_cols, axis=1, inplace=True)

invite_info['idx'] = invite_info.index  # 加编号 方便等下concat回去
invite_info_evaluate['idx'] = invite_info_evaluate.index  # 加编号 方便等下concat回去

q_key_words_df = pd.read_hdf('./my_feat/key_words_q_title_gensim.h5', key='data')
q_key_words_df['key_words_q_title'] = q_key_words_df['key_words_q_title'].map(lambda x: ','.join([str(i) for i in x]))

q_key_words_df = pd.concat([q_info, q_key_words_df], axis=1)

invite_info = invite_info.merge(q_key_words_df, on='question_id', how='left')
invite_info_evaluate = invite_info_evaluate.merge(q_key_words_df, on='question_id', how='left')

train_df = invite_info['key_words_q_title'].str.split(',', expand=True).stack() \
    .reset_index(level=0).set_index('level_0') \
    .rename(columns={0: 'key_words_q_title'}).join(invite_info.drop('key_words_q_title', axis=1)) \
    .reset_index(drop=True)
train_df['key_words_q_title'] = train_df['key_words_q_title'].map(int)

test_df = invite_info_evaluate['key_words_q_title'].str.split(',', expand=True).stack() \
    .reset_index(level=0).set_index('level_0') \
    .rename(columns={0: 'key_words_q_title'}).join(invite_info_evaluate.drop('key_words_q_title', axis=1)) \
    .reset_index(drop=True)

test_df['key_words_q_title'] = test_df['key_words_q_title'].map(int)

test_df['label'] = -1
total_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
total_df['kw_total_count'] = total_df.groupby('key_words_q_title')['author_id'].transform('count')

train_df = total_df[total_df['label'] != -1]
test_df = total_df[total_df['label'] == -1]

test_df['kw_max_count'] = test_df.groupby('idx')['kw_total_count'].transform('max')
test_df['kw_mean_count'] = test_df.groupby('idx')['kw_total_count'].transform('mean')
test_df['kw_min_count'] = test_df.groupby('idx')['kw_total_count'].transform('min')
test_df['kw_sum_count'] = test_df.groupby('idx')['kw_total_count'].transform('sum')

test_df.drop_duplicates(['idx'], inplace=True)
test_df.reset_index(drop=True, inplace=True)

save_cols = ['kw_max_count', 'kw_mean_count', 'kw_min_count', 'kw_sum_count']
test_df[save_cols].to_hdf('./my_feat/kw_count_statistics_test_b.h5', key='data', index=None)
