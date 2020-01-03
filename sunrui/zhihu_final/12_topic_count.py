import pandas as pd
import pickle

with open('./pkl/question_info.pkl', 'rb') as file:
    question_info = pickle.load(file)

invite_info = pd.read_hdf('./my_feat/convert_train.h5', key='data')
invite_info_evaluate = pd.read_hdf('./my_feat/convert_test_b.h5', key='data')

question_info = question_info[['question_id', 'topic']]

invite_info = invite_info[['question_id', 'author_id', 'label']]
invite_info['idx'] = invite_info.index  # 加编号 方便等下concat回去
invite_info = invite_info.merge(question_info, on='question_id', how='left')

invite_info['topic'] = invite_info['topic'].map(lambda x: ','.join([str(i) for i in x]))

train_df = invite_info['topic'].str.split(',', expand=True).stack() \
    .reset_index(level=0).set_index('level_0') \
    .rename(columns={0: 'topic'}).join(invite_info.drop('topic', axis=1)) \
    .reset_index(drop=True)
train_df['topic'] = train_df['topic'].map(int)

invite_info_evaluate = invite_info_evaluate[['question_id', 'author_id']]
invite_info_evaluate['idx'] = invite_info_evaluate.index  # 加编号 方便等下concat回去
invite_info_evaluate = invite_info_evaluate.merge(question_info, on='question_id', how='left')
invite_info_evaluate['topic'] = invite_info_evaluate['topic'].map(lambda x: ','.join([str(i) for i in x]))

test_df = invite_info_evaluate['topic'].str.split(',', expand=True).stack() \
    .reset_index(level=0).set_index('level_0') \
    .rename(columns={0: 'topic'}).join(invite_info_evaluate.drop('topic', axis=1)) \
    .reset_index(drop=True)

test_df['topic'] = test_df['topic'].map(int)

test_df['label'] = -1

total_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

total_df['topic_all_count'] = total_df.groupby('topic')['author_id'].transform('count')

train_df = total_df[total_df['label'] != -1]
test_df = total_df[total_df['label'] == -1]

test_df['topic_all_max_count'] = test_df.groupby('idx')['topic_all_count'].transform('max')
test_df['topic_all_mean_count'] = test_df.groupby('idx')['topic_all_count'].transform('mean')
test_df['topic_all_min_count'] = test_df.groupby('idx')['topic_all_count'].transform('min')
test_df['topic_all_sum_count'] = test_df.groupby('idx')['topic_all_count'].transform('sum')

train_df.drop_duplicates(['idx'], inplace=True)
test_df.drop_duplicates(['idx'], inplace=True)

train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

save_cols = ['topic_all_max_count', 'topic_all_mean_count', 'topic_all_min_count', 'topic_all_sum_count']
test_df[save_cols].to_hdf('./my_feat/topic_count_statistics_test_b.h5', key='data')
