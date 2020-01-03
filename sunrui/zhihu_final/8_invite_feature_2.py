import pandas as pd
from tqdm import tqdm

invite_info = pd.read_hdf('./my_feat/convert_train.h5', key='data')
invite_info_evaluate = pd.read_hdf('./my_feat/convert_test_b.h5', key='data')

tqdm.pandas()

invite_info['invite_time'] = invite_info.progress_apply(lambda row: (row['invite_day'] * 24 + row['invite_hour']),
                                                        axis=1)

invite_info_evaluate['invite_time'] = invite_info_evaluate.progress_apply(
    lambda row: (row['invite_day'] * 24 + row['invite_hour']), axis=1)

invite_info.drop(['label', 'invite_day', 'invite_hour', 'author_id_convert', 'author_id_label_count'], axis=1,
                 inplace=True)
invite_info_evaluate.drop(['invite_day', 'invite_hour', 'author_id_convert', 'author_id_label_count'], axis=1,
                          inplace=True)

invite_data_all = pd.concat([invite_info, invite_info_evaluate], ignore_index=True)

invite_merge_df = pd.merge(invite_data_all, invite_data_all, on='author_id', how='inner')

invite_merge_df['time_gap'] = invite_merge_df['invite_time_x'] - invite_merge_df['invite_time_y']

filter_invite_df = invite_merge_df[invite_merge_df['time_gap'] > 0].reset_index(drop=True)

filter_invite_df['current_invite_count'] = filter_invite_df.groupby(['question_id_x', 'author_id'])[
    'author_id'].transform('count')

invite_drop_duplicates = filter_invite_df.drop_duplicates(['question_id_x', 'author_id'])

invite_drop_duplicates = invite_drop_duplicates[['question_id_x', 'author_id', 'current_invite_count']]

invite_drop_duplicates.reset_index(drop=True, inplace=True)

invite_drop_duplicates.columns = ['question_id', 'author_id', 'current_invite_count']

invite_drop_duplicates.to_hdf('./my_feat/current_invite_count_b.h5', key='data', index=None)
