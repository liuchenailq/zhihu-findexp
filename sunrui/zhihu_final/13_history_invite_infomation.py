import pandas as pd

from tqdm import tqdm

tqdm.pandas()

invite_info = pd.read_hdf('./my_feat/convert_train.h5', key='data')
invite_info_evaluate = pd.read_hdf('./my_feat/convert_test_b.h5', key='data')

invite_info['invite_time'] = invite_info.progress_apply(lambda row: (row['invite_day'] * 24 + row['invite_hour']),
                                                        axis=1)
invite_info_evaluate['invite_time'] = invite_info_evaluate.progress_apply(
    lambda row: (row['invite_day'] * 24 + row['invite_hour']), axis=1)
invite_info_evaluate['label'] = -1
invite_info.drop(['invite_day', 'invite_hour', 'author_id_convert', 'author_id_label_count'], axis=1, inplace=True)
invite_info_evaluate.drop(['invite_day', 'invite_hour', 'author_id_convert', 'author_id_label_count'], axis=1,
                          inplace=True)

invite_merge_user_df = pd.merge(invite_info, invite_info, on='author_id', how='left')

invite_merge_user_test_df = pd.merge(invite_info_evaluate, invite_info, on='author_id', how='left')

# 在当前的时间点上，这个用户被邀请了多少次
invite_merge_user_test_df['current_invite_user_count'] = \
    invite_merge_user_test_df.groupby(['question_id_x', 'author_id'])['author_id'].transform('count')
invite_merge_user_test_df['current_invite_user_convert'] = \
    invite_merge_user_test_df.groupby(['question_id_x', 'author_id'])['label_y'].transform('sum')
invite_merge_user_test_df['current_invite_user_success_rate'] = invite_merge_user_test_df[
                                                                    'current_invite_user_convert'] / \
                                                                invite_merge_user_test_df['current_invite_user_count']

invite_merge_user_test_df.drop_duplicates(['question_id_x', 'author_id'], inplace=True)
filter_invite_user_df_test = invite_merge_user_test_df[
    ['question_id_x', 'author_id', 'current_invite_user_convert', 'current_invite_user_success_rate']]
filter_invite_user_df_test.columns = ['question_id', 'author_id', 'current_invite_user_convert',
                                      'current_invite_user_success_rate']
filter_invite_user_df_test.to_hdf('./my_feat/current_invite_user_situation_test_b.h5', key='data', index=None)
