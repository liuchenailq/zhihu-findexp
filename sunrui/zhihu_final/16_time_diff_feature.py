import pandas as pd
import numpy as np

train = pd.read_hdf('./my_feat/convert_train.h5', key='data')
test_a = pd.read_hdf('./my_feat/convert_test.h5', key='data')
test = pd.read_hdf('./my_feat/convert_test_b.h5', key='data')

train.drop(['author_id_convert', 'author_id_label_count'], axis=1, inplace=True)
test_a.drop(['author_id_convert', 'author_id_label_count'], axis=1, inplace=True)
test.drop(['author_id_convert', 'author_id_label_count'], axis=1, inplace=True)

train.columns = ['qid', 'uid', 'ctime', 'label', 'day', 'hour']
test_a.columns = ['qid', 'uid', 'ctime', 'day', 'hour']
test.columns = ['qid', 'uid', 'ctime', 'day', 'hour']

a1 = train.shape[0]
# a2 = dev.shape[0]
a3 = test.shape[0]
feature = pd.concat([train, test_a, test], axis=0).reset_index(drop=True)
feature['invite_time'] = feature['day'] + 0.04166 * feature['hour']

total = pd.concat([train, test], axis=0).reset_index(drop=True)
total['invite_time'] = total['day'] + 0.04166 * total['hour']
lens_df = pd.DataFrame({'src_i': np.arange(a1 + a3)})
total = pd.concat([total, lens_df], axis=1)

# 计算用户的未来访问时间差
total_sorted = feature.sort_values(by=['uid', 'invite_time'], axis=0, ascending=True).reset_index(drop=True)

t = total_sorted.groupby('uid')['invite_time'] \
    .apply(lambda x: np.subtract(x.tolist()[1:] + [4100], x.tolist()).tolist()).reset_index() \
    .rename(columns={'invite_time': 'key'})

t1 = pd.DataFrame(
    {'uid': t.uid.repeat(t.key.str.len()), 'merge_uid_time_feature': np.concatenate(t.key.values)}).reset_index(
    drop=True)
print(t1.shape)
print(t1.head(10))
t = total_sorted.groupby('uid')['invite_time'] \
    .apply(lambda x: np.subtract(x.tolist(), [2900] + x.tolist()[:-1]).tolist()).reset_index() \
    .rename(columns={'invite_time': 'key'})

t2 = pd.DataFrame(
    {'uid': t.uid.repeat(t.key.str.len()), 'merge_uid_time_past': np.concatenate(t.key.values)}).reset_index(
    drop=True)
print(t2.shape)
print(t2.head(10))

total_sorted = pd.concat([total_sorted, t1[['merge_uid_time_feature']], t2[['merge_uid_time_past']]], axis=1)[
    ['uid', 'qid', 'ctime', 'label', 'merge_uid_time_feature', 'merge_uid_time_past']]

mf = total_sorted.drop_duplicates(subset=['qid', 'uid', 'ctime', 'label'], keep='first')

total = pd.merge(total, mf, how='left', on=['qid', 'uid', 'ctime', 'label'])

featlist = ['merge_uid_time_feature', 'merge_uid_time_past']

print(total.shape)

train_save_df = total.iloc[:a1][featlist]
test_b_save_df = total.iloc[a1:][featlist]

print(test.shape)
print(test_a.shape)

test_b_save_df.reset_index(drop=True, inplace=True)
train_save_df.reset_index(drop=True, inplace=True)

train_save_df.to_hdf('./my_feat/time_diff_train.h5', key='data')
test_b_save_df.to_hdf('./my_feat/time_diff_test_b.h5', key='data')
