import pandas as pd

invite_info = pd.read_hdf('./my_feat/convert_train.h5', key='data')
invite_info_evaluate = pd.read_hdf('./my_feat/convert_test_b.h5', key='data')
invite_info_evaluate['label'] = -1
total = pd.concat([invite_info, invite_info_evaluate], axis=0)

total.drop(['author_id_convert', 'author_id_label_count', 'invite_time'], axis=1, inplace=True)

total = total.rename({
    'author_id': 'uid',
    'question_id': 'qid',
    'invite_hour': 'hour',
    'invite_day': 'day'
})

total.columns = ['uid', 'day', 'hour', 'label', 'qid']

from tqdm import tqdm

stat_feat = [
    (['uid'], ['label'], ['count']),
    (['qid'], ['label'], ['count']),
    (['uid', 'day'], ['label'], ['count']),
    (['qid', 'day'], ['label'], ['count']),
    (['uid', 'day', 'hour'], ['label'], ['count']),
    (['qid', 'day', 'hour'], ['label'], ['count']),
    (['uid', 'hour'], ['label'], ['count']),
    (['qid', 'hour'], ['label'], ['count']),
    (['qid'], ['hour'], ['nunique']),
    (['qid'], ['day'], ['nunique']),
    (['uid'], ['day'], ['nunique']),
    (['uid'], ['hour'], ['nunique'])
]
data = total  # total是把训练集，测试机，验证机拼接起来了
fea_list = []
for stat in tqdm(stat_feat):
    fea_name = '_'.join(stat[0]) + '_' + '_'.join(stat[1]) + '_' + '_'.join(stat[2])
    print(fea_name)
    fea_list.append(fea_name)
    t = total.groupby(stat[0])[stat[1][0]].agg(stat[2]).reset_index() \
        .rename(columns={stat[2][0]: fea_name})
    data = pd.merge(data, t, how='left', on=stat[0]).fillna(0)

train = data[data['label'] != -1]
test = data[data['label'] == -1]

train.drop(['uid', 'qid', 'hour', 'day', 'label'], axis=1, inplace=True)
test.drop(['uid', 'qid', 'hour', 'day', 'label'], axis=1, inplace=True)
train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

train.to_hdf('./my_feat/count_and_unique_feature_train_final_train.h5', key='data')
test.to_hdf('./my_feat/count_and_unique_feature_train_final_test.h5', key='data')
