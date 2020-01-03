import pandas as pd
from tqdm import tqdm


def id_seq_feature(train_feats, train_labels, test_feats):
    feat_seq = {}
    train_seq = []
    train_seq_3 = []  # 取后3个
    train_seq_5 = []  # 取后3个

    for i, feat in tqdm(enumerate(train_feats)):
        if feat not in feat_seq.keys():
            feat_seq[feat] = []
        train_seq.append(str(feat_seq[feat]))
        train_seq_3.append(str(feat_seq[feat][-3:]))
        train_seq_5.append(str(feat_seq[feat][-5:]))
        feat_seq[feat].append(int(train_labels[i]))

    test_seq = []
    test_seq_3 = []
    test_seq_5 = []
    for i, feat in tqdm(enumerate(test_feats)):
        if feat not in feat_seq.keys():
            feat_seq[feat] = []
        test_seq.append(str(feat_seq[feat]))
        test_seq_3.append(str(feat_seq[feat][-3:]))
        test_seq_5.append(str(feat_seq[feat][-5:]))

    train_seq.extend(test_seq)
    train_seq_3.extend(test_seq_3)
    train_seq_5.extend(test_seq_5)
    return train_seq, train_seq_3, train_seq_5


invite_info = pd.read_hdf('./my_feat/convert_train.h5', key='data')
invite_info_evaluate = pd.read_hdf('./my_feat/convert_test_b.h5', key='data')

invite_info['old_idx'] = invite_info.index
invite_info['tag'] = 1

invite_info_evaluate['old_idx'] = invite_info_evaluate.index
invite_info_evaluate['tag'] = 0

drop_cols = ['invite_time', 'author_id_convert', 'author_id_label_count']
invite_info.drop(drop_cols, inplace=True, axis=1)
invite_info_evaluate.drop(drop_cols, inplace=True, axis=1)

total = pd.concat([invite_info, invite_info_evaluate], axis=0, ignore_index=True, sort=False)

total['invite_time'] = total['invite_day'] + total['invite_hour'] * 0.04166
total.sort_values('invite_time', inplace=True)

total['uid_seq_feature'], total['uid_seq_feature_3'], total['uid_seq_feature_5'] = id_seq_feature(
    total[~total['label'].isnull()]['author_id'].values,
    total[~total['label'].isnull()]['label'].values,
    total[total['label'].isnull()]['author_id'].values)

from sklearn.preprocessing import LabelEncoder

total['uid_seq_feature'] = LabelEncoder().fit_transform(total['uid_seq_feature'])
total['uid_seq_feature_3'] = LabelEncoder().fit_transform(total['uid_seq_feature_3'])
total['uid_seq_feature_5'] = LabelEncoder().fit_transform(total['uid_seq_feature_5'])

train_df = total[total['tag'] == 1]
test_df = total[total['tag'] == 0]

train_df.sort_values('old_idx', inplace=True)
test_df.sort_values('old_idx', inplace=True)

train_df[['uid_seq_feature', 'uid_seq_feature_3', 'uid_seq_feature_5']].to_hdf('./my_feat/uid_seq_feature_train_b.h5',
                                                                               key='data')
test_df[['uid_seq_feature', 'uid_seq_feature_3', 'uid_seq_feature_5']].to_hdf('./my_feat/uid_seq_feature_test_b.h5',
                                                                              key='data')
