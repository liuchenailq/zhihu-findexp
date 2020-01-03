import pandas as pd
import numpy as np
import datetime


import os

import pickle
from ctr_utils import *
from parallel_tool_df import multiprocessing_apply_data_frame


mode = 'train'

def print_time(name):
    time = datetime.datetime.now()
    a = time.strftime('%Y-%m-%d %H:%M')
    print(a,'---' + name)


src_train = pd.read_csv('../datasets1/invite_train.csv')
src_dev = pd.read_csv('../datasets1/invite_dev.csv')
src_test = pd.read_csv('../datasets3/test_final.csv')


use_train = pd.read_csv('../datasets3/invite_train_use.csv')

totals = pd.concat([src_train,src_dev],axis=0).reset_index(drop=True)

train_shape = use_train.shape[0]


ans_info = pd.read_csv('../datasets2/answer_info.csv')
                        #预测区间      特征构造区间   ans表使用时间
train_time_windows = [([3858, 3864], [3840, 3857], [3827, 3857])]
dev_time_windows = [([3865, 3867], [3846, 3863], [3833, 3863])]
test_time_windows = [([3868, 3874], [3850, 3867], [3837, 3867])]


def save_feature(mode, df):
    featlist = df.columns.tolist()
    for f in featlist:
        if f in ['src_i', 'qid', 'uid', 'day', 'label', 'hour', 'ctime']:
            continue
        s = df[[f]]
        print(mode, f, s.shape)
        s.to_csv('../datasets3/feature/' + mode + '/' + f + '.csv', index=False)




def get_time_1day_before(df):
    time_list = df['answer_time_list']
    if time_list == 0 or time_list == '0':
        time_list = []
    cur_time = df['i_time']
    list1 = [time for time in time_list if (cur_time - time) >= 1]
    if len(list1) > 0:
        return np.max(list1)
    else:
        return np.NAN

def tmp_func1(df1):
    df1['merge_last_ans_time_1b'] = df1.apply(get_time_1day_before, axis=1)
    return df1


def get_time_7day_before(df):
    time_list = df['answer_time_list']

    if time_list == 0 or time_list == '0':
        time_list = []

    time_list = time_list[:-1]
    cur_time = df['i_time']
    list1 = [time for time in time_list if (cur_time - time) >= 1]
    if len(list1) > 0:
        return np.max(list1)
    else:
        return np.NAN

def tmp_func7(df1):
    df1['merge_last_ans_time_7b'] = df1.apply(get_time_7day_before, axis=1)
    return df1



def ans_time_feat(train, feature, mode, ans):
    feature['i_time'] = feature['day'] + 0.04166 * feature['hour']
    train['i_time'] = train['day'] + 0.04166 * train['hour']
    if mode == 'test_B':
        s = pd.read_csv('../datasets1/invite_test.csv')
        s_score = pd.read_csv('../datasets3/feature/test/p_score')
        s['i_time'] = s['day'] + 0.04166 * s['hour']
        s = pd.concat([s, s_score], axis=1)
        s = s[s['p_score'] >= 0.42]

    train_score = pd.read_csv('../datasets3/feature/' + mode + '/p_score')

    print(train_score.shape)

    s1 = feature[feature['label'] == 1]
    train = pd.concat([train, train_score], axis=1)
    s2 = train[train['p_score'] >= 0.42]
    s1 = s1[['uid', 'i_time']]
    s2 = s2[['uid', 'i_time']]

    if mode == 'test_B':
        s = s[['uid', 'i_time']]
        s = pd.concat([s1, s2, s], axis=0)
    else:
        s = pd.concat([s1,s2],axis=0)

    # 得到每个用户7天前的回答列表
    def get_time_list(df):
        return sorted(df['i_time'].tolist())

    t = s.groupby(['uid']).apply(get_time_list).reset_index().rename(columns={0: 'answer_time_list'})

    train = pd.merge(train, t, on='uid', how='left')
    train['answer_time_list'] = train['answer_time_list'].fillna(0)

    train = multiprocessing_apply_data_frame(tmp_func1, train, 10)

    train = multiprocessing_apply_data_frame(tmp_func7, train, 10)

    train['merge_ans_dif_1'] = train['i_time'] - train['merge_last_ans_time_1b']

    train['merge_ans_dif_7'] = train['i_time'] - train['merge_last_ans_time_7b']

    train = train[['merge_ans_dif_1', 'merge_ans_dif_7']]

    return train




qus_info = pd.read_csv('../datasets/question_info.csv',usecols=['qid','topic_id'])

q_dict = dict(zip(qus_info['qid'].tolist(),qus_info['topic_id'].tolist()))


def get_time_topic_before(df):
    time_list = df['answer_time_list']
    topic_id = df['topic_id']
    if time_list == 0 or time_list == '0' or topic_id == '0':
        return np.NAN

    cur_time = df['i_time']
    rlist = []

    topic_id = set(topic_id.split(','))
    for item in time_list:
        t = q_dict[item[0]]
        t = set(t.split(','))
        if len(t & topic_id) > 0:
            rlist.append(item[1])

    list1 = [time for time in rlist if (cur_time - time) >= 1]
    if len(list1) > 0:
        return np.max(list1)
    else:
        return np.NAN

def tmp_func3(df1):
    df1['merge_last_ans_time_topic_1b'] = df1.apply(get_time_topic_before, axis=1)
    return df1

def _ans_topic_feat(train,feature,mode,ans):

    feature['i_time'] = feature['day'] + 0.04166 * feature['hour']
    train['i_time'] = train['day'] + 0.04166 * train['hour']
    feature = pd.merge(feature,qus_info,how='left',on='qid').fillna('0')
    train = pd.merge(train, qus_info, how='left', on='qid').fillna('0')

    if mode == 'test_B':
        s = pd.read_csv('../datasets/invite_test.csv')
        s_score = pd.read_csv('../datasets/feature/test/p_score')
        s['i_time'] = s['day'] + 0.04166 * s['hour']
        s = pd.concat([s, s_score], axis=1)
        s = s[s['p_score'] >= 0.42]
        s = pd.merge(s, qus_info, how='left', on='qid').fillna('0')

    train_score = pd.read_csv('../datasets/feature/' + mode + '/p_score')

    print(train_score.shape)

    s1 = feature[feature['label'] == 1]
    train = pd.concat([train, train_score], axis=1)
    s2 = train[train['p_score'] >= 0.42]
    s1 = s1[['uid', 'qid','i_time']]
    s2 = s2[['uid', 'qid','i_time']]


    if mode == 'test_B':
        s = s[['uid', 'qid', 'i_time']]
        s = pd.concat([s1, s2, s], axis=0)
    else:
        s = pd.concat([s1,s2],axis=0)




    s = pd.concat([s1, s2, s], axis=0)

    # 得到每个用户7天前的回答列表
    def get_time_list(df):

        qid_list = df['qid'].tolist()
        time_list = df['i_time'].tolist()
        tmp = []
        for i in range(len(qid_list)):
            tmp.append((qid_list[i],time_list[i]))
        return sorted(tmp,key=lambda x:x[1])

    t = s.groupby(['uid']).apply(get_time_list).reset_index().rename(columns={0: 'answer_time_list'})

    train = pd.merge(train, t, on='uid', how='left')
    train['answer_time_list'] = train['answer_time_list'].fillna(0)

    train = multiprocessing_apply_data_frame(tmp_func3, train, 10)

    train['merge_ans_dif_topic_1'] = train['i_time'] - train['merge_last_ans_time_topic_1b']

    train = train[['merge_ans_dif_topic_1']]

    return train



def get_seq_click_before(df):
    time_list = df['answer_time_list']
    if time_list == 0 or time_list == '0':
        return np.NAN
    cur_time = df['i_time']
    result = []
    for item in time_list:
        time = item[1]
        if cur_time - time > 0 and cur_time - time < 8:
            result.append(item[0])
    if len(result) > 0:
        return np.mean(result)
    else:
        return np.NAN

def tmp_func4(df1):
    df1['merge_recent_7_day_click_rate'] = df1.apply(get_seq_click_before, axis=1)
    return df1



def uid_seq_mlabel(train,feature,mode,ans):

    feature['i_time'] = feature['day'] + 0.04166 * feature['hour']
    train['i_time'] = train['day'] + 0.04166 * train['hour']


    if mode == 'test_B':
        s = pd.read_csv('../datasets/invite_test.csv')
        s_score = pd.read_csv('../datasets/feature/test/p_score')
        s['i_time'] = s['day'] + 0.04166 * s['hour']
        s = pd.concat([s, s_score], axis=1)
        s['m_label'] = s['p_score'].apply(lambda x: 1 if x >= 0.42 else 0)

    train_score = pd.read_csv('../datasets/feature/' + mode + '/p_score')

    print(train_score.shape)

    feature['m_label'] = feature['label']
    train = pd.concat([train, train_score], axis=1)
    train['m_label'] = train['p_score'].apply(lambda x: 1 if x >= 0.42 else 0)

    train = train[['uid','qid','i_time','m_label']]
    feature = feature[['uid', 'qid', 'i_time', 'm_label']]

    if mode == 'test_B':
        s = s[['uid', 'qid', 'i_time', 'm_label']]

        s = pd.concat([train, feature,s], axis=0)
    else:
        s = pd.concat([train, feature], axis=0)

    # 得到每个用户7天前的回答列表
    def get_time_list(df):

        qid_list = df['m_label'].tolist()
        time_list = df['i_time'].tolist()
        tmp = []
        for i in range(len(qid_list)):
            tmp.append((qid_list[i],time_list[i]))
        tmp = sorted(tmp,key=lambda x:x[1])
        return tmp
    t = s.groupby(['uid']).apply(get_time_list).reset_index().rename(columns={0: 'answer_time_list'})

    train = pd.merge(train, t, on='uid', how='left')
    train['answer_time_list'] = train['answer_time_list'].fillna(0)

    train = multiprocessing_apply_data_frame(tmp_func4, train, 10)


    train = train[['merge_recent_7_day_click_rate']]

    return train



def get_ans_hour_feat(train, feature, mode, ans):


    print(feature.head(1))
    s_feature = feature[feature['label'] == 1]
    t = s_feature.groupby(['uid'])['hour'].agg(['mean','std']).reset_index()\
        .rename(columns={'mean':'uid_invite_ans_hour_mean','std':'uid_invite_ans_hour_std'})

    train = pd.merge(train,t,how='left',on='uid')

    t = feature.groupby(['uid'])['hour'].agg(['mean', 'std']).reset_index() \
        .rename(columns={'mean': 'uid_invite_hour_mean', 'std': 'uid_invite_hour_std'})

    train = pd.merge(train, t, how='left', on='uid')

    t = ans.groupby(['uid'])['a_hour'].agg(['mean', 'std']).reset_index() \
        .rename(columns={'mean': 'uid_ans_hour_mean', 'std': 'uid_ans_hour_std'})

    train = pd.merge(train, t, how='left', on='uid')

    train['uid_ans_hour_mean_diff'] = np.fabs(train['uid_ans_hour_mean'] - train['hour'])
    train['uid_invite_ans_hour_mean_diff'] = np.fabs(train['uid_invite_ans_hour_mean'] - train['hour'])
    train['uid_invite_hour_mean_diff'] = np.fabs(train['uid_invite_hour_mean'] - train['hour'])


    return train


print('train')
window = train_time_windows[0]
target_range = window[0]
train_target = use_train
feature_range = window[1]
train_feature = src_train[(src_train['day'] >= feature_range[0]) & (src_train['day'] <= feature_range[1])]

ans_range = window[2]
ans_feature = ans_info[(ans_info['a_day'] >= ans_range[0]) & (ans_info['a_day'] <= ans_range[1])]





result = ans_time_feat(train_target,train_feature,'train',ans_feature)
save_feature('train',result)

result = _ans_topic_feat(train_target,train_feature,'train',ans_feature)
save_feature('train',result)

result = uid_seq_mlabel(train_target,train_feature,'train',ans_feature)
save_feature('train',result)
#
result = get_ans_hour_feat(train_target,train_feature,'train',ans_feature)
save_feature('train',result)
# #
# #
# #
# # # # 构造验证机
print('dev')
window = dev_time_windows[0]
train_target = src_dev
feature_range = window[1]
train_feature = src_train[(src_train['day'] >= feature_range[0]) & (src_train['day'] <= feature_range[1])]

ans_range = window[2]
ans_feature = ans_info[(ans_info['a_day'] >= ans_range[0]) & (ans_info['a_day'] <= ans_range[1])]

result = ans_time_feat(train_target,train_feature,'dev',ans_feature)
save_feature('dev',result)

#
result = _ans_topic_feat(train_target,train_feature,'dev',ans_feature)
save_feature('dev',result)

result = uid_seq_mlabel(train_target,train_feature,'dev',ans_feature)
save_feature('dev',result)

result = get_ans_hour_feat(train_target,train_feature,'dev',ans_feature)
save_feature('dev',result)
#


# # 构造验证机
print('test')
window = test_time_windows[0]
train_target = src_test
feature_range = window[1]
train_feature = totals[(totals['day'] >= feature_range[0]) & (totals['day'] <= feature_range[1])]

af = train_feature.copy()

ans_range = window[2]
ans_feature = ans_info[(ans_info['a_day'] >= ans_range[0]) & (ans_info['a_day'] <= ans_range[1])]

bf = ans_feature.copy()

result = ans_time_feat(train_target,train_feature,'test_B',ans_feature)
save_feature('test_B',result)

result = _ans_topic_feat(train_target,train_feature,'test_B',ans_feature)
save_feature('test_B',result)

result = uid_seq_mlabel(train_target,train_feature,'test_B',ans_feature)
save_feature('test_B',result)

result = get_ans_hour_feat(train_target,af,'test_B',bf)
save_feature('test_B',result)

