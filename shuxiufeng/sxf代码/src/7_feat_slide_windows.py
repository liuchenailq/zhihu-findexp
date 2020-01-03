#import pandas as pd
import modin.pandas as pd
import numpy as np
import datetime
import os
import pickle
from ctr_utils import *
from parallel_tool_df import multiprocessing_apply_data_frame


test_file_path = '../datasets3/test_final.csv'

mode = 'test_B'

def print_time(name):
    time = datetime.datetime.now()
    a = time.strftime('%Y-%m-%d %H:%M')
    print(a,'---' + name)
print('----------------------start------------------------')


src_train = pd.read_csv('../datasets/invite_train.csv')
src_dev = pd.read_csv('../datasets/invite_dev.csv')
src_test = pd.read_csv('../datasets/invite_test.csv')


use_train = pd.read_csv('../datasets/invite_train_use.csv')

totals = pd.concat([src_train,src_dev],axis=0).reset_index(drop=True)

train_shape = use_train.shape[0]


ans_info = pd.read_csv('../datasets/answer_info.csv')
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
        s.to_csv('../datasets/feature/' + mode + '/' + f + '.csv', index=False)


#####交叉转化率代码##########
def extruct_cross_feature_topic(data, feature):
    target = data.copy()

    print_time('extract cross feature topic')
    que = pd.read_csv('../datasets/question_info.csv', usecols=['qid', 'topic_id'])
    member_info = pd.read_csv('../datasets/member_info.csv'
            , usecols=['uid', 'sex','visit','CA', 'CB', 'CC', 'CD', 'CE'])

    feature = pd.merge(feature, que, how='left', on='qid').fillna('0')
    feature = pd.merge(feature, member_info, how='left', on='uid').fillna('0')

    target = pd.merge(target, que, how='left', on='qid').fillna('0')
    target = pd.merge(target, member_info, how='left', on='uid').fillna('0')

    target['flag'] = 1
    feature['flag'] = 1

    total_extend = feature['topic_id'].str.split(',', expand=True).stack() \
        .reset_index(level=0).set_index('level_0') \
        .rename(columns={0: 'topic'}).join(feature.drop('topic_id', axis=1)) \
        .reset_index(drop=True)

    topic_df = target['topic_id'].str.split(',', expand=True)
    target = pd.concat([target, topic_df], axis=1)

    fea_list = ['flag', 'uid', 'sex','visit', 'CA', 'CB', 'CC', 'CD', 'CE']

    result_list = []
    for fea in fea_list:
        fea_name = 'topic_' + fea + '_rate'
        print(fea_name)
        t = total_extend.groupby(['topic', fea])['label'].agg(['count','sum']).reset_index() \
            .rename(columns={'count':'count_s','sum':'sum_s'})

        HP = HyperParam(1, 1)
        HP.update_from_data_by_moment(t['count_s'].values,
                                      t['sum_s'].values)  # 矩估计
        t[fea_name] = np.divide(t['sum_s'] + HP.alpha,t['count_s'] + HP.alpha + HP.beta)
        t = t.drop(['count_s','sum_s'],axis=1)

        tmp_name = []
        for field in [0, 1, 2, 3, 4, 5]:
            target = pd.merge(target, t, how='left', left_on=[fea, field], right_on=[fea, 'topic']).rename(
                columns={fea_name: fea_name + str(field)})
            tmp_name.append(fea_name + str(field))

        target[fea_name + '_max'] = target[tmp_name].max(axis=1)
        target[fea_name + '_mean'] = target[tmp_name].mean(axis=1)
        result_list.append(fea_name + '_max')
        result_list.append(fea_name + '_mean')

        for field in [0, 1, 2, 3, 4, 5]:
            target = target.drop([fea_name + str(field)], axis=1)

    return target[result_list]



def merge_func(df1):
    for field in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
        df1 = pd.merge(df1, t, how='left', left_on=[fea, field], right_on=[fea, 'word']).rename(
            columns={fea_name: fea_name + str(field)})
    return df1


def extruct_cross_feature_word(data, feature):
    # global t
    # global fea_name
    # global fea

    target = data.copy()
    print_time('extract cross feature topic')
    que = pd.read_csv('../datasets/question_info.csv', usecols=['qid', 'q_word_seq'])
    member_info = pd.read_csv('../datasets/member_info.csv'
            , usecols=['uid', 'sex', 'visit','CA', 'CB', 'CC', 'CD', 'CE'])

    feature = pd.merge(feature, que, how='left', on='qid').fillna('0')
    feature = pd.merge(feature, member_info, how='left', on='uid').fillna('0')

    target = pd.merge(target, que, how='left', on='qid').fillna('0')
    target = pd.merge(target, member_info, how='left', on='uid').fillna('0')

    target = target.drop(['qid','ctime','day','hour'],axis=1)
    feature = feature.drop(['qid','ctime','day','hour'],axis=1)
    target['flag'] = 1
    feature['flag'] = 1

    print_time('extend')

    total_extend = feature['q_word_seq'].str.split(',', expand=True).stack() \
        .reset_index(level=0).set_index('level_0') \
        .rename(columns={0: 'word'}).join(feature.drop('q_word_seq', axis=1)) \
        .reset_index(drop=True)

    print('extend_finish')

    topic_df = target['q_word_seq'].str.split(',', expand=True)
    target = pd.concat([target, topic_df], axis=1)

    fea_list = ['uid','flag', 'sex','visit', 'CA', 'CB', 'CC', 'CD', 'CE']

    result_list = []
    for s in fea_list:
        fea = s
        fea_name = 'word_' + fea + '_rate'
        print(fea_name)
        s_total_extend = total_extend[['word',fea,'label']]
        t = s_total_extend.groupby(['word', fea])['label'].agg(['count','sum']).reset_index() \
            .rename(columns={'count':'count_s','sum':'sum_s'})

        HP = HyperParam(1, 1)
        HP.update_from_data_by_moment(t['count_s'].values,
                                      t['sum_s'].values)  # 矩估计
        t[fea_name] = np.divide(t['sum_s'] + HP.alpha,t['count_s'] + HP.alpha + HP.beta)

        print('pinghua', HP.alpha, HP.beta)
        t = t.drop(['count_s','sum_s'],axis=1)

        tmp_name = []
        for field in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
            target = pd.merge(target, t, how='left', left_on=[fea, field], right_on=[fea, 'word']).rename(
                columns={fea_name: fea_name + str(field)})
            tmp_name.append(fea_name + str(field))

        target[fea_name + '_max'] = target[tmp_name].max(axis=1)
        target[fea_name + '_mean'] = target[tmp_name].mean(axis=1)
        result_list.append(fea_name + '_max')
        result_list.append(fea_name + '_mean')

        print(target[fea_name + '_max'].head(10))
        print(target[fea_name + '_max'].tail(10))

        for field in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
            target = target.drop([fea_name + str(field)], axis=1)

    return target[result_list]


def extract_feature_ans_dif(data,feature,ans_feature):
    target = data.copy()
    print_time('extract feature ans dif')

    que = pd.read_csv('../datasets/question_info.csv', usecols=['qid', 'topic_id'])
    ans_feature = pd.merge(ans_feature, que, how='left', on='qid').fillna(0)
    target = pd.merge(target, que, how='left', on='qid').fillna(0)
    ans_feature['a_time'] = ans_feature['a_day'] + 0.04166 * ans_feature['a_hour']
    target['i_time'] = target['day'] + 0.04166 * target['hour']

    total_extend = ans_feature['topic_id'].str.split(',', expand=True).stack() \
        .reset_index(level=0).set_index('level_0') \
        .rename(columns={0: 'topic'}).join(ans_feature.drop('topic_id', axis=1)) \
        .reset_index(drop=True)

    t = total_extend.groupby(['uid','topic'])['a_time'].agg(['max']).reset_index().rename(columns={'max':'uid_topic_ans_recent_time'})

    topic_df = target['topic_id'].str.split(',', expand=True)
    topic_df = topic_df.fillna(0)
    target = pd.concat([target, topic_df], axis=1)
    fea_name = 'uid_topic_ans_recent_time'
    tmp_name = []
    result_list = []
    for field in [0, 1, 2, 3, 4, 5]:
        target = pd.merge(target, t, how='left', left_on=['uid', field], right_on=['uid', 'topic']).rename(
                columns={fea_name: fea_name + str(field)}).fillna(1000)
        target['s' + str(field)] = target['i_time'] - target[fea_name + str(field)]
        tmp_name.append('s' + str(field))

        target[fea_name + '_mean'] = target[tmp_name].mean(axis=1)
        target[fea_name + '_min'] = target[tmp_name].min(axis=1)
        target[fea_name + '_max'] = target[tmp_name].max(axis=1)
    result_list.append(fea_name + '_min')
    result_list.append(fea_name + '_mean')
    result_list.append(fea_name + '_max')

    return target[result_list]


def extract_usr_unique(data,feature,ans_feature):

    target = data.copy()
    que = pd.read_csv('../datasets/question_info.csv', usecols=['qid', 'topic_id'])
    ans = ans_feature[['uid','qid']]
    ans = pd.merge(ans,que,how='left',on='qid')
    feature = pd.merge(feature, que, how='left', on='qid').fillna('0')
    ## 获取用户被邀请的话题种类
    t = feature.groupby(['uid'])['topic_id'].apply(lambda x: len(set(','.join(x.tolist()).split(',')))).reset_index().rename(columns={'topic_id':'sw_uid_invite_topic_unique'})
    target = pd.merge(target,t,how='left',on='uid')

    t = ans.groupby(['uid'])['topic_id'].apply(lambda x: len(set(','.join(x.tolist())))).reset_index().rename(columns={'topic_id':'sw_uid_ans_topic_unique'})
    target = pd.merge(target, t, how='left', on='uid')

    fealist = ['sw_uid_invite_topic_unique','sw_uid_ans_topic_unique']
    return target[fealist]


def extruct_feature(data, feature, ans_feature):
    target = data.copy()
    print_time('extract feature ')
    # 统计uid
    t = feature.groupby('uid')['label'].agg(['count','sum','mean','std']).reset_index()\
        .rename(columns={'count':'sw_' + 'ulc','sum':'sw_' + 'uls', 'mean':'sw_' + 'ulm','std':'sw_' + 'uld'})

    print_time('pinghua')
    HP = HyperParam(1, 1)
    HP.update_from_data_by_moment(t['sw_ulc'].values,
                                  t['sw_uls'].values)  # 矩估计
    t['sw_uid_rate_hp'] = np.divide(t['sw_uls'] + HP.alpha, t['sw_ulc'] + HP.alpha + HP.beta)
    print('pinghua',HP.alpha,HP.beta)

    target = pd.merge(target,t,how='left',on='uid')

    # 统计qid
    t = feature.groupby('qid')['label'].agg(['count', 'sum', 'mean', 'std']).reset_index() \
        .rename(columns={'count': 'sw_' + 'qlc', 'sum': 'sw_' + 'qls', 'mean': 'sw_' + 'qlm', 'std': 'sw_' + 'qld'})

    HP = HyperParam(1, 1)
    HP.update_from_data_by_moment(t['sw_qlc'].values,
                                  t['sw_qls'].values)  # 矩估计
    t['sw_qid_rate_hp'] = np.divide(t['sw_qls'] + HP.alpha, t['sw_qlc'] + HP.alpha + HP.beta)
    print('pinghua', HP.alpha, HP.beta)
    target = pd.merge(target, t, how='left', on='qid')

    #统计ansinfo
    gu = ans_feature.groupby('uid')
    t = gu['qid'].agg(['count']).reset_index().rename(columns={'count': 'sw_' + 'u_ans_q_num'})
    target = pd.merge(target, t, how='left', on='uid')

    for feat in ['bit7','bit8','bit10','bit11','bit12','bit13','bit15','bit16','bit17']:
        t = gu[feat].agg(['sum','mean']).reset_index().rename(columns={'sum': 'sw_uc_' + feat,'mean': 'sw_um_' + feat})
        target = pd.merge(target, t, how='left', on='uid')



    feature['i_time'] = feature['day'] * 24 + feature['hour']
    feature_sorted = feature.sort_values(by=['i_time'], axis=0, ascending=True).reset_index(drop=True)
    feature_sorted['slabel'] = feature_sorted['label'].astype(str)
    t = feature_sorted.groupby('uid')['slabel'].apply(lambda x: '-' + ''.join(x.tolist())).reset_index().rename(columns={'slabel': 'sw_uid_seq'})
    t['sw_uid_seq_5'] = t['sw_uid_seq'].apply(lambda x: x[-5:])
    t['sw_uid_recent_uclick'] = t['sw_uid_seq'].apply(lambda x: len(x) - 1 - x.rfind('1') if x.rfind('1') != -1 else len(x))
    target = pd.merge(target,t,how='left',on='uid')


    t = feature_sorted.groupby('qid')['slabel'].apply(lambda x: '-' + ''.join(x.tolist())).reset_index().rename(
        columns={'slabel': 'sw_qid_seq'})
    t['sw_qid_seq_5'] = t['sw_qid_seq'].apply(lambda x: x[-5:])
    t['sw_qid_recent_uclick'] = t['sw_qid_seq'].apply(lambda x: len(x) - 1 - x.rfind('1') if x.rfind('1') != -1 else len(x))
    target = pd.merge(target, t, how='left', on='qid')



    ans_feature['ans_time'] = ans_feature['a_day'] + 0.04166 * ans_feature['a_hour']
    qustion_info = pd.read_csv('../datasets2/question_info.csv',usecols=['qid','q_day','q_hour'])
    qustion_info['qus_time'] = qustion_info['q_day'] + 0.04166 * qustion_info['q_hour']
    ans_feature = pd.merge(ans_feature,qustion_info,how='left',on='qid')
    print(ans_feature['qus_time'].isnull().sum(),ans_feature.shape)
    ans_feature['qus_time'] = ans_feature['qus_time'].fillna(3000)
    ans_feature['ans_dif_time'] = ans_feature['ans_time'] - ans_feature['qus_time']
    gu = ans_feature.groupby('uid')['ans_dif_time'].agg(['mean','max','min','std']).reset_index()\
        .rename(columns={'mean': 'at_mean','max':'at_max','min':'at_min','std':'at_std'})
    target = pd.merge(target,gu,how='left',on='uid')


    ans_feature['week'] = ans_feature['a_day'] % 7
    ans_feature['new_hour'] = ans_feature['a_hour'].apply(lambda x: int(x / 6))
    target['week'] = target['day'] % 7
    target['new_hour'] = target['hour'].apply(lambda x: int(x / 6))
    t = ans_feature.groupby(['uid'])['qid'].agg(['count']).reset_index().rename(columns={'count': 'u_t_count'})
    t1 = ans_feature.groupby(['uid', 'week'])['qid'].agg(['count']).reset_index().rename(
        columns={'count': 'uid_week_count'})
    target = pd.merge(target, t, how='left', on='uid').fillna(0)
    target = pd.merge(target, t1, how='left', on=['uid', 'week']).fillna(0)
    print(target.columns.tolist())

    target['uid_week_ans_rate'] = np.divide(target['uid_week_count'], target['u_t_count'] + 0.001)
    t1 = ans_feature.groupby(['uid', 'new_hour'])['qid'].agg(['count']).reset_index().rename(
        columns={'count': 'uid_nhour_count'})
    target = pd.merge(target, t1, how='left', on=['uid', 'new_hour'])
    target['uid_week_ans_rate'] = np.divide(target['uid_nhour_count'], target['u_t_count'] + 0.001)

    return target




# 构建question的mean_vector


from sklearn.metrics.pairwise import cosine_similarity


with open('../datasets/qestion_mean_vector.pkl', 'rb') as f:
    q_dict = pickle.load(f)


def __ans_topic_smilar(df):
    ans_list = df['uid_qid_list']
    if ans_list == 0 or ans_list == '0':
        return 0
    q_vector = q_dict[df['qid']]
    if len(q_vector) == 0:
        return 0
    score_list = []
    for s in ans_list.split(','):
        vector = q_dict[s]
        if len(vector) == 0:
            return 0
        score = cosine_similarity([q_vector, vector])[0, 1]
        score_list.append(score)
    return score_list


def tmp_func(df1):
    df1['score_list'] = df1.apply(__ans_topic_smilar, axis=1)
    df1['word_smilar_mean'] = df1['score_list'].apply(lambda x: np.mean(x) if x != 0 else np.NAN)
    df1['word_smilar_sum'] = df1['score_list'].apply(lambda x: np.sum(x) if x != 0 else np.NAN)
    df1['word_smilar_max'] = df1['score_list'].apply(lambda x: np.max(x) if x != 0 else np.NAN)
    df1['word_smilar_recent'] = df1['score_list'].apply(lambda x: x[-1] if x != 0 else np.NAN)
    df1 = df1.drop(['score_list'], axis=1)
    return df1


def extract_feature_smilar(data, feature, ans):
    target = data.copy()
    print_time('extract feature smilar')
    uid_qid_list = ans.groupby(['uid'])['qid'] \
        .apply(lambda x: ','.join(x.tolist())).reset_index().rename(columns={'qid': 'uid_qid_list'})

    target = pd.merge(target, uid_qid_list, how='left', on='uid').fillna(0)


    print_time('start')

    target = multiprocessing_apply_data_frame(tmp_func, target, 10)
    print_time('end')

    return target




def extract_topic_count_feature(data,ttt,feature):
    print('extract_topic count feature')

    target = data.copy()
    feature['label'] = 1
    que = pd.read_csv('../datasets/question_info.csv', usecols=['qid', 'topic_id'])
    m_list = ['uid', 'sex','visit', 'CA', 'CB', 'CC', 'CD', 'CE']
    meb = pd.read_csv('../datasets/member_info.csv', usecols=m_list)
    target = pd.merge(target, que, how='left', on='qid').fillna('0')
    feature = pd.merge(feature, que, how='left', on='qid').fillna('0')
    target = pd.merge(target, meb, how='left', on='uid').fillna('0')
    feature = pd.merge(feature, meb, how='left', on='uid').fillna('0')

    total_extend = feature['topic_id'].str.split(',', expand=True).stack() \
        .reset_index(level=0).set_index('level_0') \
        .rename(columns={0: 'topic'}).join(feature.drop('topic_id', axis=1)) \
        .reset_index(drop=True)

    topic_df = target['topic_id'].str.split(',', expand=True)
    target = pd.concat([target, topic_df], axis=1)

    fealist = m_list
    final_list = []

    ###统计topic的总量
    t1 = total_extend.groupby(['topic'])['label'].agg(['count']).reset_index().rename(columns={'count': 'topic_count'})
    t1.loc[t1['topic'] == '0', 'topic_count'] = 0

    for stat in fealist:

        s_total_extend = total_extend[['topic',stat,'label']]
        fea_name = stat + '_ans_topic_count_ratio'
        print('extract',fea_name)
        ###统计话题和用户属性的交叉量
        t = total_extend.groupby(['topic',stat])['label'].agg(['count']).reset_index().rename(columns={'count': 'sum_count'})
        t.loc[t['topic'] == '0', 'sum_count'] = 0
        t = pd.merge(t,t1,how='left',on='topic')

        #平滑求占比
        HP = HyperParam(1, 1)
        HP.update_from_data_by_moment(t['topic_count'].values,
                                      t['sum_count'].values)  # 矩估计
        t[fea_name] = np.divide(t['sum_count'] + HP.alpha, t['topic_count'] + HP.alpha + HP.beta)
        t = t.drop(['topic_count', 'sum_count'], axis=1)
        stat = ['topic',stat]
        tmp_name = []
        for field in [0, 1, 2, 3, 4]:
            lefton = []
            for i in stat:
                if i == 'topic':
                    lefton.append(field)
                else:
                    lefton.append(i)
            target = pd.merge(target, t, how='left', left_on=lefton, right_on=stat).rename(
                columns={fea_name: fea_name + str(field)})
            tmp_name.append(fea_name + str(field))

        target[fea_name + '_max'] = target[tmp_name].max(axis=1)
        target[fea_name + '_mean'] = target[tmp_name].mean(axis=1)
        final_list.append(fea_name + '_max')
        final_list.append(fea_name + '_mean')

        for field in [0, 1, 2, 3, 4]:
            target = target.drop([fea_name + str(field)], axis=1)

    return target[final_list]





def extract_topic_score(data,ttt,feature):
    print('extract_topic count feature')

    target = data.copy()
    feature['label'] = 1

    que = pd.read_csv('../datasets2/question_info.csv', usecols=['qid', 'topic_id'])
    m_list = ['uid', 'SCORE']
    meb = pd.read_csv('../datasets2/member_info.csv', usecols=m_list)
    target = pd.merge(target, que, how='left', on='qid').fillna('0')
    feature = pd.merge(feature, que, how='left', on='qid').fillna('0')
    target = pd.merge(target, meb, how='left', on='uid').fillna('0')
    feature = pd.merge(feature, meb, how='left', on='uid').fillna('0')


    total_extend = feature['topic_id'].str.split(',', expand=True).stack() \
        .reset_index(level=0).set_index('level_0') \
        .rename(columns={0: 'topic'}).join(feature.drop('topic_id', axis=1)) \
        .reset_index(drop=True)

    topic_df = target['topic_id'].str.split(',', expand=True)
    target = pd.concat([target, topic_df], axis=1)

    fealist = m_list
    final_list = []

    ###统计topic的总量

    fealist = []

    for stat in fealist:

        s_total_extend = total_extend[['topic',stat,'label']]
        fea_name = stat + '_ans_topic_count_ratio'
        print('extract',fea_name)
        ###统计话题和用户属性的交叉量
        t = total_extend.groupby(['topic',stat])['label'].agg(['count']).reset_index().rename(columns={'count': 'sum_count'})
        t.loc[t['topic'] == '0', 'sum_count'] = 0
        t = pd.merge(t,t1,how='left',on='topic')

        #平滑求占比
        HP = HyperParam(1, 1)
        HP.update_from_data_by_moment(t['topic_count'].values,
                                      t['sum_count'].values)  # 矩估计
        t[fea_name] = np.divide(t['sum_count'] + HP.alpha, t['topic_count'] + HP.alpha + HP.beta)
        t = t.drop(['topic_count', 'sum_count'], axis=1)
        stat = ['topic',stat]
        tmp_name = []
        for field in [0, 1, 2, 3, 4]:
            lefton = []
            for i in stat:
                if i == 'topic':
                    lefton.append(field)
                else:
                    lefton.append(i)
            target = pd.merge(target, t, how='left', left_on=lefton, right_on=stat).rename(
                columns={fea_name: fea_name + str(field)})
            tmp_name.append(fea_name + str(field))

        target[fea_name + '_max'] = target[tmp_name].max(axis=1)
        target[fea_name + '_mean'] = target[tmp_name].mean(axis=1)
        final_list.append(fea_name + '_max')
        final_list.append(fea_name + '_mean')

        for field in [0, 1, 2, 3, 4]:
            target = target.drop([fea_name + str(field)], axis=1)

    return target[final_list]





with open('../datasets/topic_mean_vector.pkl','rb') as f:
    t_dict = pickle.load(f)


def __ans_topic_smilar1(df):
    ans_list = df['uid_qid_list']
    if ans_list == 0 or ans_list == '0':
        return 0
    q_vector = t_dict[df['qid']]
    if len(q_vector) == 0:
        return 0
    score_list = []
    for s in ans_list.split(','):
        vector = t_dict[s]
        if len(vector) == 0:
            return 0
        score = cosine_similarity([q_vector, vector])[0, 1]
        score_list.append(score)
    return score_list


def tmp_func1(df1):
    df1['score_list'] = df1.apply(__ans_topic_smilar1, axis=1)
    df1['topic_smilar_mean'] = df1['score_list'].apply(lambda x: np.mean(x) if x != 0 else np.NAN)
    df1['topic_smilar_sum'] = df1['score_list'].apply(lambda x: np.sum(x) if x != 0 else np.NAN)
    df1['topic_smilar_max'] = df1['score_list'].apply(lambda x: np.max(x) if x != 0 else np.NAN)
    df1['topic_smilar_recent'] = df1['score_list'].apply(lambda x: x[-1] if x != 0 else np.NAN)
    df1 = df1.drop(['score_list'], axis=1)
    return df1

def extract_topic_smilar(data, feature, ans):
    target = data.copy()
    print_time('extract feature smilar')

    uid_qid_list = ans.groupby(['uid'])['qid'].apply(lambda x: ','.join(x.tolist())).reset_index().rename(columns={'qid': 'uid_qid_list'})

    target = pd.merge(target, uid_qid_list, how='left', on='uid').fillna(0)

    print_time('start')

    target = multiprocessing_apply_data_frame(tmp_func1, target, 10)
    print_time('end')

    return target



def extract_uid_seq_more(target):
    target = target.reset_index(drop=True)
    target['ii'] = target.index
    target['i_time'] = target['day'] + 0.04166 * target['hour']

    ds = target.sort_values(by=['uid', 'i_time'], axis=0, ascending=True).reset_index(drop=True)
    ds['slabel'] = ds['m_label'].astype(str)
    t = ds.groupby('uid')['slabel'].apply(lambda x: '-' + ''.join(x.tolist())).reset_index().rename(
        columns={'slabel': 'uid_m_seq'})
    t1 = ds.groupby('uid')['label'].apply(lambda x: [i for i in range(len(x.tolist()))]).reset_index().rename(
        columns={'label': 'uid_rank'})
    t_rank = pd.DataFrame(
        {'uid': t1.uid.repeat(t1.uid_rank.str.len()), 'uid_rank': np.concatenate(t1.uid_rank.values)}).reset_index(
        drop=True)

    ds = pd.merge(ds, t, how='left', on='uid').reset_index(drop=True)
    ds = pd.concat([ds, t_rank], axis=1)
    # 计算用户的访问序列
    ds['uid_mm_seq'] = ds.apply(lambda x: '-' + (x['uid_m_seq'])[:x['uid_rank']], axis=1)
    target = pd.merge(target, ds[['ii', 'uid_mm_seq']], how='left', on='ii')

    return target

def extract_feature_hour(data,feature,ans):

    target = data.copy()

    t = feature.groupby(['uid'])['hour'].agg(['mean','std']).reset_index().rename(columns={'mean':'uid_invite_hour_mean'
        ,'std':'uid_invite_hour_std'})
    s_feature = feature[feature['label'] == 1]
    target = pd.merge(target,t,how='left',on='uid')
    t = s_feature.groupby(['uid'])['hour'].agg(['mean', 'std']).reset_index().rename(
        columns={'mean': 'uid_invite_ans_hour_mean'
            , 'std': 'uid_invite_ans_hour_std'})
    target = pd.merge(target, t, how='left', on='uid')

    t = ans.groupby(['uid'])['hour'].agg(['mean', 'std']).reset_index().rename(
        columns={'mean': 'uid_ans_hour_mean'
            , 'std': 'uid_ans_hour_std'})
    target = pd.merge(target, t, how='left', on='uid')

    return target











# # 构造验证机
print('dev')
window = dev_time_windows[0]
train_target = src_dev
feature_range = window[1]
train_feature = totals[(totals['day'] >= feature_range[0]) & (totals['day'] <= feature_range[1])]

ans_range = window[2]
ans_feature = ans_info[(ans_info['a_day'] >= ans_range[0]) & (ans_info['a_day'] <= ans_range[1])]



result = extruct_cross_feature_word(train_target,train_feature)
save_feature('dev',result)
# # #
result = extruct_cross_feature_topic(train_target,train_feature)
save_feature('dev',result)
# # # #
result = extract_feature_ans_dif(train_target,train_feature,ans_feature)
save_feature('dev',result)
# # # #
result = extract_feature_smilar(train_target,train_feature,ans_feature)
save_feature('dev',result)
#
result = extract_topic_smilar(train_target,train_feature,ans_feature)
save_feature('dev',result)

result = extract_usr_unique(train_target,train_feature,ans_feature)
save_feature('dev',result)

result = extruct_feature(train_target,train_feature,ans_feature)
save_feature('dev',result)

result = extract_topic_count_feature(train_target,train_feature,ans_feature)
save_feature('dev',result)






# # 构造测试机机
print('test')
window = test_time_windows[0]
train_target = src_test
feature_range = window[1]
train_feature = totals[(totals['day'] >= feature_range[0]) & (totals['day'] <= feature_range[1])]

ans_range = window[2]
ans_feature = ans_info[(ans_info['a_day'] >= ans_range[0]) & (ans_info['a_day'] <= ans_range[1])]



result = extruct_cross_feature_word(train_target,train_feature)
save_feature('test_B',result)
# # #
result = extruct_cross_feature_topic(train_target,train_feature)
save_feature('test_B',result)
# # # #
result = extract_feature_ans_dif(train_target,train_feature,ans_feature)
save_feature('test_B',result)
# # # #
result = extract_feature_smilar(train_target,train_feature,ans_feature)
save_feature('test_B',result)
#
result = extract_topic_smilar(train_target,train_feature,ans_feature)
save_feature('test_B',result)

result = extract_usr_unique(train_target,train_feature,ans_feature)
save_feature('test_B',result)

result = extruct_feature(train_target,train_feature,ans_feature)
save_feature('test_B',result)

result = extract_topic_count_feature(train_target,train_feature,ans_feature)
save_feature('test_B',result)



print_time('-------------end-------------------')