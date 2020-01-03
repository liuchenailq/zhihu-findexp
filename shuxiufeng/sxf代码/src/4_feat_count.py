#import pandas as pd
import modin.pandas as pd
import numpy as np
import datetime
from ctr_utils import *


def print_time(name):
    time = datetime.datetime.now()
    a = time.strftime('%Y-%m-%d %H:%M')
    print(name,a)


train  = pd.read_csv('../datasets/invite_train.csv')
dev  = pd.read_csv('../datasets/invite_dev.csv')
test_a = pd.read_csv('../datasets/invite_test.csv')
test  = pd.read_csv('../datasets/test_final.csv')


total_feature = pd.concat([train,dev,test_a,test],axis = 0).reset_index(drop=True)


use_train = pd.read_csv('../datasets/invite_train_use.csv')

total_label = pd.concat([use_train,dev,test],axis=0).reset_index(drop=True)

train_shape = use_train.shape[0]
dev_shape   = dev.shape[0]
test_shape  = test.shape[0]





def save_num_feat(featlist, data, mode):
    base_path = '../datasets/feature/' + mode + '/'
    print(mode,data.shape)
    for feat in featlist:

        data[[feat]].to_csv(base_path + feat + '.csv', index=False, float_format='%.4f')


#全局count特征
def extract_id_whole_count(data, feature):
    target = data.copy()
    fea_list = []
    feaname = 'merge_uid_min_day'
    fea_list.append(feaname);print('extract',feaname)
    t = feature.groupby(['uid'])['day'].agg(['min']).reset_index().rename(columns={'min': feaname})
    target = pd.merge(target, t, how='left', on='uid')
    #
    feaname = 'merge_uid_count'
    fea_list.append(feaname);print('extract',feaname)
    t = feature.groupby(['uid'])['label'].agg(['count']).reset_index().rename(columns={'count': feaname})
    target = pd.merge(target,t,how='left',on='uid')
    #
    feaname = 'merge_uid_day_count'
    fea_list.append(feaname);print('extract',feaname)
    t = feature.groupby(['uid','day'])['label'].agg(['count']).reset_index().rename(columns={'count': feaname})
    target = pd.merge(target, t, how='left', on=['uid','day'])
    #
    feaname = 'merge_uid_day_hour_count'
    fea_list.append(feaname);print('extract',feaname)
    t = feature.groupby(['uid', 'day','hour'])['label'].agg(['count']).reset_index().rename(columns={'count': feaname})
    target = pd.merge(target, t, how='left', on=['uid', 'day','hour'])
    #
    feaname = 'merge_uid_min_day_count'
    fea_list.append(feaname);print('extract',feaname)
    t = feature.groupby(['uid', 'day'])['label'].agg(['count']).reset_index().rename(columns={'count': feaname,'day':'merge_uid_min_day'})
    target = pd.merge(target, t, how='left', on=['uid','merge_uid_min_day'])

    feaname = 'merge_uid_day_nuinque'
    fea_list.append(feaname);print('extract', feaname)
    t = feature.groupby(['uid'])['day'].agg(['nunique']).reset_index().rename(
        columns={'nunique': feaname, 'day': feaname})
    target = pd.merge(target, t, how='left', on=['uid'])

    feaname = 'merge_qid_day_nuinque'
    fea_list.append(feaname);
    print('extract', feaname)
    t = feature.groupby(['qid'])['day'].agg(['nunique']).reset_index().rename(
        columns={'nunique': feaname, 'day': feaname})
    target = pd.merge(target, t, how='left', on=['qid'])


    print_time(target.columns.tolist())

    feaname = 'merge_qid_min_day'
    fea_list.append(feaname);print('extract',feaname)
    t = feature.groupby(['qid'])['day'].agg(['min']).reset_index().rename(columns={'min': feaname})
    target = pd.merge(target, t, how='left', on='qid')

    print_time(target.columns.tolist())

    feaname = 'merge_qid_count'
    fea_list.append(feaname);print('extract',feaname)
    t = feature.groupby(['qid'])['label'].agg(['count']).reset_index().rename(columns={'count': feaname})
    target = pd.merge(target, t, how='left', on='qid')

    feaname = 'merge_qid_day_count'
    fea_list.append(feaname);print('extract',feaname)
    t = feature.groupby(['qid', 'day'])['label'].agg(['count']).reset_index().rename(columns={'count': feaname})
    target = pd.merge(target, t, how='left', on=['qid', 'day'])

    feaname = 'merge_qid_day_hour_count'
    fea_list.append(feaname);print('extract',feaname)
    t = feature.groupby(['qid', 'day', 'hour'])['label'].agg(['count']).reset_index().rename(columns={'count': feaname})
    target = pd.merge(target, t, how='left', on=['qid', 'day', 'hour'])

    feaname = 'merge_qid_min_day_count'
    fea_list.append(feaname);print('extract',feaname)
    t = feature.groupby(['qid', 'day'])['label'].agg(['count']).reset_index().rename(columns={'count': feaname,'day':'merge_qid_min_day'})
    target = pd.merge(target, t, how='left', on=['qid', 'merge_qid_min_day'])

    target['day+1'] = target['day'] + 1
    target['day-1'] = target['day'] - 1

    feaname = 'merge_qid_day+1_count'
    fea_list.append(feaname);
    print('extract', feaname)
    t = feature.groupby(['qid', 'day'])['label'].agg(['count']).reset_index().rename(columns={'count': 'a1', 'day': 'day+1'})
    target = pd.merge(target, t, how='left', on=['qid', 'day+1']).rename(columns={'a1': feaname})
    #
    feaname = 'merge_qid_day-1_count'
    fea_list.append(feaname);
    print('extract', feaname)
    t = feature.groupby(['qid', 'day'])['label'].agg(['count']).reset_index().rename(columns={'count': 'a1', 'day': 'day-1'})
    target = pd.merge(target, t, how='left', on=['qid', 'day-1']).rename(columns={'a1': feaname})

    feature['s_hour'] = feature['hour'].apply(lambda x: int(x / 6))
    target['s_hour'] = target['hour'].apply(lambda x: int(x / 6))


    feaname = 'merge_qid_day_shour_count'
    fea_list.append(feaname);
    print('extract', feaname)
    t2 = feature.groupby(['qid', 'day', 's_hour'])['label'].agg(['count']).reset_index().rename(columns={'count': 'a3'})
    target = pd.merge(target, t2, how='left', on=['qid', 'day', 's_hour']).rename(columns={'a3': feaname})

    feaname = 'merge_qid_day-1_shour_count'
    fea_list.append(feaname);
    print('extract', feaname)
    t2 = feature.groupby(['qid', 'day', 's_hour'])['label'].agg(['count']).reset_index().rename(
        columns={'count': 'a3', 'day': 'day-1'})
    target = pd.merge(target, t2, how='left', on=['qid', 'day-1', 's_hour']).rename(
        columns={'a3': feaname})

    return target[fea_list],fea_list




#对topic的统计
def extract_topic_whole_count(data,feature):
    print('extract_feature')

    target = data.copy()

    que = pd.read_csv('../datasets/question_info.csv', usecols=['qid', 'topic_id'])
    target = pd.merge(target, que, how='left', on='qid').fillna('0')
    feature = pd.merge(feature, que, how='left', on='qid').fillna('0')

    total_extend = feature['topic_id'].str.split(',', expand=True).stack() \
        .reset_index(level=0).set_index('level_0') \
        .rename(columns={0: 'topic'}).join(feature.drop('topic_id', axis=1)) \
        .reset_index(drop=True)

    topic_df = target['topic_id'].str.split(',', expand=True)
    target = pd.concat([target, topic_df], axis=1)


    stat_feat = [
        (['topic'], ['label'], ['count']),
        (['topic'], ['qid'], ['nunique']),
        (['topic'], ['uid'], ['nunique']),
        (['topic', 'day', 'hour'], ['uid'], ['nunique']),
    ]
    final_list = []

    for stat in stat_feat:
        fea_name = 'merge_' + '_'.join(stat[0]) + '_' + '_'.join(stat[1]) + '_' + '_'.join(stat[2])
        print('extract',fea_name)
        t = total_extend.groupby(stat[0])[stat[1][0]].agg(stat[2]).reset_index() \
            .rename(columns={stat[2][0]: fea_name})
        t.loc[t['topic'] == '0', fea_name] = 0
        tmp_name = []
        for field in [0, 1, 2, 3, 4]:
            lefton = []
            for i in stat[0]:
                if i == 'topic':
                    lefton.append(field)
                else:
                    lefton.append(i)
            target = pd.merge(target, t, how='left', left_on=lefton, right_on=stat[0]).rename(
                columns={fea_name: fea_name + str(field)})
            tmp_name.append(fea_name + str(field))

        target[fea_name + '_max'] = target[tmp_name].max(axis=1)
        target[fea_name + '_sum'] = target[tmp_name].sum(axis=1)
        final_list.append(fea_name + '_max')
        final_list.append(fea_name + '_sum')

        for field in [0, 1, 2, 3, 4]:
            target = target.drop([fea_name + str(field)], axis=1)

    return target[final_list], final_list
########  #####


#和时间相关的特征1
def extract_time_feature(data,feature):
    target = data.copy()

    s_feature = feature.copy()

    print(target.columns.tolist())
    print(feature.columns.tolist())

    print(target.info())

    qinfo = pd.read_csv('../datasets/question_info.csv',usecols=['qid','q_day','q_hour'])
    qinfo['q_time'] = qinfo['q_day'] + 0.04166 * qinfo['q_hour']
    target['i_time'] = target['day'] + 0.04166 * target['hour']
    feature['i_time'] = feature['day'] + 0.04166 * feature['hour']
    s_feature['i_time'] = s_feature['day'] + 0.04166 * s_feature['hour']

    t1 = feature.groupby(['uid'])['i_time'].min().reset_index().rename(columns={'i_time':'merge_uid_first_time'})
    target = pd.merge(target,t1,how='left',on='uid')
    t1 = feature.groupby(['qid'])['i_time'].min().reset_index().rename(columns={'i_time': 'merge_qid_first_time'})
    target = pd.merge(target, t1, how='left', on='qid')

    target['merge_qid_cur_diff_first'] = target['i_time'] - target['merge_qid_first_time']
    target['merge_uid_cur_diff_first'] = target['i_time'] - target['merge_uid_first_time']

    target = pd.merge(target,qinfo,how='left',on='qid')

    target['time_diff'] = target['i_time'] - target['q_time']

    feature['merge_qid_rank'] = feature.groupby('qid')['i_time'].rank(method='max')
    feature['merge_uid_rank'] = feature.groupby('uid')['i_time'].rank(method='max')
    feature['merge_qid_day_rank'] = feature.groupby(['qid','day'])['hour'].rank(method='max')
    feature['merge_uid_day_rank'] = feature.groupby(['uid','day'])['hour'].rank(method='max')


    fea_list = ['qid','uid','ctime','label','merge_qid_rank','merge_uid_rank','merge_qid_day_rank','merge_uid_day_rank']
    mf = feature[fea_list].drop_duplicates(subset=['qid','uid','ctime','label'],keep='first')
    target = pd.merge(target,mf,how='left',on=['qid','uid','ctime','label'])

    target['merge_uid_avg_stack_count'] = np.divide(target['merge_uid_rank'],target['merge_uid_cur_diff_first'] + 1)
    target['merge_qid_avg_stack_count'] = np.divide(target['merge_qid_rank'], target['merge_qid_cur_diff_first'] + 1)





    feature_sorted = s_feature.sort_values(by=['uid', 'i_time'], axis=0, ascending=True).reset_index(drop=True)
    feature_sorted['uid_rank_tmp'] = feature_sorted.groupby(['uid'])['i_time'].rank(method='dense')
    feature_sorted = feature_sorted.drop_duplicates(subset=['uid','uid_rank_tmp'],keep='first')
    feature_sorted['a2'] = feature_sorted.groupby('uid')['i_time'].shift(1).fillna(3000)

    print(feature_sorted.head(10))
    feature_sorted['merge_uid_past_invite_time'] = feature_sorted['i_time'] - feature_sorted['a2']

    fea_list = ['uid','qid','ctime','label','merge_uid_past_invite_time']
    mf = feature_sorted[fea_list].drop_duplicates(subset=['qid', 'uid', 'ctime', 'label'], keep='first')
    target = pd.merge(target, mf, how='left', on=['qid', 'uid', 'ctime', 'label'])

    feature_sorted = s_feature.sort_values(by=['qid', 'i_time'], axis=0, ascending=True).reset_index(drop=True)
    feature_sorted['qid_rank_tmp'] = feature_sorted.groupby(['qid'])['i_time'].rank(method='dense')
    feature_sorted = feature_sorted.drop_duplicates(subset=['qid', 'qid_rank_tmp'], keep='first')
    feature_sorted['a2'] = feature_sorted.groupby('qid')['i_time'].shift(1).fillna(3000)
    feature_sorted['merge_qid_past_invite_time'] = feature_sorted['i_time'] - feature_sorted['a2']
    fea_list = ['qid', 'uid','ctime', 'label', 'merge_qid_past_invite_time']
    mf = feature_sorted[fea_list].drop_duplicates(subset=['qid', 'uid', 'ctime', 'label'], keep='first')
    target = pd.merge(target, mf, how='left', on=['qid', 'uid', 'ctime', 'label'])

    final_list = ['merge_qid_rank','merge_uid_rank','merge_qid_day_rank','merge_uid_day_rank','merge_qid_past_invite_time', 'merge_uid_past_invite_time',
                  'merge_uid_avg_stack_count','merge_qid_avg_stack_count','merge_qid_cur_diff_first','merge_uid_cur_diff_first']

    return target,final_list


#和时间相关的特征2
def time_feat(target,feature):
    data = target.copy()
    q_info = pd.read_csv('../datasets/question_info.csv',usecols=['qid','q_day','q_hour'])
    q_info['q_times'] = q_info['q_day'] + 0.04166 * q_info['q_hour']
    feature['i_times'] = feature['day'] + 0.04166 * feature['hour']
    print(q_info.head(10))
    data['i_time'] = data['day'] + 0.04166 * data['hour']
    data_min = np.min(q_info['q_day'])
    print(data_min)
    data = pd.merge(data, q_info, how='left',on='qid')
    data['q_times'] = data['q_times'].fillna(data_min)
    data['q_hour'] = data['q_hour'].fillna(0)
    print(data.head(5))
    data['merge_time_diff'] = data['i_time'] - data['q_times']

    t = feature.groupby(['qid'])['i_times'].agg(['min','max']).reset_index().rename(columns={'min': 'q_invite_min_time','max':'q_invite_max_time'})
    data = pd.merge(data,t,how='left',on='qid')
    data['merge_qid_diff_first_invite_time'] = data['i_time'] - data['q_invite_min_time']
    data['merge_qid_diff_first_invite_time_sub_max'] = np.divide(data['merge_qid_diff_first_invite_time'] ,data['q_invite_max_time'] - data['q_invite_min_time'])

    t = feature.groupby(['uid'])['i_times'].agg(['min','max']).reset_index().rename(columns={'min': 'u_invite_min_time','max':'u_invite_max_time'})
    data = pd.merge(data, t, how='left', on='uid')
    data['merge_uid_diff_first_invite_time'] = data['i_time'] - data['u_invite_min_time']
    data['merge_uid_diff_first_invite_time_sub_max'] = np.divide(data['merge_qid_diff_first_invite_time'] ,data['q_invite_max_time'] - data['q_invite_min_time'])

    return data,['merge_uid_diff_first_invite_time','merge_qid_diff_first_invite_time'
        ,'merge_uid_diff_first_invite_time_sub_max','merge_qid_diff_first_invite_time_sub_max']



#两次邀请的时间差

def get_seq_uid():
    train = pd.read_csv('../datasets/invite_' + 'train' + '.csv')
    dev = pd.read_csv('../datasets/invite_' + 'dev' + '.csv')
    test_a = pd.read_csv('../datasets/invite_test.csv')
    test = pd.read_csv('../datasets/test_final.csv')

    use_train = pd.read_csv('../datasets/invite_train_use.csv')

    # total1 = pd.concat([use_train,dev,test],axis=0).reset_index(drop=True)

    a1 = train.shape[0]
    a2 = dev.shape[0]
    a3 = test.shape[0]
    feature = pd.concat([train,dev,test_a,test],axis=0).reset_index(drop=True)
    feature['invite_time'] = feature['day'] + 0.04166 * feature['hour']

    total = pd.concat([train, dev,test], axis=0).reset_index(drop=True)
    total['invite_time'] = total['day'] + 0.04166 * total['hour']
    lens_df = pd.DataFrame({'src_i':np.arange(a1 + a2 + a3)})
    total = pd.concat([total,lens_df],axis=1)

    # 计算用户的未来访问时间差
    total_sorted = feature.sort_values(by=['uid','invite_time'],axis=0 ,ascending=True).reset_index(drop=True)

    t = total_sorted.groupby('uid')['invite_time'] \
        .apply(lambda x: np.subtract(x.tolist()[1:] + [4100], x.tolist()).tolist()).reset_index() \
        .rename(columns={'invite_time': 'key'})

    t1 = pd.DataFrame({'uid': t.uid.repeat(t.key.str.len()), 'merge_uid_time_feature': np.concatenate(t.key.values)}).reset_index(drop=True)
    print(t1.shape)
    print(t1.head(10))
    t = total_sorted.groupby('uid')['invite_time'] \
        .apply(lambda x: np.subtract(x.tolist(),[2900] + x.tolist()[:-1]).tolist()).reset_index() \
        .rename(columns={'invite_time': 'key'})

    t2 = pd.DataFrame({'uid': t.uid.repeat(t.key.str.len()), 'merge_uid_time_past': np.concatenate(t.key.values)}).reset_index(drop=True)
    print(t2.shape)
    print(t2.head(10))

    total_sorted = pd.concat([total_sorted,t1[['merge_uid_time_feature']],t2[['merge_uid_time_past']]],axis=1)[['uid','qid','ctime','label','merge_uid_time_feature','merge_uid_time_past']]

    mf = total_sorted.drop_duplicates(subset=['qid', 'uid', 'ctime', 'label'], keep='first')

    print(mf.head(5))
    print(total.head(5))
    total = pd.merge(total, mf, how='left', on=['qid', 'uid', 'ctime', 'label'])


    #total1 = pd.merge(total1,total,how='left',on=['uid','qid','ctime'])

    featlist = ['merge_uid_time_feature', 'merge_uid_time_past']

    s = total.iloc[0:a1]
    s = s[s['day'] >= 3858]

    print(s.head(5))
    print(s.tail(5))

    print(use_train.head(5))
    print(use_train.tail(5))


    print(total.shape)

    save_num_feat(featlist, s, 'train')
    save_num_feat(featlist, total.iloc[a1:a1 + a2], 'dev')
    save_num_feat(featlist, total.iloc[a1 + a2:], 'test_B')













print_time('extract_id_whole_count')
####提取uid qid的全局统计特征
result,fealist = extract_id_whole_count(total_label,total_feature)
print_time('save_feature')
save_num_feat(fealist,result.iloc[:train_shape],'train')
save_num_feat(fealist,result.iloc[train_shape:train_shape + dev_shape],'dev')
save_num_feat(fealist,result.iloc[train_shape + dev_shape:],'test_B')
print_time('save_finish')

#
#

result,fealist = extract_topic_whole_count(total_label,total_feature)
print_time('save_feature')
save_num_feat(fealist,result.iloc[:train_shape],'train')
save_num_feat(fealist,result.iloc[train_shape:train_shape + dev_shape],'dev')
save_num_feat(fealist,result.iloc[train_shape + dev_shape:],'test_B')
print_time('save_finish')


result,fealist = extract_time_feature(total_label,total_feature)
print_time('save_feature')
save_num_feat(fealist,result.iloc[:train_shape],'train')
save_num_feat(fealist,result.iloc[train_shape:train_shape + dev_shape],'dev')
save_num_feat(fealist,result.iloc[train_shape + dev_shape:],'test_B')
print_time('save_finish')
#

result,fealist = time_feat(total_label,total_feature)
print_time('save_feature')
save_num_feat(fealist,result.iloc[:train_shape],'train')
save_num_feat(fealist,result.iloc[train_shape:train_shape + dev_shape],'dev')
save_num_feat(fealist,result.iloc[train_shape + dev_shape:],'test_B')
print_time('save_finish')

#
get_seq_uid()

















# topic_count('train')
# topic_count('dev')
# topic_count('test')













