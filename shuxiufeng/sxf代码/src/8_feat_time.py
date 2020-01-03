import pandas as pd
import numpy as np
import datetime


def print_time(name):
    time = datetime.datetime.now()
    a = time.strftime('%Y-%m-%d %H:%M')
    print(name,a)

print_time('--------------start---------------')


def save_num_feat(featlist, data, mode):
    base_path = '../datasets/feature/' + mode + '/'
    print(data.shape)
    for feat in featlist:
        data[[feat]].to_csv(base_path + feat + '.csv', index=False, float_format='%.4f')


train  = pd.read_csv('../datasets/invite_train.csv')
dev  = pd.read_csv('../datasets/invite_dev.csv')
test  = pd.read_csv('../datasets/test_final.csv')
a = train.shape[0]
b = dev.shape[0]

# u1 = train[train['day'] >= 3858]
#
# u1.to_csv('../datasets3/invite_train_use.csv',index=False)

total_feature = pd.concat([train,dev,test],axis = 0).reset_index(drop=True)
print(total_feature.shape)

use_train = pd.read_csv('../datasets/invite_train_use.csv')
total_label = pd.concat([use_train,dev,test],axis=0).reset_index(drop=True)
print(total_label.shape)

train_shape = use_train.shape[0]
dev_shape   = dev.shape[0]
test_shape  = test.shape[0]




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

    t1 = feature.groupby(['uid'])['i_time'].min().reset_index().rename(columns={'i_time':'uid_first_time'})
    target = pd.merge(target,t1,how='left',on='uid')
    t1 = feature.groupby(['qid'])['i_time'].min().reset_index().rename(columns={'i_time': 'qid_first_time'})
    target = pd.merge(target, t1, how='left', on='qid')

    target['qid_cur_diff_first'] = target['i_time'] - target['qid_first_time']
    target['uid_cur_diff_first'] = target['i_time'] - target['uid_first_time']

    target = pd.merge(target,qinfo,how='left',on='qid')

    target['time_diff'] = target['i_time'] - target['q_time']

    feature['qid_rank'] = feature.groupby('qid')['i_time'].rank(method='max')
    feature['uid_rank'] = feature.groupby('uid')['i_time'].rank(method='max')
    feature['qid_day_rank'] = feature.groupby(['qid','day'])['hour'].rank(method='max')
    feature['uid_day_rank'] = feature.groupby(['uid','day'])['hour'].rank(method='max')
    # feature['qid_day_rank_rv'] = feature.groupby(['qid', 'day'])['hour'].rank(method='max',ascending=False)
    # feature['uid_day_rank_rv'] = feature.groupby(['uid', 'day'])['hour'].rank(method='max',ascending=False)


    fea_list = ['qid','uid','ctime','label','qid_rank','uid_rank','qid_day_rank','uid_day_rank']
    mf = feature[fea_list].drop_duplicates(subset=['qid','uid','ctime','label'],keep='first')
    target = pd.merge(target,mf,how='left',on=['qid','uid','ctime','label'])

    target['uid_avg_stack_count'] = np.divide(target['uid_rank'],target['uid_cur_diff_first'] + 1)
    target['qid_avg_stack_count'] = np.divide(target['qid_rank'], target['qid_cur_diff_first'] + 1)





    feature_sorted = s_feature.sort_values(by=['uid', 'i_time'], axis=0, ascending=True).reset_index(drop=True)
    feature_sorted['uid_rank_tmp'] = feature_sorted.groupby(['uid'])['i_time'].rank(method='dense')
    feature_sorted = feature_sorted.drop_duplicates(subset=['uid','uid_rank_tmp'],keep='first')
    feature_sorted['a2'] = feature_sorted.groupby('uid')['i_time'].shift(1).fillna(3000)

    print(feature_sorted.head(10))
    feature_sorted['uid_past_invite_time'] = feature_sorted['i_time'] - feature_sorted['a2']

    fea_list = ['uid','qid','ctime','label','uid_past_invite_time']
    mf = feature_sorted[fea_list].drop_duplicates(subset=['qid', 'uid', 'ctime', 'label'], keep='first')
    target = pd.merge(target, mf, how='left', on=['qid', 'uid', 'ctime', 'label'])

    feature_sorted = s_feature.sort_values(by=['qid', 'i_time'], axis=0, ascending=True).reset_index(drop=True)
    feature_sorted['qid_rank_tmp'] = feature_sorted.groupby(['qid'])['i_time'].rank(method='dense')
    feature_sorted = feature_sorted.drop_duplicates(subset=['qid', 'qid_rank_tmp'], keep='first')
    feature_sorted['a2'] = feature_sorted.groupby('qid')['i_time'].shift(1).fillna(3000)
    feature_sorted['qid_past_invite_time'] = feature_sorted['i_time'] - feature_sorted['a2']
    fea_list = ['qid', 'uid','ctime', 'label', 'qid_past_invite_time']
    mf = feature_sorted[fea_list].drop_duplicates(subset=['qid', 'uid', 'ctime', 'label'], keep='first')
    target = pd.merge(target, mf, how='left', on=['qid', 'uid', 'ctime', 'label'])

    final_list = ['qid_rank','uid_rank','qid_day_rank','uid_day_rank','qid_past_invite_time', 'uid_past_invite_time',
                  'uid_avg_stack_count','qid_avg_stack_count','qid_cur_diff_first','uid_cur_diff_first']

    return target,final_list,feature



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
    data['time_diff'] = data['i_time'] - data['q_times']

    t = feature.groupby(['qid'])['i_times'].agg(['min','max']).reset_index().rename(columns={'min': 'q_invite_min_time','max':'q_invite_max_time'})
    data = pd.merge(data,t,how='left',on='qid')
    data['qid_diff_first_invite_time'] = data['i_time'] - data['q_invite_min_time']
    data['qid_diff_first_invite_time_sub_max'] = np.divide(data['qid_diff_first_invite_time'] ,data['q_invite_max_time'] - data['q_invite_min_time'])

    t = feature.groupby(['uid'])['i_times'].agg(['min','max']).reset_index().rename(columns={'min': 'u_invite_min_time','max':'u_invite_max_time'})
    data = pd.merge(data, t, how='left', on='uid')
    data['uid_diff_first_invite_time'] = data['i_time'] - data['u_invite_min_time']
    data['uid_diff_first_invite_time_sub_max'] = np.divide(data['qid_diff_first_invite_time'] ,data['q_invite_max_time'] - data['q_invite_min_time'])

    return data,['time_diff','uid_diff_first_invite_time','qid_diff_first_invite_time'
        ,'uid_diff_first_invite_time_sub_max','qid_diff_first_invite_time_sub_max']





def get_seq_uid():
    train = pd.read_csv('../datasets/invite_' + 'train' + '.csv')
    dev = pd.read_csv('../datasets/invite_' + 'dev' + '.csv')
    test = pd.read_csv('../datasets/test_final.csv')

    use_train = pd.read_csv('../datasets/invite_train_use.csv')

    # total1 = pd.concat([use_train,dev,test],axis=0).reset_index(drop=True)

    a1 = train.shape[0]
    a2 = dev.shape[0]
    a3 = test.shape[0]
    total = pd.concat([train, dev,test], axis=0).reset_index(drop=True)
    total['invite_time'] = total['day'] + 0.04166 * total['hour']
    lens_df = pd.DataFrame({'src_i':np.arange(a1 + a2 + a3)})
    total = pd.concat([total,lens_df],axis=1)

    # 计算用户的未来访问时间差
    total_sorted = total.sort_values(by=['uid','invite_time'],axis=0 ,ascending=True).reset_index(drop=True)

    t = total_sorted.groupby('uid')['invite_time'] \
        .apply(lambda x: np.subtract(x.tolist()[1:] + [4100], x.tolist()).tolist()).reset_index() \
        .rename(columns={'invite_time': 'key'})

    t1 = pd.DataFrame({'uid': t.uid.repeat(t.key.str.len()), 'uid_time_feature': np.concatenate(t.key.values)}).reset_index(drop=True)
    print(t1.shape)
    print(t1.head(10))
    t = total_sorted.groupby('uid')['invite_time'] \
        .apply(lambda x: np.subtract(x.tolist(),[2900] + x.tolist()[:-1]).tolist()).reset_index() \
        .rename(columns={'invite_time': 'key'})

    t2 = pd.DataFrame({'uid': t.uid.repeat(t.key.str.len()), 'uid_time_past': np.concatenate(t.key.values)}).reset_index(drop=True)
    print(t2.shape)
    print(t2.head(10))

    total_sorted = pd.concat([total_sorted,t1,t2],axis=1)[['src_i','uid_time_feature','uid_time_past']]

    total = pd.merge(total,total_sorted,how='left',on='src_i')

    #total1 = pd.merge(total1,total,how='left',on=['uid','qid','ctime'])

    featlist = ['uid_time_feature', 'uid_time_past']

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




get_seq_uid()

print_time('extract_id_whole_count')


# ####提取uid qid的全局统计特征
result,fealist,feature = extract_time_feature(total_label,total_feature)
save_num_feat(fealist,result.iloc[:train_shape],'train')
save_num_feat(fealist,result.iloc[train_shape:train_shape+dev_shape],'dev')
save_num_feat(fealist,result.iloc[train_shape+dev_shape:],'test_B')





result,fealist = time_feat(total_label,total_feature)
print_time('save_feature')
save_num_feat(fealist,result.iloc[:train_shape],'train')
save_num_feat(fealist,result.iloc[train_shape:train_shape+dev_shape],'dev')
save_num_feat(fealist,result.iloc[train_shape+dev_shape:],'test_B')
print_time('save_finish')


print_time('--------------end---------------')















# topic_count('train')
# topic_count('dev')
# topic_count('test')













