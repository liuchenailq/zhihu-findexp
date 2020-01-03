#mport pandas as pd
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
test  = pd.read_csv('../datasets/test_final.csv')


total_feature = pd.concat([train,dev,test],axis = 0).reset_index(drop=True)


use_train = pd.read_csv('../datasets/invite_train_use.csv')

total_label = pd.concat([use_train,dev,test],axis=0).reset_index(drop=True)

train_shape = use_train.shape[0]
dev_shape   = dev.shape[0]
test_shape  = test.shape[0]





def save_num_feat(featlist, data, mode):
    base_path = '../datasets/feature/' + mode + '/'
    print(data.shape)
    for feat in featlist:
        data[[feat]].to_csv(base_path + feat + '.csv', index=False, float_format='%.4f')



def extract_id_whole_count(data, feature):
    target = data.copy()
    fea_list = []
    feaname = 'uid_min_day'
    fea_list.append(feaname);print('extract',feaname)
    t = feature.groupby(['uid'])['day'].agg(['min']).reset_index().rename(columns={'min': feaname})
    target = pd.merge(target, t, how='left', on='uid')
    #
    feaname = 'uid_count'
    fea_list.append(feaname);print('extract',feaname)
    t = feature.groupby(['uid'])['label'].agg(['count']).reset_index().rename(columns={'count': feaname})
    target = pd.merge(target,t,how='left',on='uid')
    #
    feaname = 'uid_day_count'
    fea_list.append(feaname);print('extract',feaname)
    t = feature.groupby(['uid','day'])['label'].agg(['count']).reset_index().rename(columns={'count': feaname})
    target = pd.merge(target, t, how='left', on=['uid','day'])
    #
    feaname = 'uid_day_hour_count'
    fea_list.append(feaname);print('extract',feaname)
    t = feature.groupby(['uid', 'day','hour'])['label'].agg(['count']).reset_index().rename(columns={'count': feaname})
    target = pd.merge(target, t, how='left', on=['uid', 'day','hour'])
    #
    feaname = 'uid_min_day_count'
    fea_list.append(feaname);print('extract',feaname)
    t = feature.groupby(['uid', 'day'])['label'].agg(['count']).reset_index().rename(columns={'count': feaname,'day':'uid_min_day'})
    target = pd.merge(target, t, how='left', on=['uid','uid_min_day'])

    feaname = 'uid_day_nuinque'
    fea_list.append(feaname);print('extract', feaname)
    t = feature.groupby(['uid'])['day'].agg(['nunique']).reset_index().rename(
        columns={'nunique': feaname, 'day': feaname})
    target = pd.merge(target, t, how='left', on=['uid'])

    feaname = 'qid_day_nuinque'
    fea_list.append(feaname);
    print('extract', feaname)
    t = feature.groupby(['qid'])['day'].agg(['nunique']).reset_index().rename(
        columns={'nunique': feaname, 'day': feaname})
    target = pd.merge(target, t, how='left', on=['qid'])


    print_time(target.columns.tolist())

    feaname = 'qid_min_day'
    fea_list.append(feaname);print('extract',feaname)
    t = feature.groupby(['qid'])['day'].agg(['min']).reset_index().rename(columns={'min': feaname})
    target = pd.merge(target, t, how='left', on='qid')

    print_time(target.columns.tolist())

    feaname = 'qid_count'
    fea_list.append(feaname);print('extract',feaname)
    t = feature.groupby(['qid'])['label'].agg(['count']).reset_index().rename(columns={'count': feaname})
    target = pd.merge(target, t, how='left', on='qid')

    feaname = 'qid_day_count'
    fea_list.append(feaname);print('extract',feaname)
    t = feature.groupby(['qid', 'day'])['label'].agg(['count']).reset_index().rename(columns={'count': feaname})
    target = pd.merge(target, t, how='left', on=['qid', 'day'])

    feaname = 'qid_day_hour_count'
    fea_list.append(feaname);print('extract',feaname)
    t = feature.groupby(['qid', 'day', 'hour'])['label'].agg(['count']).reset_index().rename(columns={'count': feaname})
    target = pd.merge(target, t, how='left', on=['qid', 'day', 'hour'])

    feaname = 'qid_min_day_count'
    fea_list.append(feaname);print('extract',feaname)
    t = feature.groupby(['qid', 'day'])['label'].agg(['count']).reset_index().rename(columns={'count': feaname,'day':'qid_min_day'})
    target = pd.merge(target, t, how='left', on=['qid', 'qid_min_day'])


    return target[fea_list],fea_list





def extract_topic_whole_count(data,feature):
    print('extract_feature')

    target = data.copy()

    que = pd.read_csv('../datasets/question_info.csv', usecols=['qid', 'topic_id'])
    target = pd.merge(target, que, how='left', on='qid').fillna('0')
    feature = pd.merge(feature, que, how='left', on='qid').fillna('0')

    print_time('extenf')

    total_extend = feature['topic_id'].str.split(',', expand=True).stack() \
        .reset_index(level=0).set_index('level_0') \
        .rename(columns={0: 'topic'}).join(feature.drop('topic_id', axis=1)) \
        .reset_index(drop=True)

    topic_df = target['topic_id'].str.split(',', expand=True)
    target = pd.concat([target, topic_df], axis=1)

    print_time('extend_finish')

    stat_feat = [
        (['topic'], ['label'], ['count']),
        (['topic'], ['qid'], ['nunique']),
        (['topic'], ['uid'], ['nunique']),
        (['topic', 'day', 'hour'], ['uid'], ['nunique']),
    ]
    final_list = []

    for stat in stat_feat:
        fea_name = '_'.join(stat[0]) + '_' + '_'.join(stat[1]) + '_' + '_'.join(stat[2])
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


'''

看这块

'''

def extract_topic_count_feature(data,feature):
    print('extract_topic count feature')

    target = data.copy()
    que = pd.read_csv('../datasets/question_info.csv', usecols=['qid', 'topic_id'])
    m_list = ['uid','sex','visit', 'CA', 'CB', 'CC', 'CD', 'CE']
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
        fea_name = stat + '_topic_count_ratio'
        print('extract',fea_name)
        ###统计话题和用户属性的交叉量
        s_total_extend = total_extend[[stat,'topic','label']]

        print('groupvy start')
        t = s_total_extend.groupby(['topic',stat])['label'].agg(['count']).reset_index().rename(columns={'count': 'sum_count'})
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
        print('merge start')
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
        target[fea_name + '_mean'] = target[tmp_name].sum(axis=1)
        final_list.append(fea_name + '_max')
        final_list.append(fea_name + '_mean')

        for field in [0, 1, 2, 3, 4]:
            target = target.drop([fea_name + str(field)], axis=1)

    return target[final_list], final_list







def extract_usr_unique(data,feature):

    target = data.copy()
    que = pd.read_csv('../datasets2/question_info.csv', usecols=['qid', 'topic_id'])
    ans = pd.read_csv('../datasets2/answer_info.csv', usecols=['uid','qid'])
    ans = pd.merge(ans,que,how='left',on='qid')
    feature = pd.merge(feature, que, how='left', on='qid').fillna('0')
    ## 获取用户被邀请的话题种类
    t = feature.groupby(['uid'])['topic_id'].apply(lambda x: len(set(','.join(x.tolist()).split(',')))).reset_index().rename(columns={'topic_id':'uid_invite_topic_unique'})
    target = pd.merge(target,t,how='left',on='uid')

    t = ans.groupby(['uid'])['topic_id'].apply(lambda x: len(set(','.join(x.tolist())))).reset_index().rename(columns={'topic_id':'uid_ans_topic_unique'})
    target = pd.merge(target, t, how='left', on='uid')

    fealist = ['uid_invite_topic_unique','uid_ans_topic_unique']
    return target[fealist],fealist

result, fealist = extract_topic_whole_count(total_label, total_feature)
print_time('save_feature')
save_num_feat(fealist,result.iloc[:train_shape],'train')
save_num_feat(fealist,result.iloc[train_shape:train_shape + dev_shape],'dev')
save_num_feat(fealist, result.iloc[train_shape + dev_shape:], 'test_B')
print_time('save_finish')


    ## 获取用户回答过的话题种类



result,fealist = extract_id_whole_count(total_label,total_feature)
print_time('save_feature')
save_num_feat(fealist,result.iloc[:train_shape],'train')
save_num_feat(fealist,result.iloc[train_shape:train_shape + dev_shape],'dev')
save_num_feat(fealist,result.iloc[train_shape + dev_shape:],'test_B')
print_time('save_finish')





result,fealist = extract_topic_count_feature(total_label,total_feature)
print_time('save_feature')
save_num_feat(fealist,result.iloc[:train_shape],'train')
save_num_feat(fealist,result.iloc[train_shape:train_shape + dev_shape],'dev')
save_num_feat(fealist,result.iloc[train_shape + dev_shape:],'test_B')
print_time('save_finish')
#
#
# ####提取话题的统计的全局统计特征
#




result,fealist = extract_usr_unique(total_label,total_feature)
print_time('save_feature')
save_num_feat(fealist,result.iloc[:train_shape],'train')
save_num_feat(fealist,result.iloc[train_shape:train_shape + dev_shape],'dev')
save_num_feat(fealist,result.iloc[train_shape + dev_shape:],'test_B')
print_time('save_finish')














# topic_count('train')
# topic_count('dev')
# topic_count('test')













