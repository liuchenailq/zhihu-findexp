import pandas as pd
import pickle




import pandas as pd
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
    for feat in featlist:
        data[[feat]].to_csv(base_path + feat + '.csv', index=False, float_format='%.4f')



def extract_score_feature(data, feature):
    target = data.copy()
    fea_list = []

    member_info = pd.read_csv('../datasets/member_info.csv')

    tmp = member_info[['uid']]

    score_list = []
    for fea in ['sex','visit','BA','BB','BC','BD','BE','CA','CB','CC','CD','CE']:
        tmp[fea + '_score_rank'] = member_info.groupby(fea)['SCORE'].rank(method='dense')
        score_list.append(fea + '_score_rank')

    target = pd.merge(target, tmp, how='left', on=['uid'])
    return target[score_list],score_list

def extract_score_feature2(data, feature):
    target = data.copy()
    fea_list = []

    member_info = pd.read_csv('../datasets/member_info.csv',usecols=['uid','SCORE'])
    target = pd.merge(target,member_info,how='left',on='uid')

    target['uid_qid_score_rank'] = target.groupby(['qid'])['SCORE'].rank(method='dense')
    target['uid_qid_day_score_rank'] = target.groupby(['qid','day'])['SCORE'].rank(method='dense')
    target['uid_day_score_rank'] = target.groupby(['uid','day'])['SCORE'].rank(method='dense')

    score_list = ['uid_qid_score_rank','uid_qid_day_score_rank','uid_day_score_rank']


    return target[score_list],score_list





print_time('extract_id_whole_count')
result,fealist = extract_score_feature(total_label,total_feature)
print_time('save_feature')
save_num_feat(fealist,result.iloc[:train_shape],'train')
save_num_feat(fealist,result.iloc[train_shape:train_shape + dev_shape],'dev')
save_num_feat(fealist,result.iloc[train_shape + dev_shape:],'test_B')
print_time('save_finish')

#
# ####提取话题的统计的全局统计特征
#
print_time('extract_id_whole_count')
result,fealist = extract_score_feature2(total_label,total_feature)
print_time('save_feature')
save_num_feat(fealist,result.iloc[:train_shape],'train')
save_num_feat(fealist,result.iloc[train_shape:train_shape + dev_shape],'dev')
save_num_feat(fealist,result.iloc[train_shape + dev_shape:],'test_B')
print_time('save_finish')
















# topic_count('train')
# topic_count('dev')
# topic_count('test')













