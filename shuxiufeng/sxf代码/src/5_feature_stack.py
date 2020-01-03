import pandas as pd
import numpy as np

def save_num_feat(featlist, data, mode):
    base_path = '../datasets/feature/' + mode + '/'
    for feat in featlist:
        data[[feat]].to_csv(base_path + feat + '.csv', index=False, float_format='%.4f')


src_train  = pd.read_csv('../datasets/invite_train.csv')
src_dev  = pd.read_csv('../datasets/invite_dev.csv')
src_test  = pd.read_csv('../datasets/test_final.csv')

use_train = pd.read_csv('../datasets/invite_train_use.csv')
ans = pd.read_csv('../datasets/answer_info.csv')


#
def extract_feature(target,feature,ans):


    t1 = feature.groupby(['uid'])['day'].agg(['nunique']).reset_index().rename(columns={'nunique': 'uid_stack_day'})
    target = pd.merge(target, t1, how='left', on='uid')
    t1 = feature.groupby(['qid'])['day'].agg(['nunique']).reset_index().rename(columns={'nunique': 'qid_stack_day'})
    target = pd.merge(target, t1, how='left', on='qid')

    t1 = feature.groupby(['uid'])['label'].agg(['mean','count','sum'])\
        .reset_index().rename(columns={'mean':'uid_hist_invite_mean',
                                       'count':'uid_hist_invite_count',
                                       'sum':'uid_hist_invite_sum'})
    target = pd.merge(target,t1,how='left',on='uid')

    t1 = ans.groupby(['uid'])['qid'].agg(['count']).reset_index().rename(columns={'count': 'uid_ans_count'})

    target = pd.merge(target, t1, how='left', on='uid')


    q_info = pd.read_csv('../datasets/question_info.csv',usecols=['qid','topic_id'])
    feature = pd.merge(feature,q_info,how='left',on='qid')
    t = feature.groupby(['uid'])['topic_id'].apply(
        lambda x: len(set(','.join(x.tolist()).split(',')))).reset_index().rename(
        columns={'topic_id': 'uid_hist_invite_topic_unique'})
    target = pd.merge(target, t, how='left', on='uid')

    ans = pd.merge(ans, q_info, how='left', on='qid')
    t = ans.groupby(['uid'])['topic_id'].apply(lambda x: len(set(','.join(x.tolist())))).reset_index().rename(
        columns={'topic_id': 'uid_hist_ans_topic_unique'})
    target = pd.merge(target, t, how='left', on='uid')

    fealist = ['uid_hist_invite_mean','uid_hist_invite_count','uid_hist_invite_sum','uid_ans_count',
        'uid_hist_invite_topic_unique', 'uid_hist_ans_topic_unique','uid_hist_invite_mean'
        ,'uid_hist_invite_count','uid_hist_invite_sum','uid_ans_count'
               ,'uid_stack_day','qid_stack_day']


    return target,fealist

use_day_min = np.min(use_train['day'])
train_feature = src_train[src_train['day'] < use_day_min]
train_ans = ans[ans['a_day'] < use_day_min]
result,fealist = extract_feature(use_train,train_feature,train_ans)
print(result.columns.tolist())
save_num_feat(fealist,result,'train')


use_day_min = np.min(src_dev['day'])
train_feature = src_train
train_ans = ans[ans['a_day'] < use_day_min]
result,fealist = extract_feature(src_dev,train_feature,train_ans)
save_num_feat(fealist,result,'dev')



use_day_min = np.min(src_test['day'])
train_feature = pd.concat([src_train,src_dev],axis=0)
train_ans = ans[ans['a_day'] < use_day_min]
result,fealist = extract_feature(src_test,train_feature,train_ans)
save_num_feat(fealist,result,'test_B')


