"""
继续构造其他特征，关于问题的count特征要格外注意线上线下一致性（因为线上测试集采样一半用户分布的）

用户特征：
    uid_ans_dif （用户本次邀请时间距离7天前最后一次回答时间的时间间隔）
    uid_ans_day_nunique (用户历史回答出现的天数)
    uid_last_invite_diff (本次邀请距离用户上次邀请的天数差)

问题特征：
    qid_min_day_count(问题第一次投放当天邀请次数)、qid_duration_days（问题投放的最后天 - 问题最开始投放的天）
    qid_to_first_days(问题当前邀请距离问题第一次邀请的天数差)
    qid_rank（qid当前的邀请时间 - qid最早的邀请时间） / （qid最晚的邀请时间 - qid最早的邀请时间） 包含了测试集的信息
"""

import pandas as pd
import gc

job = 'test'  # train0、train1、dev0、dev1、test
path = '/cu04_nfs/lchen/data/data_set_0926/'

target_start = None
target_end = None
feature_start = None
feature_end = None
answer_start = None
answer_end = None
target_data = None
feature_data = None
answer_data = None

temp_feature = pd.read_csv(open(path + 'invite_info_0926.txt', "r", encoding='utf-8'), sep='\t', header=None, names=['questionID', 'memberID', 'time', 'label'])
temp_feature['day'] = temp_feature['time'].apply(lambda x: int(x.split('-')[0][1:]))
temp_answer = pd.read_csv(open(path + 'answer_info_0926.txt', "r", encoding='utf-8'), sep='\t', header=None,
                          names=['answerID', 'questionID', 'memberID', 'time', 'chars',
                                 'words', 'isexcellent', 'isrecommend', 'isaccept',
                                 'havePicture', 'haveVideo', 'char_len', 'like_count',
                                 'get_like_count', 'comment_count', 'collect_count', 'thanks_count',
                                 'report_count', 'no_help_count', 'oppose_count'])
temp_answer['day'] = temp_answer['time'].apply(lambda x: int(x.split('-')[0][1:]))

if job == 'train0':
    target_start = 3858
    target_end = 3864
    feature_start = 3840
    feature_end = 3857
    answer_start = 3827
    answer_end = 3857
    target_data = pd.read_csv(path + 'features/train0.txt', sep='\t')
    feature_data = temp_feature[(temp_feature['day'] >= feature_start) & (temp_feature['day'] <= feature_end)]
    answer_data = temp_answer[(temp_answer['day'] >= answer_start) & (temp_answer['day'] <= answer_end)]
if job == 'train1':
    target_start = 3858
    target_end = 3864
    feature_start = 3840
    feature_end = 3857
    answer_start = 3827
    answer_end = 3857
    target_data = pd.read_csv(path + 'features/train1.txt', sep='\t')
    feature_data = temp_feature[(temp_feature['day'] >= feature_start) & (temp_feature['day'] <= feature_end)]
    answer_data = temp_answer[(temp_answer['day'] >= answer_start) & (temp_answer['day'] <= answer_end)]
if job == 'dev0':
    target_start = 3865
    target_end = 3867
    feature_start = 3846
    feature_end = 3863
    answer_start = 3833
    answer_end = 3863
    target_data = pd.read_csv(path + 'features/dev0.txt', sep='\t')
    feature_data = temp_feature[(temp_feature['day'] >= feature_start) & (temp_feature['day'] <= feature_end)]
    answer_data = temp_answer[(temp_answer['day'] >= answer_start) & (temp_answer['day'] <= answer_end)]
if job == 'dev1':
    target_start = 3865
    target_end = 3867
    feature_start = 3846
    feature_end = 3863
    answer_start = 3833
    answer_end = 3863
    target_data = pd.read_csv(path + 'features/dev1.txt', sep='\t')
    feature_data = temp_feature[(temp_feature['day'] >= feature_start) & (temp_feature['day'] <= feature_end)]
    answer_data = temp_answer[(temp_answer['day'] >= answer_start) & (temp_answer['day'] <= answer_end)]
if job == 'test':
    target_start = 3868
    target_end = 3874
    feature_start = 3850
    feature_end = 3867
    answer_start = 3837
    answer_end = 3867
    target_data = pd.read_csv(path + 'features/test.txt', sep='\t')
    feature_data = temp_feature[(temp_feature['day'] >= feature_start) & (temp_feature['day'] <= feature_end)]
    answer_data = temp_answer[(temp_answer['day'] >= answer_start) & (temp_answer['day'] <= answer_end)]

del temp_feature
del temp_answer
gc.collect()
gc.collect()

print(job, "target time windows is {}-{}, feature time window is {}-{}, answer time window is {}-{}".format(target_data['day'].min(), target_data['day'].max(), feature_data['day'].min(), feature_data['day'].max(), answer_data['day'].min(), answer_data['day'].max()))
print("shape: ", target_data.shape)

print("问题特征")
total = pd.concat([target_data[['questionID', 'memberID', 'time', 'day']], feature_data], ignore_index=True, sort=False)
question_min_day = total.groupby('questionID')['day'].agg(['min','max']).reset_index().rename(columns={'min': 'qid_min', 'max': 'qid_max'})
question_min_day['qid_duration_days'] = question_min_day['qid_max'] - question_min_day['qid_min']
target_data = target_data.merge(question_min_day, how='left', on='questionID')
temp = total.groupby(['questionID', 'day'], as_index=False)['memberID'].agg({"qid_min_day_count": "count"})
temp['qid_min'] = temp['day']
del temp['day']
target_data = target_data.merge(temp, how='left', on=['questionID', 'qid_min'])
target_data['qid_to_first_days'] = target_data['day'] - target_data['qid_min']
target_data['qid_rank'] = target_data.apply(lambda x: 0 if x['qid_duration_days'] == 0 else x['qid_to_first_days'] / x['qid_duration_days'], axis=1)


print("用户特征")
print("用户当前邀请时间距离7天前最后一次回答时间的时间间隔")
def get_time_list(df):
    return sorted(df['day'].tolist())
temp = answer_data.groupby(['memberID']).apply(get_time_list).reset_index().rename(columns={0: 'answer_time_list'})
target_data = target_data.merge(temp, how='left', on="memberID")
target_data['answer_time_list'] = target_data['answer_time_list'].fillna(0)
def get_uid_ans_dif(answer_time_list, cur_time):
    if answer_time_list == 0 or answer_time_list == '0':
        return "MAX"   # 用户7天前没有回答
    n = len(answer_time_list)
    for i in range(n-1, -1, -1):
        if cur_time >= answer_time_list[i] + 7:
            return str(cur_time - answer_time_list[i])
    return "MAX"
target_data['uid_ans_dif'] = target_data.apply(lambda x: get_uid_ans_dif(x['answer_time_list'], x['day']), axis=1)
target_data['uid_ans_day_nunique'] = target_data.apply(lambda x: 0 if x['answer_time_list'] == 0 or x['answer_time_list'] == '0' else len(set(x['answer_time_list'])), axis=1)

temp = total.groupby(['memberID', 'day'], as_index=False)['questionID'].agg({"count": 'count'})
temp.sort_values('day', inplace=True)
uid_last_invite_diff = []
member_invite_time_dict = {}
for memberID, day in temp[['memberID', 'day']].values:
    if memberID not in member_invite_time_dict.keys():
        uid_last_invite_diff.append("first")
        member_invite_time_dict[memberID] = day
    else:
        uid_last_invite_diff.append(str(day - member_invite_time_dict[memberID]))
        member_invite_time_dict[memberID] = day
temp['uid_last_invite_diff'] = uid_last_invite_diff
target_data = target_data.merge(temp[['memberID', 'day', 'uid_last_invite_diff']], how='left', on=['memberID', 'day'])

del target_data['qid_min']
del target_data['qid_max']
del target_data['answer_time_list']

print("shape: ", target_data.shape)

for col in target_data.columns:
    print(target_data[[col]].describe())

target_data.to_csv(path + "features/" + job + ".txt", sep='\t', index=False)
print("finish!")
