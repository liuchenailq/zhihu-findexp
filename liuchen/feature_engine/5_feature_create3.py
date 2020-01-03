"""
member_continuous_reject_times (用户连续拒绝邀请的次数)
member_continuous_accept_times (用户连续接受邀请的次数)

uid_after_invite_diff (本次邀请距离用户下次邀请的天数)
qid_after_invite_diff(本次邀请距离问题下次邀请的天数)
qid_last_invite_diff(本次邀请距离问题上次邀请的天数)

uid_ans_topic_count (用户回答过多少种话题 在回答记录表种统计)
"""

import pandas as pd
import gc
import numpy as np

job = 'test'  # train0、train1、dev0、dev1、test
# path = 'E:\\competition\\看山杯\\data\\data_set_0926\\'
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

print(job, "target time windows is {}-{}, feature time window is {}-{}".format(target_data['day'].min(), target_data['day'].max(), feature_data['day'].min(), feature_data['day'].max()))
print("shape: ", target_data.shape)

save_cols = ['questionID', 'memberID', 'time', 'index', 'day']
if job != 'test':
    save_cols.append("label")
for col in target_data.columns:
    if col not in save_cols:
        del target_data[col]
for col in answer_data.columns:
    if col not in ['questionID', 'memberID', 'time', 'day']:
        del answer_data[col]
gc.collect()
gc.collect()

print("读取问题信息")
question_data = pd.read_csv(open(path + 'question_info_0926.txt', "r", encoding='utf-8'), sep='\t', header=None,
                               names=['questionID', 'q_createTime', 'q_title_chars', 'q_title_words', 'q_desc_chars',
                               'q_desc_words', 'q_topic_IDs'])


feature_data['hour'] = feature_data['time'].apply(lambda x: int(x.split('-')[1][1:]))
feature_data['invite_time'] = feature_data['day'] * 100 + feature_data['hour']
feature_data.sort_values('invite_time', inplace=True)

# print("用户点击序列")  # 去除
# def user_click_seq(memberIDs, features):
#     member_seq_dict = {}
#     for memberID, label in features[['memberID', 'label']].values:
#         if memberID not in member_seq_dict.keys():
#             member_seq_dict[memberID] = []
#         member_seq_dict[memberID].append(str(int(label)))
#     member_seq = []
#     member_seq_3 = []
#     for memberID in memberIDs:
#         if memberID not in member_seq_dict.keys():
#             member_seq.append("__UNKNOWN__")
#             member_seq_3.append("__UNKNOWN__")
#         else:
#             member_seq.append(",".join(member_seq_dict[memberID]))
#             member_seq_3.append(",".join(member_seq_dict[memberID][-3:]))
#     return member_seq, member_seq_3
# target_data['uid_seq_feature'], target_data['uid_seq_feature_3'] = user_click_seq(target_data['memberID'].values, feature_data)


print("用户连续拒绝邀请的次数 用户连续接受邀请的次数")
member_continuous_reject_times_dict = {}
member_continuous_accept_times_dict = {}
for memberID, label in feature_data[['memberID', 'label']].values:
    if int(label) == 1:
        member_continuous_reject_times_dict[memberID] = 0
        member_continuous_accept_times_dict[memberID] = member_continuous_accept_times_dict.get(memberID, 0) + 1
    else:
        member_continuous_accept_times_dict[memberID] = 0
        member_continuous_reject_times_dict[memberID] = member_continuous_reject_times_dict.get(memberID, 0) + 1
target_data['member_continuous_reject_times'] = target_data['memberID'].apply(lambda x: member_continuous_reject_times_dict.get(x, 0))
target_data['member_continuous_accept_times'] = target_data['memberID'].apply(lambda x: member_continuous_accept_times_dict.get(x, 0))
save_cols.append("member_continuous_reject_times")
save_cols.append("member_continuous_accept_times")

print("本次邀请距离用户下次邀请的天数")
temp = target_data.groupby(['memberID', 'day'], as_index=False)['questionID'].agg({"count": 'count'})
temp.sort_values('day', inplace=True)
memberID_day = temp[['memberID', 'day']].values
uid_after_invite_diff = []
member_invite_time_dict = {}
for i in range(len(memberID_day)-1, -1, -1):
    memberID = memberID_day[i][0]
    day = memberID_day[i][1]
    if memberID not in member_invite_time_dict.keys():
        uid_after_invite_diff.append("last")
        member_invite_time_dict[memberID] = day
    else:
        uid_after_invite_diff.append(str(member_invite_time_dict[memberID] - day))
        member_invite_time_dict[memberID] = day

uid_after_invite_diff = uid_after_invite_diff[::-1]
temp['uid_after_invite_diff'] = uid_after_invite_diff
target_data = target_data.merge(temp[['memberID', 'day', 'uid_after_invite_diff']], how='left', on=['memberID', 'day'])
save_cols.append("uid_after_invite_diff")

print("本次邀请距离问题上次邀请的天数")
total = pd.concat([target_data[['questionID', 'memberID', 'time', 'day']], feature_data[['questionID', 'memberID', 'time', 'day']]], ignore_index=True, sort=False)
temp = total.groupby(['questionID', 'day'], as_index=False)['memberID'].agg({"count": 'count'})
temp.sort_values('day', inplace=True)
qid_last_invite_diff = []
question_invite_time_dict = {}
for questionID, day in temp[['questionID', 'day']].values:
    if questionID not in question_invite_time_dict.keys():
        qid_last_invite_diff.append("first")
        question_invite_time_dict[questionID] = day
    else:
        qid_last_invite_diff.append(str(day - question_invite_time_dict[questionID]))
        question_invite_time_dict[questionID] = day
temp['qid_last_invite_diff'] = qid_last_invite_diff
target_data = target_data.merge(temp[['questionID', 'day', 'qid_last_invite_diff']], how='left', on=['questionID', 'day'])
save_cols.append("qid_last_invite_diff")

print("本次邀请距离问题下次邀请的天数差")
temp = target_data.groupby(['questionID', 'day'], as_index=False)['memberID'].agg({"count": 'count'})
temp.sort_values('day', inplace=True)
questionID_day = temp[['questionID', 'day']].values
qid_after_invite_diff = []
question_invite_time_dict = {}
for i in range(len(questionID_day)-1, -1, -1):
    questionID = questionID_day[i][0]
    day = questionID_day[i][1]
    if questionID not in question_invite_time_dict.keys():
        qid_after_invite_diff.append("last")
        question_invite_time_dict[questionID] = day
    else:
        qid_after_invite_diff.append(str(question_invite_time_dict[questionID] - day))
        question_invite_time_dict[questionID] = day
qid_after_invite_diff = qid_after_invite_diff[::-1]
temp['qid_after_invite_diff'] = qid_after_invite_diff
target_data = target_data.merge(temp[['questionID', 'day', 'qid_after_invite_diff']], how='left', on=['questionID', 'day'])
save_cols.append("qid_after_invite_diff")

print("用户回答过多少种话题")
answer_data = answer_data.merge(question_data[["questionID", 'q_topic_IDs']], how='left', on='questionID')
uid_ans_topic_count_dict = {}
for memberID, topics in answer_data[['memberID', 'q_topic_IDs']].values:
    if memberID not in uid_ans_topic_count_dict.keys():
        uid_ans_topic_count_dict[memberID] = set()
    if topics != "-1":
        for topic in topics.split(","):
            uid_ans_topic_count_dict[memberID].add(topic)
target_data['uid_ans_topic_count'] = target_data['memberID'].apply(lambda x: 0 if x not in uid_ans_topic_count_dict.keys() else len(uid_ans_topic_count_dict[x]))
save_cols.append("uid_ans_topic_count")

print(save_cols)
target_data = target_data[save_cols]
print("shape: ", target_data.shape)

for col in target_data.columns:
    print(target_data[[col]].describe())

target_data.to_csv(path + "features/feature3_" + job + ".txt", sep='\t', index=False)
print("finish!")
