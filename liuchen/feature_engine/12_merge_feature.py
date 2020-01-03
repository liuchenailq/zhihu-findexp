import pandas as pd

path = 'E:\\competition\\看山杯\\data\\data_set_0926\\'
train = pd.read_csv(path + "features\\train_dev.txt", sep='\t')
train['index'] = list(range(1, train.shape[0] + 1))

test1 = pd.read_csv(path + 'invite_info_evaluate_1_0926.txt', sep='\t', header=None, names=['questionID', 'memberID', 'time'])
test1['index'] = -1

test = pd.read_csv(path + 'invite_info_evaluate_2_0926.txt', sep='\t', header=None, names=['questionID', 'memberID', 'time'])
test['index'] = list(range(train.shape[0] + 1, train.shape[0] + test.shape[0] + 1))

data = pd.concat([train, test, test1], ignore_index=True, sort=False)
data['day'] = data['time'].apply(lambda x: int(x.split('-')[0][1:]))
data['hour'] = data['time'].apply(lambda x: int(x.split('-')[1][1:]))
data['invite_time'] = data['day'] * 100 + data['hour']

save_cols = ['questionID', 'memberID', 'time', 'index']
print("用户当天的邀请次数、用户当天当小时的邀请次数、问题当天的邀请次数、问题当天当小时的邀请次数")
for feat, time in [('memberID', 'day'), ('memberID', 'day_hour'), ('questionID', 'day'), ('questionID', 'day_hour')]:
    group1 = feat
    group2 = time
    if group2 == 'day_hour':
        group2 = 'invite_time'
    feat_name = "new_{}_{}_count".format(feat, time)
    temp = data.groupby([group1, group2], as_index=False)['index'].agg({feat_name: 'count'})
    data = data.merge(temp[[group1, group2, feat_name]], how='left', on=[group1, group2])
    save_cols.append(feat_name)

print(save_cols)

print("qid_min_day_count")
question_min_day = data.groupby('questionID')['day'].agg(['min','max']).reset_index().rename(columns={'min': 'qid_min', 'max': 'qid_max'})
question_min_day['new_qid_duration_days'] = question_min_day['qid_max'] - question_min_day['qid_min']
data = data.merge(question_min_day, how='left', on='questionID')
data['new_qid_to_first_days'] = data['day'] - data['qid_min']

save_cols.append("new_qid_to_first_days")
save_cols.append("new_qid_duration_days")

temp = data.groupby(['questionID', 'day'], as_index=False)['memberID'].agg({"new_qid_min_day_count": "count"})
data = data.merge(temp, how='left', left_on=['questionID', 'qid_min'], right_on=['questionID', 'day'])
save_cols.append("new_qid_min_day_count")
data['day'] = data['time'].apply(lambda x: int(x.split('-')[0][1:]))


print("qid_rank")
data['new_qid_rank'] = data.apply(lambda x: 0 if x['new_qid_duration_days'] == 0 else x['new_qid_to_first_days'] / x['new_qid_duration_days'], axis=1)
save_cols.append("new_qid_rank")

memberID_min_day = data.groupby('memberID')['day'].agg(['min','max']).reset_index().rename(columns={'min': 'uid_min', 'max': 'uid_max'})
memberID_min_day['new_uid_duration_days'] = memberID_min_day['uid_max'] - memberID_min_day['uid_min']
data = data.merge(memberID_min_day, how='left', on='memberID')
data['new_uid_to_first_days'] = data['day'] - data['uid_min']

save_cols.append("new_uid_duration_days")
save_cols.append("new_uid_to_first_days")

data['new_uid_rank'] = data.apply(lambda x: 0 if x['new_uid_duration_days'] == 0 else x['new_uid_to_first_days'] / x['new_uid_duration_days'], axis=1)
save_cols.append("new_uid_rank")


print("qid_after_invite_diff qid_last_invite_diff")
data.sort_values('day', inplace=True)
questionID_day = data[['questionID', 'day']].values
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
data['new_qid_after_invite_diff'] = qid_after_invite_diff
save_cols.append("new_qid_after_invite_diff")

qid_last_invite_diff = []
question_invite_time_dict = {}
for questionID, day in data[['questionID', 'day']].values:
    if questionID not in question_invite_time_dict.keys():
        qid_last_invite_diff.append("first")
        question_invite_time_dict[questionID] = day
    else:
        qid_last_invite_diff.append(str(day - question_invite_time_dict[questionID]))
        question_invite_time_dict[questionID] = day
data['new_qid_last_invite_diff'] = qid_last_invite_diff
save_cols.append("new_qid_last_invite_diff")

print("uid_last_invite_diff uid_after_invite_diff")

memberID_day = data[['memberID', 'day']].values
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
data['new_uid_after_invite_diff'] = uid_after_invite_diff
save_cols.append("new_uid_after_invite_diff")

uid_last_invite_diff = []
member_invite_time_dict = {}
for memberID, day in data[['memberID', 'day']].values:
    if memberID not in member_invite_time_dict.keys():
        uid_last_invite_diff.append("first")
        member_invite_time_dict[memberID] = day
    else:
        uid_last_invite_diff.append(str(day - member_invite_time_dict[memberID]))
        member_invite_time_dict[memberID] = day
data['new_uid_last_invite_diff'] = uid_last_invite_diff
save_cols.append("new_uid_last_invite_diff")

target_data = data[save_cols]
print(save_cols)
print("shape: ", target_data.shape)

for col in target_data.columns:
    print(target_data[[col]].describe())

target_data.to_csv(path + "features\\merge_feature.txt", sep='\t', index=False)