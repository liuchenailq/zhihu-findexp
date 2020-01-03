"""
train、dev、test用预测的概率填充label后，构造一些特征或者重新生成以前的特征

new_ans_dif_1： 本次邀请距离用户上次回答的天数差
new_memberID_times、new_memberID_pos_times、new_memberID_neg_times：用户历史邀请数、邀请成功数、邀请失败数
new_uid_seq_feature、new_uid_seq_feature_3：用户邀请序列
new_member_continuous_reject_times (用户连续拒绝邀请的次数)
new_member_continuous_accept_times (用户连续接受邀请的次数)
"""
import pandas as pd
import gc


job = 'test'  # train0、train1、dev0、dev1、test
# path = 'E:\\competition\\看山杯\\data\\data_set_0926\\'
path = '/cu04_nfs/lchen/data/data_set_0926/'

target_start = None
target_end = None
feature_start = None
feature_end = None
answer_start = None
answer_end = None
target_data = None   # 预测区间
feature_data = None  # 特征提取区间
pred_data = None  # 预测的结果  'questionID', 'memberID', 'time', 'index', 'result'

temp_feature = pd.read_csv(open(path + 'invite_info_0926.txt', "r", encoding='utf-8'), sep='\t', header=None, names=['questionID', 'memberID', 'time', 'label'])
temp_feature['day'] = temp_feature['time'].apply(lambda x: int(x.split('-')[0][1:]))

if job == 'train0':
    target_start = 3858
    target_end = 3864
    feature_start = 3840
    feature_end = 3857
    answer_start = 3827
    answer_end = 3857
    target_data = pd.read_csv(path + 'features/train0.txt', sep='\t')
    feature_data = temp_feature[(temp_feature['day'] >= feature_start) & (temp_feature['day'] <= feature_end)]
    pred_data = pd.read_csv(path + 'features/pred_train.txt', sep='\t')
    pred_data = pred_data[pred_data['index'].isin(target_data['index'])]
if job == 'train1':
    target_start = 3858
    target_end = 3864
    feature_start = 3840
    feature_end = 3857
    answer_start = 3827
    answer_end = 3857
    target_data = pd.read_csv(path + 'features/train1.txt', sep='\t')
    feature_data = temp_feature[(temp_feature['day'] >= feature_start) & (temp_feature['day'] <= feature_end)]
    pred_data = pd.read_csv(path + 'features/pred_train.txt', sep='\t')
    pred_data = pred_data[pred_data['index'].isin(target_data['index'])]
if job == 'dev0':
    target_start = 3865
    target_end = 3867
    feature_start = 3846
    feature_end = 3863
    answer_start = 3833
    answer_end = 3863
    target_data = pd.read_csv(path + 'features/dev0.txt', sep='\t')
    feature_data = temp_feature[(temp_feature['day'] >= feature_start) & (temp_feature['day'] <= feature_end)]
    pred_data = pd.read_csv(path + 'features/pred_dev.txt', sep='\t')
    pred_data = pred_data[pred_data['index'].isin(target_data['index'])]
if job == 'dev1':
    target_start = 3865
    target_end = 3867
    feature_start = 3846
    feature_end = 3863
    answer_start = 3833
    answer_end = 3863
    target_data = pd.read_csv(path + 'features/dev1.txt', sep='\t')
    feature_data = temp_feature[(temp_feature['day'] >= feature_start) & (temp_feature['day'] <= feature_end)]
    pred_data = pd.read_csv(path + 'features/pred_dev.txt', sep='\t')
    pred_data = pred_data[pred_data['index'].isin(target_data['index'])]
if job == 'test':
    target_start = 3868
    target_end = 3874
    feature_start = 3850
    feature_end = 3867
    answer_start = 3837
    answer_end = 3867
    target_data = pd.read_csv(path + 'features/test.txt', sep='\t')
    feature_data = temp_feature[(temp_feature['day'] >= feature_start) & (temp_feature['day'] <= feature_end)]
    pred_data = pd.read_csv(path + 'features/pred_test.txt', sep='\t')

del temp_feature
gc.collect()
gc.collect()

save_cols = ['questionID', 'memberID', 'time', 'day', 'index']
for col in target_data.columns:
    if col not in save_cols:
        del target_data[col]

gc.collect()

pred_data['label'] = pred_data['result'].apply(lambda x: 1 if x > 0.4 else 0)
pred_data['day'] = pred_data['time'].apply(lambda x: int(x.split('-')[0][1:]))

total_data = pd.concat([feature_data, pred_data], ignore_index=True, sort=False)  # 全部的特征提取区间
total_data.sort_values('day', inplace=True)

print("本次邀请距离用户上次回答的天数差")
def get_time_list(df):
    return sorted(df['day'].tolist())
temp = total_data[total_data['label'] == 1].groupby(['memberID']).apply(get_time_list).reset_index().rename(columns={0: 'answer_time_list'})
target_data = target_data.merge(temp, how='left', on="memberID")
target_data['answer_time_list'] = target_data['answer_time_list'].fillna(0)
def get_uid_ans_dif(answer_time_list, cur_time):
    if answer_time_list == 0 or answer_time_list == '0':
        return "MAX"   # 用户之前没有回答
    n = len(answer_time_list)
    for i in range(n-1, -1, -1):
        if cur_time >= answer_time_list[i] + 1:
            return str(cur_time - answer_time_list[i])
    return "MAX"
target_data['new_ans_dif_1'] = target_data.apply(lambda x: get_uid_ans_dif(x['answer_time_list'], x['day']), axis=1)
save_cols.append("new_ans_dif_1")

target_data.sort_values('day', inplace=True)
print("用户邀请序列")
def uid_seq(data_members, data_days, feature_members, feature_days, feature_labels):
    uid_seq_dict = {}  # memberID: []
    new_uid_seq_feature = []
    new_uid_seq_feature_3 = []

    j = 0
    day = feature_days[j]
    for i, uid in enumerate(data_members):
        cur_day = data_days[i]
        while j < len(feature_days) and cur_day > day:
            memberID = feature_members[j]
            if memberID not in uid_seq_dict.keys():
                uid_seq_dict[memberID] = []
            uid_seq_dict[memberID].append(str(int(feature_labels[j])))
            j += 1
            if j == len(feature_days):
                break
            day = feature_days[j]

        if uid not in uid_seq_dict.keys():
            new_uid_seq_feature.append("__UNKNOWN__")
            new_uid_seq_feature_3.append("__UNKNOWN__")
        else:
            new_uid_seq_feature.append(",".join(uid_seq_dict[uid]))
            new_uid_seq_feature_3.append(",".join(uid_seq_dict[uid][-3:]))
    return new_uid_seq_feature, new_uid_seq_feature_3, uid_seq_dict

target_data['new_uid_seq_feature'], target_data['new_uid_seq_feature_3'], uid_seq_dict = uid_seq(target_data['memberID'].values,
                                                                                                 target_data['day'].values,
                                                                                                 total_data['memberID'].values,
                                                                                                 total_data['day'].values,
                                                                                                 total_data['label'].values)
save_cols.append("new_uid_seq_feature")
save_cols.append("new_uid_seq_feature_3")

print("new_memberID_times、new_memberID_pos_times、new_memberID_neg_times")
target_data['new_memberID_times'] = target_data['new_uid_seq_feature'].apply(lambda x: 0 if x == '__UNKNOWN__' else len(x.split(",")))
target_data['new_memberID_pos_times'] = target_data['new_uid_seq_feature'].apply(lambda x: 0 if x == '__UNKNOWN__' else sum([int(i) for i in x.split(",")]))
target_data['new_memberID_neg_times'] = target_data['new_memberID_times'] - target_data['new_memberID_pos_times']

save_cols.append("new_memberID_times")
save_cols.append("new_memberID_pos_times")
save_cols.append("new_memberID_neg_times")

print("用户连续拒绝邀请的次数 用户连续接受邀请的次数")
def calc_member_continuous_times(x, flag):
    if x == "__UNKNOWN__":
        return 0
    temp = [int(i) for i in x.split(",")]
    count = 0
    for i in range(len(temp)-1, -1, -1):
        if temp[i] == flag:
            count += 1
        else:
            break
    return count
target_data['new_member_continuous_reject_times'] = target_data['new_uid_seq_feature'].apply(lambda x: calc_member_continuous_times(x, 0))
target_data['new_member_continuous_accept_times'] = target_data['new_uid_seq_feature'].apply(lambda x: calc_member_continuous_times(x, 1))
save_cols.append("new_member_continuous_reject_times")
save_cols.append("new_member_continuous_accept_times")

print(save_cols)
target_data = target_data[save_cols]
print("shape: ", target_data.shape)

for col in target_data.columns:
    print(target_data[[col]].describe())

target_data.to_csv(path + "features/new_feature_" + job + ".txt", sep='\t', index=False)
print("finish!")
