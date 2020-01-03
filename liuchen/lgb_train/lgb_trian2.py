import pandas as pd
import gc
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

path = '/cu04_nfs/lchen/data/data_set_0926/'
result_path = '/cu04_nfs/lchen/cache/result.txt'


print("读取训练集、验证集、测试集")
train0 = pd.read_csv(path + 'features/new_train0.txt', sep='\t')
train1 = pd.read_csv(path + 'features/new_train1.txt', sep='\t')
dev0 = pd.read_csv(path + 'features/new_dev0.txt', sep='\t')
dev1 = pd.read_csv(path + 'features/new_dev1.txt', sep='\t')
test = pd.read_csv(path + 'features/new_test.txt', sep='\t')

print("读取topic和用户属性的转化率")
train0_topic_user = pd.read_csv(path + 'features/topic_user_rate_train0.txt', sep='\t')
train1_topic_user = pd.read_csv(path + 'features/topic_user_rate_train1.txt', sep='\t')
dev0_topic_user = pd.read_csv(path + 'features/topic_user_rate_dev0.txt', sep='\t')
dev1_topic_user = pd.read_csv(path + 'features/topic_user_rate_dev1.txt', sep='\t')
test_topic_user = pd.read_csv(path + 'features/topic_user_rate_test.txt', sep='\t')


topic_user_rate_cols = ['topic_memberID_rate_max', 'topic_m_sex_rate_max', 'topic_m_twoA_rate_max', 'topic_m_twoB_rate_max', 'topic_m_twoC_rate_max', 'topic_m_twoD_rate_max', 'topic_m_twoE_rate_max', 'topic_m_categoryA_rate_max', 'topic_m_categoryB_rate_max', 'topic_m_categoryC_rate_max', 'topic_m_categoryD_rate_max', 'topic_m_categoryE_rate_max', 'topic_m_access_frequencies_rate_max']
for col in topic_user_rate_cols:
    del train0[col]
    del train1[col]
    del dev0[col]
    del dev1[col]

train0 = train0.merge(train0_topic_user[topic_user_rate_cols + ['index']], how='left', on='index')
train1 = train1.merge(train1_topic_user[topic_user_rate_cols + ['index']], how='left', on='index')
dev0 = dev0.merge(dev0_topic_user[topic_user_rate_cols + ['index']], how='left', on='index')
dev1 = dev1.merge(dev1_topic_user[topic_user_rate_cols + ['index']], how='left', on='index')
test = test.merge(test_topic_user[topic_user_rate_cols + ['index']], how='left', on='index')

print("topic和用户属性的转化率 缺失值用中位数填充")
for col in topic_user_rate_cols:
    train0[col].fillna(train0[col].median(), inplace=True)
    train1[col].fillna(train1[col].median(), inplace=True)
    dev0[col].fillna(dev0[col].median(), inplace=True)
    dev1[col].fillna(dev1[col].median(), inplace=True)
    test[col].fillna(test[col].median(), inplace=True)

print("读取问题标题词和用户属性的转化率")
train0_title_word_user = pd.read_csv(path + 'features/title_word_user_rate_train0.txt', sep='\t')
train1_title_word_user = pd.read_csv(path + 'features/title_word_user_rate_train1.txt', sep='\t')
dev0_title_word_user = pd.read_csv(path + 'features/title_word_user_rate_dev0.txt', sep='\t')
dev1_title_word_user = pd.read_csv(path + 'features/title_word_user_rate_dev1.txt', sep='\t')
test_title_word_user = pd.read_csv(path + 'features/title_word_user_rate_test.txt', sep='\t')

title_word_user_rate_cols = ['title_word_memberID_rate_max', 'title_word_m_sex_rate_max', 'title_word_m_twoA_rate_max', 'title_word_m_twoB_rate_max', 'title_word_m_twoC_rate_max', 'title_word_m_twoD_rate_max', 'title_word_m_twoE_rate_max', 'title_word_m_categoryA_rate_max', 'title_word_m_categoryB_rate_max', 'title_word_m_categoryC_rate_max', 'title_word_m_categoryD_rate_max', 'title_word_m_categoryE_rate_max', 'title_word_m_access_frequencies_rate_max']
train0 = train0.merge(train0_title_word_user[title_word_user_rate_cols + ['index']], how='left', on='index')
train1 = train1.merge(train1_title_word_user[title_word_user_rate_cols + ['index']], how='left', on='index')
dev0 = dev0.merge(dev0_title_word_user[title_word_user_rate_cols + ['index']], how='left', on='index')
dev1 = dev1.merge(dev1_title_word_user[title_word_user_rate_cols + ['index']], how='left', on='index')
test = test.merge(test_title_word_user[title_word_user_rate_cols + ['index']], how='left', on='index')

print("问题标题词和用户属性的转化率 缺失值用中位数填充")
for col in title_word_user_rate_cols:
    train0[col].fillna(train0[col].median(), inplace=True)
    train1[col].fillna(train1[col].median(), inplace=True)
    dev0[col].fillna(dev0[col].median(), inplace=True)
    dev1[col].fillna(dev1[col].median(), inplace=True)
    test[col].fillna(test[col].median(), inplace=True)


print("读取问题描述词和用户属性的转化率")
train0_desc_word_user = pd.read_csv(path + 'features/desc_word_user_rate_train0.txt', sep='\t')
train1_desc_word_user = pd.read_csv(path + 'features/desc_word_user_rate_train1.txt', sep='\t')
dev0_desc_word_user = pd.read_csv(path + 'features/desc_word_user_rate_dev0.txt', sep='\t')
dev1_desc_word_user = pd.read_csv(path + 'features/desc_word_user_rate_dev1.txt', sep='\t')
test_desc_word_user = pd.read_csv(path + 'features/desc_word_user_rate_test.txt', sep='\t')

desc_word_user_rate_cols = ['desc_word_memberID_rate_max', 'desc_word_m_sex_rate_max', 'desc_word_m_twoA_rate_max', 'desc_word_m_twoB_rate_max', 'desc_word_m_twoC_rate_max', 'desc_word_m_twoD_rate_max', 'desc_word_m_twoE_rate_max', 'desc_word_m_categoryA_rate_max', 'desc_word_m_categoryB_rate_max', 'desc_word_m_categoryC_rate_max', 'desc_word_m_categoryD_rate_max', 'desc_word_m_categoryE_rate_max', 'desc_word_m_access_frequencies_rate_max']
train0 = train0.merge(train0_desc_word_user[desc_word_user_rate_cols + ['index']], how='left', on='index')
train1 = train1.merge(train1_desc_word_user[desc_word_user_rate_cols + ['index']], how='left', on='index')
dev0 = dev0.merge(dev0_desc_word_user[desc_word_user_rate_cols + ['index']], how='left', on='index')
dev1 = dev1.merge(dev1_desc_word_user[desc_word_user_rate_cols + ['index']], how='left', on='index')
test = test.merge(test_desc_word_user[desc_word_user_rate_cols + ['index']], how='left', on='index')
print("问题描述词和用户属性的转化率 缺失值用中位数填充")
for col in desc_word_user_rate_cols:
    train0[col].fillna(train0[col].median(), inplace=True)
    train1[col].fillna(train1[col].median(), inplace=True)
    dev0[col].fillna(dev0[col].median(), inplace=True)
    dev1[col].fillna(dev1[col].median(), inplace=True)
    test[col].fillna(test[col].median(), inplace=True)




# print("读取topic和用户属性的3交叉转化率")
# train0_topic3_user = pd.read_csv(path + 'features/topic_user_3rate_train0.txt', sep='\t')
# train1_topic3_user = pd.read_csv(path + 'features/topic_user_3rate_train1.txt', sep='\t')
# dev0_topic3_user = pd.read_csv(path + 'features/topic_user_3rate_dev0.txt', sep='\t')
# dev1_topic3_user = pd.read_csv(path + 'features/topic_user_3rate_dev1.txt', sep='\t')
# test_topic3_user = pd.read_csv(path + 'features/topic_user_3rate_test.txt', sep='\t')
#
# topic3_user_rate_cols = []
# fea_list = ['m_sex', 'm_twoA', 'm_twoB', 'm_twoC', 'm_twoD', 'm_twoE', 'm_categoryB', 'm_categoryC', 'm_categoryE', 'm_access_frequencies']
# for i in range(0, len(fea_list)-1):
#     for j in range(i+1, len(fea_list)):
#         fea1 = fea_list[i]
#         fea2 = fea_list[j]
#         feature_name = "topic_" + fea1 + "_" + fea2 + "_rate_max"
#         topic3_user_rate_cols.append(feature_name)
#
# train0 = train0.merge(train0_topic3_user[topic3_user_rate_cols + ['index']], how='left', on='index')
# train1 = train1.merge(train1_topic3_user[topic3_user_rate_cols + ['index']], how='left', on='index')
# dev0 = dev0.merge(dev0_topic3_user[topic3_user_rate_cols + ['index']], how='left', on='index')
# dev1 = dev1.merge(dev1_topic3_user[topic3_user_rate_cols + ['index']], how='left', on='index')
# test = test.merge(test_topic3_user[topic3_user_rate_cols + ['index']], how='left', on='index')
#
# print("topic和用户属性的3交叉转化率 缺失值用中位数填充")
# for col in topic3_user_rate_cols:
#     train0[col].fillna(train0[col].median(), inplace=True)
#     train1[col].fillna(train1[col].median(), inplace=True)
#     dev0[col].fillna(dev0[col].median(), inplace=True)
#     dev1[col].fillna(dev1[col].median(), inplace=True)
#     test[col].fillna(test[col].median(), inplace=True)


# print("读取问题标题字和用户属性的转化率")
# train0_title_char_user = pd.read_csv(path + 'features/title_char_user_rate_train0.txt', sep='\t')
# train1_title_char_user = pd.read_csv(path + 'features/title_char_user_rate_train1.txt', sep='\t')
# dev0_title_char_user = pd.read_csv(path + 'features/title_char_user_rate_dev0.txt', sep='\t')
# dev1_title_char_user = pd.read_csv(path + 'features/title_char_user_rate_dev1.txt', sep='\t')
# test_title_char_user = pd.read_csv(path + 'features/title_char_user_rate_test.txt', sep='\t')
#
# title_char_user_rate_cols = ['title_char_memberID_rate_max', 'title_char_m_sex_rate_max', 'title_char_m_twoA_rate_max', 'title_char_m_twoB_rate_max', 'title_char_m_twoC_rate_max', 'title_char_m_twoD_rate_max', 'title_char_m_twoE_rate_max', 'title_char_m_categoryA_rate_max', 'title_char_m_categoryB_rate_max', 'title_char_m_categoryC_rate_max', 'title_char_m_categoryD_rate_max', 'title_char_m_categoryE_rate_max', 'title_char_m_access_frequencies_rate_max']
# train0 = train0.merge(train0_title_char_user[title_char_user_rate_cols + ['index']], how='left', on='index')
# train1 = train1.merge(train1_title_char_user[title_char_user_rate_cols + ['index']], how='left', on='index')
# dev0 = dev0.merge(dev0_title_char_user[title_char_user_rate_cols + ['index']], how='left', on='index')
# dev1 = dev1.merge(dev1_title_char_user[title_char_user_rate_cols + ['index']], how='left', on='index')
# test = test.merge(test_title_char_user[title_char_user_rate_cols + ['index']], how='left', on='index')
#
# print("问题标题字和用户属性的转化率 缺失值用中位数填充")
# for col in title_char_user_rate_cols:
#     train0[col].fillna(train0[col].median(), inplace=True)
#     train1[col].fillna(train1[col].median(), inplace=True)
#     dev0[col].fillna(dev0[col].median(), inplace=True)
#     dev1[col].fillna(dev1[col].median(), inplace=True)
#     test[col].fillna(test[col].median(), inplace=True)

print("读取特征3生成的特征")
feature3_cols = ['member_continuous_reject_times', 'member_continuous_accept_times', 'uid_after_invite_diff', 'qid_after_invite_diff', 'qid_last_invite_diff', 'uid_ans_topic_count']
feature3_train0 = pd.read_csv(path + 'features/feature3_train0.txt', sep='\t')
feature3_train1 = pd.read_csv(path + 'features/feature3_train1.txt', sep='\t')
feature3_dev0 = pd.read_csv(path + 'features/feature3_dev0.txt', sep='\t')
feature3_dev1 = pd.read_csv(path + 'features/feature3_dev1.txt', sep='\t')
feature3_test = pd.read_csv(path + 'features/feature3_test.txt', sep='\t')

train0 = train0.merge(feature3_train0[feature3_cols + ['index']], how='left', on='index')
train1 = train1.merge(feature3_train1[feature3_cols + ['index']], how='left', on='index')
dev0 = dev0.merge(feature3_dev0[feature3_cols + ['index']], how='left', on='index')
dev1 = dev1.merge(feature3_dev1[feature3_cols + ['index']], how='left', on='index')
test = test.merge(feature3_test[feature3_cols + ['index']], how='left', on='index')


print("读取在回答记录表中统计的用户属性在话题的占比特征")
topic_user_ratio_cols = []
fea_list = ['memberID', 'm_sex', 'm_twoA', 'm_twoB', 'm_twoC', 'm_twoD', 'm_twoE', 'm_categoryA', 'm_categoryB', 'm_categoryC', 'm_categoryD', 'm_categoryE', 'm_access_frequencies']
for feat in fea_list:
    topic_user_ratio_cols.append(feat + '_topic_count_ratio_max')
temp_train0 = pd.read_csv(path + 'features/topic_user_ratio_train0.txt', sep='\t')
temp_train1 = pd.read_csv(path + 'features/topic_user_ratio_train1.txt', sep='\t')
temp_dev0 = pd.read_csv(path + 'features/topic_user_ratio_dev0.txt', sep='\t')
temp_dev1 = pd.read_csv(path + 'features/topic_user_ratio_dev1.txt', sep='\t')
temp_test = pd.read_csv(path + 'features/topic_user_ratio_test.txt', sep='\t')

train0 = train0.merge(temp_train0[topic_user_ratio_cols + ['index']], how='left', on='index')
train1 = train1.merge(temp_train1[topic_user_ratio_cols + ['index']], how='left', on='index')
dev0 = dev0.merge(temp_dev0[topic_user_ratio_cols + ['index']], how='left', on='index')
dev1 = dev1.merge(temp_dev1[topic_user_ratio_cols + ['index']], how='left', on='index')
test = test.merge(temp_test[topic_user_ratio_cols + ['index']], how='left', on='index')
for col in topic_user_ratio_cols:
    train0[col].fillna(train0[col].median(), inplace=True)
    train1[col].fillna(train1[col].median(), inplace=True)
    dev0[col].fillna(dev0[col].median(), inplace=True)
    dev1[col].fillna(dev1[col].median(), inplace=True)
    test[col].fillna(test[col].median(), inplace=True)

print("在回答记录中统计回答该话题的用户盐值分数的平均分、最低分、最高分")
topic_m_salt_score_cols = ['topic_m_salt_score_min', 'topic_m_salt_score_max', 'topic_m_salt_score_mean', 'diff_topic_m_salt_score_min', 'diff_topic_m_salt_score_max', 'diff_topic_m_salt_score_mean']
temp_train0 = pd.read_csv(path + 'features/topic_salt_score_train0.txt', sep='\t')
temp_train1 = pd.read_csv(path + 'features/topic_salt_score_train1.txt', sep='\t')
temp_dev0 = pd.read_csv(path + 'features/topic_salt_score_dev0.txt', sep='\t')
temp_dev1 = pd.read_csv(path + 'features/topic_salt_score_dev1.txt', sep='\t')
temp_test = pd.read_csv(path + 'features/topic_salt_score_test.txt', sep='\t')

train0 = train0.merge(temp_train0[topic_m_salt_score_cols + ['index']], how='left', on='index')
train1 = train1.merge(temp_train1[topic_m_salt_score_cols + ['index']], how='left', on='index')
dev0 = dev0.merge(temp_dev0[topic_m_salt_score_cols + ['index']], how='left', on='index')
dev1 = dev1.merge(temp_dev1[topic_m_salt_score_cols + ['index']], how='left', on='index')
test = test.merge(temp_test[topic_m_salt_score_cols + ['index']], how='left', on='index')

print("读取在回答记录表中统计的用户属性在问题标题词的占比特征")
title_word_user_ratio_cols = []
fea_list = ['memberID', 'm_sex', 'm_twoA', 'm_twoB', 'm_twoC', 'm_twoD', 'm_twoE', 'm_categoryA', 'm_categoryB', 'm_categoryC', 'm_categoryD', 'm_categoryE', 'm_access_frequencies']
for feat in fea_list:
    title_word_user_ratio_cols.append(feat + '_title_word_count_ratio_max')
temp_train0 = pd.read_csv(path + 'features/title_word_user_ratio_train0.txt', sep='\t')
temp_train1 = pd.read_csv(path + 'features/title_word_user_ratio_train1.txt', sep='\t')
temp_dev0 = pd.read_csv(path + 'features/title_word_user_ratio_dev0.txt', sep='\t')
temp_dev1 = pd.read_csv(path + 'features/title_word_user_ratio_dev1.txt', sep='\t')
temp_test = pd.read_csv(path + 'features/title_word_user_ratio_test.txt', sep='\t')

train0 = train0.merge(temp_train0[title_word_user_ratio_cols + ['index']], how='left', on='index')
train1 = train1.merge(temp_train1[title_word_user_ratio_cols + ['index']], how='left', on='index')
dev0 = dev0.merge(temp_dev0[title_word_user_ratio_cols + ['index']], how='left', on='index')
dev1 = dev1.merge(temp_dev1[title_word_user_ratio_cols + ['index']], how='left', on='index')
test = test.merge(temp_test[title_word_user_ratio_cols + ['index']], how='left', on='index')
for col in title_word_user_ratio_cols:
    train0[col].fillna(train0[col].median(), inplace=True)
    train1[col].fillna(train1[col].median(), inplace=True)
    dev0[col].fillna(dev0[col].median(), inplace=True)
    dev1[col].fillna(dev1[col].median(), inplace=True)
    test[col].fillna(test[col].median(), inplace=True)

print("读取利用预测概率填充后新的特征")
new_feature_cols = ['new_ans_dif_1', 'new_memberID_times', 'new_memberID_pos_times', 'new_memberID_neg_times', 'new_uid_seq_feature', 'new_uid_seq_feature_3', 'new_member_continuous_reject_times', 'new_member_continuous_accept_times']
temp_train0 = pd.read_csv(path + 'features/new_feature_train0.txt', sep='\t')
temp_train1 = pd.read_csv(path + 'features/new_feature_train1.txt', sep='\t')
temp_dev0 = pd.read_csv(path + 'features/new_feature_dev0.txt', sep='\t')
temp_dev1 = pd.read_csv(path + 'features/new_feature_dev1.txt', sep='\t')
temp_test = pd.read_csv(path + 'features/new_feature_test.txt', sep='\t')

train0 = train0.merge(temp_train0[new_feature_cols + ['index']], how='left', on='index')
train1 = train1.merge(temp_train1[new_feature_cols + ['index']], how='left', on='index')
dev0 = dev0.merge(temp_dev0[new_feature_cols + ['index']], how='left', on='index')
dev1 = dev1.merge(temp_dev1[new_feature_cols + ['index']], how='left', on='index')
test = test.merge(temp_test[new_feature_cols + ['index']], how='left', on='index')

print("读取merge")
merge_feature_data = pd.read_csv(path + 'features/merge_feature.txt', sep='\t')
merge_feature_cols = ['index', 'new_memberID_day_count', 'new_memberID_day_hour_count', 'new_questionID_day_count', 'new_questionID_day_hour_count', 'new_qid_to_first_days', 'new_qid_duration_days', 'new_qid_min_day_count', 'new_qid_rank', 'new_uid_duration_days', 'new_uid_to_first_days', 'new_uid_rank', 'new_qid_after_invite_diff', 'new_qid_last_invite_diff', 'new_uid_after_invite_diff', 'new_uid_last_invite_diff']
train0 = train0.merge(merge_feature_data[merge_feature_cols], how='left', on='index')
train1 = train1.merge(merge_feature_data[merge_feature_cols], how='left', on='index')
dev0 = dev0.merge(merge_feature_data[merge_feature_cols], how='left', on='index')
dev1 = dev1.merge(merge_feature_data[merge_feature_cols], how='left', on='index')
test = test.merge(merge_feature_data[merge_feature_cols], how='left', on='index')

# print("读取嫁接学习预测的概率")
# graft_lgb = pd.read_csv(path + 'features/graft_lgb.txt', sep='\t')
# graft_lgb['graft_lgb_predict'] = graft_lgb['result']
# train0 = train0.merge(graft_lgb[['graft_lgb_predict', 'index']], how='left', on='index')
# train1 = train1.merge(graft_lgb[['graft_lgb_predict', 'index']], how='left', on='index')
# dev0 = dev0.merge(graft_lgb[['graft_lgb_predict', 'index']], how='left', on='index')
# dev1 = dev1.merge(graft_lgb[['graft_lgb_predict', 'index']], how='left', on='index')
# test = test.merge(graft_lgb[['graft_lgb_predict', 'index']], how='left', on='index')

all_feature = pd.read_csv(open(path + 'invite_info_0926.txt', "r", encoding='utf-8'), sep='\t', header=None, names=['questionID', 'memberID', 'time', 'label'])
all_feature['day'] = all_feature['time'].apply(lambda x: int(x.split('-')[0][1:]))
all_feature['hour'] = all_feature['time'].apply(lambda x: int(x.split('-')[1][1:]))
all_feature['invite_time'] = all_feature['day'] * 100 + all_feature['hour']
all_feature.sort_values('invite_time', inplace=True)

print("train0的大小：", train0.shape[0])
print("train1的大小：", train1.shape[0])
print("dev0的大小：", dev0.shape[0])
print("dev1的大小：", dev1.shape[0])
print("test的大小：", test.shape[0])

print("读取用户数据")
member_path = path + 'member_info_0926.txt'
member_info = pd.read_csv(open(member_path, "r", encoding='utf-8'), sep='\t', header=None,
                          names=['memberID', 'm_sex', 'm_keywords', 'm_amount_grade', 'm_hot_grade', 'm_registry_type',
                                 'm_registry_platform', 'm_access_frequencies', 'm_twoA', 'm_twoB', 'm_twoC', 'm_twoD',
                                 'm_twoE', 'm_categoryA', 'm_categoryB', 'm_categoryC', 'm_categoryD', 'm_categoryE',
                                 'm_salt_score', 'm_attention_topics', 'm_interested_topics'])
del member_info['m_keywords']
del member_info['m_amount_grade']
del member_info['m_hot_grade']
del member_info['m_registry_type']
del member_info['m_registry_platform']
gc.collect()
print("读取问题数据")
question_path = path + 'question_info_0926.txt'
question_info_data = pd.read_csv(open(question_path, "r", encoding='utf-8'), sep='\t', header=None,
                                 names=['questionID', 'q_createTime', 'q_title_chars', 'q_title_words', 'q_desc_chars',
                                 'q_desc_words', 'q_topic_IDs'])
del question_info_data['q_createTime']
del question_info_data['q_title_chars']
del question_info_data['q_title_words']
del question_info_data['q_desc_chars']
gc.collect()

'''
print("用户点击序列特征构造")
def user_click_seq(memberIDs, features):
    member_seq_dict = {}
    for memberID, label in features[['memberID', 'label']].values:
        if memberID not in member_seq_dict.keys():
            member_seq_dict[memberID] = []
        member_seq_dict[memberID].append(str(int(label)))
    member_seq = []
    member_seq_3 = []
    for memberID in memberIDs:
        if memberID not in member_seq_dict.keys():
            member_seq.append("__UNKNOWN__")
            member_seq_3.append("__UNKNOWN__")
        else:
            member_seq.append(",".join(member_seq_dict[memberID]))
            member_seq_3.append(",".join(member_seq_dict[memberID][-3:]))
    return member_seq, member_seq_3
train0['uid_seq_feature'], train0['uid_seq_feature_3'] = user_click_seq(train0['memberID'].values, all_feature[(all_feature['day'] >= 3840) & (all_feature['day'] <= 3857)])
train1['uid_seq_feature'], train1['uid_seq_feature_3'] = user_click_seq(train1['memberID'].values, all_feature[(all_feature['day'] >= 3840) & (all_feature['day'] <= 3857)])
dev0['uid_seq_feature'], dev0['uid_seq_feature_3'] = user_click_seq(dev0['memberID'].values, all_feature[(all_feature['day'] >= 3846) & (all_feature['day'] <= 3863)])
dev1['uid_seq_feature'], dev1['uid_seq_feature_3'] = user_click_seq(dev1['memberID'].values, all_feature[(all_feature['day'] >= 3846) & (all_feature['day'] <= 3863)])
test['uid_seq_feature'], test['uid_seq_feature_3'] = user_click_seq(test['memberID'].values, all_feature[(all_feature['day'] >= 3850) & (all_feature['day'] <= 3867)])
'''

data = pd.concat([train0, train1, dev0, dev1, test], ignore_index=True, sort=False)
data = data.merge(member_info, how='left', on='memberID')
data = data.merge(question_info_data, how='left', on='questionID')
del member_info
del question_info_data
gc.collect()
gc.collect()

"""定义特征"""
class_feat = ['m_sex', 'm_access_frequencies', 'm_twoA', 'm_twoB', 'm_twoC', 'm_twoD', 'm_twoE', 'm_categoryA',
              'm_categoryB', 'm_categoryC', 'm_categoryD', 'm_categoryE',
              'm_num_interest_topic', 'num_topic_attention_intersection',
              'q_num_topic_words', 'num_topic_interest_intersection', 'hour',
              'new_uid_seq_feature', 'new_uid_seq_feature_3', 'new_memberID_day_count',
              'new_memberID_day_hour_count', 'new_qid_duration_days', 'new_qid_to_first_days', 'uid_ans_dif', 'uid_ans_day_nunique', 'new_uid_last_invite_diff',
              'new_uid_after_invite_diff', 'new_qid_after_invite_diff', 'new_qid_last_invite_diff', 'new_ans_dif_1','new_uid_duration_days', 'new_uid_to_first_days'
            ]

fixlen_number_columns = ['m_salt_score', 'm_num_atten_topic', 'q_num_title_chars_words', 'q_num_desc_chars_words', 'q_num_desc_words',
                         'q_num_title_words', 'days_to_invite', 'new_memberID_times', 'new_memberID_pos_times', 'new_memberID_neg_times', 'member_answer_times',
                         'member_like_times', 'new_questionID_day_count', 'new_questionID_day_hour_count', 'member_answer_topic_atten', 'member_answer_topic_interest',
                         'member_topic_answer_times', 'new_qid_min_day_count', 'new_qid_rank', 'topic_day_qid_nunique_max', 'topic_day_hour_qid_nunique_max',
                         'topic_memberID_rate_max', 'topic_m_sex_rate_max', 'topic_m_twoA_rate_max', 'topic_m_twoB_rate_max', 'topic_m_twoC_rate_max',
                         'topic_m_twoD_rate_max', 'topic_m_twoE_rate_max', 'topic_m_categoryA_rate_max', 'topic_m_categoryB_rate_max', 'topic_m_categoryC_rate_max',
                         'topic_m_categoryD_rate_max', 'topic_m_categoryE_rate_max', 'topic_m_access_frequencies_rate_max',
                         'new_member_continuous_reject_times', 'new_member_continuous_accept_times', 'uid_ans_topic_count', 'new_uid_rank'
                        ]

fixlen_number_columns.extend(title_word_user_rate_cols)
fixlen_number_columns.extend(desc_word_user_rate_cols)
# fixlen_number_columns.extend(topic3_user_rate_cols)
# fixlen_number_columns.extend(title_char_user_rate_cols)
fixlen_number_columns.extend(topic_user_ratio_cols)
fixlen_number_columns.extend(title_word_user_ratio_cols)
fixlen_number_columns.extend(topic_m_salt_score_cols)

all_features = class_feat + fixlen_number_columns

print("特征维数：", len(all_features))

encoder = LabelEncoder()
for feat in class_feat:
    encoder.fit(data[feat])
    data[feat] = encoder.transform(data[feat])

del train0
del train1
del dev0
del dev1
gc.collect()
gc.collect()

train = data[~data['label'].isnull()]
y_train = train['label'].values
X_train = train[all_features].values
y_valid = train[train['day'] >= 3865]['label'].values
X_valid = train[train['day'] >= 3865][all_features].values
X_test = data[data['label'].isnull()][all_features].values
pred_train = train[train['day'] <= 3864][['questionID', 'memberID', 'time', 'index']]
pred_dev = train[train['day'] >= 3865][['questionID', 'memberID', 'time', 'index']]
print("训练集大小：", len(X_train))
print("验证集大小：", len(X_valid))
print("测试集大小：", len(X_test))
del data
gc.collect()

model_lgb = LGBMClassifier(boosting_type='gbdt', num_leaves=120, learning_rate=0.01, n_estimators=2000,
                           max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                           min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,
                           colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, n_jobs=16, silent=True)

model_lgb.fit(X_train, y_train,
              eval_names=['train'],
              eval_metric=['logloss', 'auc'],
              eval_set=[(X_train, y_train)])

features_score = pd.DataFrame({
        'column':all_features ,
        'importance': model_lgb.feature_importances_,
}).sort_values(by='importance')

# features_score.to_csv("/home/lchen/zhihu/lgb/features.txt", sep="\t", index=False)

# print("训练集开始预测")
# y_pred = model_lgb.predict_proba(X_train)[:, 1]
# pred_train['result'] = y_pred
# pred_train.to_csv('/cu04_nfs/lchen/data/data_set_0926/features/pred_train.txt', index=False, sep='\t')
#
# print("验证集开始预测")
# y_pred = model_lgb.predict_proba(X_valid)[:, 1]
# pred_dev['result'] = y_pred
# pred_dev.to_csv('/cu04_nfs/lchen/data/data_set_0926/features/pred_dev.txt', index=False, sep='\t')

print("测试集开始预测")
y_pred = model_lgb.predict_proba(X_test)[:, 1]
test['result'] = y_pred
test[['questionID', 'memberID', 'time', 'result']].to_csv(result_path, index=False, header=False, sep='\t')
print("结果保存成功！")
