"""
继续构造特征

topic_day_qid_nunique_max(话题当天出现了多少种qid (对于问题绑定的topic_id 求max) )
topic_day_qid_nunique_sum(话题当天出现了多少种qid (对于问题绑定的topic_id 求sum) )

topic_day_hour_qid_nunique_max  当天当小时话题下有多少个qid (问题绑定qid求max)
topic_day_hour_qid_nunique_sum  当天当小时话题下有多少个qid (问题绑定qid求sum)


"""

import pandas as pd
import gc
import numpy as np

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
if job == 'train1':
    target_start = 3858
    target_end = 3864
    feature_start = 3840
    feature_end = 3857
    answer_start = 3827
    answer_end = 3857
    target_data = pd.read_csv(path + 'features/train1.txt', sep='\t')
    feature_data = temp_feature[(temp_feature['day'] >= feature_start) & (temp_feature['day'] <= feature_end)]
if job == 'dev0':
    target_start = 3865
    target_end = 3867
    feature_start = 3846
    feature_end = 3863
    answer_start = 3833
    answer_end = 3863
    target_data = pd.read_csv(path + 'features/dev0.txt', sep='\t')
    feature_data = temp_feature[(temp_feature['day'] >= feature_start) & (temp_feature['day'] <= feature_end)]
if job == 'dev1':
    target_start = 3865
    target_end = 3867
    feature_start = 3846
    feature_end = 3863
    answer_start = 3833
    answer_end = 3863
    target_data = pd.read_csv(path + 'features/dev1.txt', sep='\t')
    feature_data = temp_feature[(temp_feature['day'] >= feature_start) & (temp_feature['day'] <= feature_end)]
if job == 'test':
    target_start = 3868
    target_end = 3874
    feature_start = 3850
    feature_end = 3867
    answer_start = 3837
    answer_end = 3867
    target_data = pd.read_csv(path + 'features/test.txt', sep='\t')
    feature_data = temp_feature[(temp_feature['day'] >= feature_start) & (temp_feature['day'] <= feature_end)]

del temp_feature

gc.collect()
gc.collect()

print(job, "target time windows is {}-{}, feature time window is {}-{}".format(target_data['day'].min(), target_data['day'].max(), feature_data['day'].min(), feature_data['day'].max()))
print("shape: ", target_data.shape)
save_cols = list(target_data.columns)

print("读取用户信息")
member_data = pd.read_csv(open(path + "member_info_0926.txt", "r", encoding='utf-8'), sep='\t', header=None,
                          names=['memberID', 'm_sex', 'm_keywords', 'm_amount_grade', 'm_hot_grade', 'm_registry_type',
                                  'm_registry_platform', 'm_access_frequencies', 'm_twoA', 'm_twoB', 'm_twoC', 'm_twoD',
                                  'm_twoE', 'm_categoryA', 'm_categoryB', 'm_categoryC', 'm_categoryD', 'm_categoryE',
                                  'm_salt_score', 'm_attention_topics', 'm_interested_topics'])
print("读取问题信息")
question_data = pd.read_csv(open(path + 'question_info_0926.txt', "r", encoding='utf-8'), sep='\t', header=None,
                               names=['questionID', 'q_createTime', 'q_title_chars', 'q_title_words', 'q_desc_chars',
                               'q_desc_words', 'q_topic_IDs'])

target_data = target_data.merge(question_data[['questionID', 'q_topic_IDs']], how='left', on='questionID')

print("话题当天出现了多少种qid (对于问题绑定的topic_id 求max) ")
topic_day_qids = {}  # key: topic_day  value: qids
for question, day, topics in target_data[['questionID', 'day', 'q_topic_IDs']].values:
    if topics != "-1":
        for topic in topics.split(","):
            key = topic + "_" + str(day)
            if key not in topic_day_qids.keys():
                topic_day_qids[key] = set()
            topic_day_qids[key].add(question)
topic_day_qid_nunique_dict = {}  # key: topic_day  value: len(qids)
for key, value in topic_day_qids.items():
    topic_day_qid_nunique_dict[key] = len(value)
def get_topic_day_qid_nunique_max(x):
    topics = x['q_topic_IDs']
    day = x['day']
    if topics == "-1":
        return 0
    t = []
    for topic in topics.split(","):
        t.append(topic_day_qid_nunique_dict[topic + "_" + str(day)])
    return np.max(t)
target_data['topic_day_qid_nunique_max'] = target_data.apply(lambda x: get_topic_day_qid_nunique_max(x), axis=1)
def get_topic_day_qid_nunique_sum(x):
    topics = x['q_topic_IDs']
    day = x['day']
    if topics == "-1":
        return 0
    t = []
    for topic in topics.split(","):
        t.append(topic_day_qid_nunique_dict[topic + "_" + str(day)])
    return np.sum(t)
target_data['topic_day_qid_nunique_sum'] = target_data.apply(lambda x: get_topic_day_qid_nunique_sum(x), axis=1)

target_data['invite_time'] = target_data['day'] * 100 + target_data['hour']
print("当天当小时话题下有多少个qid")
topic_day_qids = {}  # key: topic_invite_time  value: qids
for question, day, topics in target_data[['questionID', 'invite_time', 'q_topic_IDs']].values:
    if topics != "-1":
        for topic in topics.split(","):
            key = topic + "_" + str(day)
            if key not in topic_day_qids.keys():
                topic_day_qids[key] = set()
            topic_day_qids[key].add(question)
topic_day_qid_nunique_dict = {}  # key: topic_day  value: len(qids)
for key, value in topic_day_qids.items():
    topic_day_qid_nunique_dict[key] = len(value)
def get_topic_day_hour_qid_nunique_max(x):
    topics = x['q_topic_IDs']
    day = x['invite_time']
    if topics == "-1":
        return 0
    t = []
    for topic in topics.split(","):
        t.append(topic_day_qid_nunique_dict[topic + "_" + str(day)])
    return np.max(t)
target_data['topic_day_hour_qid_nunique_max'] = target_data.apply(lambda x: get_topic_day_hour_qid_nunique_max(x), axis=1)
def get_topic_day_hour_qid_nunique_sum(x):
    topics = x['q_topic_IDs']
    day = x['invite_time']
    if topics == "-1":
        return 0
    t = []
    for topic in topics.split(","):
        t.append(topic_day_qid_nunique_dict[topic + "_" + str(day)])
    return np.sum(t)
target_data['topic_day_hour_qid_nunique_sum'] = target_data.apply(lambda x: get_topic_day_hour_qid_nunique_sum(x), axis=1)
del topic_day_qid_nunique_dict
del topic_day_qids
gc.collect()
save_cols.append("topic_day_qid_nunique_max")
save_cols.append("topic_day_qid_nunique_sum")
save_cols.append("topic_day_hour_qid_nunique_max")
save_cols.append("topic_day_hour_qid_nunique_sum")


# print("话题和用户属性的转化率")
# feature_data = feature_data.merge(question_data[['questionID', 'q_topic_IDs']], how='left', on='questionID')
# feature_data = feature_data.merge(member_data, how='left', on='memberID')
# target_data = target_data.merge(member_data, how='left', on='memberID')
#
# # 将q_topic_IDs拆开 一条记录变为多条
# total_extend = feature_data['q_topic_IDs'].str.split(',', expand=True).stack() \
#         .reset_index(level=0).set_index('level_0') \
#         .rename(columns={0: 'topic'}).join(feature_data.drop('q_topic_IDs', axis=1)) \
#         .reset_index(drop=True)
#
# # 将target_data的q_topic_IDs展开
# topic_df = target_data['q_topic_IDs'].str.split(',', expand=True)
# topic_df = topic_df.fillna(0)
# target_data = pd.concat([target_data, topic_df], axis=1)
#
# fea_list = ['memberID', 'm_sex', 'm_twoA', 'm_twoB', 'm_twoC', 'm_twoD', 'm_twoE', 'm_categoryA', 'm_categoryB', 'm_categoryC', 'm_categoryD', 'm_categoryE', 'm_access_frequencies']
# for fea in fea_list:
#     fea_name = 'topic_' + fea + '_rate'
#     print(fea_name)
#     # 计算转化率
#     t = total_extend.groupby(['topic', fea])['label'].agg(['mean']).reset_index() \
#         .rename(columns={'mean': fea_name})
#
#     tmp_name = []
#     for field in [0, 1, 2, 3, 4]:
#         target_data = pd.merge(target_data, t, how='left', left_on=[fea, field], right_on=[fea, 'topic']).rename(
#             columns={fea_name: fea_name + str(field)})
#         if 'topic' in target_data.columns:
#             del target_data['topic']
#         tmp_name.append(fea_name + str(field))
#
#     target_data[fea_name + '_max'] = target_data[tmp_name].max(axis=1)
#     save_cols.append(fea_name + '_max')
#
#     for field in [0, 1, 2, 3, 4]:
#         target_data = target_data.drop([fea_name + str(field)], axis=1)

target_data = target_data[save_cols]
print("shape: ", target_data.shape)

for col in target_data.columns:
    print(target_data[[col]].describe())

target_data.to_csv(path + "features/new_" + job + ".txt", sep='\t', index=False)
print("finish!")
