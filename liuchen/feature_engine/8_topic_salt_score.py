"""
在回答记录中统计回答该话题的用户盐值分数的平均分、最低分、最高分
用户的盐值分数 - 平均分、最低分、最高分
"""

import pandas as pd
import gc
import numpy as np
import scipy.special as special

job = 'test'  # train0、train1、dev0、dev1、test
path = '/cu04_nfs/lchen/data/data_set_0926/'

target_start = None
target_end = None
feature_start = None
feature_end = None
answer_start = None
answer_end = None
target_data = None
answer_data = None

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
    answer_data = temp_answer[(temp_answer['day'] >= answer_start) & (temp_answer['day'] <= answer_end)]
if job == 'train1':
    target_start = 3858
    target_end = 3864
    feature_start = 3840
    feature_end = 3857
    answer_start = 3827
    answer_end = 3857
    target_data = pd.read_csv(path + 'features/train1.txt', sep='\t')
    answer_data = temp_answer[(temp_answer['day'] >= answer_start) & (temp_answer['day'] <= answer_end)]
if job == 'dev0':
    target_start = 3865
    target_end = 3867
    feature_start = 3846
    feature_end = 3863
    answer_start = 3833
    answer_end = 3863
    target_data = pd.read_csv(path + 'features/dev0.txt', sep='\t')
    answer_data = temp_answer[(temp_answer['day'] >= answer_start) & (temp_answer['day'] <= answer_end)]
if job == 'dev1':
    target_start = 3865
    target_end = 3867
    feature_start = 3846
    feature_end = 3863
    answer_start = 3833
    answer_end = 3863
    target_data = pd.read_csv(path + 'features/dev1.txt', sep='\t')
    answer_data = temp_answer[(temp_answer['day'] >= answer_start) & (temp_answer['day'] <= answer_end)]
if job == 'test':
    target_start = 3868
    target_end = 3874
    feature_start = 3850
    feature_end = 3867
    answer_start = 3837
    answer_end = 3867
    target_data = pd.read_csv(path + 'features/test.txt', sep='\t')
    answer_data = temp_answer[(temp_answer['day'] >= answer_start) & (temp_answer['day'] <= answer_end)]

del temp_answer
gc.collect()
gc.collect()

print(job, "target time windows is {}-{}, answer time window is {}-{}".format(target_data['day'].min(), target_data['day'].max(), answer_data['day'].min(), answer_data['day'].max()))
print("shape: ", target_data.shape)

save_cols = ['questionID', 'memberID', 'time', 'index']
for col in target_data.columns:
    if col not in save_cols:
        del target_data[col]
gc.collect()

for col in answer_data.columns:
    if col not in ['questionID', 'memberID']:
        del answer_data[col]
gc.collect()

print("读取问题信息")
question_data = pd.read_csv(open(path + 'question_info_0926.txt', "r", encoding='utf-8'), sep='\t', header=None,
                               names=['questionID', 'q_createTime', 'q_title_chars', 'q_title_words', 'q_desc_chars',
                               'q_desc_words', 'q_topic_IDs'])
for col in question_data.columns:
    if col not in ['questionID', 'q_topic_IDs']:
        del question_data[col]
print("读取用户信息")
member_data = pd.read_csv(open(path + "member_info_0926.txt", "r", encoding='utf-8'), sep='\t', header=None,
                          names=['memberID', 'm_sex', 'm_keywords', 'm_amount_grade', 'm_hot_grade', 'm_registry_type',
                                  'm_registry_platform', 'm_access_frequencies', 'm_twoA', 'm_twoB', 'm_twoC', 'm_twoD',
                                  'm_twoE', 'm_categoryA', 'm_categoryB', 'm_categoryC', 'm_categoryD', 'm_categoryE',
                                  'm_salt_score', 'm_attention_topics', 'm_interested_topics'])

target_data = target_data.merge(question_data[['questionID', 'q_topic_IDs']], how='left', on='questionID')
answer_data = answer_data.merge(question_data[['questionID', 'q_topic_IDs']], how='left', on='questionID')
answer_data = answer_data.merge(member_data[['memberID', 'm_salt_score']], how='left', on='memberID')
target_data = target_data.merge(member_data[['memberID', 'm_salt_score']], how='left', on='memberID')

del question_data
del member_data
gc.collect()
gc.collect()

total_extend = answer_data['q_topic_IDs'].str.split(',', expand=True).stack() \
        .reset_index(level=0).set_index('level_0') \
        .rename(columns={0: 'topic'}).join(answer_data.drop('q_topic_IDs', axis=1)) \
        .reset_index(drop=True)

topic_df = target_data['q_topic_IDs'].str.split(',', expand=True)
topic_df = topic_df.fillna(0)
target_data = pd.concat([target_data, topic_df], axis=1)

t1 = total_extend.groupby(['topic'])['m_salt_score'].agg({"topic_m_salt_score_min": "min", "topic_m_salt_score_max": "max", "topic_m_salt_score_mean": "mean"})

target_data = pd.merge(target_data, t1, how='left', left_on=0, right_on='topic')

save_cols.append("topic_m_salt_score_min")
save_cols.append("topic_m_salt_score_max")
save_cols.append("topic_m_salt_score_mean")

print("缺失值 中位数填充")
for feat in ['topic_m_salt_score_min', 'topic_m_salt_score_max', 'topic_m_salt_score_mean']:
    target_data[feat].fillna(target_data[feat].median(), inplace=True)

target_data['diff_topic_m_salt_score_min'] = target_data['m_salt_score'] - target_data['topic_m_salt_score_min']
target_data['diff_topic_m_salt_score_max'] = target_data['m_salt_score'] - target_data['topic_m_salt_score_max']
target_data['diff_topic_m_salt_score_mean'] = target_data['m_salt_score'] - target_data['topic_m_salt_score_mean']

save_cols.append("diff_topic_m_salt_score_min")
save_cols.append("diff_topic_m_salt_score_max")
save_cols.append("diff_topic_m_salt_score_mean")

print(save_cols)
target_data = target_data[save_cols]
print("shape: ", target_data.shape)

for col in target_data.columns:
    print(target_data[[col]].describe())

target_data.to_csv(path + "features/topic_salt_score_" + job + ".txt", sep='\t', index=False)
print("finish!")
