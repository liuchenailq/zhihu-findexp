"""
特征构造
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
temp_target = pd.read_csv(path + "features/train_dev.txt", sep='\t')
temp_target['index'] = list(range(1, temp_target.shape[0] + 1))
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
    target_data = temp_target[(temp_target['day'] >= target_start) & (temp_target['day'] <= target_end) & (temp_target['flag'] == 0)]
    feature_data = temp_feature[(temp_feature['day'] >= feature_start) & (temp_feature['day'] <= feature_end)]
    answer_data = temp_answer[(temp_answer['day'] >= answer_start) & (temp_answer['day'] <= answer_end)]
if job == 'train1':
    target_start = 3858
    target_end = 3864
    feature_start = 3840
    feature_end = 3857
    answer_start = 3827
    answer_end = 3857
    target_data = temp_target[(temp_target['day'] >= target_start) & (temp_target['day'] <= target_end) & (temp_target['flag'] == 1)]
    feature_data = temp_feature[(temp_feature['day'] >= feature_start) & (temp_feature['day'] <= feature_end)]
    answer_data = temp_answer[(temp_answer['day'] >= answer_start) & (temp_answer['day'] <= answer_end)]
if job == 'dev0':
    target_start = 3865
    target_end = 3867
    feature_start = 3846
    feature_end = 3863
    answer_start = 3833
    answer_end = 3863
    target_data = temp_target[(temp_target['day'] >= target_start) & (temp_target['day'] <= target_end) & (temp_target['flag'] == 0)]
    feature_data = temp_feature[(temp_feature['day'] >= feature_start) & (temp_feature['day'] <= feature_end)]
    answer_data = temp_answer[(temp_answer['day'] >= answer_start) & (temp_answer['day'] <= answer_end)]
if job == 'dev1':
    target_start = 3865
    target_end = 3867
    feature_start = 3846
    feature_end = 3863
    answer_start = 3833
    answer_end = 3863
    target_data = temp_target[(temp_target['day'] >= target_start) & (temp_target['day'] <= target_end) & (temp_target['flag'] == 1)]
    feature_data = temp_feature[(temp_feature['day'] >= feature_start) & (temp_feature['day'] <= feature_end)]
    answer_data = temp_answer[(temp_answer['day'] >= answer_start) & (temp_answer['day'] <= answer_end)]
if job == 'test':
    target_start = 3868
    target_end = 3874
    feature_start = 3850
    feature_end = 3867
    answer_start = 3837
    answer_end = 3867
    target_data = pd.read_csv(open(path + 'invite_info_evaluate_2_0926.txt', "r", encoding='utf-8'), sep='\t', header=None, names=['questionID', 'memberID', 'time'])
    target_data['day'] = target_data['time'].apply(lambda x: int(x.split('-')[0][1:]))
    target_data['index'] = list(range(temp_target.shape[0] + 1, target_data.shape[0] + temp_target.shape[0] + 1))
    feature_data = temp_feature[(temp_feature['day'] >= feature_start) & (temp_feature['day'] <= feature_end)]
    answer_data = temp_answer[(temp_answer['day'] >= answer_start) & (temp_answer['day'] <= answer_end)]

target_data['hour'] = target_data['time'].apply(lambda x: int(x.split('-')[1][1:]))
target_data['invite_time'] = target_data['day'] * 100 + target_data['hour']

del temp_target
del temp_feature
del temp_answer
gc.collect()

"""特征构造从此开始"""
print(job, "target time windows is {}-{}, feature time window is {}-{}, answer time window is {}-{}".format(target_data['day'].min(), target_data['day'].max(), feature_data['day'].min(), feature_data['day'].max(), answer_data['day'].min(), answer_data['day'].max()))

print("读取用户信息")
member_data = pd.read_csv(open(path + "member_info_0926.txt", "r", encoding='utf-8'), sep='\t', header=None,
                          names=['memberID', 'm_sex', 'm_keywords', 'm_amount_grade', 'm_hot_grade', 'm_registry_type',
                                  'm_registry_platform', 'm_access_frequencies', 'm_twoA', 'm_twoB', 'm_twoC', 'm_twoD',
                                  'm_twoE', 'm_categoryA', 'm_categoryB', 'm_categoryC', 'm_categoryD', 'm_categoryE',
                                  'm_salt_score', 'm_attention_topics', 'm_interested_topics'])
print("用户关注的、感兴趣的话题数")
member_data['m_num_atten_topic'] = member_data['m_attention_topics'].apply(lambda x: len(str(x).split(',')) if str(x) != '-1' else 0)
member_data['m_num_interest_topic'] = member_data['m_interested_topics'].apply(lambda x: len(str(x).split(',')) if str(x) != '-1' else 0)
target_data = target_data.merge(member_data[['memberID', 'm_num_atten_topic', 'm_num_interest_topic', 'm_attention_topics', 'm_interested_topics']], how='left', on='memberID')
print("用户基本特征生成完毕")
print()
del member_data
gc.collect()


print("读取问题信息")
question_data = pd.read_csv(open(path + 'question_info_0926.txt', "r", encoding='utf-8'), sep='\t', header=None,
                               names=['questionID', 'q_createTime', 'q_title_chars', 'q_title_words', 'q_desc_chars',
                               'q_desc_words', 'q_topic_IDs'])
question_data['q_day'] = question_data['q_createTime'].apply(lambda x: int(x.split('-')[0][1:]))
print("问题标题、描述字词长度")
question_data['q_num_title_chars_words'] = question_data['q_title_chars'].apply(lambda x: len(str(x).split(',')) if str(x) != '-1' else 0)
question_data['q_num_desc_chars_words'] = question_data['q_desc_chars'].apply(lambda x: len(str(x).split(',')) if str(x) != '-1' else 0)
question_data['q_num_desc_words'] = question_data['q_desc_words'].apply(lambda x: len(str(x).split(',')) if str(x) != '-1' else 0)
question_data['q_num_title_words'] = question_data['q_title_words'].apply(lambda x: len(str(x).split(',')) if str(x) != '-1' else 0)
print("问题绑定的话题个数")
question_data['q_num_topic_words'] = question_data['q_topic_IDs'].apply(lambda x: len(str(x).split(',')) if str(x) != '-1' else 0)
target_data = target_data.merge(question_data[['questionID', 'q_day', 'q_num_title_chars_words', 'q_num_desc_chars_words', 'q_num_desc_words', 'q_num_title_words', 'q_num_topic_words', 'q_topic_IDs']], how='left', on='questionID')
print("问题基本特征生成完毕")
print()
answer_data = answer_data.merge(question_data[['questionID', 'q_topic_IDs']], how='left', on='questionID')
del question_data
gc.collect()

print("用户历史邀请次数、用户历史邀请负样本数、用户历史邀请正样本数")
member_label_count = feature_data.groupby(['memberID'], as_index=False)['label'].agg({'memberID_times': 'count', 'memberID_pos_times': 'sum'})
target_data = target_data.merge(member_label_count[['memberID', 'memberID_times', 'memberID_pos_times']], how='left', on='memberID')
target_data[['memberID_times', 'memberID_pos_times']] = target_data[['memberID_times', 'memberID_pos_times']].fillna(0)
target_data['memberID_neg_times'] = target_data['memberID_times'] - target_data['memberID_pos_times']
del member_label_count
gc.collect()

print("用户历史回答次数")
member_answer_count = answer_data.groupby(['memberID'], as_index=False)['questionID'].agg({'member_answer_times': 'count'})
target_data = target_data.merge(member_answer_count[['memberID', 'member_answer_times']], how='left', on='memberID')
target_data[['member_answer_times']] = target_data[['member_answer_times']].fillna(0)
del member_answer_count
gc.collect()

print("用户历史点赞数")
member_like_count = answer_data.groupby(['memberID'], as_index=False)['like_count'].agg({'member_like_times': 'sum'})
target_data = target_data.merge(member_like_count[['memberID', 'member_like_times']], how='left', on='memberID')
target_data[['member_like_times']] = target_data[['member_like_times']].fillna(0)
del member_like_count
gc.collect()

print("用户当天的邀请次数、用户当天当小时的邀请次数、问题当天的邀请次数、问题当天当小时的邀请次数")
for feat, time in [('memberID', 'day'), ('memberID', 'day_hour'), ('questionID', 'day'), ('questionID', 'day_hour')]:
    group1 = feat
    group2 = time
    if group2 == 'day_hour':
        group2 = 'invite_time'
    feat_name = "{}_{}_count".format(feat, time)
    temp = target_data.groupby([group1, group2], as_index=False)['index'].agg({feat_name: 'count'})
    target_data = target_data.merge(temp[[group1, group2, feat_name]], how='left', on=[group1, group2])

del temp
gc.collect()

print("用户关注的话题和问题绑定的话题交集个数 、用户感兴趣的话题和问题绑定的话题交集个数")
def topic_intersection(x, y, flag=0):
    """
    :param x: 问题绑定的话题
    :param y: 用户感兴趣的话题或者关注的话题  0：感兴趣 1：关注的
    :param flag:
    :return:
    """
    x = str(x)
    y = str(y)
    if x == '-1' or y == '-1':
        return 0
    q_topics = set()
    for t in x.split(","):
        q_topics.add(t)
    m_topics = set()
    for t in y.split(","):
        if flag == 1:
            m_topics.add(t)
        else:
            m_topics.add(t.split(":")[0])
    return len(q_topics.intersection(m_topics))
target_data['num_topic_interest_intersection'] = target_data.apply(lambda row: topic_intersection(row['q_topic_IDs'], row['m_interested_topics'], 0), axis=1)
target_data['num_topic_attention_intersection'] = target_data.apply(lambda row: topic_intersection(row['q_topic_IDs'], row['m_attention_topics'], 1), axis=1)

print("用户回答过的话题和用户关注的话题的交集、用户回答过的话题和用户感兴趣的话题的交集、用户回答过的话题和问题话题的交集")
member_topic_count = {} # key: member_topic value：回答次数
for memberID, topics in answer_data[['memberID', 'q_topic_IDs']].values:
    if topics != '-1':
        for topic in topics.split(","):
            key = memberID + "_" + topic
            member_topic_count[key] = member_topic_count.get(key, 0) + 1


def calc_intersection(memberID, topics, isinterested_topics=False):
    if topics == '-1':
        return 0
    t = 0
    for topic in topics.split(","):
        if isinterested_topics is True:
            key = memberID + "_" + topic.split(":")[0]
        else:
            key = memberID + "_" + topic
        t += member_topic_count.get(key, 0)
    return t
target_data['member_answer_topic_atten'] = target_data.apply(lambda x: calc_intersection(x['memberID'], x['m_attention_topics'], False), axis=1)
target_data['member_answer_topic_interest'] = target_data.apply(lambda x: calc_intersection(x['memberID'], x['m_interested_topics'], True), axis=1)
target_data['member_topic_answer_times'] = target_data.apply(lambda x: calc_intersection(x['memberID'], x['q_topic_IDs'], False), axis=1)

print("邀请发送距问题创建的天数")
target_data['days_to_invite'] = target_data['day'] - target_data['q_day']

del target_data['m_attention_topics']
del target_data['m_interested_topics']
del target_data['q_topic_IDs']

for col in target_data.columns:
    print(target_data[[col]].describe())

target_data.to_csv(path + "features/" + job + ".txt", sep='\t', index=False)
print("finish!")

