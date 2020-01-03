"""
话题和用户属性的转化率
利用贝叶斯平滑
"""
import pandas as pd
import gc
import numpy as np
import scipy.special as special

job = 'test'  # train0、train1、dev0、dev1、test
path = '/cu04_nfs/lchen/data/data_set_0926/'

def log(log: str):
    print(log)


def time_log(time_elapsed):
    print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))  # 打印出来时间


def log_event(event: str):
    log(event)


class HyperParam(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample_from_beta(self, alpha, beta, num, imp_upperbound):
        sample = np.random.beta(alpha, beta, num)
        I = []
        C = []
        for click_ratio in sample:
            imp = np.random.random() * imp_upperbound
            # imp = imp_upperbound
            click = imp * click_ratio
            I.append(imp)
            C.append(click)
        return I, C

    def update_from_data_by_FPI(self, tries, success, iter_num, epsilon):
        '''estimate alpha, beta using fixed point iteration'''
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(tries, success, self.alpha, self.beta)
            if abs(new_alpha - self.alpha) < epsilon and abs(new_beta - self.beta) < epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, tries, success, alpha, beta):
        '''fixed point iteration'''
        sumfenzialpha = 0.0
        sumfenzibeta = 0.0
        sumfenmu = 0.0
        for i in range(len(tries)):
            sumfenzialpha += (special.digamma(success[i] + alpha) - special.digamma(alpha))
            sumfenzibeta += (special.digamma(tries[i] - success[i] + beta) - special.digamma(beta))
            sumfenmu += (special.digamma(tries[i] + alpha + beta) - special.digamma(alpha + beta))

        return alpha * (sumfenzialpha / sumfenmu), beta * (sumfenzibeta / sumfenmu)

    def update_from_data_by_moment(self, tries, success):  # tries尝试了多少次ctr  success 命中了多少次ctr
        '''estimate alpha, beta using moment estimation'''
        mean, var = self.__compute_moment(tries, success)
        # print 'mean and variance: ', mean, var
        # self.alpha = mean*(mean*(1-mean)/(var+0.000001)-1)
        self.alpha = (mean + 0.000001) * ((mean + 0.000001) * (1.000001 - mean) / (var + 0.000001) - 1)
        # self.beta = (1-mean)*(mean*(1-mean)/(var+0.000001)-1)
        self.beta = (1.000001 - mean) * ((mean + 0.000001) * (1.000001 - mean) / (var + 0.000001) - 1)

    def __compute_moment(self, tries, success):
        '''
        moment estimation
        '''
        ctr_list = []
        var = 0.0
        for i in range(len(tries)):
            ctr_list.append(float(success[i]) / (tries[i] + 0.000000001))
        mean = sum(ctr_list) / len(ctr_list)
        for ctr in ctr_list:
            var += pow(ctr - mean, 2)

        return mean, var / (len(ctr_list) - 1)


def merge_test(train_df, test_df, feature_name, is_fill_na=True):
    temp = train_df.groupby(feature_name, as_index=False)['label'].agg(
        {feature_name + '_click_count': 'sum', feature_name + '_all_count': 'count'})
    HP = HyperParam(1, 1)
    HP.update_from_data_by_moment(temp[feature_name + '_all_count'].values,
                                  temp[feature_name + '_click_count'].values)  # 矩估计
    temp[feature_name + '_convert'] = (temp[feature_name + '_click_count'] + HP.alpha) / (
            temp[feature_name + '_all_count'] + HP.alpha + HP.beta)
    temp = temp[[feature_name, feature_name + '_convert', feature_name + '_click_count']].drop_duplicates()
    test_df = pd.merge(test_df, temp, on=[feature_name], how='left')
    print(test_df.columns.tolist())
    if is_fill_na:
        test_df[feature_name + '_convert'].fillna(HP.alpha / (HP.alpha + HP.beta), inplace=True)
    return test_df


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

save_cols = ['questionID', 'memberID', 'time', 'index']
for col in target_data.columns:
    if col not in save_cols:
        del target_data[col]

gc.collect()

print("读取问题信息")
question_data = pd.read_csv(open(path + 'question_info_0926.txt', "r", encoding='utf-8'), sep='\t', header=None,
                               names=['questionID', 'q_createTime', 'q_title_chars', 'q_title_words', 'q_desc_chars',
                               'q_desc_words', 'q_topic_IDs'])
print("读取用户信息")
member_data = pd.read_csv(open(path + "member_info_0926.txt", "r", encoding='utf-8'), sep='\t', header=None,
                          names=['memberID', 'm_sex', 'm_keywords', 'm_amount_grade', 'm_hot_grade', 'm_registry_type',
                                  'm_registry_platform', 'm_access_frequencies', 'm_twoA', 'm_twoB', 'm_twoC', 'm_twoD',
                                  'm_twoE', 'm_categoryA', 'm_categoryB', 'm_categoryC', 'm_categoryD', 'm_categoryE',
                                  'm_salt_score', 'm_attention_topics', 'm_interested_topics'])
del member_data['m_keywords']
del member_data['m_amount_grade']
del member_data['m_hot_grade']
del member_data['m_registry_type']
del member_data['m_registry_platform']

target_data = target_data.merge(question_data[['questionID', 'q_topic_IDs']], how='left', on='questionID')
feature_data = feature_data.merge(question_data[['questionID', 'q_topic_IDs']], how='left', on='questionID')
feature_data = feature_data.merge(member_data, how='left', on='memberID')
target_data = target_data.merge(member_data, how='left', on='memberID')

del target_data['m_attention_topics']
del target_data['m_interested_topics']
del feature_data['m_attention_topics']
del feature_data['m_interested_topics']
del question_data
del member_data
gc.collect()
gc.collect()

# 将q_topic_IDs拆开 一条记录变为多条
total_extend = feature_data['q_topic_IDs'].str.split(',', expand=True).stack() \
        .reset_index(level=0).set_index('level_0') \
        .rename(columns={0: 'topic'}).join(feature_data.drop('q_topic_IDs', axis=1)) \
        .reset_index(drop=True)

topic_df = target_data['q_topic_IDs'].str.split(',', expand=True)
topic_df = topic_df.fillna(0)
target_data = pd.concat([target_data, topic_df], axis=1)
del topic_df
del feature_data
gc.collect()

target_data['join'] = '_'
total_extend['join'] = '_'
fea_list = ['memberID', 'm_sex', 'm_twoA', 'm_twoB', 'm_twoC', 'm_twoD', 'm_twoE', 'm_categoryA', 'm_categoryB', 'm_categoryC', 'm_categoryD', 'm_categoryE', 'm_access_frequencies']
for fea in fea_list:
    feature_name = "topic_" + fea
    print(feature_name)
    total_extend[feature_name] = total_extend["topic"].map(str) + total_extend['join'].map(str) + total_extend[fea].map(str)
    temp = total_extend.groupby(feature_name, as_index=False)['label'].agg(
        {feature_name + '_click_count': 'sum', feature_name + '_all_count': 'count'})
    HP = HyperParam(1, 1)
    HP.update_from_data_by_moment(temp[feature_name + '_all_count'].values,
                                  temp[feature_name + '_click_count'].values)  # 矩估计
    temp[feature_name + '_rate'] = (temp[feature_name + '_click_count'] + HP.alpha) / (
                temp[feature_name + '_all_count'] + HP.alpha + HP.beta)
    temp = temp[[feature_name, feature_name + '_rate']].drop_duplicates()
    tmp_name = []
    for field in [0, 1, 2, 3, 4]:
        target_data[feature_name] = target_data[field].map(str) + target_data['join'].map(str) + target_data[fea].map(str)
        target_data = pd.merge(target_data, temp, how='left', on=feature_name).rename(columns={feature_name + '_rate': feature_name + '_rate' + str(field)})
        tmp_name.append(feature_name + '_rate' + str(field))
    target_data[feature_name + '_rate_max'] = target_data[tmp_name].max(axis=1)
    save_cols.append(feature_name + '_rate_max')
    for field in [0, 1, 2, 3, 4]:
        target_data = target_data.drop([feature_name + '_rate' + str(field)], axis=1)
    del total_extend[feature_name]
    gc.collect()

print(save_cols)
target_data = target_data[save_cols]
print("shape: ", target_data.shape)

for col in target_data.columns:
    print(target_data[[col]].describe())

target_data.to_csv(path + "features/topic_user_rate_" + job + ".txt", sep='\t', index=False)
print("finish!")
