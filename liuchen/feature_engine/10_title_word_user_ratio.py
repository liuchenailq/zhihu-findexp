"""
用户属性在问题标题词下的比例
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
question_data = pd.read_csv(path + 'features/question_word_tfidf.txt', sep='\t')

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
del member_data['m_attention_topics']
del member_data['m_interested_topics']
gc.collect()

target_data = target_data.merge(question_data[['questionID', 'q_title_words']], how='left', on='questionID')
answer_data = answer_data.merge(question_data[['questionID', 'q_title_words']], how='left', on='questionID')
answer_data = answer_data.merge(member_data, how='left', on='memberID')
target_data = target_data.merge(member_data, how='left', on='memberID')

del member_data
del question_data
gc.collect()

# 将q_title_words拆开 一条记录变为多条
total_extend = answer_data['q_title_words'].str.split(' ', expand=True).stack() \
        .reset_index(level=0).set_index('level_0') \
        .rename(columns={0: 'title_word'}).join(answer_data.drop('q_title_words', axis=1)) \
        .reset_index(drop=True)

topic_df = target_data['q_title_words'].str.split(' ', expand=True)
topic_df = topic_df.fillna(0)
target_data = pd.concat([target_data, topic_df], axis=1)
del topic_df
del answer_data
gc.collect()

t1 = total_extend.groupby(['title_word'])['memberID'].agg(['count']).reset_index().rename(columns={'count': 'title_word_count'})

fea_list = ['memberID', 'm_sex', 'm_twoA', 'm_twoB', 'm_twoC', 'm_twoD', 'm_twoE', 'm_categoryA', 'm_categoryB', 'm_categoryC', 'm_categoryD', 'm_categoryE', 'm_access_frequencies']
for stat in fea_list:
    fea_name = stat + '_title_word_count_ratio'
    print('extract', fea_name)

    # 统计词和用户属性的交叉量
    t = total_extend.groupby(['title_word', stat])['questionID'].agg(['count']).reset_index().rename(
        columns={'count': 'sum_count'})
    t = pd.merge(t, t1, how='left', on='title_word')  # title_word stat sum_count title_word_count

    # 平滑求占比
    HP = HyperParam(1, 1)
    HP.update_from_data_by_moment(t['title_word_count'].values,
                                  t['sum_count'].values)  # 矩估计
    t[fea_name] = np.divide(t['sum_count'] + HP.alpha, t['title_word_count'] + HP.alpha + HP.beta)
    t = t.drop(['title_word_count', 'sum_count'], axis=1)  # title_word stat ratio

    stat = ['title_word', stat]
    tmp_name = []
    for field in [0, 1, 2, 3, 4]:
        lefton = []
        for i in stat:
            if i == 'title_word':
                lefton.append(field)
            else:
                lefton.append(i)
        target_data = pd.merge(target_data, t, how='left', left_on=lefton, right_on=stat).rename(
            columns={fea_name: fea_name + str(field)})
        if "title_word" in target_data.columns:
            del target_data['title_word']
        tmp_name.append(fea_name + str(field))

    target_data[fea_name + '_max'] = target_data[tmp_name].max(axis=1)
    target_data[fea_name + '_mean'] = target_data[tmp_name].mean(axis=1)
    save_cols.append(fea_name + '_max')
    save_cols.append(fea_name + '_mean')

    for field in [0, 1, 2, 3, 4]:
        target_data = target_data.drop([fea_name + str(field)], axis=1)


print(save_cols)
target_data = target_data[save_cols]
print("shape: ", target_data.shape)

for col in target_data.columns:
    print(target_data[[col]].describe())

target_data.to_csv(path + "features/title_word_user_ratio_" + job + ".txt", sep='\t', index=False)
print("finish!")


