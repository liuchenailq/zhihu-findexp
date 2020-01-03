# @author     : srtianxia
# @time       : 2019/12/3 16:36
# @description:

import numpy as np
import pandas as pd
import scipy.special as special


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


def merge_train(kfold_4_df, kfold_1_df, feature_name, is_fill_na):  # 用其他训练集的四折的数据去merge一折的
    temp = kfold_4_df.groupby(feature_name, as_index=False)['label'].agg(
        {feature_name + '_click_count': 'sum', feature_name + '_all_count': 'count'})
    HP = HyperParam(1, 1)
    HP.update_from_data_by_moment(temp[feature_name + '_all_count'].values,
                                  temp[feature_name + '_click_count'].values)  # 矩估计
    temp[feature_name + '_convert'] = (temp[feature_name + '_click_count'] + HP.alpha) / (
            temp[feature_name + '_all_count'] + HP.alpha + HP.beta)
    temp = temp[[feature_name, feature_name + '_convert', feature_name + '_click_count']].drop_duplicates()
    kfold_1_df = pd.merge(kfold_1_df, temp, on=[feature_name], how='left')
    if is_fill_na:
        kfold_1_df[feature_name + '_convert'].fillna(HP.alpha / (HP.alpha + HP.beta), inplace=True)
    return kfold_1_df
