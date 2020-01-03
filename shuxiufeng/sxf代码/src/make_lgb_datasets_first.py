import pandas as pd
import pickle
import numpy as np

from sklearn import preprocessing


class CateEnconder():
    def __init__(self, method='identiy',ratio = 1.0,bins=None, max=-1, threshold = 1000):
        self.method = method
        self.ratio = ratio
        self.bins = bins
    def transform(self,data,feat):
        if self.method == 'log2':
            data[feat] = np.round(np.log2(data[feat]) * self.ratio)
            return np.max(data[feat])
        if self.method == 'log1p':
            data[feat] = np.round(np.log1p(data[feat]) * self.ratio)
            return np.max(data[feat])
        if self.method == 'linear':
            data[feat] = data[feat].apply(lambda x: x if x <= 200 else 200)
            data[feat] = data[feat].apply(lambda x: x if x <= 100 else 100 + (x % 10) )
            return np.max(data[feat])
        if self.method == 'percent':
            data[feat] = np.digitize(data[feat], self.bins, right=True)
            return np.max(data[feat])
        else:
            return 0
    def getmethod(self):
        return self.method


class DigtEncoder():
    def __init__(self, method='linear', split_max = -1, split_min = -1):
        self.method = method

    def transform(self,data,feat):
        if self.method == 'log1p':
            data[feat] = np.log1p(data[feat])
        else:
            data[feat] = data[feat]
    def getmethod(self):
        return self.method
    def getlogtransform(self):
        return 0



def make_lgb_data(mode, isSave =True,hasFile = False):


    if hasFile == True:
        return pd.read_csv('../datasets/feature/' + mode + '/nn_data.csv')

    if mode == 'train':
        mode1 = mode + '_use'
        data = pd.read_csv('../datasets/invite_' + mode1 + '.csv')
    elif mode == 'test_B':
        data = pd.read_csv('../datasets/test_final.csv')
    else:
        mode1 = mode
        data = pd.read_csv('../datasets/invite_' + mode1 + '.csv')

    data = data[['label']]

    print('..load id features..')
    id_feat_list = make_id_feature(mode,data)

    print('..load number features..')
    digt_feat_list = make_number_feature(mode,data)

    # print('..load muti features..')
    # muti_feat_list, muti_p = make_multi_feature(mode,data)

    if isSave == True:
        data.to_csv('../datasets/feature/' + mode + '/nn_data.csv', index=False)

    data = data.drop(['qid_count', 'uid_count','qid_min_day', 'uid_min_day', 'qid_day_nuinque', 'uid_day_nuinque'], axis=1)

    return data,id_feat_list,digt_feat_list






def make_number_feature(mode,data):
    feat_list = []
    load_number_feature('SCORE', DigtEncoder('linear'), mode, data,feat_list)
    load_number_feature('a_topic_num', DigtEncoder('linear'), mode, data, feat_list)
    load_number_feature('l_topic_num', DigtEncoder('linear'), mode, data, feat_list)
    load_number_feature('q_word_num', DigtEncoder('linear'), mode, data, feat_list)
    load_number_feature('d_word_num', DigtEncoder('log1p'), mode, data, feat_list)
    load_number_feature('l_topic_cross_topic_id', DigtEncoder('linear'), mode, data, feat_list)
    load_number_feature('a_topic_cross_topic_id', DigtEncoder('log1p'), mode, data, feat_list)
    load_number_feature('l_topic_meanscore', DigtEncoder(), mode, data, feat_list)
    load_number_feature('l_topic_maxscore', DigtEncoder(), mode, data, feat_list)



    load_number_feature('uid_rank', DigtEncoder('linear'), mode, data, feat_list)
    load_number_feature('qid_rank', DigtEncoder('linear'), mode, data, feat_list)
    load_number_feature('uid_day_rank', DigtEncoder('linear'), mode, data, feat_list)

    load_number_feature('uid_past_invite_time', DigtEncoder('linear'), mode, data, feat_list)
    load_number_feature('qid_past_invite_time', DigtEncoder('linear'), mode, data, feat_list)


    load_number_feature('uid_time_past', DigtEncoder('linear'), mode, data, feat_list)
    load_number_feature('uid_time_feature', DigtEncoder('linear'), mode, data, feat_list)

    load_number_feature('qid_diff_first_invite_time', DigtEncoder('linear'), mode, data, feat_list)
    load_number_feature('uid_diff_first_invite_time', DigtEncoder('linear'), mode, data, feat_list)

    load_number_feature('uid_avg_stack_count', DigtEncoder('linear'), mode, data, feat_list)
    load_number_feature('qid_avg_stack_count', DigtEncoder('linear'), mode, data, feat_list)
    load_number_feature('qid_cur_diff_first', DigtEncoder('linear'), mode, data, feat_list)
    load_number_feature('uid_cur_diff_first', DigtEncoder('linear'), mode, data, feat_list)

    load_number_feature('uid_hist_invite_mean', DigtEncoder('linear'), mode, data, feat_list)
    load_number_feature('uid_hist_invite_count', DigtEncoder('linear'), mode, data, feat_list)
    load_number_feature('uid_hist_invite_sum', DigtEncoder('linear'), mode, data, feat_list)
    load_number_feature('uid_ans_count', DigtEncoder('linear'), mode, data, feat_list)
    load_number_feature('uid_hist_invite_topic_unique', DigtEncoder(), mode, data, feat_list)
    load_number_feature('uid_hist_ans_topic_unique', DigtEncoder(), mode, data, feat_list)

    data['uid_hist_ab'] = np.divide(data['uid_hist_invite_topic_unique'] ,data['uid_hist_invite_sum'] + 1)
    data['uid_hist_as_ab'] = np.divide(data['uid_hist_ans_topic_unique'],data['uid_ans_count'] + 1)


    load_number_feature('sw_uid_invite_topic_unique', DigtEncoder('linear'), mode, data, feat_list)
    load_number_feature('sw_uid_ans_topic_unique', DigtEncoder('linear'), mode, data, feat_list)

    load_number_feature('uid_diff_first_invite_time_sub_max', DigtEncoder('linear'), mode, data, feat_list)
    load_number_feature('qid_diff_first_invite_time_sub_max', DigtEncoder('linear'), mode, data, feat_list)

    data['uid_time_past'] = data['uid_time_past'].apply(lambda x: x if x < 30 else np.NAN)
    data['uid_time_feature'] = data['uid_time_feature'].apply(lambda x: x if x <= 4 else np.NAN)

    data['uid_past_invite_time'] = data['uid_past_invite_time'].apply(lambda x: x if x < 30 else np.NAN)
    data['qid_past_invite_time'] = data['qid_past_invite_time'].apply(lambda x: x if x < 30 else np.NAN)



    load_number_feature('time_diff', DigtEncoder('linear'), mode, data, feat_list)

    load_number_feature('qid_count', DigtEncoder('log1p'), mode, data, feat_list)
    load_number_feature('uid_count', DigtEncoder('linear'), mode, data, feat_list)
    load_number_feature('uid_day_count', DigtEncoder('linear'), mode, data, feat_list)
    load_number_feature('uid_day_hour_count', DigtEncoder('linear'), mode, data, feat_list)
    load_number_feature('uid_min_day_count', DigtEncoder('linear'), mode, data, feat_list)
    load_number_feature('half_qid_day_count', DigtEncoder('log1p'), mode, data, feat_list)
    load_number_feature('half_qid_day_hour_count', DigtEncoder('linear'), mode, data, feat_list)
    load_number_feature('half_qid_min_day_count', DigtEncoder('linear'), mode, data, feat_list)

    load_number_feature('half_qid_day_shour_count', DigtEncoder(), mode, data, feat_list)
    load_number_feature('half_qid_day-1_shour_count', DigtEncoder(), mode, data, feat_list)
    load_number_feature('half_qid_day-1_count', DigtEncoder(), mode, data, feat_list)
    load_number_feature('half_qid_day+1_count', DigtEncoder(), mode, data, feat_list)

    data['qid_next_today_count_dif'] = data['half_qid_day+1_count'] - data['half_qid_day_count']
    data['qid_last_today_count_dif'] = data['half_qid_day_count'] - data['half_qid_day-1_count']


    load_number_feature('uid_day_nuinque', DigtEncoder('linear'), mode, data, feat_list)
    load_number_feature('qid_day_nuinque', DigtEncoder('linear'), mode, data, feat_list)
    load_number_feature('uid_min_day', DigtEncoder('linear'), mode, data, feat_list)
    load_number_feature('qid_min_day', DigtEncoder('linear'), mode, data, feat_list)
    data['qid_count_nunique'] = np.divide(data['qid_count'],data['qid_day_nuinque'])
    data['uid_count_nunique'] = np.divide(data['uid_count'], data['uid_day_nuinque'])
    data['qid_by_day'] = np.divide(data['qid_rank'] - 1, data['time_diff'] + 0.00001)


    data['cur_day_diff'] = data['half_qid_day_count'] - data['half_qid_min_day_count']

    load_number_feature('sw_um_bit7', DigtEncoder(), mode, data, feat_list)
    load_number_feature('sw_um_bit8', DigtEncoder(), mode, data, feat_list)
    load_number_feature('sw_um_bit10', DigtEncoder(), mode, data, feat_list)
    load_number_feature('sw_um_bit11', DigtEncoder(), mode, data, feat_list)
    load_number_feature('sw_um_bit12', DigtEncoder(), mode, data, feat_list)
    load_number_feature('sw_um_bit13', DigtEncoder(), mode, data, feat_list)
    load_number_feature('sw_um_bit15', DigtEncoder(), mode, data, feat_list)
    load_number_feature('sw_um_bit16', DigtEncoder(), mode, data, feat_list)
    load_number_feature('sw_um_bit17', DigtEncoder(), mode, data, feat_list)
    #
    load_number_feature('sw_uc_bit7', DigtEncoder(), mode, data, feat_list)
    load_number_feature('sw_uc_bit8', DigtEncoder(), mode, data, feat_list)
    load_number_feature('sw_uc_bit10', DigtEncoder(), mode, data, feat_list)
    load_number_feature('sw_uc_bit11', DigtEncoder(), mode, data, feat_list)
    load_number_feature('sw_uc_bit12', DigtEncoder(), mode, data, feat_list)
    load_number_feature('sw_uc_bit13', DigtEncoder(), mode, data, feat_list)
    load_number_feature('sw_uc_bit15', DigtEncoder(), mode, data, feat_list)
    load_number_feature('sw_uc_bit16', DigtEncoder(), mode, data, feat_list)
    load_number_feature('sw_uc_bit17', DigtEncoder(), mode, data, feat_list)


    load_number_feature('topic_flag_rate_max', DigtEncoder(), mode, data, feat_list)
    load_number_feature('topic_uid_rate_max', DigtEncoder(), mode, data, feat_list)
    load_number_feature('topic_sex_rate_max', DigtEncoder(), mode, data, feat_list)
    load_number_feature('topic_visit_rate_max', DigtEncoder(), mode, data, feat_list)
    load_number_feature('topic_CA_rate_max', DigtEncoder(), mode, data, feat_list)
    load_number_feature('topic_CB_rate_max', DigtEncoder(), mode, data, feat_list)
    load_number_feature('topic_CC_rate_max', DigtEncoder(), mode, data, feat_list)
    load_number_feature('topic_CD_rate_max', DigtEncoder(), mode, data, feat_list)
    load_number_feature('topic_CE_rate_max', DigtEncoder(), mode, data, feat_list)

    load_number_feature('topic_flag_rate_mean', DigtEncoder(), mode, data, feat_list)
    load_number_feature('topic_uid_rate_mean', DigtEncoder(), mode, data, feat_list)
    load_number_feature('topic_sex_rate_mean', DigtEncoder(), mode, data, feat_list)
    load_number_feature('topic_visit_rate_mean', DigtEncoder(), mode, data, feat_list)
    load_number_feature('topic_CA_rate_mean', DigtEncoder(), mode, data, feat_list)
    load_number_feature('topic_CB_rate_mean', DigtEncoder(), mode, data, feat_list)
    load_number_feature('topic_CC_rate_mean', DigtEncoder(), mode, data, feat_list)
    load_number_feature('topic_CD_rate_mean', DigtEncoder(), mode, data, feat_list)
    load_number_feature('topic_CE_rate_mean', DigtEncoder(), mode, data, feat_list)
    #
    #
    #
    load_number_feature('word_flag_rate_max', DigtEncoder(), mode, data, feat_list)
    load_number_feature('word_uid_rate_max', DigtEncoder(), mode, data, feat_list)
    load_number_feature('word_sex_rate_max', DigtEncoder(), mode, data, feat_list)
    load_number_feature('word_visit_rate_max', DigtEncoder(), mode, data, feat_list)
    load_number_feature('word_CE_rate_max', DigtEncoder(), mode, data, feat_list)
    load_number_feature('word_CA_rate_max', DigtEncoder(), mode, data, feat_list)
    load_number_feature('word_CB_rate_max', DigtEncoder(), mode, data, feat_list)
    load_number_feature('word_CC_rate_max', DigtEncoder(), mode, data, feat_list)
    load_number_feature('word_CD_rate_max', DigtEncoder(), mode, data, feat_list)
    #
    load_number_feature('word_flag_rate_mean', DigtEncoder(), mode, data, feat_list)
    load_number_feature('word_uid_rate_mean', DigtEncoder(), mode, data, feat_list)
    load_number_feature('word_sex_rate_mean', DigtEncoder(), mode, data, feat_list)
    load_number_feature('word_visit_rate_mean', DigtEncoder(), mode, data, feat_list)
    load_number_feature('word_CE_rate_mean', DigtEncoder(), mode, data, feat_list)
    load_number_feature('word_CA_rate_mean', DigtEncoder(), mode, data, feat_list)
    load_number_feature('word_CB_rate_mean', DigtEncoder(), mode, data, feat_list)
    load_number_feature('word_CC_rate_mean', DigtEncoder(), mode, data, feat_list)
    load_number_feature('word_CD_rate_mean', DigtEncoder(), mode, data, feat_list)




    load_number_feature('at_std', DigtEncoder(), mode, data, feat_list)
    load_number_feature('at_min', DigtEncoder(), mode, data, feat_list)
    load_number_feature('at_max', DigtEncoder(), mode, data, feat_list)
    load_number_feature('at_mean', DigtEncoder(), mode, data, feat_list)
    data['at_dif'] = data['time_diff'] - data['at_mean']

    load_number_feature('sw_qid_recent_uclick', DigtEncoder(), mode, data, feat_list)
    load_number_feature('sw_uid_recent_uclick', DigtEncoder(), mode, data, feat_list)
    load_number_feature('sw_u_ans_q_num', DigtEncoder(), mode, data, feat_list)
    load_number_feature('sw_qid_rate_hp', DigtEncoder(), mode, data, feat_list)
    load_number_feature('sw_qld', DigtEncoder(), mode, data, feat_list)
    load_number_feature('sw_qlm', DigtEncoder(), mode, data, feat_list)
    load_number_feature('sw_qls', DigtEncoder(), mode, data, feat_list)
    load_number_feature('sw_qlc', DigtEncoder(), mode, data, feat_list)
    load_number_feature('sw_uid_rate_hp', DigtEncoder(), mode, data, feat_list)
    load_number_feature('sw_uld', DigtEncoder(), mode, data, feat_list)
    load_number_feature('sw_ulm', DigtEncoder(), mode, data, feat_list)
    load_number_feature('sw_uls', DigtEncoder(), mode, data, feat_list)
    load_number_feature('sw_ulc', DigtEncoder(), mode, data, feat_list)

    load_number_feature('uid_week_ans_rate', DigtEncoder(), mode, data, feat_list)
    load_number_feature('uid_week_count', DigtEncoder(), mode, data, feat_list)
    load_number_feature('uid_nhour_count', DigtEncoder(), mode, data, feat_list)

    load_number_feature('topic_day_hour_uid_nunique_max', DigtEncoder(), mode, data, feat_list)
    load_number_feature('topic_uid_nunique_max', DigtEncoder(), mode, data, feat_list)
    load_number_feature('topic_qid_nunique_max', DigtEncoder(), mode, data, feat_list)
    load_number_feature('topic_label_count_max', DigtEncoder(), mode, data, feat_list)

    load_number_feature('topic_day_hour_uid_nunique_sum', DigtEncoder(), mode, data, feat_list)

    load_number_feature('word_smilar_sum', DigtEncoder(), mode, data, feat_list)
    load_number_feature('word_smilar_max', DigtEncoder(), mode, data, feat_list)
    load_number_feature('word_smilar_recent', DigtEncoder(), mode, data, feat_list)
    load_number_feature('word_smilar_mean', DigtEncoder(), mode, data, feat_list)

    load_number_feature('topic_smilar_sum', DigtEncoder(), mode, data, feat_list)
    load_number_feature('topic_smilar_max', DigtEncoder(), mode, data, feat_list)
    load_number_feature('topic_smilar_recent', DigtEncoder(), mode, data, feat_list)
    load_number_feature('topic_smilar_mean', DigtEncoder(), mode, data, feat_list)

    load_number_feature('CE_topic_count_ratio_max', DigtEncoder(), mode, data, feat_list)
    load_number_feature('CD_topic_count_ratio_max', DigtEncoder(), mode, data, feat_list)
    load_number_feature('CC_topic_count_ratio_max', DigtEncoder(), mode, data, feat_list)
    load_number_feature('CB_topic_count_ratio_max', DigtEncoder(), mode, data, feat_list)
    load_number_feature('CA_topic_count_ratio_max', DigtEncoder(), mode, data, feat_list)
    load_number_feature('visit_topic_count_ratio_max', DigtEncoder(), mode, data, feat_list)
    load_number_feature('uid_topic_count_ratio_max', DigtEncoder(), mode, data, feat_list)
    #
    load_number_feature('CE_topic_count_ratio_mean', DigtEncoder(), mode, data, feat_list)
    load_number_feature('CD_topic_count_ratio_mean', DigtEncoder(), mode, data, feat_list)
    load_number_feature('CC_topic_count_ratio_mean', DigtEncoder(), mode, data, feat_list)
    load_number_feature('CB_topic_count_ratio_mean', DigtEncoder(), mode, data, feat_list)
    load_number_feature('CA_topic_count_ratio_mean', DigtEncoder(), mode, data, feat_list)
    load_number_feature('sex_topic_count_ratio_mean', DigtEncoder(), mode, data, feat_list)
    load_number_feature('visit_topic_count_ratio_mean', DigtEncoder(), mode, data, feat_list)
    load_number_feature('uid_topic_count_ratio_mean', DigtEncoder(), mode, data, feat_list)

    load_number_feature('qid_bind_topic_num', DigtEncoder(), mode, data, feat_list)

    data['CD_topic_count_ratio_mean'] = np.divide(data['CD_topic_count_ratio_mean'], data['qid_bind_topic_num'] + 1)
    data['CE_topic_count_ratio_mean'] = np.divide(data['CE_topic_count_ratio_mean'], data['qid_bind_topic_num'] + 1)
    data['CB_topic_count_ratio_mean'] = np.divide(data['CB_topic_count_ratio_mean'], data['qid_bind_topic_num'] + 1)
    data['CC_topic_count_ratio_mean'] = np.divide(data['CC_topic_count_ratio_mean'], data['qid_bind_topic_num'] + 1)
    data['CA_topic_count_ratio_mean'] = np.divide(data['CA_topic_count_ratio_mean'], data['qid_bind_topic_num'] + 1)
    data['sex_topic_count_ratio_mean'] = np.divide(data['sex_topic_count_ratio_mean'], data['qid_bind_topic_num'] + 1)
    data['visit_topic_count_ratio_mean'] = np.divide(data['visit_topic_count_ratio_mean'], data['qid_bind_topic_num'] + 1)
    data['uid_topic_count_ratio_mean'] = np.divide(data['uid_topic_count_ratio_mean'], data['qid_bind_topic_num'] + 1)

    load_number_feature('CA_score_rank', DigtEncoder(), mode, data, feat_list)
    load_number_feature('CE_score_rank', DigtEncoder(), mode, data, feat_list)
    load_number_feature('CD_score_rank', DigtEncoder(), mode, data, feat_list)
    load_number_feature('CC_score_rank', DigtEncoder(), mode, data, feat_list)

    load_number_feature('uid_qid_score_rank', DigtEncoder(), mode, data, feat_list)
    load_number_feature('uid_qid_day_score_rank', DigtEncoder(), mode, data, feat_list)

    load_number_feature('CE_ans_topic_count_ratio_max', DigtEncoder(), mode, data, feat_list)
    load_number_feature('CD_ans_topic_count_ratio_max', DigtEncoder(), mode, data, feat_list)
    load_number_feature('CC_ans_topic_count_ratio_max', DigtEncoder(), mode, data, feat_list)
    load_number_feature('CB_ans_topic_count_ratio_max', DigtEncoder(), mode, data, feat_list)
    load_number_feature('CA_ans_topic_count_ratio_max', DigtEncoder(), mode, data, feat_list)
    load_number_feature('visit_ans_topic_count_ratio_max', DigtEncoder(), mode, data, feat_list)
    load_number_feature('uid_ans_topic_count_ratio_max', DigtEncoder(), mode, data, feat_list)
    #
    load_number_feature('CE_ans_topic_count_ratio_mean', DigtEncoder(), mode, data, feat_list)
    load_number_feature('CD_ans_topic_count_ratio_mean', DigtEncoder(), mode, data, feat_list)
    load_number_feature('CC_ans_topic_count_ratio_mean', DigtEncoder(), mode, data, feat_list)
    load_number_feature('CB_ans_topic_count_ratio_mean', DigtEncoder(), mode, data, feat_list)
    load_number_feature('CA_ans_topic_count_ratio_mean', DigtEncoder(), mode, data, feat_list)
    load_number_feature('visit_ans_topic_count_ratio_mean', DigtEncoder(), mode, data, feat_list)
    load_number_feature('uid_ans_topic_count_ratio_mean', DigtEncoder(), mode, data, feat_list)

    #-----------这里考虑家还是不夹--------------------
    # load_number_feature('uid_ans_hour_mean_diff', DigtEncoder(), mode, data, feat_list)
    # load_number_feature('uid_invite_ans_hour_mean_diff', DigtEncoder(), mode, data, feat_list)
    # load_number_feature('uid_invite_hour_mean_diff', DigtEncoder(), mode, data, feat_list)
    # load_number_feature('uid_ans_hour_std', DigtEncoder(), mode, data, feat_list)
    # load_number_feature('uid_invite_ans_hour_std', DigtEncoder(), mode, data, feat_list)
    # load_number_feature('uid_invite_hour_std', DigtEncoder(), mode, data, feat_list)
    # load_number_feature('recent_7_day_click_rate', DigtEncoder(), mode, data, feat_list,fillnas=0)
    # load_number_feature('ans_dif_1', DigtEncoder(), mode, data, feat_list)
    # load_number_feature('ans_dif_7', DigtEncoder(), mode, data, feat_list)
    # load_number_feature('ans_dif_topic_1', DigtEncoder(), mode, data, feat_list)



















    return feat_list






def make_id_feature(mode, data):

    # if mode == 'train':
    #     id_lengths = {}
    # else:
    #     with open('../datasets2/feature/id_length_lgb.pkl','rb') as f:
    #         id_lengths = pickle.load(f)
    id_lengths = {}
    feat_list = []

    load_id_feature('sex', CateEnconder(),  mode, data, id_lengths,feat_list)
    load_id_feature('visit', CateEnconder(), mode, data, id_lengths,feat_list)
    load_id_feature('BA', CateEnconder(), mode, data, id_lengths,feat_list)
    load_id_feature('BB', CateEnconder(), mode, data, id_lengths,feat_list)
    load_id_feature('BC', CateEnconder(), mode, data, id_lengths,feat_list)
    load_id_feature('BD', CateEnconder(), mode, data, id_lengths,feat_list)
    # #load_id_feature('id_uid_mm_seq', CateEnconder(), mode, data, id_lengths, feat_list)
    # #load_id_feature('id_uid_qq_seq', CateEnconder(), mode, data, id_lengths, feat_list)
    # #load_id_feature('CA', CateEnconder(), mode, data, id_lengths,feat_list)
    # #load_id_feature('CB', CateEnconder(), mode, data, id_lengths,feat_list)
    #
    #
    # #load_id_feature('CC', CateEnconder(), mode, data, id_lengths,feat_list)
    # #load_id_feature('CD', CateEnconder(), mode, data, id_lengths,feat_list)
    load_id_feature('CE', CateEnconder(), mode, data, id_lengths,feat_list)
    # #load_id_feature('uid_seq_8', CateEnconder(), mode, data, id_lengths, feat_list)
    # #load_id_feature('uid_seq_5', CateEnconder(), mode, data, id_lengths, feat_list)
    # #load_id_feature('q_hour', CateEnconder('linear'), mode, data, id_lengths, feat_list)
    # #load_id_feature('hour', CateEnconder('linear'), mode, data, id_lengths, feat_list)
    #
    # #load_id_feature('id_sk_uid_seq_8', CateEnconder(), mode, data, id_lengths, feat_list)
    load_id_feature('sw_uid_seq_5', CateEnconder(), mode, data, id_lengths, feat_list)
    # #load_id_feature('sw_qid_seq_5', CateEnconder('linear'), mode, data, id_lengths, feat_list)


    # with open('../datasets2/feature/id_length_lgb.pkl','wb+') as f:
    #     pickle.dump(id_lengths,f,pickle.HIGHEST_PROTOCOL)
    return feat_list




# def make_multi_feature(mode,data):
#     feat_list = []
#     muti_p = []
#
#     load_muti_feature('topic_id', mode, data,5,'topic',feat_list,muti_p)
#     load_muti_feature('a_topic', mode, data, 8, 'topic', feat_list, muti_p) #用户
#     load_muti_feature('l_topic_top5', mode, data, 5, 'topic', feat_list, muti_p) #用户的topic
#     load_muti_feature('last_topic', mode, data, 15, 'topic', feat_list, muti_p) #用户点击过的topic
#     return feat_list,muti_p


def load_muti_feature(feat ,mode, data,  maxlen = 5 ,use_init = '-', feat_list = None, muti_p = None):
    feat_list.append(feat)
    muti_p.append([maxlen,use_init])
    t = pd.read_csv('../datasets/feature/' + mode + '/' + feat + '.csv',skip_blank_lines=False)
    t = t.fillna('0')
    data[feat] = t
    print(feat, t.shape, t[feat].isnull().sum())




def load_id_feature(feat ,CateEncoder, mode, data, id_map,feat_list):
    feat_list.append(feat)
    t = pd.read_csv('../datasets/feature/' + mode + '/' + feat + '.csv',skip_blank_lines=False)
    #对数据进行转换
    t = t.fillna(0)

    print(feat, t.shape, t[feat].isnull().sum())

    max_value = CateEncoder.transform(t,feat)

    if mode != 'train':
        # max_value = id_map[feat] - 1
        data[feat] = t[feat]
        return
    else:
        data[feat] = t[feat]

    if CateEncoder.getmethod() == 'identiy':
        a = 0
        # with open('../datasets2/feature/' + feat + '.pkl','rb') as f:
        #     maps = pickle.load(f)
        # id_map[feat] = len(maps) + 1
    else:
        a = 0




def load_number_feature(feat,DigtEncoder, mode, data,feat_list,fillnas = -2):

    feat_list.append(feat)
    t = pd.read_csv('../datasets/feature/' + mode + '/' + feat + '.csv',skip_blank_lines=False)
    if mode == 'test' or mode == 'test_B':
        print(feat,t.shape)
    if fillnas != -2:
        t = t.fillna(fillnas)
    DigtEncoder.transform(t,feat)
    data[feat] = t[feat]





