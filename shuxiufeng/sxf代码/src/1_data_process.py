# -*- coding: utf-8 -*-

from csv import DictWriter
from tqdm import tqdm
import mmap
import pandas as pd


def get_file_lines(file_path):
    '''
    获取文件行数
    '''
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def process_vowpal(file_path, out_path):
    '''
    去除 qestion_info 中的单字编码序列
    '''
    headers = ['qid', 'ctime', 'q_char_seq', 'q_word_seq', 'd_char_seq', 'd_word_seq', 'topic_id']
    fo = open(out_path, 'wt')
    writer = DictWriter(fo, fieldnames=headers, lineterminator='\n')
    writer.writeheader()

    with open(file_path, 'rt') as f:
        for line in tqdm(f, total=get_file_lines(file_path)):
            feature_groups = line.strip().split('\t')
            fea_dict = {}
            for index,feas in enumerate(feature_groups):

                if index in [2,4]:
                    fea_dict[headers[index]] = -1
                else:
                    fea_dict[headers[index]] = feas
            writer.writerow(fea_dict)
    fo.close()

def process_vowpal1(file_path, out_path):
    '''
    去除 qestion_info 中的单字编码序列
    '''
    headers = ['aid','qid', 'uid','atime', 'a_char_seq', 'a_word_seq', 'bit7', 'bit8',
               'bit9', 'bit10','bit11', 'bit12','bit13', 'bit14','bit15', 'bit16','bit17', 'bit18',
               'bit19', 'bit20']
    fo = open(out_path, 'wt')
    writer = DictWriter(fo, fieldnames=headers, lineterminator='\n')
    writer.writeheader()

    with open(file_path, 'rt') as f:
        for line in tqdm(f, total=get_file_lines(file_path)):
            feature_groups = line.strip().split('\t')
            fea_dict = {}
            for index,feas in enumerate(feature_groups):

                if index in [4,5]:
                    fea_dict[headers[index]] = -1
                else:
                    fea_dict[headers[index]] = feas
            writer.writerow(fea_dict)
    fo.close()


if __name__ == '__main__':

    #处理question_info
    file_path = '../datasets/question_info_0926.txt'
    out_path = '../datasets/question_info.csv'
    process_vowpal(file_path, out_path)
    #处理answer_info
    file_path1 = '../datasets/answer_info_0926.txt'
    out_path1 = '../datasets/answer_info.csv'
    process_vowpal1(file_path1, out_path1)
    #处理member_info
    header = ['uid', 'sex', 'c_word_seq', 'c_degree', 'c_hot'
        , 'register', 'platform', 'visit', 'BA', 'BB', 'BC', 'BD', 'BE',
              'CA', 'CB', 'CC', 'CD', 'CE', 'SCORE', 'a_topic', 'l_topic']
    member_info = pd.read_csv('../datasets/member_info_0926.txt',sep='\t',header=None,names=header)
    member_info.to_csv('../datasets/member_info.csv',index=False)


    def get_hour(df):
        hour = df['ctime'].split('H')[1]
        return int(hour)


    def get_day(df):
        time = df['ctime']
        day = time.split('-')[0]
        day = day.split('D')[1]
        return int(day)

    #处理邀请数据集
    header = ['qid','uid','ctime','label']
    invite_info = pd.read_csv('../datasets/invite_info_0926.txt', sep='\t', header=None, names=header)
    invite_info['hour'] = invite_info.apply(get_hour, axis=1)
    invite_info['day'] = invite_info.apply(get_day, axis=1)

    invite_dev = invite_info[invite_info['day']   >=3865]
    invite_train = invite_info[invite_info['day'] < 3865]
    invite_train_use = invite_train[invite_train['day'] > 3858]

    #保存验证机，取后三天
    invite_dev.to_csv('../datasets/invite_dev.csv', index=False)
    invite_train.to_csv('../datasets/invite_train.csv', index=False)
    invite_train_use.to_csv('../datasets/invite_train_use.csv', index=False)






    #处理测试机
    header = ['qid', 'uid', 'ctime']
    invite_info = pd.read_csv('../datasets/invite_info_evaluate.txt', sep='\t', header=None, names=header)
    invite_info['hour'] = invite_info.apply(get_hour, axis=1)
    invite_info['day'] = invite_info.apply(get_day, axis=1)
    invite_info.to_csv('../datasets/invite_test.csv', index=False)

    #处理测试集B榜
    header = ['qid', 'uid', 'ctime']

    test_df = pd.read_csv('../datasets/invite_info_evaluate_2_0926.txt', sep='\t', header=None, names=header)
    test_df['label'] = -1
    test_df['hour'] = test_df.apply(get_hour, axis=1)
    test_df['day'] = test_df.apply(get_day, axis=1)
    test_df.to_csv('../datasets/test_final.csv', index=False)

