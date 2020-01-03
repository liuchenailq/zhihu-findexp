import random

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm


def deepwalk(df, f1, f2, L):
    # Deepwalk算法，
    print("deepwalk:", f1, f2)  # uid aid
    # 构建图
    dic = {}
    for item in tqdm(df[[f1, f2]].values):
        try:  # uid aid
            str(item[1])
            str(item[0])
        except:
            continue
        try:  # 在问题里面加入用户
            dic['question_' + str(item[1])].add('user_' + str(item[0]))
        except:
            dic['question_' + str(item[1])] = set(['user_' + str(item[0])])
        try:  # 在用户里加入广告
            dic['user_' + str(item[0])].add('question_' + str(item[1]))
        except:
            dic['user_' + str(item[0])] = set(['question_' + str(item[1])])
    dic_cont = {}
    for key in dic:
        dic[key] = list(dic[key])  # set 转换为list
        dic_cont[key] = len(dic[key])  # 保存的是长度吗
    print("creating")
    # 构建路径
    path_length = 10
    sentences = []
    length = []
    for key in dic:  # key是user id 或者 a id
        sentence = [key]
        while len(sentence) != path_length:
            key = dic[sentence[-1]][random.randint(0, dic_cont[sentence[-1]] - 1)]
            if len(sentence) >= 2 and key == sentence[-2]:
                break
            else:
                sentence.append(key)
        sentences.append(sentence)
        length.append(len(sentence))
        if len(sentences) % 100000 == 0:
            print(len(sentences))
    print(np.mean(length))
    print(len(sentences))
    # 训练Deepwalk模型
    print('training...')
    random.shuffle(sentences)
    model = Word2Vec(sentences, size=L, window=4, min_count=1, sg=1, workers=10, iter=20)
    print('outputing...')
    # 输出
    values = set(df[f1].values)
    w2v = []
    for v in values:
        try:
            a = [v]
            a.extend(model['user_' + v])
            w2v.append(a)
        except:
            pass
    out_df = pd.DataFrame(w2v)
    names = [f1]
    for i in range(L):
        names.append(f1 + '_' + f2 + '_' + names[0] + '_deepwalk_embedding_' + str(L) + '_' + str(i))
    out_df.columns = names
    print(out_df.head())
    out_df.to_pickle('deepwalk/' + f1 + '_' + f2 + '_' + f1 + '_deepwalk_' + str(L) + '_final.pkl')
    ########################
    values = set(df[f2].values)
    w2v = []
    for v in values:
        try:
            a = [v]
            a.extend(model['question_' + v])
            w2v.append(a)
        except:
            pass
    out_df = pd.DataFrame(w2v)
    names = [f2]
    for i in range(L):
        names.append(f1 + '_' + f2 + '_' + names[0] + '_deepwalk_embedding_' + str(L) + '_' + str(i))
    out_df.columns = names
    print(out_df.head())
    out_df.to_pickle('deepwalk/' + f1 + '_' + f2 + '_' + f2 + '_deepwalk_' + str(L) + '_final.pkl')


invite_info = pd.read_hdf('./my_feat/convert_train.h5', key='data')
invite_info_evaluate = pd.read_hdf('./my_feat/convert_test_b.h5', key='data')

member_feat = pd.read_hdf('./feats/member_feat.h5', key='data')
invite_info = pd.merge(invite_info, member_feat, on='author_id', how='left')
invite_info_evaluate = pd.merge(invite_info_evaluate, member_feat, on='author_id', how='left')

invite_info_evaluate['label'] = -1

df = pd.concat([invite_info, invite_info_evaluate], ignore_index=True)

print('start deepwalk function')
deepwalk(df, 'author_id', 'question_id', 8)
