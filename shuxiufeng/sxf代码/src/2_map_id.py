import pandas as pd
import pickle
import numpy as np

id_all_map = {}

print('###########读取word的映射表###########')
path = '../datasets/word_vectors_64d.txt'
vector_list = []
word_list = []
word_list.append('-1')
vector_list.append([0 for i in range(0, 64)])
with open(path,'r') as f:
    for line in f.readlines():
        line = line.strip().split('\t')
        name = line[0]
        word = line[1].split(' ')
        vector = [float(word[i]) for i in range(0, 64)]
        word_list.append(name)
        vector_list.append(vector)

word_map = dict(zip(word_list,range(len(word_list))))
id_all_map['word'] = word_map

vector_list[0] = np.mean(vector_list[1:],axis=0).tolist()

print(np.mean(vector_list[1:],axis=0))

with open('../datasets/word_vector.pkl','wb+') as f:
    pickle.dump(vector_list,f,pickle.HIGHEST_PROTOCOL)
with open('../datasets/word_map.pkl','wb+') as f:
    pickle.dump(word_map,f,pickle.HIGHEST_PROTOCOL)


print('############读取topic的映射表###########')
path = '../datasets/topic_vectors_64d.txt'
vector_list = []
topic_list = []
topic_list.append('-1')
vector_list.append([0 for i in range(0, 64)])
with open(path,'r') as f:
    for line in f.readlines():
        line = line.strip().split('\t')
        name = line[0]
        topic = line[1].split(' ')
        vector = [float(topic[i]) for i in range(0, 64)]
        topic_list.append(name)
        vector_list.append(vector)

topic_map = dict(zip(topic_list,range(len(topic_list))))
id_all_map['topic'] = topic_map

vector_list[0] = np.mean(vector_list[1:],axis=0).tolist()

print(np.mean(vector_list[1:],axis=0))

with open('../datasets/topic_vector.pkl','wb+') as f:
    pickle.dump(vector_list,f,pickle.HIGHEST_PROTOCOL)
with open('../datasets/topic_map.pkl','wb+') as f:
    pickle.dump(topic_map,f,pickle.HIGHEST_PROTOCOL)


# print('##########对member_info进行映射##################')

m_info = pd.read_csv('../datasets/member_info.csv')
m_id_feat = ['sex', 'visit', 'BA', 'BB', 'BC', 'BD', 'BE', 'CE', 'CA', 'CB', 'CC', 'CD']
for feat in m_id_feat:
    u = m_info[feat].unique().tolist()
    map = dict(zip(u, range(1,len(u) + 1)))
    map['-1'] = 0
    id_all_map[feat] = map
    m_info[feat] = m_info[feat].apply(lambda x: map[x])



m_topic_feat = ['a_topic','l_topic']
m_info['a_topic'] = \
    m_info['a_topic'].apply(lambda x: ','.join([str(topic_map[i]) for i in x.split(',')]))


def __map_ltopic(df):
    topic = df['l_topic']
    if topic == -1 or topic == '-1':
        return '0:0.0'
    else:
        topic_list = topic.split(',')
        tmp = []
        for t in topic_list:
            t = t.split(':')
            tmp.append((topic_map[t[0]], float(t[1])))
        tmp = sorted(tmp, key=lambda x: x[1], reverse=True)
        result = ','.join([str(item[0]) + ':' + str(item[1]) for item in tmp])
        return result
m_info['l_topic'] = m_info.apply(__map_ltopic,axis=1)


m_info[['uid','SCORE'] + m_id_feat + m_topic_feat].to_csv('../datasets/member_info.csv',index=False)

with open('../datasets/id_all_map.pkl','wb+') as f:
    pickle.dump(id_all_map,f,pickle.HIGHEST_PROTOCOL)


print('##########对question_info进行映射##################')

q_info = pd.read_csv('../datasets/question_info.csv')
q_info['topic_id'] = \
    q_info['topic_id'].apply(lambda x: ','.join([str(topic_map[i]) for i in x.split(',')]))

q_info['q_word_seq'] = \
    q_info['q_word_seq'].apply(lambda x: ','.join([str(word_map[i]) for i in x.split(',')]))

q_info['d_word_seq'] = \
    q_info['d_word_seq'].apply(lambda x: ','.join([str(word_map[i]) for i in x.split(',')]))

q_info['q_day'] = q_info['ctime'].apply(lambda x: int((x.split('-')[0]).split('D')[1]))
q_info['q_hour'] = q_info['ctime'].apply(lambda x: int(x.split('H')[1]))

q_info.to_csv('../datasets/question_info.csv',index=False)

#
#
print('##########对answer_info进行映射##################')

a_info = pd.read_csv('../datasets/answer_info.csv')
a_info['a_word_seq'] = \
    a_info['a_word_seq'].apply(lambda x: ','.join([str(word_map[i]) for i in x.split(',')]) if x != -1 else '0')

a_info['a_day'] = a_info['atime'].apply(lambda x: int((x.split('-')[0]).split('D')[1]))
a_info['a_hour'] = a_info['atime'].apply(lambda x: int(x.split('H')[1]))
a_info.to_csv('../datasets/answer_info.csv',index=False)





















