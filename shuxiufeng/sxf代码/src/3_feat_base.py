import pandas as pd
import numpy as np
import pickle


def print_time(name):
    time = datetime.datetime.now()
    a = time.strftime('%Y-%m-%d %H:%M')
    print(name,a)




def save_id_feat(featlist, data, mode, id_map):
    base_path = '../datasets/feature/' + mode + '/'
    print(data.shape)
    for feat in featlist:
        data[[feat]].to_csv(base_path + feat + '.csv',index=False)
        with open('../datasets/feature/' + feat + '.pkl','wb+') as f:
            pickle.dump(id_map[feat],f,pickle.HIGHEST_PROTOCOL)

def save_num_feat(featlist, data, mode):
    base_path = '../datasets/feature/' + mode + '/'
    for feat in featlist:
        data[[feat]].to_csv(base_path + feat + '.csv',index=False)




def user_base_feat(mode):
    if mode == 'train':

        m_info = pd.read_csv('../datasets/member_info.csv')
        #关注的话题数
        m_info['a_topic_num'] = m_info['a_topic'].apply(lambda x: len(x.split(',')) if x != '0' else 0)
        #ltopic的数量
        m_info['l_topic_num'] = m_info['l_topic'].apply(lambda x: len(x.split(',')) if x != '0:0.0' else 0)
        #ltopic的最大分数
        m_info['l_topic_maxscore'] = m_info['l_topic'].apply(lambda x: np.max([float(item.split(':')[1]) for item in x.split(',')])if x != '0:0.0' else np.NAN)
        #ltioic的平均分数
        m_info['l_topic_meanscore'] = m_info['l_topic'].apply(lambda x: np.mean([float(item.split(':')[1]) for item in x.split(',')] )if x != '0:0.0' else np.NAN)

        save_feat = ['uid','a_topic_num','l_topic_num','l_topic_maxscore','l_topic_meanscore']
        print(m_info.head(10))
        m_info[save_feat].to_csv('../datasets/tmp/m_info_num.csv',index=False,float_format='%.3f')


    if mode == 'train':
        mode1 = mode + '_use'
        data = pd.read_csv('../datasets/invite_' + mode1 + '.csv')
    elif mode == 'test_B':
        data = pd.read_csv('../datasets/test_final.csv')
    else:
        mode1 = mode
        data = pd.read_csv('../datasets/invite_' + mode1 + '.csv')


    m_info_num = pd.read_csv('../datasets/tmp/m_info_num.csv')
    data = pd.merge(data,m_info_num,how='left',on='uid')



    use_col = ['uid', 'sex', 'visit', 'BA', 'BB', 'BC', 'BD', 'BE', 'CE', 'CA', 'CB', 'CC', 'CD', 'SCORE']
    m_info = pd.read_csv('../datasets/member_info.csv', usecols=use_col)
    data = pd.merge(data, m_info, how='left', on='uid')

    save_num_feat(['sex','visit','BA','BB','BC','BD','BE','CE'],data,mode)
    save_num_feat(['a_topic_num','l_topic_num','l_topic_maxscore','l_topic_meanscore','SCORE'],data,mode)




def user_qid_cross(mode):
    '''
    计算用户和问题对应的话题相交数
    :param model:
    :return:

    '''
    q_info = pd.read_csv('../datasets/question_info.csv',usecols=['qid','topic_id','q_word_seq','d_word_seq'])
    m_info = pd.read_csv('../datasets/member_info.csv',usecols=['uid','a_topic','l_topic'])

    if mode == 'train':
        mode1 = mode + '_use'
        data = pd.read_csv('../datasets/invite_' + mode1 + '.csv')
    elif mode == 'test_B':
        data = pd.read_csv('../datasets/test_final.csv')
    else:
        mode1 = mode
        data = pd.read_csv('../datasets/invite_' + mode1 + '.csv')
    data = pd.merge(data,q_info,how='left',on='qid').fillna('0')
    data = pd.merge(data,m_info, how='left', on='uid').fillna('0')

    # a topic cross topic_id

    data['a_topic_cross_topic_id'] =  data.apply(
        lambda x: len(set(x['a_topic'].split(',')).intersection(set(x['topic_id'].split(','))))
        if x['a_topic'] != '0' and x['topic_id'] != '0' else 0,axis=1)

    data['l_topic_cross_topic_id'] = data.apply(
        lambda x: len(set([item.split(':')[0] for item in x['l_topic'].split(',')]).intersection(set(x['topic_id'].split(','))))
        if x['a_topic'] != '0:0.0' and x['topic_id'] != '0' else 0, axis=1)

    data['q_word_num'] = data['q_word_seq'].apply(lambda x: len(x.split(',')) if x != '0' else 0)
    data['d_word_num'] = data['d_word_seq'].apply(lambda x: len(x.split(',')) if x != '0' else 0)

    save_feat = ['a_topic_cross_topic_id','l_topic_cross_topic_id','q_word_num','d_word_num']
    data = data[save_feat]

    save_num_feat(save_feat,data,mode)
    #data.to_csv('../datasets2/feature/' + mode + '/qus_num_feat.csv', index=False, float_format='%.3f')


def feat_topic(mode):
    m_info = pd.read_csv('../datasets/member_info.csv',usecols=['uid','a_topic','l_topic'])
    m_info['l_topic_top5'] = m_info['l_topic'].apply(lambda x:','.join([s.split(':')[0] for s in x.split(',')[:5]]))
    if mode == 'train':
        mode1 = mode + '_use'
        data = pd.read_csv('../datasets/invite_' + mode1 + '.csv')
    elif mode == 'test_B':
        data = pd.read_csv('../datasets/test_final.csv')
    else:
        mode1 = mode
        data = pd.read_csv('../datasets/invite_' + mode1 + '.csv')

    data = pd.merge(data, m_info, how='left', on='uid').fillna('0')
    q_info = pd.read_csv('../datasets/question_info.csv', usecols=['qid', 'topic_id'])

    data = pd.merge(data, q_info, how='left', on='qid').fillna('0')
    data['qid_bind_topic_num'] = data['topic_id'].apply(lambda x: len(x.split(',')) if x != -1 and x != '0' else 0)
    featlist = ['a_topic','topic_id','l_topic_top5','qid_bind_topic_num']


    save_num_feat(featlist,data,mode)


if __name__ == '__main__':

    for mode in ['train','dev','test_B']:
        print('current mode :', mode)

        user_base_feat(mode)

        feat_topic(mode)

        user_qid_cross(mode)

