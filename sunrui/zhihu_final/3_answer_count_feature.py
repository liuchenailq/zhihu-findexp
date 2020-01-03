# 生成final test的answer info
# 包含几个方面
# 1. 回答次数
# 2. 历史回答的topic的话题的embedding平均的distance
# 3. 历史回答的topic的LR的输出probs
import pandas as pd
from tqdm import tqdm
import pickle

tqdm.pandas()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)

with open('./pkl/question_info.pkl', 'rb') as file:
    question_info = pickle.load(file)

question_info.drop(['title_sw_series', 'title_w_series', 'desc_sw_series', 'desc_w_series'], axis=1,
                   inplace=True)

with open('./pkl/answer_info.pkl', 'rb') as file:
    answer_info = pickle.load(file)

invite_info_final_path = './my_feat/convert_test_b.h5'
invite_info_evaluate_final = pd.read_hdf(invite_info_final_path, key='data')

# 获取邀请时间
invite_info_evaluate_final['i_time'] = invite_info_evaluate_final.progress_apply(
    lambda row: (row['invite_day'] * 24 + row['invite_hour']), axis=1)

# 获取回答时间
answer_info['a_time'] = answer_info.progress_apply(lambda row: (row['answer_day'] * 24 + row['answer_hour']), axis=1)

merge_df = pd.merge(invite_info_evaluate_final, answer_info, on='author_id', how='inner')
merge_df['current_answer_count'] = merge_df.groupby(['question_id_x', 'author_id'])['question_id_x'].transform(
    'count')

merge_df.drop_duplicates(['question_id_x', 'author_id'], inplace=True)

merge_df = merge_df[['question_id_x', 'author_id', 'current_answer_count']]
merge_df.columns = ['question_id', 'author_id', 'current_answer_count']
merge_df.to_hdf('./my_feat/user_current_answer_count_test_b.h5', key='data')
