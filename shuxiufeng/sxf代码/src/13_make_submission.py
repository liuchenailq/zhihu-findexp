import pandas as pd

f1 = pd.read_csv('../datasets/invite_info_evaluate_2_0926.txt',sep='\t',header = None, names=['a1','a2','a3'])
f1['label'] = -1
f2 = pd.read_csv('lgb_result_final.csv')
f1['label'] = f2['pred_lgb']
print(f1.shape)
f1.to_csv('sxf_result.txt',sep='\t',header=False,index=False)