import pandas as pd
import numpy as np
import pickle as pickle
from make_lgb_datasets_first import make_lgb_data
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import lightgbm as lgb
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

data_train = None
data_dev = None
data_test = None
train_dev= None
idlist = None
dglist = None
mtlist = None
mtp = None
mode = 'train'

if mode == 'train':

    data_test, _, _ = make_lgb_data('test_B', False, False)

    data_train,idlist,dglist = make_lgb_data('train', False, False)
    print('1')
    data_dev,_,_ = make_lgb_data('dev', False, False)
    print('2')


elif mode == 'test':
    data_test,idlist,dglist = make_lgb_data('test_B', False, False)
    data_dev,_,_= make_lgb_data('dev', False, False)

data_train = data_train.reset_index(drop=True)
data_dev = data_dev.reset_index(drop=True)

y_train = data_train['label']
X_train = data_train.drop(['label'],axis=1)



s = X_train.shape[0]



y_dev = data_dev['label']
X_dev = data_dev.drop(['label'],axis=1)

s1 = X_dev.shape[0]

y_test = data_test['label']
X_test = data_test.drop(['label'],axis=1)



print(X_train.shape,X_dev.shape)

total = pd.concat([X_train,X_dev,X_test],axis=0).reset_index(drop=True)

total = total[['sw_uid_seq_5']]
total['sw_uid_seq_5'] = total['sw_uid_seq_5'].astype(str)
total = pd.get_dummies(total)

d1 = total.iloc[:s].reset_index(drop=True)
d2 = total.iloc[s:s + s1].reset_index(drop=True)
d3 = total.iloc[s + s1:].reset_index(drop=True)
print(d1.shape[0],d2.shape[0])
print(d1.head(5))

X_train = pd.concat([X_train,d1],axis=1)
X_dev = pd.concat([X_dev,d2],axis=1)
X_test = pd.concat([X_test,d3],axis=1)
print(X_train.shape,X_dev.shape)



params_lgb = {'num_leaves':128,
         #'min_data_in_leaf': ,
         'objective':'binary',
         'max_depth': -1,
         'learning_rate': 0.05,
         #"min_child_samples": 10,
         "boosting": "gbdt",
         #'is_unbalance':True,
         'max_bin': 256,
         "feature_fraction": 0.6,
         "bagging_fraction": 1 ,
         "bagging_seed": 11,
         "metric": 'auc',
         "lambda_l1": 0.1,
         'lambda_l2': 0.9,
         "verbosity": -1,
         "nthread": 4,
         "random_state": 666}






def lgb_train_predict(train_x, train_y, dev_x,dev_y ,test_x,params, cf,rounds):

    dtrain = lgb.Dataset(train_x, label=train_y)
    dev = lgb.Dataset(dev_x,label=dev_y)
    model = lgb.train(params, dtrain, rounds, valid_sets=[dev], verbose_eval=1,
                      categorical_feature=cf,
                      early_stopping_rounds=20)
    model.save_model('model_feature_t_1.txt')
    ans = model.predict(dev_x)
    test_ans = model.predict(test_x)
    train_ans = model.predict(train_x)
    ans = ans
    ac = pd.DataFrame({
        'column': list(train_x.columns.values),
        'importance': model.feature_importance(),
    }).sort_values(by='importance')
    print(ac)
    return model, ans, test_ans, train_ans

cf = ['sex','visit','BA','BB','BC','BD','CE']
model_lgb, pred_lgb, test_ans, train_ans = lgb_train_predict(X_train, y_train, X_dev,y_dev, X_test, params_lgb, cf,2000)

#pred_lgb = ctb_train_predict(X_train,y_train,X_dev,y_dev,categorical_features)

result = pd.DataFrame({'pred_lgb':test_ans})
result.to_csv('lgbfinal_1test.csv',index=False)

from sklearn.metrics import roc_auc_score

auc_score = roc_auc_score(y_dev, pred_lgb)
auc_score1 = roc_auc_score(y_train, train_ans)
print('dev_auc',auc_score,'train_auc',auc_score1)


test_score = pd.DataFrame({'p_score':test_ans})
test_score.to_csv('../datasets/feature/test_B/p_score',index=False)

dev_score = pd.DataFrame({'p_score':pred_lgb})
dev_score.to_csv('../datasets/feature/dev/p_score',index=False)

train_score = pd.DataFrame({'p_score':train_ans})
train_score.to_csv('../datasets/feature/train/p_score',index=False)

