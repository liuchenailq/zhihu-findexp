# 运行环境
pandas   
sklearn   
numpy   
scipy   
lightgbm   

# 训练集验证集划分方式
邀请数据表中的第[3858, 3864]天作为训练集，第[3865, 3867]作为线下验证集，测试集为官方发布的。   

为了保证数据分布一致，将训练集和验证集进行了用户采样，一份训练集按用户采样分成两个文件train0和train1，一份验证集按用户采样分成dev0和dev1两个文件。所以现在一共有train0,trian1,dev0,dev1,test四份文件。    

每个文件都要进行特征生成，在代码中只需要修改job参数，指定是train0、train1、dev0、dev1、test即可。   

# 运行步骤
1. 在feature_engine文件下依次运行如下文件生成特征。
2. 在 lgb_train 文件下运行 lgb_train1.py得到训练集、验证集、测试集的预测结果。
3. 在feature_engine文件运行 new_feature.py文件生成特征
4. 在 lgb_train 文件下运行 lgb_train2.py，得到的result.txt就是最后的预测结果。
