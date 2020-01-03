---运行环境说明-----

系统： CentOS Linux release 7.3.1611 (Core)
关键包版本: 
pandas     0.25.3      （pandas处理库）
lightgbm   2.3.0       (不同版本可能模型不兼容)
modin      0.6.3       (pandas加速处理库)
numpy      1.17.1      (科学计算库)





结构说明
datasets
    --feature  保存生成的特征，每个特征存储为一个csv文件
        --train  保存训练集的特征，每个特征单独保存为一个csv文件
        --dev    保存验证机集的特征，每个特征单独保存为一个csv文件
        --test   保存A榜测试集的特征，每个特征单独保存为一个csv文件
        --testB     保存B榜测试集的特征，每个特征单独保存为一个csv文件
    --tmp  保存临时文件
src 保存源代码

---代码运行说明，由于比赛时间有限，最后一天多台机器并行运行以下代码，大概时间为4至5个小时-----
按照如下顺序执行代码

1_data_process.py 
    处理数据集，将数据集处理成csv格式，划分出训练集，验证机，并从训练集中划分出子训练集
2_map_id.py
    对除了uid和qid之外的类别特征进行整数映射，处理官方预先训练向量
3_feat_base.py
    提取一些基本的特征
4_feat_count.py
    提取一些计数特征，这部分是联合a榜b榜数据集进行统计的
5_feat_count1.py
    没有联合a榜数据集统计的特征
6_make_mean_vector.py
    对问题绑定的问题词序列和话题序列分布求平均并保存
7_feat_slide_windows.py
    通过滑窗的方式构建特征
8_feat_time
    提取和时间性有关的一些特征
9_feat_more_here.py
    提取额外的一些补充特征
10_train_lgb_first.py
    运行第一遍lgb,产生对数据的预测分数，用于下一步构造一些特征
11_feat_jiajie.py
    考虑到测试集有7天，因此用预测得到的分数模拟测试集的标签，构造一些特征
12_train_lgb_final.py
    训练最后的lgb模型

            
