"""
进行用户采样
"""
import pandas as pd
import random

path = "E:\\competition\\看山杯\\data\\data_set_0926\\"
train = pd.read_csv(open(path + 'invite_info_0926.txt', "r", encoding='utf-8'), sep='\t', header=None, names=['questionID', 'memberID', 'time', 'label'])
train['day'] = train['time'].apply(lambda x: int(x.split('-')[0][1:]))
train = train[train['day'] >= 3858]
print(train.shape)

day_users = {}  # key: day value:set of users
for day, user in train[['day', 'memberID']].values:
    if day not in day_users.keys():
        day_users[day] = set()
    day_users[day].add(user)

## 采样
day_user_flag = {}  # day_user_flag[day][user] = 0 or 1
for day, users in day_users.items():
    day_user_flag[day] = {}
    for user in users:
        day_user_flag[day][user] = random.randint(0, 1)

train['flag'] = train.apply(lambda x: day_user_flag[x['day']][x['memberID']], axis=1)
print(train[train['flag'] == 1].shape)

train.to_csv(path + "features\\train_dev.txt", sep='\t', index=False)  # questionID memberID time label day flag
