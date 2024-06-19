import pandas as pd
import numpy as np

# attack = 0.01 # 0.02
# thre_ = 4
# load_path = './Data/Original/Amazon_Electronics.csv'
# save_path = './Data/Process/Amazon/' + str(attack) + '/'

attack = 0.01 # 0.02
thre_ = 5
load_path = './Data/Original/BookCrossing.csv'
save_path = './Data/Process/BookCrossing/' + str(attack) + '/'

if __name__ == '__main__':

    if thre_ == 4:
        print('Amazon')
        data = pd.read_csv(load_path, delimiter=',', encoding='ISO-8859-1').iloc[:, :3]  # 第四列是浏览时间
    elif thre_ == 5:
        print('BookCrossing')
        data = pd.read_csv(load_path, delimiter=';', encoding='ISO-8859-1')
    else:
        raise ValueError('Invalid datasets. ')

    data.columns = ['user', 'item', 'label']

    print("user num:", data['user'].unique().shape[0])
    print("item num:", data['item'].unique().shape[0])
    print("intersection num:", data.shape[0])
    print("data sparse:", data.shape[0] / data['user'].unique().shape[0] / data['item'].unique().shape[0])
    print("label type:", sorted(data['label'].unique()))

    def k_core_filtering(select_data, k=5):
        itr = 0
        while True:
            pre_shape = select_data.shape[0]

            user_info = select_data.groupby('user').agg({'label': ['count', 'mean']})
            s_u = user_info[user_info[('label', 'count')] >= k].index
            select_data = select_data[select_data['user'].isin(s_u)]

            item_info = select_data.groupby('item').agg({'label': ['count', 'mean']})
            s_i = item_info[item_info[('label', 'count')] >= k].index
            select_data = select_data[select_data['item'].isin(s_i)]

            aft_shape = select_data.shape[0]
            print("itr:", itr, 'pre-shape:', pre_shape, 'aft_shape:', aft_shape)
            if pre_shape == aft_shape:
                break
        return select_data

    select_data = k_core_filtering(data, k=5)
    print("user num:", select_data['user'].unique().shape[0])
    print("item num:", select_data['item'].unique().shape[0])
    print("intersection num:", select_data.shape[0])
    print("data sparse:", select_data.shape[0] / select_data['user'].unique().shape[0] / select_data['item'].unique().shape[0])

    idx = np.arange(select_data.shape[0])
    np.random.seed(1024)
    np.random.shuffle(idx)
    n1 = int(select_data.shape[0] * 0.6)
    n2 = int(select_data.shape[0] * 0.8)
    idx1 = idx[:n1]
    idx2 = idx[n1:n2]
    idx3 = idx[n2:]
    train, valid, test = select_data.iloc[idx1], select_data.iloc[idx2], select_data.iloc[idx3]

    print("user num:", train['user'].unique().shape[0])
    print("item num:", train['item'].unique().shape[0])
    print("intersection num:", train.shape[0])

    user_map = dict(zip(train['user'].unique(), np.arange(train['user'].unique().shape[0])))  # 对user序号进行重新编码
    item_map = dict(zip(train['item'].unique(), np.arange(train['item'].unique().shape[0])))  # 对item序号进行重新编码

    def enconding_f(pd_data, user_map, item_map, thre=5):  # thre代表隐反馈阈值，对于BookCrossing而言评分为0到10（5）, 对于amazon而言评分为1-5（4）
        pd_data = pd_data.copy()
        pd_data['user'] = pd_data['user'].map(user_map)
        pd_data['item'] = pd_data['item'].map(item_map)
        pd_data['label'] = pd_data['label'].apply(lambda x: 1 if x > thre else 0)
        return pd_data
    
    train_user = train['user'].unique()
    train_item = train['item'].unique()
    valid_ = valid[valid['user'].isin(train_user)]
    valid_ = valid_[valid_['item'].isin(train_item)]
    test_ = test[test['user'].isin(train_user)]
    test_ = test_[test_['item'].isin(train_item)]

    nn = int(train.shape[0] * (1-attack))
    idx1 = np.arange(nn)
    idx2 = np.arange(nn, train.shape[0])
    train_normal, train_unlearn = train.iloc[idx1].copy(), train.iloc[idx2].copy()

    train_normal_save = enconding_f(train_normal, user_map, item_map, thre=thre_)
    train_unlearn_save = enconding_f(train_unlearn, user_map, item_map, thre=thre_)
    train_save = enconding_f(train, user_map, item_map, thre=thre_)
    valid_save = enconding_f(valid_, user_map, item_map, thre=thre_)
    test_save = enconding_f(test_, user_map, item_map, thre=thre_)

    train_unlearn_save['label'] = 1 - train_unlearn_save['label']

    train_save.to_csv(save_path + 'train.csv', index=None)
    train_normal_save.to_csv(save_path + 'train_normal.csv', index=None)
    train_unlearn_save.to_csv(save_path + 'train_random.csv', index=None)
    test_save.to_csv(save_path + 'test.csv', index=None)
    valid_save.to_csv(save_path + 'valid.csv', index=None)