import pickle
import random
import numpy as np


def E_score1(a,b):
    return np.sum(a * b) / (np.sqrt(
                    np.sum(np.power(a, 2))) * np.sqrt(np.sum(np.power(b, 2))) )

def E_score2(a,b):
    return np.sum(np.power(a-b, 2))


def data_partition_1_withpath(data_path, train, k, T, itr_type='all'):
    '''
    itr_type: 'all': both positive and negative, 'positive': only positive interactions will be considered
    '''
    
    # load the pretrained embedding
    with open(data_path + '/user_pretrain.pk', 'rb') as f:
        uidW = pickle.load(f)
    with open(data_path + '/item_pretrain.pk', 'rb') as f:
        iidW = pickle.load(f)
    
    if itr_type == 'all':
        data = train[['user','item','label']].values.astype(int)
    elif itr_type == 'positive':
        data = train[train['label'].isin([1])]['user','item','label'].values.astype(int)
    else:
        raise NotImplementedError("you must set the ity_type as all or positive!")
    
    # Randomly select k centroids
    data =  data.tolist()
    max_data = 1.2 * len(data) / k
    centroids = random.sample(data, k)

    # centro emb
    centroembs = []
    for i in range(k):
        temp_u = uidW[centroids[i][0]]
        temp_i = iidW[centroids[i][1]]
        centroembs.append([temp_u, temp_i])

    for tt in range(T):
        print(tt)
        C = [{} for i in range(k)]
        C_itr = [{} for i in range(k)]
        C_num=[0 for i in range(k)]
        Scores = {}
        for i in range(len(data)):
            for j in range(k):
                
                score_u = E_score2(uidW[data[i][0]],centroembs[j][0])
                score_i = E_score2(iidW[data[i][1]],centroembs[j][1])
                Scores[i, j] = -score_u * score_i

        Scores = sorted(Scores.items(), key=lambda x: x[1], reverse=True)

        fl = set()
        for i in range(len(Scores)):
            if Scores[i][0][0] not in fl:

                if C_num[Scores[i][0][1]] < max_data:
                    if data[Scores[i][0][0]][0] not in C[Scores[i][0][1]]:
                        C[Scores[i][0][1]][data[Scores[i][0][0]][0]]=[data[Scores[i][0][0]][1]]
                        
                    else:
                        C[Scores[i][0][1]][data[Scores[i][0][0]][0]].append(data[Scores[i][0][0]][1])
                         #[[0]].append(data[Scores[i][0][0]][1])
                    try:
                        C_itr[Scores[i][0][1]].append(data[Scores[i][0][0]])
                    except:
                        C_itr[Scores[i][0][1]] = [data[Scores[i][0][0]]]

                    fl.add(Scores[i][0][0])
                    C_num[Scores[i][0][1]] +=1

        centroembs_next = []
        for i in range(k):
            temp_u = []
            temp_i = []

            for j in C[i].keys():
                for l in C[i][j]:
                    temp_u.append(uidW[j])
                    temp_i.append(iidW[l])
            centroembs_next.append([np.mean(temp_u), np.mean(temp_i)])

        loss = 0.0

        for i in range(k):
            score_u = E_score2(centroembs_next[i][0],centroembs[i][0])

            score_i = E_score2(centroembs_next[i][1],centroembs[i][1])

            loss += (score_u * score_i)

        centroembs = centroembs_next
        for i in range(k):
            print(C_num[i])

        print(tt, loss)

    users = [[] for i in range(k)]
    items = [[] for i in range(k)]
    
    for i in range(k):
        users[i] = list(C[i].keys())
        for j in C[i].keys():
            for l in C[i][j]:
                if l not in items[i]:
                    items[i].append(l)
    return (C,C_itr), users,items

def data_partition_3_withpath(data_path, train, k, T, itr_type='all'):
    '''
    itr_type: 'all': both positive and negative, 'positive': only positive interactions will be considered
    '''
    
    # load the pretrained embedding
    with open(data_path + '/user_pretrain.pk', 'rb') as f:
        uidW = pickle.load(f)
    with open(data_path + '/item_pretrain.pk', 'rb') as f:
        iidW = pickle.load(f)
    
    if itr_type == 'all':
        data = train[['user','item','label']].values.astype(int)
    elif itr_type == 'positive':
        data = train[train['label'].isin([1])]['user','item','label'].values.astype(int)
    else:
        raise NotImplementedError("you must set the ity_type as all or positive!")
    
    # Randomly select k centroids
    data =  data.tolist()
    index = list(range(len(data)))

    random.shuffle(index) 

    elem_num =len(data) / k

    C = [{} for i in range(k)]
    C_itr = [[] for i in range(k)]


    for idx in range(k):
        start = int(idx*elem_num)
        if idx!= k-1:
            end = int((idx+1)*elem_num)
        else:
            end = int(len(data))
        for i in index[start:end]:
            if data[i][0] not in C[idx]:
                C[idx][data[i][0]]=[data[i][1]]
            else:
                C[idx][data[i][0]].append(data[i][1])
            C_itr[idx].append([data[i][0], data[i][1], data[i][2]])

    users = [[] for i in range(k)]
    items = [[] for i in range(k)]
    for i in range(k):
        users[i]=list(C[i].keys())
        for j in C[i].keys():
            for l in C[i][j]:
                if l not in items[i]:
                    items[i].append(l)
    return (C,C_itr), users,items