import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score


def compute_neighbor(data_generator, k_hop=0):
    assert k_hop == 0
    train_data = data_generator.train.values.copy()
    matrix_size = data_generator.n_users + data_generator.n_items
    train_data[:,1] += data_generator.n_users
    train_data[:,-1] = np.ones_like(train_data[:,-1])

    train_data2 = np.ones_like(train_data)
    train_data2[:,0] = train_data[:,1]
    train_data2[:,1] = train_data[:,0]

    paddding = np.concatenate([np.arange(matrix_size).reshape(-1,1), np.arange(matrix_size).reshape(-1,1), np.ones(matrix_size).reshape(-1,1)],axis=-1)
    data = np.concatenate([train_data, train_data2, paddding],axis=0).astype(int)
    train_matrix = sp.csc_matrix((data[:,-1],(data[:,0],data[:,1])),shape=(matrix_size,matrix_size))
    
    neighbor_set = list()
    init_users = data_generator.train_random['user'].values.reshape(-1)
    neighbor_set.extend(np.unique(init_users))
    init_items = data_generator.train_random['item'].values.reshape(-1) + data_generator.n_users
    neighbor_set.extend(np.unique(init_items))
    # print("neighbor_set size:", len(neighbor_set))

    neighbor_set = np.array(neighbor_set)
    return neighbor_set[np.where(neighbor_set<data_generator.n_users)], neighbor_set[np.where(neighbor_set>=data_generator.n_users)] - data_generator.n_users


def get_eval_mask(data_generator):

    valid_data = data_generator.valid[['user', 'item', 'label']].values
    test_data = data_generator.test[['user', 'item', 'label']].values

    nei_users, nei_items = compute_neighbor(data_generator)
    nei_users = torch.from_numpy(nei_users).cuda().long()
    nei_items = torch.from_numpy(nei_items).cuda().long()

    # mask or
    mask_1 = np.zeros(valid_data.shape[0])
    for ii in range(valid_data.shape[0]):
        if valid_data[ii,0] in nei_users or valid_data[ii,1] in nei_items:
            mask_1[ii] = 1
    mask_1 = np.where(mask_1>0)[0] 

    mask_2 = np.zeros(test_data.shape[0])
    for ii in range(test_data.shape[0]):
        if test_data[ii,0] in nei_users or test_data[ii,1] in nei_items:
            mask_2[ii] = 1
    mask_2 = np.where(mask_2>0)[0] 
    
    # mask and
    mask_3 = np.zeros(valid_data.shape[0])
    for ii in range(valid_data.shape[0]):
        if valid_data[ii,0] in nei_users and valid_data[ii,1] in nei_items:
            mask_3[ii] = 1
    mask_3 = np.where(mask_3>0)[0] 

    mask_4 = np.zeros(test_data.shape[0])
    for ii in range(test_data.shape[0]):
        if test_data[ii,0] in nei_users and test_data[ii,1] in nei_items:
            mask_4[ii] = 1
    mask_4 = np.where(mask_4>0)[0] 

    return (mask_1, mask_2, mask_3, mask_4)


def get_eval_result(data_generator, model, mask):
    valid_data = data_generator.valid[['user', 'item', 'label']].values
    test_data = data_generator.test[['user', 'item', 'label']].values

    nei_users, nei_items = compute_neighbor(data_generator)
    nei_users = torch.from_numpy(nei_users).cuda().long()
    nei_items = torch.from_numpy(nei_items).cuda().long()

    mask_1, mask_2, mask_3, mask_4 = mask[0], mask[1], mask[2], mask[3]

    with torch.no_grad():
        valid_predictions =  model.predict(valid_data[:,0], valid_data[:,1])
        test_predictions =  model.predict(test_data[:,0], test_data[:,1])

    valid_auc = roc_auc_score(valid_data[:,-1],valid_predictions)
    valid_auc_or = roc_auc_score(valid_data[:,-1][mask_1], valid_predictions[mask_1])
    valid_auc_and = roc_auc_score(valid_data[:,-1][mask_3], valid_predictions[mask_3])

    test_auc = roc_auc_score(test_data[:,-1],test_predictions)
    test_auc_or = roc_auc_score(test_data[:,-1][mask_2], test_predictions[mask_2])
    test_auc_and = roc_auc_score(test_data[:,-1][mask_4], test_predictions[mask_4])

    return valid_auc, valid_auc_or, valid_auc_and, test_auc, test_auc_or, test_auc_and
    
