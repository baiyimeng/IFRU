import numpy as np
import random as rd
import scipy.sparse as sp
from time import *
from utility.data_partition import *
import pickle
import os
import pandas as pd
import torch
import torch.utils.data
from scipy.sparse import csr_matrix
from pandas.core.frame import DataFrame


def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)


class Data_for_MF(object):
    def __init__(self, data_path, batch_size):
        self.batch_size = batch_size
        self.train_normal = pd.read_csv(data_path + "/train_normal.csv", index_col=None)
        self.train_random = pd.read_csv(data_path + "/train_random.csv", index_col=None)
        self.valid = pd.read_csv(data_path + "/valid.csv", index_col=None)
        self.test = pd.read_csv(data_path + "/test.csv", index_col=None)
        self.data_size()

    def data_size(self):
        data = pd.concat([self.train_normal, self.train_random, self.valid, self.test], axis=0, ignore_index=True)
        self.n_users = data['user'].max() + 1
        self.n_items = data['item'].max() + 1

    def set_train_mode(self, mode='full'):
        self.train_mode = mode
        if mode == 'full':
            self.train = pd.concat([self.train_normal, self.train_random], axis=0, ignore_index=True)
        else:
            self.train = self.train_normal.copy()
        self.n_train = self.train.shape[0]
        self.n_valid = self.valid.shape[0]
        self.n_test = self.test.shape[0]
        self.train_loader = None

    def batch_generator(self):
        if self.train_loader is None:
            n_sampels = self.train.shape[0]
            idx = np.arange(n_sampels)
            np.random.shuffle(idx)
            data = torch.from_numpy(self.train[['user', 'item', 'label']].values)
            self.train_loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=True, drop_last=False)
        return self.train_loader


class Data_for_RecEraser_MF(object):
    def __init__(self, data_path, batch_size, part_type=1, part_num=10, part_T=1):
        self.batch_size = batch_size
        self.path = data_path
        self.train_normal = pd.read_csv(data_path + "/train_normal.csv", index_col=None)
        self.train_random = pd.read_csv(data_path + "/train_random.csv", index_col=None)
        self.valid = pd.read_csv(data_path + "/valid.csv", index_col=None)
        self.test = pd.read_csv(data_path + "/test.csv", index_col=None)
        self.part_num = part_num
        self.part_type = part_type
        self.data_size()
        if self.part_type != 0:
            try:
                with open(self.path + '/C_type-' + str(part_type) + '_num-' + str(part_num) + '.pk', 'rb') as f:
                    self.C, self.C_itr = pickle.load(f)
                with open(self.path + '/C_U_type-' + str(part_type) + '_num-' + str(part_num) + '.pk', 'rb') as f:
                    self.C_U = pickle.load(f)
                with open(self.path + '/C_I_type-' + str(part_type) + '_num-' + str(part_num) + '.pk', 'rb') as f:
                    self.C_I = pickle.load(f)
            except Exception:  # partition
                train_all = pd.concat([self.train_normal, self.train_random], axis=0, ignore_index=True)
                if part_type == 1:
                    CC_, self.C_U, self.C_I = data_partition_1_withpath(self.path, train_all, part_num, part_T)
                    if isinstance(CC_, tuple):
                        self.C, self.C_itr = CC_[0], CC_[1]
                if part_type == 2:
                    raise NotImplementedError("not implement this partion methods")
                if part_type == 3:
                    CC_, self.C_U, self.C_I = data_partition_3_withpath(self.path, train_all, part_num, part_T)
                    if isinstance(CC_, tuple):
                        self.C, self.C_itr = CC_[0], CC_[1]

                with open(self.path + '/C_type-' + str(part_type) + '_num-' + str(part_num) + '.pk', 'wb') as f:
                    pickle.dump(CC_, f)
                with open(self.path + '/C_U_type-' + str(part_type) + '_num-' + str(part_num) + '.pk', 'wb') as f:
                    pickle.dump(self.C_U, f)
                with open(self.path + '/C_I_type-' + str(part_type) + '_num-' + str(part_num) + '.pk', 'wb') as f:
                    pickle.dump(self.C_I, f)
            self.n_C = []
            for i in range(len(self.C)):
                t = 0
                for j in self.C[i]:
                    t += len(self.C[i][j])
                self.n_C.append(t)

    def data_size(self):
        data = pd.concat([self.train_normal, self.train_random, self.valid, self.test], axis=0, ignore_index=True)
        self.n_users = data['user'].max() + 1
        self.n_items = data['item'].max() + 1

    def set_train_mode(self, mode='all'):
        self.train_mode = mode
        if mode == 'full':
            self.train = pd.concat([self.train_normal, self.train_random], axis=0, ignore_index=True)
        else:
            self.train = self.train_normal.copy()
            self.remove_unlearning_data()
        self.n_train = self.train.shape[0]
        self.n_valid = self.valid.shape[0]
        self.n_test = self.test.shape[0]
        self.train_loader = None

    def batch_generator(self):
        if self.train_loader is None:
            n_sampels = self.train.shape[0]
            idx = np.arange(n_sampels)
            np.random.shuffle(idx)
            data = torch.from_numpy(self.train[['user', 'item', 'label']].values)
            self.train_loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=True, drop_last=False)
        return self.train_loader

    def batch_generator_local(self, local_id=None):
        local_data = self.C_itr[local_id]
        data = torch.from_numpy(np.array(local_data))
        train_loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=True, drop_last=False)
        return train_loader

    def remove_unlearning_data(self):
        C_itr_new = [[] for i in range(self.part_num)]
        unlearn_data = self.train_random[['user', 'item', 'label']].values.tolist()
        for local_id in range(self.part_num):
            local_data = self.C_itr[local_id]
            for data_ in local_data:
                if data_ in unlearn_data:
                    continue
                C_itr_new[local_id].append(data_)
        self.C_itr = C_itr_new
        self.train = self.train_normal


class Data_for_RecEraser_LightGCN(object):
    def __init__(self, data_path, batch_size, part_type=1, part_num=10, part_T=1):
        self.batch_size = batch_size
        self.path = data_path
        self.train_normal = pd.read_csv(data_path + "/train_normal.csv", index_col=None)
        self.train_random = pd.read_csv(data_path + "/train_random.csv", index_col=None)
        self.valid = pd.read_csv(data_path + "/valid.csv", index_col=None)
        self.test = pd.read_csv(data_path + "/test.csv", index_col=None)
        self.data_size()
        self.part_num = part_num
        self.Graph = [0] * part_num
        self.part_type = part_type
        if self.part_type != 0:
            try:
                with open(self.path + '/C_type-' + str(part_type) + '_num-' + str(part_num) + '.pk', 'rb') as f:
                    self.C, self.C_itr = pickle.load(f)
                with open(self.path + '/C_U_type-' + str(part_type) + '_num-' + str(part_num) + '.pk', 'rb') as f:
                    self.C_U = pickle.load(f)
                with open(self.path + '/C_I_type-' + str(part_type) + '_num-' + str(part_num) + '.pk', 'rb') as f:
                    self.C_I = pickle.load(f)
                print("load part finsihed")
            except Exception:  # partition
                train_all = pd.concat([self.train_normal, self.train_random], axis=0, ignore_index=True)
                if part_type == 1:
                    CC_, self.C_U, self.C_I = data_partition_1_withpath(self.path, train_all, part_num, part_T)
                    if isinstance(CC_, tuple):
                        self.C, self.C_itr = CC_[0], CC_[1]
                if part_type == 2:
                    raise NotImplementedError("not implement this partion methods")
                if part_type == 3:
                    CC_, self.C_U, self.C_I = data_partition_3_withpath(self.path, train_all, part_num, part_T)
                    if isinstance(CC_, tuple):
                        self.C, self.C_itr = CC_[0], CC_[1]

                with open(self.path + '/C_type-' + str(part_type) + '_num-' + str(part_num) + '.pk', 'wb') as f:
                    pickle.dump(CC_, f)
                with open(self.path + '/C_U_type-' + str(part_type) + '_num-' + str(part_num) + '.pk', 'wb') as f:
                    pickle.dump(self.C_U, f)
                with open(self.path + '/C_I_type-' + str(part_type) + '_num-' + str(part_num) + '.pk', 'wb') as f:
                    pickle.dump(self.C_I, f)

            self.n_C = []

            for i in range(len(self.C)):
                t = 0
                for j in self.C[i]:
                    t += len(self.C[i][j])
                self.n_C.append(t)

    def data_size(self):
        data = pd.concat([self.train_normal, self.train_random, self.valid, self.test], axis=0, ignore_index=True)
        self.n_users = data['user'].max() + 1
        self.n_items = data['item'].max() + 1

    def set_train_mode(self, mode='full'):
        self.train_mode = mode
        if mode == 'retraining':  # 设置train，移去unlearn，更新Graph
            self.train = self.train_normal
            self.remove_unlearning_data()
            for i in range(self.part_num):
                self.Graph[i] = self.getSparseGraph_mode(i, mode)
        elif mode == 'full':  # 设置train，更新Graph
            self.train = pd.concat([self.train_normal, self.train_random], axis=0)
            for i in range(self.part_num):
                self.Graph[i] = self.getSparseGraph_mode(i, mode)
        else:
            raise NotImplementedError("please select the mode in: [full, retraining, unlearning]")
        self.n_train = self.train.shape[0]
        self.n_valid = self.valid.shape[0]
        self.n_test = self.test.shape[0]
        self.train_loader = None

    def batch_generator(self):
        if self.train_loader is None:
            n_sampels = self.train.shape[0]
            idx = np.arange(n_sampels)
            np.random.shuffle(idx)
            data = torch.from_numpy(self.train[['user', 'item', 'label']].values)
            self.train_loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=True, drop_last=False)
        return self.train_loader

    def batch_generator_local(self, local_id=None):
        local_data = self.C_itr[local_id]
        data = torch.from_numpy(np.array(local_data))
        train_loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=True, drop_last=False)
        return train_loader

    def remove_unlearning_data(self):
        C_itr_new = [[] for i in range(self.part_num)]
        unlearn_data = self.train_random[['user', 'item', 'label']].values.tolist()
        for local_id in range(self.part_num):
            local_data = self.C_itr[local_id]
            for data_ in local_data:
                if data_ in unlearn_data:
                    continue
                C_itr_new[local_id].append(data_)
        self.C_itr = C_itr_new

    def getSparseGraph_mode(self, local_id, mode):
        data = self.C_itr[local_id]
        data = DataFrame(data)
        data.rename(columns={0: 'user', 1: 'item', 2: 'label'}, inplace=True)
        pos_train = data[data['label'] > 0].values.copy()

        pos_train[:, 1] += self.n_users

        print("loading adjacency matrix")
        if True:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat_' + str(local_id) + mode + '.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacency matrix")
                s = time()
                pos_train_t = pos_train.copy()
                pos_train_t[:, 0] = pos_train[:, 1]
                pos_train_t[:, 1] = pos_train[:, 0]
                pos = np.concatenate([pos_train, pos_train_t], axis=0)

                adj_mat = sp.csr_matrix((pos[:, 2], (pos[:, 0], pos[:, 1])), shape=(self.n_users + self.n_items, self.n_users + self.n_items))
                adj_mat = adj_mat.todok()
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat_' + str(local_id) + mode + '.npz', norm_adj)
            graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            graph = graph.coalesce().cuda()
        return graph

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))


class Data_for_LightGCN(object):
    def __init__(self, config, path):
        print("loading: ", path)
        self.split = config.A_split
        self.folds = config.A_n_fold
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']

        train_normal_file = path + "/train_normal.csv"
        train_unlearn_file = path + "/train_random.csv"
        valid_file = path + "/valid.csv"
        test_file = path + "/test.csv"
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0

        self.train_normal = pd.read_csv(train_normal_file)[['user', 'item', 'label']]
        self.train_random = pd.read_csv(train_unlearn_file)[['user', 'item', 'label']]
        self.valid = pd.read_csv(valid_file)[['user', 'item', 'label']]
        self.test = pd.read_csv(test_file)[['user', 'item', 'label']]

        self.n_users = 1 + max([self.train_normal['user'].max(), self.train_random['user'].max(), self.valid['user'].max(), self.test['user'].max()])
        self.n_items = 1 + max([self.train_normal['item'].max(), self.train_random['item'].max(), self.valid['item'].max(), self.test['item'].max()])

        self.testDataSize = self.test.shape[0]
        self.validDataSize = self.valid.shape[0]
        self.train_normal_size = self.train_normal.shape[0]
        self.train_random_size = self.train_random.shape[0]

        self.Graph = None
        self.ChangedGraph = None
        print(f"{self.train_normal_size} interactions for normal training")
        print(f"{self.train_random_size} interactions for unlearning")
        print(f"{self.validDataSize} interactions for validation")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{config.dataset} Sparsity : {(self.validDataSize + self.testDataSize+self.train_normal_size+self.train_random_size) / self.n_users / self.n_items}")
        print("%s is ready to go" % (config.dataset))

    def set_train_mode(self, mode):
        if mode == 'retraining':
            self.train = self.train_normal
            self.getSparseGraph_mode_a(mode)
            self.n_train = self.train.shape[0]
        elif mode == 'full':
            self.train = pd.concat([self.train_normal, self.train_random], axis=0)
            self.getSparseGraph_mode_a(mode)
            self.getSparseGraph_mode_b(mode)
            self.n_train = self.train.shape[0]
        else:
            raise NotImplementedError("please select the mode in: [full, retraining]")

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_users + self.n_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().cuda())
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))

    def getSparseGraph_mode_a(self, mode):
        pos_train = self.train[self.train['label'] > 0].values

        self.trainUser = self.train['user'].values.squeeze()
        self.trainItem = self.train['item']

        UserItemNet = csr_matrix((np.ones(pos_train.shape[0]), (pos_train[:, 0], pos_train[:, 1])), shape=(self.n_users, self.n_items))
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat_' + mode + '.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat_' + mode + '.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().cuda()
                print("don't split the matrix")
    
    def getSparseGraph_mode_b(self, mode, ratio_flag=0.01, recomputed=False):
        '''
        computed the changed graph, only considering the item that will be influenced
        '''
        print("loading adjacency matrix")
        if self.ChangedGraph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_changed_adj_mat_'+ mode +'.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
                self.changed_data = pd.read_csv(self.path+'/changed_data_'+mode+str(ratio_flag)+'.csv')
                if recomputed:
                    a_ = [1]
                    print(a_[3])
            except :
                print("generating adjacency matrix")
                s = time()   
                train_normal_pos = self.train_normal[self.train_normal['label']>0].values[:, 0:3]
                train_unlearn_pos = self.train_random[self.train_random['label']>0].values[:, 0:3]
                train_all_pos = np.concatenate([train_normal_pos, train_unlearn_pos], axis=0)

                UserItemNet_all = csr_matrix((np.ones(train_all_pos.shape[0]), (train_all_pos[:,0],train_all_pos[:,1])),shape=(self.n_users,self.n_items))
                UserItemNet_normal = csr_matrix((np.ones(train_normal_pos.shape[0]), (train_normal_pos[:,0], train_normal_pos[:,1])), shape=(self.n_users,self.n_items))

                # matrix all
                adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R_all = UserItemNet_all.tolil()
                adj_mat[:self.n_users, self.n_users:] = R_all
                adj_mat[self.n_users:, :self.n_users] = R_all.T
                adj_mat = adj_mat.todok()
                rowsum_all = np.array(adj_mat.sum(axis=1))

                # matrix changed
                adj_mat_ch = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
                adj_mat_ch = adj_mat_ch.tolil()
                R = UserItemNet_normal.tolil()
                adj_mat_ch[:self.n_users, self.n_users:] = R
                adj_mat_ch[self.n_users:, :self.n_users] = R.T

                unlearn_user = train_unlearn_pos[:,0]
                unlearn_item = train_unlearn_pos[:,1] + self.n_users
                unlearned_idx = np.concatenate([np.unique(unlearn_user), np.unique(unlearn_item)], axis=-1).astype(np.int32)
                ch_adj_mat = adj_mat_ch.todok()
                rowsum = np.array(ch_adj_mat.sum(axis=1))

                # changed information:
                changed_ratio = 1 -  rowsum * 1.0 / (rowsum_all+1e-9)
                nonzero_idx = np.where(rowsum_all==0)[0]
                changed_ratio = changed_ratio.squeeze()
                changed_ratio[nonzero_idx] = 0.0
                print("changed ratio info: (mean, median, max, min):(%.5f, %.5f, %.5f, %.5f)"%(changed_ratio[unlearned_idx].mean(), np.median(changed_ratio[unlearned_idx]), changed_ratio[unlearned_idx].max(), changed_ratio[unlearned_idx].min()))
                # generated changed data
                changed_ui = unlearned_idx[np.where(changed_ratio[unlearned_idx]>ratio_flag)]
                changed_u = changed_ui[np.where(changed_ui< self.n_users)]
                changed_i = changed_ui[np.where(changed_ui>=self.n_users)] - self.n_users
                train_ch_u = self.train_normal[self.train_normal['user'].isin(changed_u)]
                train_ch_i = self.train_normal[self.train_normal['item'].isin(changed_i)]
                self.changed_data = pd.concat([train_ch_u,train_ch_i],axis=0,ignore_index=True).drop_duplicates(keep='last',ignore_index=True)

                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(ch_adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_changed_adj_mat_'+mode+'.npz', norm_adj)
                self.changed_data.to_csv(self.path+'/changed_data_'+mode+str(ratio_flag)+'.csv',index=False)
            if self.split == True:
                self.ChangedGraph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.ChangedGraph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.ChangedGraph = self.ChangedGraph.coalesce().cuda()
                print("don't split the matrix")

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1, ))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    def generate_train_dataloader(self, batch_size=1024):
        '''
        generate minibatch data for full training and retrianing
        '''
        data = torch.from_numpy(self.train[['user', 'item', 'label']].values)
        train_loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=batch_size, drop_last=False, num_workers=2)
        return train_loader

