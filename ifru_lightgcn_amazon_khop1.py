import os
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from utility.load_data import * 
import scipy.sparse as sp
import torch.nn.functional as F
from Model.MF import MF
import time
from sklearn.metrics import roc_auc_score
import time
from torch.autograd import Variable
from Model.Lightgcn import LightGCN
from utility.compute import *


class model_hyparameters(object):
    def __init__(self):
        super().__init__()
        self.embed_size = 48
        self.batch_size = 2048
        self.epoch = 5000
        self.data_path = 'Data/Process/'
        self.dataset = 'BookCrossing'
        self.attack = '0.02'
        self.k_hop = 0
        self.data_type = 'full'
        self.if_epoch = 5000
        self.if_lr = 1e4
        self.if_init_std = 0
        self.seed = 1024
        self.lr = 1e-3
        self.regs = 0
        self.init_std = 0

        # lightgcn hyper-parameters
        self.gcn_layers = 1
        self.keep_prob = 1
        self.A_n_fold = 10
        self.A_split = False
        self.dropout = False
        self.pretrain = 0
        
    def reset(self, config):
        for name,val in config.items():
            setattr(self,name,val)


class influence_unlearn(nn.Module):
    def __init__(self,save_name,if_epoch=100,if_lr=1e-2,k_hop=1,init_range=1e-4) -> None:
        super(influence_unlearn).__init__()
        self.if_epoch = if_epoch
        self.if_lr = if_lr
        self.k_hop = k_hop
        self.range = init_range
        self.save_name = save_name
        self.p = None

    def compute_hessian_with_test(self, model=None, data_generator=None):
        nei_users, nei_items = compute_neighbor(data_generator, 0)
        nei_users = torch.from_numpy(nei_users).cuda().long()
        nei_items = torch.from_numpy(nei_items).cuda().long()

        mask = get_eval_mask(data_generator)

        valid_auc, valid_auc_or, valid_auc_and, test_auc, test_auc_or, test_auc_and = get_eval_result(data_generator, model, mask)
        print("results before unlearning: valid auc:%.6f, valid auc or:%.6f, valid auc and:%.6f" %(valid_auc, valid_auc_or, valid_auc_and))
        print("results before unlearning: test auc:%.6f, test auc or:%.6f, test auc and:%.6f" %(test_auc, test_auc_or, test_auc_and))
        
        nei_users, nei_items = self.compute_neighbor_influence_clip(data_generator, k_hop=self.k_hop)
        nei_users = torch.from_numpy(nei_users).cuda().long()
        nei_items = torch.from_numpy(nei_items).cuda().long()
        un_u_para = model.embedding_user.weight[nei_users].reshape(-1)
        un_i_para = model.embedding_item.weight[nei_items].reshape(-1)
        u_para_num = un_u_para.shape[0]
        i_para_num = un_i_para.shape[0]

        un_ui_para = torch.cat([un_u_para,un_i_para],dim=-1)
        u_para = model.embedding_user.weight.clone().detach()
        i_para = model.embedding_item.weight.clone().detach()
        u_para[nei_users] = un_ui_para[:u_para_num].reshape(-1, u_para.shape[-1])
        i_para[nei_items] = un_ui_para[u_para_num:].reshape(-1, i_para.shape[-1])

        def loss_fun(para1, para2):
            u_para,i_para = para1,para2
            learned_data = data_generator.train[['user','item','label']].values
            learned_data = torch.from_numpy(learned_data).cuda()
            user_embs = u_para[learned_data[:,0].long()]
            item_embs = i_para[learned_data[:,1].long()]
            scores = torch.mul(user_embs, item_embs).sum(dim=-1)
            bce_loss = F.binary_cross_entropy_with_logits(scores, learned_data[:,-1].float(), reduction='mean')
            return bce_loss
        
        def unlearn_loss_fun(para1,para2):
            u_para, i_para = para1, para2
            rm_data = data_generator.train_random.values
            rm_data = torch.from_numpy(rm_data).cuda()
            adj_graph = model.Graph
            user_emb_gnn, item_emb_gnn = model.F_computer(u_para, i_para, adj_graph)
            user_embs_rm = user_emb_gnn[rm_data[:,0].long()]
            item_embs_rm = item_emb_gnn[rm_data[:,1].long()]
            scores = torch.mul(user_embs_rm, item_embs_rm).sum(dim=-1)
            rm_loss = F.binary_cross_entropy_with_logits(scores, rm_data[:,2].float(), reduction='sum')
            
            # the loss of changed data
            ### original loss
            ch_data = data_generator.changed_data.values[:,0:3]
            ch_data = torch.from_numpy(ch_data).cuda()
            user_emb_ch1 = user_emb_gnn[ch_data[:,0].long()]
            item_emb_ch1 = item_emb_gnn[ch_data[:,1].long()]
            ch_score1 = torch.mul(user_emb_ch1,item_emb_ch1).sum(dim=-1)
            ch_loss_1 = F.binary_cross_entropy_with_logits(ch_score1, ch_data[:,2].float(), reduction='sum')
            ### changed loss
            ch_graph = data_generator.ChangedGraph
            user_emb_gnn, item_emb_gnn = model.F_computer(u_para, i_para, ch_graph)
            user_emb_ch2 = user_emb_gnn[ch_data[:,0].long()]
            item_emb_ch2 = item_emb_gnn[ch_data[:,1].long()]
            ch_score_2 = torch.mul(user_emb_ch2,item_emb_ch2).sum(dim=-1)
            ch_loss_2 = F.binary_cross_entropy_with_logits(ch_score_2, ch_data[:,2].float(), reduction='sum')
            ### loss for changed data
            ch_loss = ch_loss_1 - ch_loss_2
            ### total unlearn loss
            unlearn_loss = rm_loss + ch_loss
            return unlearn_loss

        total_loss = loss_fun(u_para, i_para)
        total_grad = torch.autograd.grad(total_loss, un_ui_para, create_graph=True, retain_graph=True)[0].reshape(-1,1)
        unlearn_loss = unlearn_loss_fun(u_para, i_para)
        unlearn_grad = torch.autograd.grad(unlearn_loss, un_ui_para,retain_graph=True)[0].reshape(-1,1)

        def hvp(grad, vec):
            vec = vec.detach()
            prod = torch.mul(vec, grad).sum()
            res = torch.autograd.grad(prod, un_ui_para, retain_graph=True)[0]
            return res.detach()
        def grad_goal(grad, vec):
            return hvp(grad, vec).unsqueeze(-1) - unlearn_grad.detach()
        
        self.p = Variable(torch.empty([unlearn_grad.shape[0],1])).cuda()
        nn.init.uniform_(self.p, -self.range, self.range)
        opt = torch.optim.Adam([self.p], lr=self.if_lr)

        best_auc = 0
        best_epoch = 0
        best_test_auc = 0
        best_nei_auc = 0
        not_change = 0
        t0 = time.time()

        res = 0
        for if_ep in range(self.if_epoch):
            s_time = time.time()
            opt.zero_grad()
            self.p.grad = grad_goal(total_grad, self.p)
            opt.step()
            with torch.no_grad():   
                un_ui_para_temp = un_ui_para + 1.0/data_generator.n_train * self.p.squeeze()
                e_time = time.time()
                model.embedding_user.weight.data[nei_users] = un_ui_para_temp[:u_para_num].reshape(-1, u_para.shape[-1]).data + 0
                model.embedding_item.weight.data[nei_items] = un_ui_para_temp[u_para_num:].reshape(-1, i_para.shape[-1]).data + 0

                valid_auc, valid_auc_or, valid_auc_and, test_auc, test_auc_or, test_auc_and = get_eval_result(data_generator, model, mask)
                
                print("epoch: %d, time: %.6f, valid auc:%.6f, valid auc or:%.6f, valid auc and:%.6f, test auc:%.6f, test auc or:%.6f, test auc and:%.6f" %(if_ep, e_time-s_time, valid_auc, valid_auc_or, valid_auc_and, test_auc, test_auc_or, test_auc_and))
               
                if valid_auc > best_auc:
                    best_auc = valid_auc
                    best_epoch = if_ep
                    best_test_auc = test_auc
                    print("save best model")
                    torch.save(model.state_dict(), self.save_name)
                    not_change = 0
                    res += e_time-s_time
                else:
                    not_change += 1

            if not_change > 10:
                break
            print('time_cost:',time.time()-t0)

    def compute_neighbor_influence_clip(self, data_generator, k_hop=1):
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

        degree = np.array(train_matrix.sum(axis=-1)).squeeze()
        
        unlearn_user = data_generator.train_random['user'].values.reshape(-1)
        unlearn_user, cunt_u = np.unique(unlearn_user, return_counts=True)
        unlearn_item = data_generator.train_random['item'].values.reshape(-1) + data_generator.n_users
        unlearn_item, cunt_i = np.unique(unlearn_item,return_counts=True)

        unlearn_ui = np.concatenate([unlearn_user, unlearn_item], axis=-1)
        unlearn_ui_cunt = np.concatenate([cunt_u, cunt_i], axis=-1)
        degree_k = degree[unlearn_ui]
        neighbor_set = dict(zip(unlearn_ui, unlearn_ui_cunt*1.0/degree_k))
        neighbor_set_list = [neighbor_set]
        pre_neighbor_set = neighbor_set
        print("neighbor_set size:", len(neighbor_set))

        for i in range(k_hop):
            neighbor_set = dict()
            existing_node = list(pre_neighbor_set.keys())
            nonzero_raw, nonzero_col = train_matrix[existing_node].nonzero()
            for kk in range(nonzero_raw.shape[0]):
                out_node =  existing_node[nonzero_raw[kk]]
                in_node = nonzero_col[kk]
                try:
                    neighbor_set[in_node] += pre_neighbor_set[out_node] *1.0/degree[in_node]
                except:
                    neighbor_set[in_node] = pre_neighbor_set[out_node] *1.0/degree[in_node]
            pre_neighbor_set = neighbor_set
            neighbor_set_list.append(neighbor_set)
        
        nei_dict = neighbor_set_list[1].copy()

        nei_weights = np.array(list(nei_dict.values()))
        nei_nodes = np.array(list(nei_dict.keys()))
        quantile_info = [np.quantile(nei_weights,m*0.1) for m in range(1,11)]
        print("quantile information (median 0.1-0.2--->1): ", quantile_info)
        
        select_index = np.where(nei_weights>quantile_info[3])
        neighbor_set = nei_nodes[select_index]
        print("neighbors before filtering:",nei_nodes.shape,"after filtering:", neighbor_set.shape)

        all_nei_ui = np.concatenate([unlearn_ui.squeeze(),neighbor_set.squeeze()])
        all_nei_ui = neighbor_set.squeeze()
        all_nei_ui = np.unique(all_nei_ui)
        print("total influenced users+items:",all_nei_ui.shape)
        return all_nei_ui[np.where(all_nei_ui<data_generator.n_users)], all_nei_ui[np.where(all_nei_ui>=data_generator.n_users)] - data_generator.n_users


def main(config_args):
    args = model_hyparameters()
    assert config_args is not None
    args.reset(config_args)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)
    
    data_generator = Data_for_LightGCN(args, path=args.data_path + args.dataset + '/' + args.attack)
    data_generator.set_train_mode(args.data_type)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    save_name = './Weights/LightGCN/LightGCN_lr-0.0001-embed_size-32-batch_size-2048-data_type-full-dataset-Amazon-attack-0.01-seed-1024-init_std-0.0001-m.pth'
    model = LightGCN(args, dataset=data_generator).cuda()
    model.load_state_dict(torch.load(save_name))

    test_data = data_generator.test[['user','item','label']].values
    test_predictions =  model.predict(test_data[:,0], test_data[:,1])
    test_auc = roc_auc_score(test_data[:,-1],test_predictions)

    valid_data = data_generator.valid[['user','item','label']].values
    valid_predictions =  model.predict(valid_data[:,0], valid_data[:,1])
    valid_auc = roc_auc_score(valid_data[:,-1],valid_predictions)

    print("***************before unlearning*************")
    print("valid auc:",valid_auc)
    print("test auc:",test_auc)
    print(config_args)

    save_name = "Weights/LightGCN_IFRU/lightgcn_dataset_{}_attack_{}_lr_{}_khop_{}_emb_{}.pth".format(args.dataset, args.attack, args.if_lr, args.k_hop, args.embed_size)
    unlearn = influence_unlearn(save_name=save_name,if_epoch=args.if_epoch, if_lr=args.if_lr, k_hop=args.k_hop, init_range=args.if_init_std)
    unlearn.compute_hessian_with_test(model=model,data_generator=data_generator)
    model.load_state_dict(torch.load(save_name))

    test_predictions =  model.predict(test_data[:,0], test_data[:,1])
    test_auc = roc_auc_score(test_data[:,-1],test_predictions)
    valid_predictions =  model.predict(valid_data[:,0], valid_data[:,1])
    valid_auc = roc_auc_score(valid_data[:,-1],valid_predictions)
    print("***************after unlearning***************")
    print("valid auc:",valid_auc)
    print("test auc:",test_auc)

if __name__=='__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    config_args = {}
    config_args['embed_size'] = 32
    config_args['batch_size'] = 2048
    config_args['epoch'] = 5000
    config_args['data_path'] = 'Data/Process/'
    config_args['dataset'] = 'Amazon'
    config_args['attack'] = '0.01'
    config_args['k_hop'] = 1
    config_args['data_type'] = 'full'
    config_args['if_epoch'] = 5000
    config_args['if_lr'] = 1e4
    config_args['if_init_std'] = 0
    config_args['seed'] = 1024
    config_args['lr'] = 1e-4
    config_args['regs'] = 0
    config_args['init_std'] = 1e-4
    main(config_args)