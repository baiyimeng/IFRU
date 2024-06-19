import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F


class RecEraser_MF(nn.Module):
    def __init__(self, data_config, args):
        super(RecEraser_MF, self).__init__()
        self.model_type = 'RecEraser_MF'
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.attention_size = args.embed_size / 2
        self.batch_size = args.batch_size
        self.bias_flag = args.biased
        self.decay = args.regs
        self.decay_agg = args.regs_agg
        self.verbose = args.verbose
        self.num_local = args.part_num
        self.drop_prob = args.drop_prob
        self.init_std = args.init_std
        self._init_weights()

    def emb_lookup(self, users, items, local_id=None):
        users_emb = self.user_embedding(users)
        items_emb = self.item_embedding(items)
        if local_id is None:
            return users_emb, items_emb
        else:
            return users_emb[:, local_id * self.emb_dim:(local_id + 1) * self.emb_dim], items_emb[:, local_id * self.emb_dim:(local_id + 1) * self.emb_dim]

    def _init_weights(self):
        all_weights = dict()

        self.user_embedding = nn.Embedding(self.n_users, self.num_local * self.emb_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.num_local * self.emb_dim)

        nn.init.normal_(self.user_embedding.weight, std=self.init_std)
        nn.init.normal_(self.item_embedding.weight, std=self.init_std)

        if self.bias_flag:
            self.local_biases = nn.parameter.Parameter(torch.zeros(self.num_local))
            self.agg_bias = nn.parameter.Parameter(torch.zeros(1))

        all_weights['WA'] = nn.parameter.Parameter(torch.empty([self.emb_dim, int(self.attention_size)]))
        nn.init.trunc_normal_(all_weights['WA'], mean=0, std=math.sqrt(2.0 / (self.attention_size + self.emb_dim)))
        all_weights['BA'] = nn.parameter.Parameter(torch.zeros((int(self.attention_size), )))
        all_weights['HA'] = nn.parameter.Parameter(torch.empty([int(self.attention_size), 1]))
        nn.init.constant_(all_weights['HA'], 0.01)
        all_weights['WB'] = nn.parameter.Parameter(torch.empty([self.emb_dim, int(self.attention_size)]))
        nn.init.trunc_normal_(all_weights['WB'], mean=0, std=math.sqrt(2.0 / (self.attention_size + self.emb_dim)))
        all_weights['BB'] = nn.parameter.Parameter(torch.zeros([int(self.attention_size)]))
        all_weights['HB'] = nn.parameter.Parameter(torch.empty([int(self.attention_size), 1]))
        nn.init.constant_(all_weights['HB'], 0.01)
        all_weights['trans_W'] = nn.parameter.Parameter(torch.empty([self.num_local, self.emb_dim, self.emb_dim]))
        nn.init.xavier_uniform_(all_weights['trans_W'])
        all_weights['trans_B'] = nn.parameter.Parameter(torch.empty(self.num_local, self.emb_dim))
        nn.init.xavier_uniform_(all_weights['trans_B'])
        self.weights = nn.ParameterDict(all_weights)

    def compute_bce_loss(self, users, items, labels, bias=None):
        if bias is None:
            predicts = torch.mul(users, items).sum(dim=-1)
        else:
            predicts = torch.mul(users, items).sum(dim=-1) + bias
        bce_loss = F.binary_cross_entropy_with_logits(predicts, labels)
        reg_loss = (users**2).sum() + (items**2).sum()
        reg_loss = self.decay * reg_loss / users.shape[0]
        return bce_loss, reg_loss

    def batch_rating_local(self, users, items, local_num):
        self.eval()
        if not isinstance(items, torch.Tensor):
            items = torch.from_numpy(np.array(items)).cuda()
        if not isinstance(users, torch.Tensor):
            users = torch.from_numpy(np.array(users)).cuda()
        u_e, i_e = self.emb_lookup(users, items, local_id=local_num)
        if self.bias_flag:
            return torch.matmul(u_e, i_e.T) + self.local_biases[local_num]
        else:
            return torch.matmul(u_e, i_e.T)

    def single_prediction(self, users, items, local_num):
        self.eval()
        users = torch.from_numpy(users).cuda().long()
        items = torch.from_numpy(items).cuda().long()
        u_e, i_e = self.emb_lookup(users, items, local_id=local_num)
        if self.bias_flag:
            predictions = torch.multiply(u_e, i_e).sum(dim=-1) + self.local_biases[local_num]
        else:
            predictions = torch.multiply(u_e, i_e).sum(dim=-1)
        return predictions.detach().cpu().numpy()

    def single_model(self, users, items, labels, local_num):
        u_e, i_e = self.emb_lookup(users, items, local_id=local_num)
        if self.bias_flag:
            bce_loss, reg_loss = self.compute_bce_loss(u_e, i_e, labels, bias=self.local_biases[local_num])
        else:
            bce_loss, reg_loss = self.compute_bce_loss(u_e, i_e, labels)
        loss = bce_loss + reg_loss
        return loss, bce_loss, reg_loss, 0

    def attention_based_agg(self, embs, flag):
        if flag == 0:
            embs_w = torch.exp(torch.einsum('abc,ck->abk', F.relu(torch.einsum('abc,ck->abk', embs, self.weights['WA']) + self.weights['BA']), self.weights['HA']))
            embs_w = torch.div(embs_w, torch.sum(embs_w, dim=1, keepdim=True))
        else:
            embs_w = torch.exp(torch.einsum('abc,ck->abk', F.relu(torch.einsum('abc,ck->abk', embs, self.weights['WB']) + self.weights['BB']), self.weights['HB']))
            embs_w = torch.div(embs_w, torch.sum(embs_w, dim=1, keepdim=True))

        agg_emb = torch.sum(torch.multiply(embs_w, embs), dim=1)
        return agg_emb, embs_w

    def batch_rating_agg(self, users, items):
        self.eval()
        with torch.no_grad():
            u_es, i_es = self.emb_lookup(users, items)
        u_es = torch.einsum('abc,bcd->abd', u_es, self.weights['trans_W']) + self.weights['trans_B']
        i_es = torch.einsum('abc,bcd->abd', i_es, self.weights['trans_W']) + self.weights['trans_B']
        if self.bias_flag:
            return torch.matmul(u_es, i_es.T) + self.agg_bias
        else:
            return torch.matmul(u_es, i_es.T)

    def agg_predict(self, users, items):
        self.eval()
        if not isinstance(users, torch.Tensor):
            users = torch.from_numpy(users).cuda().long()
        if not isinstance(items, torch.Tensor):
            items = torch.from_numpy(items).cuda().long()
        with torch.no_grad():
            u_es, i_es = self.emb_lookup(users, items)
            u_es = u_es.view(-1, self.num_local, self.emb_dim)
            i_es = i_es.view(-1, self.num_local, self.emb_dim)
        u_es = torch.einsum('abc,bcd->abd', u_es, self.weights['trans_W']) + self.weights['trans_B']
        i_es = torch.einsum('abc,bcd->abd', i_es, self.weights['trans_W']) + self.weights['trans_B']
        u_e, u_w = self.attention_based_agg(u_es, 0)
        i_e, i_w = self.attention_based_agg(i_es, 1)
        u_e_drop = F.dropout(u_e, self.drop_prob, training=self.training)
        if self.bias_flag:
            batch_ratings = torch.mul(u_e_drop, i_e).sum(dim=-1) + self.agg_bias
        else:
            batch_ratings = torch.mul(u_e_drop, i_e).sum(dim=-1)
        return batch_ratings.detach().cpu().numpy()

    def compute_agg_model(self, users, items, labels):
        with torch.no_grad():
            u_es, i_es = self.emb_lookup(users, items)
            u_es = u_es.view(-1, self.num_local, self.emb_dim)
            i_es = i_es.view(-1, self.num_local, self.emb_dim)

        u_es = torch.einsum('abc,bcd->abd', u_es, self.weights['trans_W']) + self.weights['trans_B']
        i_es = torch.einsum('abc,bcd->abd', i_es, self.weights['trans_W']) + self.weights['trans_B']

        u_e, u_w = self.attention_based_agg(u_es, 0)
        i_e, i_w = self.attention_based_agg(i_es, 1)
        u_e_drop = F.dropout(u_e, self.drop_prob, training=self.training)

        if self.bias_flag:
            bce_loss, reg_loss = self.compute_bce_loss(u_e_drop, i_e, labels, bias=self.agg_bias)
        else:
            bce_loss, reg_loss = self.compute_bce_loss(u_e_drop, i_e, labels)
        reg_loss = self.decay_agg * (self.l2_loss(self.weights['trans_W']) + self.l2_loss(self.weights['trans_B']))

        batch_ratings = None
        loss = bce_loss + reg_loss
        return loss, bce_loss, reg_loss, 0, batch_ratings, u_w

    def l2_loss(self, x):
        return (x**2).sum() / 2
    
    def predict(self, users, items):
        return self.agg_predict(users, items)
    

class RecEraser_LightGCN(nn.Module):
    def __init__(self, data_config, args):
        super(RecEraser_LightGCN, self).__init__()
        self.model_type = 'RecEraser_LightGCN'
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.attention_size = args.embed_size / 2
        self.batch_size = args.batch_size
        self.n_layers = 1
        self.bias_flag = args.biased
        self.decay = args.regs
        self.decay_agg = args.regs_agg
        self.verbose = args.verbose
        self.num_local = args.part_num
        self.drop_prob = args.drop_prob
        self.init_std = args.init_std
        self.Graph = []
        self._init_weights()

    def emb_lookup(self, users, items, local_id=None):  # 查询某个子模型的最终层emb
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight

        if local_id is None:
            users_emb_final = torch.ones_like(users_emb)
            items_emb_final = torch.ones_like(items_emb)
            for i in range(self.num_local):
                temp_u = users_emb[:, i * self.emb_dim:(i + 1) * self.emb_dim]
                temp_i = items_emb[:, i * self.emb_dim:(i + 1) * self.emb_dim]
                users_emb_final[:, i * self.emb_dim:(i + 1) * self.emb_dim], items_emb_final[:, i * self.emb_dim:(i + 1) * self.emb_dim] = self.computer(temp_u, temp_i, graph=self.Graph[i])
            return users_emb_final[users], items_emb_final[items]
        else:
            temp_u = users_emb[:, local_id * self.emb_dim:(local_id + 1) * self.emb_dim]
            temp_i = items_emb[:, local_id * self.emb_dim:(local_id + 1) * self.emb_dim]
            users_emb_final, items_emb_final = self.computer(temp_u, temp_i, graph=self.Graph[local_id])
            return users_emb_final[users], items_emb_final[items]

    def _init_weights(self):
        all_weights = dict()

        self.user_embedding = nn.Embedding(self.n_users, self.num_local * self.emb_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.num_local * self.emb_dim)
        nn.init.normal_(self.user_embedding.weight, std=self.init_std)
        nn.init.normal_(self.item_embedding.weight, std=self.init_std)

        if self.bias_flag:
            self.local_biases = nn.parameter.Parameter(torch.zeros(self.num_local))
            self.agg_bias = nn.parameter.Parameter(torch.zeros(1))
        all_weights['WA'] = nn.parameter.Parameter(torch.empty([self.emb_dim, int(self.attention_size)]))
        nn.init.trunc_normal_(all_weights['WA'], mean=0, std=math.sqrt(2.0 / (self.attention_size + self.emb_dim)))
        all_weights['BA'] = nn.parameter.Parameter(torch.zeros((int(self.attention_size), )))
        all_weights['HA'] = nn.parameter.Parameter(torch.empty([int(self.attention_size), 1]))
        nn.init.constant_(all_weights['HA'], 0.01)
        all_weights['WB'] = nn.parameter.Parameter(torch.empty([self.emb_dim, int(self.attention_size)]))
        nn.init.trunc_normal_(all_weights['WB'], mean=0, std=math.sqrt(2.0 / (self.attention_size + self.emb_dim)))
        all_weights['BB'] = nn.parameter.Parameter(torch.zeros([int(self.attention_size)]))
        all_weights['HB'] = nn.parameter.Parameter(torch.empty([int(self.attention_size), 1]))
        nn.init.constant_(all_weights['HB'], 0.01)
        all_weights['trans_W'] = nn.parameter.Parameter(torch.empty([self.num_local, self.emb_dim, self.emb_dim]))
        nn.init.xavier_uniform_(all_weights['trans_W'])
        all_weights['trans_B'] = nn.parameter.Parameter(torch.empty(self.num_local, self.emb_dim))
        nn.init.xavier_uniform_(all_weights['trans_B'])

        self.weights = nn.ParameterDict(all_weights)

    def compute_bce_loss(self, users, items, labels, bias=None):
        if bias is None:
            predicts = torch.mul(users, items).sum(dim=-1)
        else:
            predicts = torch.mul(users, items).sum(dim=-1) + bias
        mf_loss = F.binary_cross_entropy_with_logits(predicts, labels)
        reg_loss = (users**2).sum() + (items**2).sum()
        reg_loss = self.decay * reg_loss / users.shape[0]
        return mf_loss, reg_loss

    def batch_rating_local(self, users, items, local_num):
        self.eval()
        if not isinstance(items, torch.Tensor):
            items = torch.from_numpy(np.array(items)).cuda()
        if not isinstance(users, torch.Tensor):
            users = torch.from_numpy(np.array(users)).cuda()
        u_e, i_e = self.emb_lookup(users, items, local_id=local_num)
        if self.bias_flag:
            return torch.matmul(u_e, i_e.T) + self.local_biases[local_num]
        else:
            return torch.matmul(u_e, i_e.T)

    def single_prediction(self, users, items, local_num):
        self.eval()
        users = torch.from_numpy(users).cuda().long()
        items = torch.from_numpy(items).cuda().long()
        u_e, i_e = self.emb_lookup(users, items, local_id=local_num)

        if self.bias_flag:
            predictions = torch.multiply(u_e, i_e).sum(dim=-1) + self.local_biases[local_num]
        else:
            predictions = torch.multiply(u_e, i_e).sum(dim=-1)
        return predictions.detach().cpu().numpy()

    def single_model(self, users, items, labels, local_num):
        u_e, i_e = self.emb_lookup(users, items, local_id=local_num)

        if self.bias_flag:
            mf_loss, reg_loss = self.compute_bce_loss(u_e, i_e, labels, bias=self.local_biases[local_num])
        else:
            mf_loss, reg_loss = self.compute_bce_loss(u_e, i_e, labels)
        loss = mf_loss + reg_loss
        return loss, mf_loss, reg_loss, 0

    def computer(self, users_emb, items_emb, graph):
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(graph, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users, self.n_items])
        return users, items

    def attention_based_agg(self, embs, flag):
        if flag == 0:
            embs_w = torch.exp(torch.einsum('abc,ck->abk', F.relu(torch.einsum('abc,ck->abk', embs, self.weights['WA']) + self.weights['BA']), self.weights['HA']))
            embs_w = torch.div(embs_w, torch.sum(embs_w, dim=1, keepdim=True))
        else:
            embs_w = torch.exp(torch.einsum('abc,ck->abk', F.relu(torch.einsum('abc,ck->abk', embs, self.weights['WB']) + self.weights['BB']), self.weights['HB']))
            embs_w = torch.div(embs_w, torch.sum(embs_w, dim=1, keepdim=True))

        agg_emb = torch.sum(torch.multiply(embs_w, embs), dim=1)
        return agg_emb, embs_w

    def attention_based_agg2(self, embs):

        embs_w = torch.exp(torch.einsum('abc,ck->abk', F.relu(torch.einsum('abc,ck->abk', embs, self.weights['WA']) + self.weights['BA']), self.weights['HA']))
        embs_w = torch.div(embs_w, torch.sum(embs_w, dim=1, keep_dims=True))
        agg_emb = torch.sum(torch.multiply(embs_w, embs), 1)

        return agg_emb, embs_w

    def batch_rating_agg(self, users, items):  # all-ranking prediction
        self.eval()
        with torch.no_grad():
            u_es, i_es = self.emb_lookup(users, items)
        u_es = torch.einsum('abc,bcd->abd', u_es, self.weights['trans_W']) + self.weights['trans_B']
        i_es = torch.einsum('abc,bcd->abd', i_es, self.weights['trans_W']) + self.weights['trans_B']
        if self.bias_flag:
            return torch.matmul(u_es, i_es.T) + self.agg_bias
        else:
            return torch.matmul(u_es, i_es.T)

    def agg_predict(self, users, items):  # generating prediction for different candidates
        self.eval()
        if not isinstance(users, torch.Tensor):
            users = torch.from_numpy(users).cuda().long()
        if not isinstance(items, torch.Tensor):
            items = torch.from_numpy(items).cuda().long()
        with torch.no_grad():
            u_es, i_es = self.emb_lookup(users, items)
            u_es = u_es.view(-1, self.num_local, self.emb_dim)
            i_es = i_es.view(-1, self.num_local, self.emb_dim)
        u_es = torch.einsum('abc,bcd->abd', u_es, self.weights['trans_W']) + self.weights['trans_B']
        i_es = torch.einsum('abc,bcd->abd', i_es, self.weights['trans_W']) + self.weights['trans_B']
        u_e, u_w = self.attention_based_agg(u_es, 0)
        i_e, i_w = self.attention_based_agg(i_es, 1)
        u_e_drop = F.dropout(u_e, self.drop_prob, training=self.training)
        if self.bias_flag:
            batch_ratings = torch.mul(u_e_drop, i_e).sum(dim=-1) + self.agg_bias
        else:
            batch_ratings = torch.mul(u_e_drop, i_e).sum(dim=-1)
        return batch_ratings.detach().cpu().numpy()

    def compute_agg_model(self, users, items, labels):
        with torch.no_grad():
            u_es, i_es = self.emb_lookup(users, items)
            u_es = u_es.view(-1, self.num_local, self.emb_dim)
            i_es = i_es.view(-1, self.num_local, self.emb_dim)

        u_es = torch.einsum('abc,bcd->abd', u_es, self.weights['trans_W']) + self.weights['trans_B']
        i_es = torch.einsum('abc,bcd->abd', i_es, self.weights['trans_W']) + self.weights['trans_B']

        u_e, u_w = self.attention_based_agg(u_es, 0)
        i_e, i_w = self.attention_based_agg(i_es, 1)
        u_e_drop = F.dropout(u_e, self.drop_prob, training=self.training)

        if self.bias_flag:
            mf_loss, reg_loss = self.compute_bce_loss(u_e_drop, i_e, labels, bias=self.agg_bias)
        else:
            mf_loss, reg_loss = self.compute_bce_loss(u_e_drop, i_e, labels)
        reg_loss = self.decay_agg * (self.l2_loss(self.weights['trans_W']) + self.l2_loss(self.weights['trans_B']))
        batch_ratings = None
        loss = mf_loss + reg_loss
        return loss, mf_loss, reg_loss, 0, batch_ratings, u_w

    def l2_loss(self, x):
        return (x**2).sum() / 2

    def predict(self, users, items):
        return self.agg_predict(users, items)
