import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class LightGCN(nn.Module):
    def __init__(self, config: dict, dataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.n_items
        self.latent_dim = self.config.embed_size  #['latent_dim_rec']
        self.n_layers = self.config.gcn_layers  #['lightGCN_n_layers']
        self.keep_prob = self.config.keep_prob  #['keep_prob']
        self.A_split = self.config.A_split  #['A_split']
        self.dropout_flag = self.config.dropout
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config.pretrain == 0:
            nn.init.normal_(self.embedding_user.weight, std=self.config.init_std)
            nn.init.normal_(self.embedding_item.weight, std=self.config.init_std)
            print('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.Graph
        print(f"lightgcn is already to go(dropout:{self.config.dropout})")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        if self.dropout_flag:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def F_computer(self,users_emb,items_emb,adj_graph):
        """
        propagate methods for lightGCN
        """       
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.dropout_flag:
            if self.training:
                print("droping")
                raise NotImplementedError("dropout methods are not implemented")
            else:
                g_droped = adj_graph        
        else:
            g_droped = adj_graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        return users_emb, items_emb

    def compute_bce_loss(self, users, items, labels):
        (users_emb, items_emb) = self.getEmbedding(users.long(), items.long())
        matching = torch.mul(users_emb, items_emb)
        scores = torch.sum(matching, dim=-1)
        bce_loss = F.binary_cross_entropy_with_logits(scores, labels, reduction='mean')
        return bce_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma

    def predict(self, users, items):
        users = torch.from_numpy(users).long().cuda()
        items = torch.from_numpy(items).long().cuda()
        with torch.no_grad():
            all_user_emb, all_item_emb = self.computer()
            users_emb = all_user_emb[users]
            items_emb = all_item_emb[items]
            inner_pro = torch.mul(users_emb, items_emb).sum(dim=-1)
            scores = torch.sigmoid(inner_pro)
        return scores.cpu().numpy()