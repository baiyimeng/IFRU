import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MF(nn.Module):
    def __init__(self, data_config, args):
        super().__init__()
        self.model_type = 'MF'
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.regs = args.regs
        self.decay = args.regs

        self.user_embeddings = nn.Embedding(self.n_users, self.emb_dim)
        self.item_embeddings = nn.Embedding(self.n_items, self.emb_dim)
        nn.init.normal_(self.user_embeddings.weight, std=args.init_std)
        nn.init.normal_(self.item_embeddings.weight, std=args.init_std)

    def forward(self, user, item):
        user_emb = self.user_embeddings(user)
        item_emb = self.item_embeddings(item)
        result = torch.mul(user_emb, item_emb).sum(-1)
        return result

    def predict(self, users, items):
        self.eval()
        users = torch.from_numpy(users).cuda().long()
        items = torch.from_numpy(items).cuda().long()
        with torch.no_grad():
            result = self.forward(users, items)
        return result.cpu().numpy()

    def batch_rating(self, users, candidated_items):
        if isinstance(users, list):
            users = torch.from_numpy(np.array(users)).cuda()
            candidated_items = torch.from_numpy(np.array(candidated_items)).cuda()
        else:
            users = torch.from_numpy(users).cuda()
            candidated_items = torch.from_numpy(candidated_items).cuda()
        self.eval()
        with torch.no_grad():
            user_embed = self.user_embeddings(users)
            item_embed = self.item_embeddings(candidated_items)
            ratings = torch.matmul(user_embed, item_embed.T)
        return ratings.cpu().numpy()

    def train_one_batch_ouput_bce(self, user, items, labels, opt):
        self.train()
        opt.zero_grad()
        user_embs = self.user_embeddings(user)
        item_embs = self.item_embeddings(items)
        scores = torch.mul(user_embs, item_embs).sum(dim=-1)
        regularizer = (user_embs**2).sum() + (item_embs**2).sum()
        regularizer = regularizer / user.shape[0]
        bce_loss = F.binary_cross_entropy_with_logits(scores, labels, reduction='mean')
        reg_loss = self.decay * regularizer
        loss = bce_loss + reg_loss
        loss.backward()
        opt.step()
        return bce_loss.item(), reg_loss.item(), loss.item()

    def _statistics_params(self):
        total_parameters = 0
        for variable in self.parameters:
            shape = variable.shape
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("#params: %d" % total_parameters)