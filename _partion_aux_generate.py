import torch
import numpy as np
import pickle
from Model.MF import MF
from utility.load_data import *

load_path = '/home/baiyimeng/2023/IFRU_copy/Weights/MF/MF_lr-0.001-embed_size-48-batch_size-2048-data_type-full-dataset-BookCrossing-attack-0.02-seed-1024-init_std-0.001-m.pth'
data_path = './Data/Process/'
dataset = 'BookCrossing'  # 'Amazon'
attack = '0.02'  # '0.01'
embed_size = 48


if __name__ == '__main__':
    data_generator = Data_for_MF(data_path=data_path + dataset + '/' + attack, batch_size=2048)
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    class MyObject:
        def __init__(self):
            self.lr = 1e-3
            self.embed_size = embed_size
            self.batch_size = 2048
            self.regs = 0
            self.init_std = 1e-4
    args = MyObject()

    model = MF(data_config=config, args=args).cuda()
    model.load_state_dict(torch.load(load_path))

    user_pretrain = model.user_embeddings.weight.cpu().detach().numpy()
    item_pretrain = model.item_embeddings.weight.cpu().detach().numpy()
    with open(data_path + dataset + '/' + attack + '/user_pretrain.pk','wb') as f:
        pickle.dump(user_pretrain,f)
    with open(data_path + dataset + '/' + attack + '/item_pretrain.pk','wb') as f:
        pickle.dump(item_pretrain,f)