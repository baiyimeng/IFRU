from distutils.command.config import config
import os
import torch
import numpy as np
from utility.load_data import *
import pandas as pd
import sys
from time import time
from sklearn.metrics import roc_auc_score
import random
from Model.MF import MF
from utility.compute import *


class model_hyparameters(object):
    def __init__(self):
        super().__init__()
        self.lr = 1e-3
        self.embed_size = 48
        self.regs = 0
        self.batch_size = 2048
        self.epoch = 5000
        self.data_path = 'Data/Process/'
        self.dataset = 'BookCrossing'
        self.attack = '0.02'
        self.data_type = 'full'
        self.seed = 1024
        self.init_std = 1e-4

    def reset(self, config):
        for name, val in config.items():
            setattr(self, name, val)


class early_stoper(object):
    def __init__(self, refer_metric='valid_auc', stop_condition=10):
        super().__init__()
        self.best_epoch = 0
        self.best_eval_result = None
        self.not_change = 0
        self.stop_condition = stop_condition
        self.init_flag = True
        self.refer_metric = refer_metric

    def update_and_isbest(self, eval_metric, epoch):
        if self.init_flag:
            self.best_epoch = epoch
            self.init_flag = False
            self.best_eval_result = eval_metric
            return True
        elif eval_metric[self.refer_metric] > self.best_eval_result[self.refer_metric]:
            self.best_eval_result = eval_metric
            self.not_change = 0
            self.best_epoch = epoch
            return True
        else:
            self.not_change += 1
            return False

    def is_stop(self):
        if self.not_change > self.stop_condition:
            return True
        else:
            return False


def main(config_args):
    args = model_hyparameters()
    args.reset(config_args)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    save_name = 'MF_'
    for name_str, name_val in config_args.items():
        save_name += name_str + '-' + str(name_val) + '-'

    data_generator = Data_for_MF(data_path=args.data_path + args.dataset + '/' + args.attack, batch_size=args.batch_size)
    data_generator.set_train_mode(args.data_type)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    model = MF(data_config=config, args=args).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=model.lr)

    best_epoch = 0
    best_valid_auc = 0
    best_test_auc = 0
    e_stoper = early_stoper(refer_metric='valid_auc', stop_condition=10)
    mask = get_eval_mask(data_generator)

    for epoch in range(args.epoch):
        t1 = time()
        loss, bce_loss, reg_loss = 0., 0., 0.
        for batch_data in data_generator.batch_generator():
            users, items, labels = batch_data[:, 0].cuda().long(), batch_data[:, 1].cuda().long(), batch_data[:, 2].cuda().float()
            batch_bce_loss, batch_reg_loss, batch_loss = model.train_one_batch_ouput_bce(users, items, labels, opt)
            loss += batch_loss
            bce_loss += batch_bce_loss
            reg_loss += batch_reg_loss

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        t2 = time()
        valid_auc, valid_auc_or, valid_auc_and, test_auc, test_auc_or, test_auc_and = get_eval_result(data_generator, model, mask)

        t3 = time()
        perf_str = "epoch: %d, time: %.6f, valid auc:%.6f, valid auc or:%.6f, valid auc and:%.6f, test auc:%.6f, test auc or:%.6f, test auc and:%.6f" %(epoch, t3-t2, valid_auc, valid_auc_or, valid_auc_and, test_auc, test_auc_or, test_auc_and)
        print(perf_str)

        one_result = {'valid_auc': valid_auc, 'test_auc': test_auc}
        is_best = e_stoper.update_and_isbest(one_result, epoch)
        if is_best:
            best_epoch = epoch
            best_valid_auc = valid_auc
            best_test_auc = test_auc
            torch.save(model.state_dict(), './Weights/MF/' + save_name + "m.pth")
            print("saving the best model")

        if e_stoper.is_stop():
            print("save path for best model:", './Weights/MF/' + save_name + "m.pth")
            break

    final_perf = 'best_epoch = {}, best_valid_auc = {}, best_test_auc = {}'.format(best_epoch, best_valid_auc, best_test_auc)
    print(final_perf)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config = {
        'lr': 1e-4,  # [1e-2, 1e-3, 1e-4]
        'embed_size': 64,  # [32, 48, 64]
        'batch_size': 2048,
        'data_type': 'retraining',
        'dataset': 'Amazon',  #[BookCrossing, Amazon]
        'attack':'0.02',  # [0.02, 0.01]
        'seed': 1024,
        'init_std': 1e-4  # [1e-2, 1e-3, 1e-4]
    }
    main(config)