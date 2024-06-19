import numpy as np
from utility.load_data import *
import os
import sys
import pickle
import torch
from sklearn.metrics import roc_auc_score
from time import time
import random
from Model.Eraser import RecEraser_LightGCN
from utility.compute import *


class model_hyparameters(object):
    def __init__(self):
        super().__init__()
        self.lr = 1e-3
        self.regs = 0
        self.regs_agg = 0
        self.embed_size = 48
        self.batch_size = 2048
        self.epoch = 5000
        self.data_path = 'Data/Process/'
        self.dataset = 'BookCrossing'
        self.attack = '0.02'
        self.pretrain = 0
        self.n_layers = 1
        self.verbose = 1
        self.data_type = 'full'
        self.save_flag = 1
        self.drop_prob = 0
        self.biased = False
        self.init_std = 1e-3
        self.part_type = 1  # 0: whole data, 1: interaction_based, 2: user_based, 3: random
        self.part_num = 10  # partition number
        self.part_T = 50  # iteraction for partition
        self.seed = 1024

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
        else:
            if eval_metric[self.refer_metric] > self.best_eval_result[self.refer_metric]:
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

    def re_init(self, stop_condition=None):
        self.best_epoch = 0
        self.best_eval_result = None
        self.not_change = 0
        self.init_flag = True
        if stop_condition is not None:
            self.stop_condition = stop_condition


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

    config = dict()
    data_generator = Data_for_RecEraser_LightGCN(args.data_path + args.dataset + '/' + args.attack, args.batch_size, args.part_type, args.part_num, args.part_T)
    data_generator.set_train_mode(args.data_type)
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    total_time = [0] * (args.part_num + 1)

    model = RecEraser_LightGCN(config, args).cuda()
    model.Graph = data_generator.Graph
    e_stoper = early_stoper(refer_metric='valid_auc', stop_condition=10)

    opt_list = [torch.optim.Adam(model.parameters(), lr=args.lr) for i in range(args.part_num)]
    opt_list.append(torch.optim.Adam(model.parameters(), lr=args.lr))

    valid_data = data_generator.valid[['user', 'item', 'label']].values
    test_data = data_generator.test[['user', 'item', 'label']].values
    mask = get_eval_mask(data_generator)

    if args.save_flag == 1:
        weights_save_path = './Weights/LightGCN_SISA/lightgcn'
        for name_, val_ in config_args.items():
            weights_save_path += name_ + '_' + str(val_) + '_'
        ensureDir(weights_save_path)

    for i in range(args.part_num):
        e_stoper.re_init()  # re-init the early-stopper
        weights_save_path_local = weights_save_path + "-local-" + str(i) + ".pk"

        print("start trainig %d-th sub-model" % (i))
        train_loader_i = data_generator.batch_generator_local(local_id=i)
        for epoch in range(args.epoch):
            t1 = time()
            loss, bce_loss, reg_loss = 0., 0., 0.
            model.train()
            for batch_data in train_loader_i:
                users, items, labels = batch_data[:, 0].cuda().long(), batch_data[:, 1].cuda().long(), batch_data[:, 2].cuda().float()
                model.zero_grad()
                batch_loss, batch_bce_loss, batch_reg_loss, _ = model.single_model(users, items, labels, i)
                batch_loss.backward()
                opt_list[i].step()
                loss += batch_loss
                bce_loss += batch_bce_loss
                reg_loss += batch_reg_loss

            if torch.isnan(loss) == True:
                print('ERROR: loss is nan.')
                sys.exit()

            t2 = time()
            valid_predictions = model.single_prediction(valid_data[:, 0], valid_data[:, 1], i)
            valid_auc = roc_auc_score(valid_data[:, -1], valid_predictions)
            test_predictions = model.single_prediction(test_data[:, 0], test_data[:, 1], i)
            test_auc = roc_auc_score(test_data[:, -1], test_predictions)
            one_result = {'valid_auc': valid_auc, 'test_auc': test_auc}

            t3 = time()
            if args.verbose > 0:
                perf_str = '[local_model %d] epoch %d [%.4fs + %.4fs]: train_loss=[%.4f=%.4f + %.4f], [valid,test] auc=[%.4f, %.4f]' \
                            % \
                           (i, epoch, t2 - t1, t3 - t2, loss, bce_loss, reg_loss, valid_auc, test_auc)
                print(perf_str)

            is_best = e_stoper.update_and_isbest(one_result, epoch)
            if is_best:
                total_time[i] += (t2-t1)
                user_emb = model.user_embedding.weight[:, i * (model.emb_dim):(i + 1) * model.emb_dim].detach().cpu().numpy()
                item_emb = model.item_embedding.weight[:, i * (model.emb_dim):(i + 1) * model.emb_dim].detach().cpu().numpy()
                with open(weights_save_path_local, 'wb') as f:
                    pickle.dump((user_emb, item_emb), f)

            if e_stoper.is_stop():
                break

    user_emb = []
    item_emb = []
    for i in range(args.part_num):
        weights_save_path_local = weights_save_path + "-local-" + str(i) + ".pk"
        with open(weights_save_path_local, 'rb') as f:
            emb1, emb2 = pickle.load(f)
            user_emb.append(emb1)
            item_emb.append(emb2)
    user_emb = np.concatenate(user_emb, axis=-1)
    item_emb = np.concatenate(item_emb, axis=-1)
    user_emb = torch.from_numpy(user_emb).float().cuda()
    item_emb = torch.from_numpy(item_emb).float().cuda()
    with torch.no_grad():
        model.user_embedding.weight.copy_(user_emb.data)
        model.item_embedding.weight.copy_(item_emb.data)

    #train agg
    e_stoper.re_init()  # re-init the early-stopper
    agg_train_loader = data_generator.batch_generator()
    for epoch in range(args.epoch):
        t1 = time()
        loss, bce_loss, reg_loss, attention_loss = 0., 0., 0., 0.
        model.train()
        for batch_data in agg_train_loader:
            users, items, labels = batch_data[:, 0].cuda().long(), batch_data[:, 1].cuda().long(), batch_data[:, 2].cuda().float()
            model.zero_grad()
            batch_loss, batch_bce_loss, batch_reg_loss, _, _, _ = model.compute_agg_model(users, items, labels)
            batch_loss.backward()
            opt_list[-1].step()
            loss += batch_loss
            bce_loss += batch_bce_loss
            reg_loss += batch_reg_loss

        if torch.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        t2 = time()
        valid_auc, valid_auc_or, valid_auc_and, test_auc, test_auc_or, test_auc_and = get_eval_result(data_generator, model, mask)
        one_result = {'valid_auc': valid_auc, 'test_auc': test_auc}

        t3 = time()
        if args.verbose > 0:
            print("epoch: %d, time: %.6f, valid auc:%.6f, valid auc or:%.6f, valid auc and:%.6f, test auc:%.6f, test auc or:%.6f, test auc and:%.6f" %(epoch, t3-t2, valid_auc, valid_auc_or, valid_auc_and, test_auc, test_auc_or, test_auc_and))
        is_best = e_stoper.update_and_isbest(one_result, epoch)
        if is_best:
            total_time[-1] += (t2-t1)
            torch.save(model.state_dict(), weights_save_path + "-m.pth")

        if e_stoper.is_stop():
            print("best epoch: ", e_stoper.best_epoch)
            break


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    config = {
        'lr': 1e-4,
        'embed_size': 32,
        'batch_size': 2048,
        'data_type': 'retraining',
        'init_std': 1e-4,
        'dataset': 'Amazon',
        'attack': '0.01',
        'seed': 1024,
        'part_type': 3,  # 0: whole data, 1: interaction_based, 2: user_based, 3: random
    }
    main(config)