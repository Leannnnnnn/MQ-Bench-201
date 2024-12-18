# Note: 新版本的训练数据生成代码，考虑主干节点，Resblock节点位宽设置，各cell独立分配量化位宽
# Date: 2024/04/13

import os
import re
import glob
import yaml
import torch
import pickle
import random
import torchvision
from torch import nn
from tqdm import tqdm
from nats_bench import create
from utils.model_utils import load_data, train_model, test_model, train_model_with_epoch_list, get_network, find_nor_conv_positions
from utils.bitassign_utils import MixBitAssign
from lib.utils.quantize_utils import calibrate




if __name__ == "__main__":
    # Prepare dataset

    yaml_meta_cache_path = '/home/dell/MP-NAS-Bench201/results/0_yaml_cache_meta_info.pkl'
    yaml_path_new = '/home/dell/MP-NAS-Bench201/results/configs/'

    yaml_meta_cache = {}

    # 使用glob找出所有匹配的文件
    pattern = os.path.join(yaml_path_new, '*_*.yaml')
    files = glob.glob(pattern)

    with open(yaml_meta_cache_path, 'rb') as f:
        yaml_meta_cache = pickle.load(f)


    api = create('/home/dell/dataset/NATS-tss-v1_0-3ffb9-full_cat/NATS-tss-v1_0-3ffb9-full', 'tss', fast_mode=True, verbose=False)

    result_save_path = '/home/dell/MP-NAS-Bench201/results/nb201_acc_reload_info.pkl'

    acc_reload = {}

    if os.path.exists(result_save_path):
        with open(result_save_path, 'rb') as f:
            acc_reload = pickle.load(f)
        

    for model_idx in range(2891, args.index):
        info = api.query_by_index(model_idx, hp = 200)

        cell_arch_str = info.arch_str
        print('Model:{}, Arch: {}'.format(model_idx, cell_arch_str))

        model = get_network(api, model_idx, dataset = target_H['dataset'], quant = False)

        info = api.get_more_info(model_idx, 'cifar10', hp = 200)
        print("Got train+valid acc: {}".format(info['train-accuracy']))
        print("Got test acc: {}".format(info["test-accuracy"]))

        print('Validate model {}.'.format(model_idx))
        # val_acc, val_loss = test_model(model, valid_loader, len(valid_loader)*target_H['batch_size'], args.device)

        val_acc, val_loss = 0.0, 0.0

        print('Test model {}.'.format(model_idx))
        # test_acc, test_loss = test_model(model, test_loader, len(test_loader)*target_H['batch_size'], args.device)
        test_acc, test_loss = 0.0, 0.0

        acc_reload[model_idx] = {'val_accuracy':val_acc, 'val_loss':val_loss, 'test_accuracy':test_acc, 'test_loss':test_loss, 'origin_info':info}

        with open(result_save_path, 'wb') as f:
            pickle.dump(acc_reload, f)
        
        print("Results saved in: ", result_save_path)






