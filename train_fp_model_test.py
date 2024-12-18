
import os
import torch
import random
import torchvision
import numpy as np
from torch import nn
from tqdm import tqdm
from nats_bench import create
from lib.utils.quantize_utils import calibrate, QConv2d, QLinear
from xautodl.models import get_cell_based_tiny_net
from utils.model_utils import load_data, test_model, train_model, get_network, convert_linear_to_qlinear, find_nor_conv_positions


if __name__ =="__main__":

    # 超参数设置
    H0 = {'dataset': 'cifar10','epochs': 50, 'lr': 0.1, 'batch_size': 256}
    H1 = {'dataset': 'cifar10','epochs': 200, 'lr': 0.1, 'batch_size': 256}
    H2 = {'dataset': 'cifar10','epochs': 150, 'lr': 0.01, 'batch_size': 512}
    H3 = {'dataset': 'cifar10','epochs': 150, 'lr': 0.01, 'batch_size': 1024}
    target_H = H1

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    train_loader, valid_loader, test_loader = load_data('cifar10', '~/dataset', target_H)
    print('Dataset prepared.')

    # api = create('/home/dell/dataset/NATS-tss-v1_0-3ffb9-simple', 'tss', fast_mode=True, verbose=False)
    api = create('/home/dell/dataset/NATS-tss-v1_0-3ffb9-full_cat/NATS-tss-v1_0-3ffb9-full', 'tss', fast_mode=True, verbose=False)

    model_save_dir = '/home/dell/MP-NAS-Bench201/results'
    model_idx = 2566

    config = api.get_net_config(model_idx, 'cifar10')
    model = get_cell_based_tiny_net(config)

    params = api.get_net_param(model_idx, 'cifar10', None, hp = 200)
    model.load_state_dict(next(iter(params.values())))

    print(model)
    for name, param in model.named_parameters():
        print(name, param.size())

    file_path = 'test_{}_fp_cifar10_model_{}.pth'.format(target_H['epochs'],model_idx)

    train_model(model, target_H, model_save_dir, train_loader, model_idx, device = '0')

    print('Validate model {}.'.format(model_idx))
    val_acc, val_loss = test_model(model, valid_loader, len(valid_loader)*target_H['batch_size'])

    print('Test model {}.'.format(model_idx))
    test_acc, test_loss = test_model(model, test_loader, len(test_loader)*target_H['batch_size'])
