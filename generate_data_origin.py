# Note: 初始版本的训练集生成代码，只考虑所有cell进行相同位宽的量化
# Date: 2024/04/13

import os
import torch
import pickle
import random
import torchvision
from torch import nn
from tqdm import tqdm
from nats_bench import create
from utils.model_utils import load_data, train_model, test_model, convert_conv2d_to_qconv2d, get_network, convert_linear_to_qlinear, find_nor_conv_positions
from utils.bitassign_utils import MixBitAssign




if __name__ == "__main__":
    # Prepare dataset
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=str, required=True, help='Model range, like 1-1000')
    parser.add_argument('--device', type=str, required=True, help='Device on which the models will be run, set 0 or 1')
    parser.add_argument('--total', type=int, default=50, help='The total number of model to train.')
    args = parser.parse_args()

    split = args.index.split('-')


    train_loader, valid_loader, test_loader = load_data('cifar10', '~/dataset')
    print('Dataset prepared.')

    api = create('/home/dell/dataset/NATS-tss-v1_0-3ffb9-full_cat/NATS-tss-v1_0-3ffb9-full', 'tss', fast_mode=True, verbose=False)

    yaml_path = '/home/dell/MP-NAS-Bench201/config/'
    meta_data_filename = '/home/dell/MP-NAS-Bench201/results/meta_data_mp_info.pkl'
    model_save_dir = '/home/dell/MP-NAS-Bench201/results/models'

    # 超参数设置
    H0 = {'epochs': 50, 'lr': 0.001, 'batch_size': 256}
    H1 = {'epochs': 50, 'lr': 0.01, 'batch_size': 256}
    H2 = {'epochs': 80, 'lr': 0.01, 'batch_size': 256}
    H3 = {'epochs': 90, 'lr': 0.01, 'batch_size': 256}
    total_model = args.total
    target_H = H1

    for i in range(total_model):
        model_idx = random.randint(int(split[0]),int(split[1]))
        # model_idx = 13472
        # 保证同一个模型仅训练一次
        # if model_idx in get_meta_key(pkl_path):
        #     print('Model {} is collected, regenerate.'.format(model_idx))
        #     continue

        info = api.query_by_index(model_idx, hp = 200)

        cell_arch_str = info.arch_str
        print('Arch: {}'.format(cell_arch_str))
        conv_positions = find_nor_conv_positions(cell_arch_str)

        model = get_network(api, model_idx, dataset = 'cifar10', quant = True)
        bit_assigner = MixBitAssign(model, model_idx, target_H, conv_positions, yaml_path = yaml_path, meta_data_filename = meta_data_filename)

        if bit_assigner.generate_random_bitwidth(cell_infer_only = True):  # 确保没有生成重复的位宽选项
            print('Set bit width: {}'.format(bit_assigner.bitwidth_table))
        else:
            print('Data is aready existent, regenerate. ')
            continue

        model_name = bit_assigner.get_dict_index()
        model = bit_assigner.get_model()

        print('Train {}/{} model of index: {}'.format(i+1, total_model, model_name))
        print(args)
        train_model(model, target_H, model_save_dir, train_loader, model_name, device = args.device)
        bit_assigner.save_to_yaml()
        acc = test_model(model, test_loader, len(test_loader)*target_H['batch_size'])

        # # Save results
        bit_assigner.generate_meta_data(acc)

