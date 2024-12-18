# 手动添加未记录的模型

import os
import re
import glob
import torch
import pickle
import random
import torchvision
from torch import nn
from tqdm import tqdm
from nats_bench import create
from xautodl.models import get_cell_based_tiny_net
from utils.model_utils import load_data, test_model, convert_conv2d_to_qconv2d, get_network, convert_linear_to_qlinear, find_nor_conv_positions
from utils.bitassign_utils_origin import MixBitAssign


if __name__ =="__main__":

    train_loader, valid_loader, test_loader = load_data('cifar10', '~/dataset')
    print('Dataset prepared.')

    api = create('/home/dell/dataset/NATS-tss-v1_0-3ffb9-simple', 'tss', fast_mode=True, verbose=False)

    yaml_path = '/home/dell/MP-NAS-Bench201/config/'
    meta_data_filename = '/home/dell/MP-NAS-Bench201/results/meta_data_mp_info.pkl'
    model_save_dir = '/home/dell/MP-NAS-Bench201/results/models'

    # 超参数设置
    H0 = {'epochs': 50, 'lr': 0.001, 'batch_size': 256}
    H1 = {'epochs': 50, 'lr': 0.01, 'batch_size': 256}
    H2 = {'epochs': 80, 'lr': 0.01, 'batch_size': 256}
    H3 = {'epochs': 90, 'lr': 0.01, 'batch_size': 256}
    total_model = 50
    target_H = H1

    match_bit_table = {'stem': {'qconv': {'w_bit': 8, 'a_bit': 8}},
    'classifier': {'qlinear': {'w_bit': 8, 'a_bit': 8}},
    'resblock': {'conv_a': {'w_bit': 8, 'a_bit': 8}, 'conv_b': {'w_bit': 8, 'a_bit': 8}, 'downsample': {'w_bit': 8, 'a_bit': 8}}}

    # 使用glob找出所有匹配的文件
    pattern = os.path.join(model_save_dir, 'test_50_quant_cifar10_model_*_*.pth')
    files = glob.glob(pattern)

    count = 0
    extracted_parts = []
    meta_data_dict = {}

    with open(meta_data_filename, 'rb') as f:
        meta_data_dict = pickle.load(f)

    for file_path in files:
        # 提取文件名中的特定部分
        match = re.search(r'test_50_quant_cifar10_model_([0-9]+_[0-9]+).pth', file_path)
        if match:
            dict_idx = match.group(1)
            if dict_idx not in meta_data_dict.keys():
                print(f"Reload model index: {dict_idx}")
                tempt = dict_idx.split('_')
                model_idx = int(tempt[0])

                info = api.query_by_index(model_idx, hp = 200)

                cell_arch_str = info.arch_str
                print(cell_arch_str)
                conv_positions = find_nor_conv_positions(cell_arch_str)

                config = api.get_net_config(model_idx, 'cifar10')
                model = get_cell_based_tiny_net(config)
                convert_conv2d_to_qconv2d(model, w_bit=8, a_bit=8 )
                convert_linear_to_qlinear(model, w_bit=8, a_bit=8)

                model_path = os.path.join(model_save_dir, file_path)

                bit_assigner = MixBitAssign(model, model_idx, target_H, conv_positions, yaml_path = yaml_path, meta_data_filename = meta_data_filename)
                bit_assigner.set_dict_index(dict_idx)
                if not bit_assigner.load_yaml(match_bit_table):
                    print('Don\'t match, get next one.')
                    continue
                bit_assigner.apply_bitwidth_to_model()
                model = bit_assigner.get_model()
                model.load_state_dict(torch.load(model_path))

                acc = test_model(model, test_loader, len(test_loader)*target_H['batch_size'])

                # # Save results
                bit_assigner.generate_meta_data(acc)

                count += 1

    print('Reload {} models. '.format(count))
