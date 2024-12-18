# 手动添加未记录的模型

import os
import re
import glob
import torch
import pickle
import random
import torchvision
import numpy as np
from torch import nn
from tqdm import tqdm
from nats_bench import create
from xautodl.models import get_cell_based_tiny_net
from utils.model_utils import load_data, test_model, convert_conv2d_to_qconv2d, get_network, convert_linear_to_qlinear, find_nor_conv_positions
from utils.bitassign_utils import MixBitAssign
from thop import profile

if __name__ =="__main__":

    # 超参数设置
    H0 = {'dataset': 'cifar10','epochs': 50, 'lr': 0.01, 'batch_size': 256}
    H1 = {'dataset': 'cifar10','epochs': 150, 'lr': 0.01, 'batch_size': 256}
    H2 = {'dataset': 'cifar10','epochs': 150, 'lr': 0.01, 'batch_size': 512}
    H3 = {'dataset': 'cifar10','epochs': 150, 'lr': 0.01, 'batch_size': 1024}
    total_model = 50
    target_H = H0

    # train_loader, valid_loader, test_loader = load_data('cifar10', '~/dataset', target_H)
    # print('Dataset prepared.')

    api = create('/home/dell/dataset/NATS-tss-v1_0-3ffb9-simple', 'tss', fast_mode=True, verbose=False)

    yaml_path = '/home/dell/MP-NAS-Bench201/config/'
    yaml_path_new = '/home/dell/MP-NAS-Bench201/results/configs/'
    meta_data_filename = '/home/dell/MP-NAS-Bench201/results/meta_data_mp_info.pkl'
    model_save_dir = '/home/dell/MP-NAS-Bench201/results/models'

    mp_nas_statistics_path = '/home/dell/MP-NAS-Bench201/results/mp_nas_statistics_150epochs.pkl'
    mp_nas_statistics = []


    match_bit_table = {'stem': {'qconv': {'w_bit': 8, 'a_bit': 8}},
    'classifier': {'qlinear': {'w_bit': 8, 'a_bit': 8}},
    'resblock': {
                    5:{'conv_a': {'w_bit': 8, 'a_bit': 8}, 'conv_b': {'w_bit': 8, 'a_bit': 8}, 'downsample': {'w_bit': 8, 'a_bit': 8}}, 
                    11:{'conv_a': {'w_bit': 8, 'a_bit': 8}, 'conv_b': {'w_bit': 8, 'a_bit': 8}, 'downsample': {'w_bit': 8, 'a_bit': 8}}}}

    # 使用glob找出所有匹配的文件
    # pattern = os.path.join(model_save_dir, 'test_50_quant_cifar10_model_*_*.pth')
    # files = glob.glob(pattern)

    count = 0
    count_all = 0
    extracted_parts = []
    meta_data_dict = {}

    yaml_meta_cache_path = '/home/dell/MP-NAS-Bench201/results/0_yaml_cache_meta_info.pkl'

    with open(yaml_meta_cache_path, 'rb') as f:
        yaml_meta_cache = pickle.load(f)

    # for file_path in files:
    #     # 提取文件名中的特定部分
    #     print('{}/{} models been reload. '.format(count_all, len(files)))
    #     count_all += 1
    #     match = re.search(r'test_50_quant_cifar10_model_([0-9]+_[0-9]+).pth', file_path)
    #     if match:
    #         dict_idx = match.group(1)
    for key in yaml_meta_cache.keys():
        if 'test_accuracy' in yaml_meta_cache[key].keys() and yaml_meta_cache[key]['train_info']['epochs'] == 150:
            print(f"Reload model index: {key}")
            tempt = key.split('_')
            model_idx = int(tempt[0])

            # info = api.query_by_index(model_idx, hp = 200)

            cell_arch_str = yaml_meta_cache[key]['arch']
            conv_positions = find_nor_conv_positions(cell_arch_str)

            config = api.get_net_config(model_idx, 'cifar10')
            model = get_cell_based_tiny_net(config)
            
            convert_conv2d_to_qconv2d(model, w_bit=8, a_bit=8 )
            convert_linear_to_qlinear(model, w_bit=8, a_bit=8)
            file_path = 'test_{}_quant_cifar10_model_{}.pth'.format(yaml_meta_cache[key]['train_info']['epochs'],key)

            model_path = os.path.join(model_save_dir, file_path)
            yaml_cache = {}

            bit_assigner = MixBitAssign(model, model_idx, target_H, conv_positions, cell_arch_str, yaml_cache, yaml_path = yaml_path_new)

            bit_assigner.set_dict_index(key)
            if not bit_assigner.load_yaml(match_bit_table):
                continue
            bit_assigner.apply_bitwidth_to_model()
            model = bit_assigner.get_model()
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path))
            else:
                print("Can not find model path: {}".format(model_path))
                continue

            # 打印model所有参数尺寸和类型
            # print(model)
            # for name, param in model.named_parameters():
            #     print(name, param.size())


            if 'test_accuracy' in yaml_meta_cache[key].keys() and yaml_meta_cache[key]['train_info']['epochs'] == 150: # cell_group, cell_uniform_op_random
                # avg_bit = []
                # for bit in yaml_meta_cache[key]['bit_width']['infercell'].keys():
                #     for b in yaml_meta_cache[key]['bit_width']['infercell'][bit].keys():
                #         avg_bit.append(yaml_meta_cache[key]['bit_width']['infercell'][bit][b]['w_bit'])
                # if(len(avg_bit)):
                #     avg_bit = np.mean(avg_bit)
                #     print('Average bit:',avg_bit)
                param = sum(p.numel() for p in model.parameters() if p.requires_grad)/1000000.0
                print('Parameters:',param, 'M')
                # 统计网络参数类型

                model_size = bit_assigner.params/10e6

                print('Got param:', model_size, 'MB')
                avg_bit = bit_assigner.params/bit_assigner.params_count*8
                print('Average bit:',avg_bit)

                mp_nas_statistics.append((key, avg_bit, param, bit_assigner.params/10e6, yaml_meta_cache[key]['test_accuracy']))

            # input = torch.randn(1, 3, 32, 32)
            # flops, _ = profile(model, inputs=(input,))
            # print('FLOPs:',flops/10e6, 'M')
            # bit_assigner.save_size_info(model_size, param, FLOPs, MACs)
            

            # print('Validate model {}.'.format(dict_idx))
            # val_acc, val_loss = test_model(model, valid_loader, len(valid_loader)*target_H['batch_size'])

            # print('Test model {}.'.format(dict_idx))
            # test_acc, test_loss = test_model(model, test_loader, len(test_loader)*target_H['batch_size'])

            # # # Save results
            # bit_assigner.save_to_yaml(val_acc, val_loss, test_acc, test_loss)
            count += 1
            print('Reload {}/{} models. '.format(count,len(yaml_meta_cache)))
            with open(mp_nas_statistics_path, 'wb') as f:
                pickle.dump(mp_nas_statistics, f)
