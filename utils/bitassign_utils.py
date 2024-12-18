# Note: 新版本的位宽分配代码，考虑主干节点，Resblock节点位宽设置，各cell独立分配量化位宽
# Date: 2024/04/13

import os
import re
import glob
import torch
import torch.nn as nn
import yaml
from collections import defaultdict
import pickle
import random

from xautodl.models.cell_operations import ReLUConvBN

class MixBitAssign:
    def __init__(self, model,  model_idx, train_info, conv_positions, cell_arch_str, yaml_cache, bitwidths={'w_bit': 8, 'a_bit': 8}, yaml_path = '~/default/'):
        self.model = model
        self.model_idx = model_idx
        self.dict_idx = str(self.model_idx)+'_1'
        self.arch_str = cell_arch_str
        self.train_info = train_info
        self.yaml_path = yaml_path
        self.table_cache = yaml_cache
        self.conv_positions = conv_positions
        self.params = 0
        self.params_count = 0
        # 初始化位宽配置字典
        self.config_table = {
            'index' : self.dict_idx,
            'arch' : self.arch_str,
            'val_accuracy': 0.9999,
            'val_loss': 0.0001,
            'test_accuracy': 0.9999,
            'test_loss': 0.0001,
            'train_info': train_info,
            'calibrate': False,
            'bit_width':{
                'assign_type': {'cell': 'cell_uniform', 'stem': 'quant_8bit'},
                'stem': {'qconv':{'w_bit': 8, 'a_bit': 8}},
                'infercell': {0:{},1:{},2:{},3:{},4:{},6:{},7:{},8:{},9:{},10:{},12:{},13:{},14:{},15:{},16:{}},
                'resblock': {
                    5:{'conv_a': {'w_bit': 8, 'a_bit': 8}, 'conv_b': {'w_bit': 8, 'a_bit': 8}, 'downsample': {'w_bit': 8, 'a_bit': 8}}, 
                    11:{'conv_a': {'w_bit': 8, 'a_bit': 8}, 'conv_b': {'w_bit': 8, 'a_bit': 8}, 'downsample': {'w_bit': 8, 'a_bit': 8}}},
                'classifier': {'qlinear':{'w_bit': 8, 'a_bit': 8}}
                },
            # 'model_size(MB)': 0.5,
            # 'param(M)': 0.5,
            # 'avg_bit': 4,
            # 'FLOPs(M)':0.5,
            # 'MACs(M)':0.5,
        }
        # 为每个infercell的conv层设置位宽
        for i in [0,1,2,3,4,6,7,8,9,10,12,13,14,15,16]:
            for pos in self.conv_positions:
                self.config_table['bit_width']['infercell'][i][pos] = bitwidths

    def set_calibrate(self):
        self.config_table['calibrate'] = True

    def save_to_yaml(self, dict_name, epoch, val_acc, val_loss, test_acc, test_loss):
        # 保存位宽配置到YAML文件
        self.dict_idx = dict_name
        self.config_table['index'] = dict_name
        self.config_table['train_info']['epochs'] = epoch
        self.config_table['val_accuracy'] = val_acc
        self.config_table['val_loss'] = val_loss
        self.config_table['test_accuracy'] = test_acc
        self.config_table['test_loss'] = test_loss
        path = self.yaml_path+'{}.yaml'.format(self.dict_idx)

        with open(path, 'w') as file:
            yaml.dump(self.config_table, file, default_flow_style=False)
        
        print('Bitwidth configuration saved to {}'.format(path))
    
    def save_size_info(self, model_size, param, FLOPs, MACs):
        # self.config_table['model_size(MB)'] = model_size
        self.config_table['param(M)'] = param
        self.config_table['FLOPs(M)'] = FLOPs
        # self.config_table['MACs(M)'] = MACs
        path = self.yaml_path+'{}.yaml'.format(self.dict_idx)
        with open(path, 'w') as file:
            yaml.dump(self.config_table, file, default_flow_style=False)

    def load_yaml(self, match):
        filename = self.yaml_path+'{}.yaml'.format(self.dict_idx)
        if not os.path.exists(filename):
            print('{} not found.'.format(filename))
            return False
        with open(filename, 'r') as file:
            self.config_table = yaml.safe_load(file)
        if self.config_table['bit_width']['classifier'] != match['classifier'] or self.config_table['bit_width']['stem'] != match['stem'] or self.config_table['bit_width']['resblock'] != match['resblock']:
            print('Don\'t match, get next one.')
            return False
        # 如果 self.config_table有'val_accuracy'键则返回False
        # if 'val_accuracy' in self.config_table:
        #     print('Result exists, get next one.')
        #     return False

        if 'model_size(MB)' in self.config_table:
            print('Result exists, get next one.')
            return False

        self.apply_bitwidth_to_model()
        return True

    def generate_random_bitwidth(self, cell_type = 'cell_uniform', stem_type = 'quant_8bit' ):
        """
        生成随机位宽, 范围在2到8位之间。
        cell_type:
            noquant: 不执行量化，位宽设为-1 
            cell_uniform: 所有cell stack执行相同量化
            cell_uniform_op_random: 所有cell stack执行相同量化, 但cell中的不同op随机选择量化
            cell_separated: 所有cell各自执行不同量化, 但cell中的不同op执行相同量化
            cell_separated_op_random: 所有cell各自执行不同量化, 且cell中的不同op随机选择量化
            cell_group: cell stack分层分组执行不同量化, 但cell中的不同op执行相同量化
            cell_group_op_random: cell stack分层分组量化, 但cell中的不同op随机选择量化
        stem_type:
            noquant: 不执行量化，位宽设为-1 
            quant_separated: 每个主干层分别量化
            quant_8bit: 各主干层保持8bit量化
        """

        if cell_type == 'cell_uniform':
            self.config_table['bit_width']["assign_type"]['cell'] = 'cell_uniform'
            bitwidths = {"w_bit": random.choice([2,4,8]), "a_bit": random.choice([8])}
            for i in [0,1,2,3,4,6,7,8,9,10,12,13,14,15,16]:
                for pos in self.conv_positions:
                    self.config_table['bit_width']['infercell'][i][pos] = bitwidths

        if cell_type == 'cell_uniform_op_random':
            self.config_table['bit_width']["assign_type"]['cell'] = 'cell_uniform_op_random'
            for pos in self.conv_positions:
                bitwidths = {"w_bit": random.choice([2,4,8]), "a_bit": random.choice([8])}
                for n in range(3):
                    for i in range(6*n, 6*n+5):
                        self.config_table['bit_width']['infercell'][i][pos] = bitwidths

        # if cell_type == 'cell_uniform_op_random':
        #     self.config_table['bit_width']["assign_type"]['cell'] = 'cell_uniform_op_random'
        #     for n in range(3):
        #         for i in range(6*n, 6*n+5):
        #             bitwidths = {"w_bit": random.choice([2]), "a_bit": random.choice([8])}
        #             self.config_table['bit_width']['infercell'][i][2] = bitwidths
        #             bitwidths = {"w_bit": random.choice([4]), "a_bit": random.choice([8])}
        #             self.config_table['bit_width']['infercell'][i][3] = bitwidths
        #             bitwidths = {"w_bit": random.choice([2]), "a_bit": random.choice([8])}
        #             self.config_table['bit_width']['infercell'][i][4] = bitwidths
        #             bitwidths = {"w_bit": random.choice([4]), "a_bit": random.choice([8])}
        #             self.config_table['bit_width']['infercell'][i][5] = bitwidths

        if cell_type == 'cell_separated':
            self.config_table['bit_width']["assign_type"]['cell'] = 'cell_separated'
            for i in [0,1,2,3,4,6,7,8,9,10,12,13,14,15,16]:
                bitwidths = {"w_bit": random.choice([2,4,8]), "a_bit": random.choice([8])}
                for pos in self.conv_positions:
                    self.config_table['bit_width']['infercell'][i][pos] = bitwidths

        if cell_type == 'cell_separated_op_random':
            self.config_table['bit_width']["assign_type"]['cell'] = 'cell_separated_op_random'
            for i in [0,1,2,3,4,6,7,8,9,10,12,13,14,15,16]:
                for pos in self.conv_positions:
                    bitwidths = {"w_bit": random.choice([2,4,8]), "a_bit": random.choice([8])}
                    self.config_table['bit_width']['infercell'][i][pos] = bitwidths
        
        if cell_type == 'cell_group':
            self.config_table['bit_width']["assign_type"]['cell'] = 'cell_group'
            for n in range(3):
                bitwidths = {"w_bit": random.choice([2,4,8]), "a_bit": random.choice([8])}
                for i in range(6*n, 6*n+5):
                    for pos in self.conv_positions:
                        self.config_table['bit_width']['infercell'][i][pos] = bitwidths

        # if cell_type == 'cell_group':
        #     self.config_table['bit_width']["assign_type"]['cell'] = 'cell_group'
        #     n = 0
        #     bitwidths = {"w_bit": random.choice([8]), "a_bit": random.choice([8])}
        #     for i in range(6*n, 6*n+5):
        #         for pos in self.conv_positions:
        #             self.config_table['bit_width']['infercell'][i][pos] = bitwidths
        #     n = 1
        #     bitwidths = {"w_bit": random.choice([8]), "a_bit": random.choice([8])}
        #     for i in range(6*n, 6*n+5):
        #         for pos in self.conv_positions:
        #             self.config_table['bit_width']['infercell'][i][pos] = bitwidths
        #     n = 2
        #     bitwidths = {"w_bit": random.choice([2]), "a_bit": random.choice([8])}
        #     for i in range(6*n, 6*n+5):
        #         for pos in self.conv_positions:
        #             self.config_table['bit_width']['infercell'][i][pos] = bitwidths
                
        if cell_type == 'cell_group_op_random':
            self.config_table['bit_width']["assign_type"]['cell'] = 'cell_group_op_random'
            for n in range(3):
                for pos in self.conv_positions:
                    bitwidths = {"w_bit": random.choice([2,4,8]), "a_bit": random.choice([8])}
                    for i in range(6*n, 6*n+5):
                            self.config_table['bit_width']['infercell'][i][pos] = bitwidths
        if cell_type == 'noquant':
            self.config_table['bit_width']["assign_type"]['cell'] = 'noquant'
            for i in [0,1,2,3,4,6,7,8,9,10,12,13,14,15,16]:
                for pos in self.conv_positions:
                    bitwidths = {"w_bit": -1, "a_bit": -1}
                    self.config_table['bit_width']['infercell'][i][pos] = bitwidths

                

        # self.config_table['bit_width']['infercell'] = infercell_bit
        if stem_type == 'quant_8bit':
            self.config_table['bit_width']["assign_type"]['stem'] = 'quant_8bit'
            self.config_table['bit_width']['stem'] = {'qconv':{'w_bit': 8, 'a_bit': 8}}
            self.config_table['bit_width']['resblock'][5] = {'conv_a': {'w_bit': 8, 'a_bit': 8}, 'conv_b': {'w_bit': 8, 'a_bit': 8}, 'downsample': {'w_bit': 8, 'a_bit': 8}}
            self.config_table['bit_width']['resblock'][11] = {'conv_a': {'w_bit': 8, 'a_bit': 8}, 'conv_b': {'w_bit': 8, 'a_bit': 8}, 'downsample': {'w_bit': 8, 'a_bit': 8}}
            self.config_table['bit_width']['classifier'] = {'qlinear':{'w_bit': 8, 'a_bit': 8}}
        
        if stem_type == 'quant_separated':
            self.config_table['bit_width']["assign_type"]['stem'] = 'quant_separated'
            self.config_table['bit_width']['stem']['qconv'] = {"w_bit": random.choice([2,4,8]), "a_bit": random.choice([8])}
            for idx in [5, 11]:
                for conv in ['conv_a', 'conv_b', 'downsample']:
                    bitwidths = {"w_bit": random.choice([2,4,8]), "a_bit": random.choice([8])}
                    self.config_table['bit_width']['resblock'][idx][conv] = bitwidths
            self.config_table['bit_width']['classifier'] = {'qlinear':{"w_bit": random.choice([2,4,8]), "a_bit": random.choice([8])}}
        
        if stem_type == 'noquant':
            self.config_table['bit_width']["assign_type"]['stem'] = 'noquant'
            self.config_table['bit_width']['stem'] = {'qconv':{'w_bit': -1, 'a_bit': -1}}
            self.config_table['bit_width']['resblock'][5] = {'conv_a': {'w_bit': -1, 'a_bit': -1}, 'conv_b': {'w_bit': -1, 'a_bit': -1}, 'downsample': {'w_bit': -1, 'a_bit': -1}}
            self.config_table['bit_width']['resblock'][11] = {'conv_a': {'w_bit': -1, 'a_bit': -1}, 'conv_b': {'w_bit': -1, 'a_bit': -1}, 'downsample': {'w_bit': -1, 'a_bit': -1}}
            self.config_table['bit_width']['classifier'] = {'qlinear':{'w_bit': -1, 'a_bit': -1}}

        # 先判断该model+位宽设置是否已存在
        models = []
        checkpoint_list = {}
        for key in self.table_cache.keys():
            split = key.split('_')
            if self.model_idx == int(split[0]):
                if self.config_table['bit_width']['infercell'] == self.table_cache[key]['bit_width']['infercell'] \
                    and self.config_table['bit_width']['stem'] == self.table_cache[key]['bit_width']['stem'] \
                    and self.config_table['bit_width']['resblock'] == self.table_cache[key]['bit_width']['resblock'] \
                    and self.config_table['bit_width']['classifier'] == self.table_cache[key]['bit_width']['classifier'] \
                    and self.config_table['train_info']['lr'] == self.table_cache[key]['train_info']['lr'] \
                    and self.config_table['train_info']['dataset'] == self.table_cache[key]['train_info']['dataset'] :
                    # if self.config_table['train_info'] == self.table_cache[key]['train_info']: # 配置已存在，包括位宽分配和训练epoch均相同
                    #     return False
                    # else: # 配置已存在，只有训练epoch不同
                    checkpoint_list[self.table_cache[key]['train_info']['epochs']] = key
                models.append(int(split[1]))

        if len(models) > 0: # 存在但bit_width不重复
            self.dict_idx = str(self.model_idx)+'_'+str(max(models)+1)
            print('get model: {}.'.format(self.dict_idx))
        else:
            self.dict_idx = str(self.model_idx)+'_1'
            print('get model: {}.'.format(self.dict_idx))
        
        self.config_table['index'] = self.dict_idx
        self.apply_bitwidth_to_model()
        # if len(checkpoint_list) > 0:
        #     return checkpoint_list
        return checkpoint_list


    def apply_bitwidth_to_model(self):
        """
        将位宽信息应用到模型层。
        """
        # for name, module in self.model.named_modules():
        #     bit_info = self.get_bitwidth_info(name)
        #     if bit_info and hasattr(module, 'w_bit') and hasattr(module, 'a_bit'):
        #         module.w_bit, module.a_bit = bit_info['w_bit'], bit_info['a_bit']
        
        for name, child in self.model.named_children():
            if name == 'stem':
                child[0].w_bit = self.config_table['bit_width']['stem']['qconv']['w_bit']
                child[0].a_bit = self.config_table['bit_width']['stem']['qconv']['a_bit']
                self.params += child[0].weight.numel() * self.config_table['bit_width']['stem']['qconv']['w_bit'] / 8
                self.params += child[1].weight.numel() * 4
                self.params += child[1].bias.numel() * 4
                self.params_count += child[0].weight.numel()+child[1].weight.numel()+child[1].bias.numel()
            if name =='cells':
                infer_list = [i for r in (range(0, 5), range(6, 11), range(12, 17)) for i in r]
                for j in infer_list:
                    for i in self.config_table['bit_width']['infercell'][j].keys():
                        self.set_quantization_bits_infercell(child[j], i, bit_width=self.config_table['bit_width']['infercell'][j][i])
                
                for idx in [5,11]:
                    for keys in self.config_table['bit_width']['resblock'][idx].keys():
                        self.set_quantization_bits_blockcell(child[idx], keys, bit_width=self.config_table['bit_width']['resblock'][idx][keys])
            if name =='classifier': 
                child.w_bit = self.config_table['bit_width']['classifier']['qlinear']['w_bit']
                child.a_bit = self.config_table['bit_width']['classifier']['qlinear']['a_bit']
                self.params += child.weight.numel() * self.config_table['bit_width']['classifier']['qlinear']['w_bit'] / 8
                self.params += child.bias.numel() * 4
                self.params_count += child.weight.numel()+child.bias.numel()
        # print('Set bit width: {}'.format(self.bitwidth_table))


    def get_bitwidth_info(self):
        return self.config_table['bit_width']

    def get_yaml_info(self):
        return self.config_table

    def get_model(self):
        return self.model
    
    def get_model_index(self, dict_idx):
        tempt = dict_idx.split('_')
        return (tempt[0], tempt[1])
    
    def set_dict_index(self, dict_idx):
        self.dict_idx = dict_idx
    
    def get_dict_index(self):
        return self.dict_idx
    

        
    def set_quantization_bits_infercell(self, module, layer_index, bit_width):
        """
        设置InferCell中QConv2d层的量化位宽。
        Args:
        - module: 模型或子模块。
        - layer_index: 要设置的QConv2d层的索引序号。
        - w_bit: 权重的量化位宽。
        - a_bit: 激活的量化位宽。
        """
        for name, child in module.named_children():
            if isinstance(child, ReLUConvBN):  # 假设ReLUConvBN是一个定义好的类
                # 检查当前的序号是否为目标层序号
                if name == str(layer_index):
                    # 设置QConv2d层的量化位宽
                    child.op[1].w_bit = bit_width['w_bit']
                    child.op[1].a_bit = bit_width['a_bit']
                    # print(f"Set QConv2d at layer {layer_index} to w_bit={w_bit}, a_bit={a_bit}")
                    self.params += child.op[1].weight.numel() * bit_width['w_bit'] / 8
                    self.params += child.op[2].weight.numel() * 4
                    self.params += child.op[2].bias.numel() * 4
                    self.params_count += child.op[1].weight.numel()+child.op[2].weight.numel()+child.op[2].bias.numel()
                    return True  # 返回True表示已成功设置位宽
            # 递归检查子模块
            elif self.set_quantization_bits_infercell(child, layer_index, bit_width):
                return True  # 如果在子模块中找到并设置了位宽，则提前返回True
        return False  # 如果没有找到目标层，返回False

    def set_quantization_bits_blockcell(self, module, model_name, bit_width):
        """
        设置ResnetBasicBlock中QConv2d层的量化位宽。
        Args:
        - module: 模型或子模块。
        - layer_index: 要设置的QConv2d层的索引序号。
        - w_bit: 权重的量化位宽。
        - a_bit: 激活的量化位宽。
        """
        for name, block_child in module.named_children(): 
            if name == model_name:
                # print(name)
                if name == 'downsample':
                    block_child[1].w_bit = bit_width['w_bit']
                    block_child[1].a_bit = bit_width['a_bit']
                    self.params += block_child[1].weight.numel() * bit_width['w_bit'] / 8
                    self.params_count += block_child[1].weight.numel()
                else:
                    block_child.op[1].w_bit = bit_width['w_bit']
                    block_child.op[1].a_bit = bit_width['a_bit']
                    self.params += block_child.op[1].weight.numel() * bit_width['w_bit'] / 8
                    self.params += block_child.op[2].weight.numel() * 4
                    self.params += block_child.op[2].bias.numel() * 4
                    self.params_count += block_child.op[1].weight.numel()+ block_child.op[2].weight.numel()+ block_child.op[2].bias.numel()
                # print("Set QConv2d at ResNetBasicblock {} to w_bit={}, a_bit={}".format(model_name,w_bit,a_bit))
                return True  # 返回True表示已成功设置位宽
            elif self.set_quantization_bits_blockcell(block_child, model_name, bit_width):
                return True
        return False