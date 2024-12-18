# Note: 初始版本的位宽分配代码，只考虑所有cell进行相同位宽的量化
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
    def __init__(self, model,  model_idx, train_info, conv_positions, bitwidths={'w_bit': 8, 'a_bit': 8}, yaml_path = '~/default/', meta_data_filename = 'meta_data.pkl',random_assign=False):
        self.model = model
        self.model_idx = model_idx
        self.dict_idx = str(self.model_idx)+'_1'
        self.train_info = train_info
        self.random_assign = random_assign
        self.meta_data_filename = meta_data_filename
        self.yaml_path = yaml_path
        # 初始化位宽配置字典
        self.bitwidth_table = {
            'stem': {'qconv':{'w_bit': 8, 'a_bit': 8}},
            'infercell': {},
            'resblock': {'conv_a': {'w_bit': 8, 'a_bit': 8}, 'conv_b': {'w_bit': 8, 'a_bit': 8}, 'downsample': {'w_bit': 8, 'a_bit': 8}},
            'classifier': {'qlinear':{'w_bit': 8, 'a_bit': 8}}
        }
        # 为每个infercell的conv层设置位宽
        for pos in conv_positions:
            self.bitwidth_table['infercell'][pos] = bitwidths

        self.meta_data_dict = {}
        # 尝试读取已有的pickle文件
        if os.path.exists(self.meta_data_filename):
            with open(self.meta_data_filename, 'rb') as f:
                try:
                    self.meta_data_dict = pickle.load(f)
                except EOFError:  # 文件为空的异常处理
                    self.meta_data_dict = {}
        

    def save_to_yaml(self):
        # 保存位宽配置到YAML文件
        path = self.yaml_path+'{}.yaml'.format(self.dict_idx)

        with open(path, 'w') as file:
            yaml.dump(self.bitwidth_table, file, default_flow_style=False)
        
        print('Bitwidth configuration saved to {}'.format(path))

    def load_yaml(self, match):
        filename = self.yaml_path+'{}.yaml'.format(self.dict_idx)
        with open(filename, 'r') as file:
            self.bitwidth_table = yaml.safe_load(file)
        if self.bitwidth_table['classifier'] != match['classifier'] or self.bitwidth_table['stem'] != match['stem'] or self.bitwidth_table['resblock'] != match['resblock']:
            return False
        self.apply_bitwidth_to_model()
        return True

    def generate_random_bitwidth(self, cell_infer_only = False):
        """
        生成随机位宽，范围在2到8位之间。
        """
        for key in self.bitwidth_table.keys():
            if isinstance(self.bitwidth_table[key], dict):
                for subkey in self.bitwidth_table[key].keys():
                    self.bitwidth_table[key][subkey] = {"w_bit": random.choice([2,4,8]), "a_bit": random.choice([8])}
            else:
                self.bitwidth_table[key] = {"w_bit": random.choice([2,4,8]), "a_bit": random.choice([8])}

        # 先判断该model+位宽设置是否已存在
        key_str = r'{}_(\d+)$'.format(self.model_idx)
        pattern = re.compile(key_str)
        # 使用列表推导式和正则表达式匹配键
        matched_numbers = []
        for key in self.meta_data_dict.keys():
            match = pattern.match(key)
            if match:
                if (self.bitwidth_table['infercell'] == self.meta_data_dict[key]['bit_width']['infercell']) and (self.train_info == self.meta_data_dict[key]['train_info']):
                    return False
                # 如果存在匹配，提取数字部分并转换为整数
                matched_numbers.append(int(match.group(1)))
        if matched_numbers:
            ct = max(matched_numbers)+1 # 存在但bit_width不重复
            self.dict_idx = str(self.model_idx)+'_'+str(ct)
            print('get model: {}.'.format(self.dict_idx))

        else:
            self.dict_idx = str(self.model_idx)+'_1'
            print('get model: {}.'.format(self.dict_idx))
        
        if cell_infer_only:
            self.bitwidth_table['stem'] = {'qconv':{'w_bit': 8, 'a_bit': 8}}
            self.bitwidth_table['resblock'] = {'conv_a': {'w_bit': 8, 'a_bit': 8}, 'conv_b': {'w_bit': 8, 'a_bit': 8}, 'downsample': {'w_bit': 8, 'a_bit': 8}}
            self.bitwidth_table['classifier'] = {'qlinear':{'w_bit': 8, 'a_bit': 8}}
        
        self.apply_bitwidth_to_model()
        return True


    def set_bitwidth(self, layer_name, bit_width, index=None, ):
        if layer_name == "infercell" and index is not None:
            self.bitwidth_table[layer_name][index] = bit_width
        elif layer_name == "stem":
            self.bitwidth_table[layer_name]['qconv'] = bit_width
        elif layer_name == "classifier":
            self.bitwidth_table[layer_name]['qlinear'] = bit_width
        elif layer_name == "resblock":
            self.bitwidth_table[layer_name] = bit_width


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
                child[0].w_bit = self.bitwidth_table['stem']['qconv']['w_bit']
                child[0].a_bit = self.bitwidth_table['stem']['qconv']['a_bit']
            if name =='cells':
                infer_list = [i for r in (range(0, 5), range(6, 11), range(12, 17)) for i in r]
                for j in infer_list:
                    for i in self.bitwidth_table['infercell'].keys():
                        set_quantization_bits_infercell(child[j], i, w_bit=self.bitwidth_table['infercell'][i]['w_bit'], a_bit=self.bitwidth_table['infercell'][i]['a_bit'])
                
                for keys in self.bitwidth_table['resblock'].keys():
                    set_quantization_bits_blockcell(child[5], keys, w_bit=self.bitwidth_table['resblock'][keys]['w_bit'], a_bit=self.bitwidth_table['resblock'][keys]['a_bit'])
                    set_quantization_bits_blockcell(child[11], keys, w_bit=self.bitwidth_table['resblock'][keys]['w_bit'], a_bit=self.bitwidth_table['resblock'][keys]['a_bit'])
            if name =='classifier': 
                child.w_bit = self.bitwidth_table['classifier']['qlinear']['w_bit']
                child.a_bit = self.bitwidth_table['classifier']['qlinear']['a_bit']
        # print('Set bit width: {}'.format(self.bitwidth_table))

    def get_bitwidth_info(self, layer_name):
        """
        根据层名获取位宽信息。
        """
        # This method needs to be customized based on the model structure
        # Here is a simplified logic
        if layer_name in self.bitwidth_table:
            return self.bitwidth_table[layer_name]
        for key in self.bitwidth_table.keys():
            if key in layer_name:  # Match layer_name with keys in bitwidth_table
                return self.bitwidth_table[key]
        return None

    def get_model(self):
        return self.model
    
    def get_model_index(self, dict_idx):
        tempt = dict_idx.split('_')
        return (tempt[0], tempt[1])
    
    def set_dict_index(self, dict_idx):
        self.dict_idx = dict_idx
    
    def get_dict_index(self):
        return self.dict_idx
    
    def generate_meta_data(self, acc):
        """以字典的形式保存网络索引号，精度信息，量化位宽信息"""
        # 将新的元数据添加到列表中
        self.meta_data_dict[self.dict_idx] = {'accuracy': acc, 'train_info': self.train_info, 'bit_width': self.bitwidth_table}
        
        # 将更新后的列表写回文件
        with open(self.meta_data_filename, 'wb') as f:
            pickle.dump(self.meta_data_dict, f)
        print('Results are saved in {}. '.format(self.meta_data_filename))


        
def set_quantization_bits_infercell(module, layer_index, w_bit, a_bit):
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
                child.op[1].w_bit = w_bit
                child.op[1].a_bit = a_bit
                # print(f"Set QConv2d at layer {layer_index} to w_bit={w_bit}, a_bit={a_bit}")
                return True  # 返回True表示已成功设置位宽
        # 递归检查子模块
        elif set_quantization_bits_infercell(child, layer_index, w_bit, a_bit):
            return True  # 如果在子模块中找到并设置了位宽，则提前返回True
    return False  # 如果没有找到目标层，返回False

def set_quantization_bits_blockcell(module, model_name, w_bit,a_bit):
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
                block_child[1].w_bit = w_bit
                block_child[1].a_bit = a_bit
            else:
                block_child.op[1].w_bit = w_bit
                block_child.op[1].a_bit = a_bit
            # print("Set QConv2d at ResNetBasicblock {} to w_bit={}, a_bit={}".format(model_name,w_bit,a_bit))
            return True  # 返回True表示已成功设置位宽
        elif set_quantization_bits_blockcell(block_child, model_name, w_bit,a_bit):
            return True
    return False