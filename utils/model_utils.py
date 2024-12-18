# MP-NAS-Bench201 test version

import os
from tqdm import tqdm
import torch
import numpy as np
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
# from torch.utils.tensorboard import SummaryWriter
from lib.utils.quantize_utils import QConv2d, QLinear

import xautodl
from xautodl.models import get_cell_based_tiny_net


def load_data(path, target_H, seed = 888):
    """
    Load dataset and return dataloader
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    print('Set random seed: ', seed)

    '''Load dataset'''
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
    ])

    if target_H['dataset'] == 'cifar10':
        # Load the CIFAR-10 datasets
        train_dataset = datasets.CIFAR10(root=path, train=True, download=False, transform=transform)
        test_dataset = datasets.CIFAR10(root=path, train=False, download=False, transform=transform)
    if target_H['dataset'] == 'cifar100':
        # Load the CIFAR-10 datasets
        train_dataset = datasets.CIFAR100(root=path, train=True, download=False, transform=transform)
        test_dataset = datasets.CIFAR100(root=path, train=False, download=False, transform=transform)

    # Splitting the training dataset into training and validation
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(0.5 * num_train))  # 50% of the training data for training

    np.random.shuffle(indices)
    train_idx, valid_idx = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    batch = target_H['batch_size']

    # Define data loaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch, sampler=train_sampler)
    valid_loader = DataLoader(train_dataset, batch_size=batch, sampler=valid_sampler)

    # Test loader remains the same
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=True)

    return train_loader, valid_loader, test_loader


def train_model(model, H, model_save_dir, train_loader, model_idx, device = '0', load = None):
    """训练模型，H是超参数设置
    H0 = {'epochs': 200, 'lr': 0.001, 'batch_size': 256}
    """

    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    assert torch.cuda.is_available(), 'CUDA is needed for CNN'


    loss_list = []
    epochs = H['epochs']

    optimizer = optim.SGD(model.parameters(), lr=H['lr'], momentum=0.9, weight_decay=0.0005, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=H['epochs'], eta_min=0)
    criterion = nn.CrossEntropyLoss()

    if load != None:
        model_save_path = os.path.join(model_save_dir, "test_{}_quant_cifar10_model_{}.pth".format(load[1], load[0]))
        model.load_state_dict(torch.load(model_save_path))
        epochs = H['epochs'] - load[1]
        print('Existing model {} with checkpoint of {} epochs'.format(load[0], load[1]))
        print('Load from {}'.format(model_save_path))

    model.cuda()
    model.train()
    for epoch in tqdm(range(epochs)):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output, logits = model(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
        loss_list.append(loss.item())
        if(epoch > 5) and sum(loss_list)/len(loss_list) > 2.0:
            print("Loss is not decreasing, early stopping")
            break

    # 保存模型
    model_save_path = os.path.join(model_save_dir, "test_{}_quant_{}_model_{}.pth".format(H['epochs'], H['dataset'], model_idx))
    torch.save(model.state_dict(), model_save_path)
    print("模型已保存在：{}".format(model_save_path))


def test_model(model, dataloder_test, test_data_size, device = '0'):
    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    assert torch.cuda.is_available(), 'CUDA is needed for CNN'

    # 损失函数
    loss_fc = nn.CrossEntropyLoss()

    model.cuda()
    # 测试步骤开始
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in dataloder_test:
            imgs, targets = data
            imgs, targets = imgs.cuda(), targets.cuda()  # 将数据移动到指定设备
            out, logits = model(imgs)
            loss = loss_fc(logits, targets)

            total_test_loss += loss.item()
            accuracy = (logits.argmax(1) == targets).sum()
            total_accuracy += accuracy.item()

    print("Average Loss：{}".format(total_test_loss/ test_data_size))
    print("Average Acc：{}".format(total_accuracy / test_data_size))
    return (total_accuracy / test_data_size),(total_test_loss/ test_data_size)

# 设计新的模型训练函数，传入参数包括递增的epoch列表，遍历epoch列表中的元素，训练到该元素时即进行测试一次，并保存该epoch时的测试精度、loss和checkpoint文件
def train_model_with_epoch_list(model, H, model_save_dir, train_loader, val_loader, test_loader, device = '0', epoch_list = [50, 100, 150],  not_train = {}, epoch_trained = {}, not_train_epoch_list = []):
    """训练模型，H是超参数设置
    H0 = {'epochs': 200, 'lr': 0.001, 'batch_size': 256}
    """

    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    assert torch.cuda.is_available(), 'CUDA is needed for CNN'

    loss_list = []
    es_flag = False
    epoch_result = {}
    epoch_bias= 0
    '''{50: {'dict_name': ' ', 'val_loss': 1, 'val_acc': -1, 'test_loss': -1, 'test_acc': -1}, 
        100: {'dict_name': ' ','val_loss': 1, 'val_acc': -1, 'test_loss': -1, 'test_acc': -1}, 
        150: {'dict_name': ' ','val_loss': 1, 'val_acc': -1, 'test_loss': -1, 'test_acc': -1}}'''


    if len(not_train) > 0:
        print('Wait to train: ', not_train)
        for i, e in enumerate(epoch_list):
            if e in not_train.keys():
                if i != 0: 
                    if epoch_list[i-1] not in not_train.keys():
                        load = (epoch_trained[epoch_list[i-1]], epoch_list[i-1])
                        epochs = not_train_epoch_list[-1]-epoch_list[i-1]
                        epoch_bias = epoch_list[i-1]

                else:
                    load = None
                    epochs = not_train_epoch_list[-1]

        if load != None:
            model_save_path = os.path.join(model_save_dir, "test_{}_quant_{}_model_{}.pth".format(load[1], H['dataset'], load[0]))
            if os.path.exists(model_save_path):
                model.load_state_dict(torch.load(model_save_path))
                print('Info: Existing model {} with checkpoint of {} epochs'.format(load[0], load[1]))
                print('Info: Load from {}'.format(model_save_path))
            else:
                print('Error: No such file or directory: {}, '.format(model_save_path))
                epoch_result = {}
                return epoch_result

        model.cuda()
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=H['lr'], momentum=0.9, weight_decay=0.0005, nesterov=True)
        scheduler = CosineAnnealingLR(optimizer, T_max=150, eta_min=0)
        criterion = nn.CrossEntropyLoss()   

        for epoch in tqdm(range(epochs)):
            if not es_flag:
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.cuda(), target.cuda()
                    optimizer.zero_grad()
                    output, logits = model(data)
                    loss = criterion(logits, target)
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
                loss_list.append(loss.item())
            if(epoch > 5) and not es_flag:
                # 计算loss_list的平均变化率
                loss_grad = 0
                for i in range(0,4):
                    loss_grad += abs(loss_list[i+1] - loss_list[i])
                loss_grad = loss_grad/4
                # 如果grad小于0.1或者是nan
                if loss_grad < 0.02 or loss_grad is None: 
                    print("Loss is not decreasing, early stopping")
                    es_flag = True
            
            if (epoch+1+epoch_bias) in not_train.keys():
                # 测试模型
                print('Validate model {} at epoch {}.'.format(not_train[epoch+1+epoch_bias], epoch+1))
                val_acc, val_loss = test_model(model, val_loader, len(val_loader)*H['batch_size'], device)

                print('Test model {} at epoch {}.'.format(not_train[epoch+1+epoch_bias], epoch+1))
                test_acc, test_loss = test_model(model, test_loader, len(test_loader)*H['batch_size'], device)

                epoch_result[epoch+1+epoch_bias] = {'dict_name': not_train[epoch+1+epoch_bias], 'val_loss': val_loss, 'val_acc': val_acc, 'test_loss': test_loss, 'test_acc': test_acc}

                model_save_path = os.path.join(model_save_dir, "test_{}_quant_{}_model_{}.pth".format(epoch+1+epoch_bias, H['dataset'], not_train[epoch+1+epoch_bias]))
                torch.save(model.state_dict(), model_save_path)
                print("模型已保存在：{}".format(model_save_path))
                model.train()
    else:
        epoch_result = {}
        print('Info: Exist all epoch choices, continue.')

    return epoch_result


def convert_conv2d_to_qconv2d(module, w_bit=8, a_bit=8, parent_name=''):
    """
    递归地将模型中的所有Conv2d层替换为QConv2d层。
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            # 创建QConv2d实例
            new_layer = QConv2d(in_channels=child.in_channels,
                                out_channels=child.out_channels,
                                kernel_size=child.kernel_size[0],
                                stride=child.stride[0],
                                padding=child.padding[0],
                                dilation=child.dilation[0],
                                groups=child.groups,
                                bias=(child.bias is not None),
                                w_bit=w_bit,  # 假设的量化位宽
                                a_bit=a_bit,  # 假设的量化位宽
                                half_wave=True)  # 假设的量化模式
            # 将QConv2d层的权重初始化为原Conv2d层的权重
            new_layer.weight.data = child.weight.data.clone()
            if child.bias is not None:
                new_layer.bias.data = child.bias.data.clone()
            
            # 替换原Conv2d层
            setattr(module, name, new_layer)
        else:
            # 递归处理子模块
            convert_conv2d_to_qconv2d(child, w_bit, a_bit, parent_name=name)


def convert_linear_to_qlinear(module, w_bit=8, a_bit=8 ):
    """
    递归地将模型中的所有Linear层替换为QLinear层。
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # 创建QLinear实例
            new_layer = QLinear(in_features=child.in_features,
                                out_features=child.out_features,
                                bias=(child.bias is not None),
                                w_bit=w_bit,  # 假设的量化位宽
                                a_bit=a_bit,  # 假设的量化位宽
                                half_wave=True)  # 假设的量化模式
            # 将QLinear层的权重和偏置初始化为原Linear层的权重和偏置
            new_layer.weight.data = child.weight.data.clone()
            if child.bias is not None:
                new_layer.bias.data = child.bias.data.clone()

            # 替换原Linear层
            setattr(module, name, new_layer)
        else:
            # 递归处理子模块
            convert_linear_to_qlinear(child, w_bit, a_bit)

def convert_linear_to_qlinear_straight(module, w_bit=8, a_bit=8 ):
    """
    递归地将模型中的所有Linear层替换为QLinear层。
    """
    # 如果module有子模块
    if module is not None:
        # 递归处理子模块
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # 创建QLinear实例
                new_layer = QLinear(in_features=child.in_features,
                                    out_features=child.out_features,
                                    bias=(child.bias is not None),
                                    w_bit=w_bit,  # 假设的量化位宽
                                    a_bit=a_bit,  # 假设的量化位宽
                                    half_wave=True)  # 假设的量化模式
                # 将QLinear层的权重和偏置初始化为原Linear层的权重和偏置
                new_layer.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    new_layer.bias.data = child.bias.data.clone()

                # 替换原Linear层
                setattr(module, name, new_layer)
            else:
                # 递归处理子模块
                convert_linear_to_qlinear_straight(child, w_bit, a_bit)
    else: 
        child = module
        if isinstance(child, nn.Linear):
            # 创建QLinear实例
            new_layer = QLinear(in_features=child.in_features,
                                out_features=child.out_features,
                                bias=(child.bias is not None),
                                w_bit=w_bit,  # 假设的量化位宽
                                a_bit=a_bit,  # 假设的量化位宽
                                half_wave=True)  # 假设的量化模式
            # 将QLinear层的权重和偏置初始化为原Linear层的权重和偏置
            new_layer.weight.data = child.weight.data.clone()
            if child.bias is not None:
                new_layer.bias.data = child.bias.data.clone()

            # 替换原Linear层
            setattr(module, name, new_layer)



def get_network(api, net_idx, dataset = 'cifar10', quant = False):
    config = api.get_net_config(net_idx, dataset)
    network = get_cell_based_tiny_net(config)

    # Load the pre-trained weights: params is a dict, where the key is the seed and value is the weights.
    params = api.get_net_param(net_idx, dataset, None, hp = 200)
    network.load_state_dict(next(iter(params.values())))

    if quant:
        convert_conv2d_to_qconv2d(network, w_bit=8, a_bit=8 )
        convert_linear_to_qlinear(network, w_bit=8, a_bit=8)
    return network


def find_nor_conv_positions(cell_structure_str):
    """
    查找并返回含有 'nor_conv' 的操作符的位置序号。
    :param cell_structure_str: 表示cell连接的字符串。
    :return: 含 'nor_conv' 操作符的位置序号列表。
    """
    nor_conv_positions = []
    # 通过 '|' 分割来定位每个算子
    parts = cell_structure_str.split('~')
    # 遍历每个部分来检查是否包含 'nor_conv'
    for i, part in enumerate(parts):
        if 'nor_conv' in part:
            nor_conv_positions.append(i)  # 根据分割结果调整索引以匹配从0开始的位置序号
    return nor_conv_positions

if __name__ =="__main__":
    print('test')