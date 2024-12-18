import os
import re
import glob
import yaml
import torch
import pickle
import random
import torchvision
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import nn
from tqdm import tqdm
from utils.model_utils import load_data, convert_conv2d_to_qconv2d, convert_linear_to_qlinear, convert_linear_to_qlinear_straight
from utils.bitassign_utils import MixBitAssign
from lib.utils.quantize_utils import QLinear

class MobileNetCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetCIFAR, self).__init__()
        self.mobilenet = torchvision.models.mobilenet_v2(pretrained=True)
        # 修改最后的分类器以适应 CIFAR-10 的 10 个类别
        self.mobilenet.classifier[1] = nn.Linear(self.mobilenet.last_channel, num_classes)

    def forward(self, x):
        return self.mobilenet(x)

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
            logits = model(imgs)
            loss = loss_fc(logits, targets)

            total_test_loss += loss.item()
            accuracy = (logits.argmax(1) == targets).sum()
            total_accuracy += accuracy.item()

    print("Average Loss：{}".format(total_test_loss/ test_data_size))
    print("Average Acc：{}".format(total_accuracy / test_data_size))
    return (total_accuracy / test_data_size),(total_test_loss/ test_data_size)
        

def train_model_with_epoch_list(model, H, model_save_dir, train_loader, val_loader, test_loader, device = '0', epoch_list = [50, 100, 150],  not_train = {}, epoch_trained = {}, not_train_epoch_list = []):
    """训练模型，H是超参数设置
    H0 = {'epochs': 200, 'lr': 0.001, 'batch_size': 256}
    """
    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    assert torch.cuda.is_available(), 'CUDA is needed for CNN'

    epoch_result = {}
    '''{50: {'dict_name': ' ', 'val_loss': 1, 'val_acc': -1, 'test_loss': -1, 'test_acc': -1}, 
        100: {'dict_name': ' ','val_loss': 1, 'val_acc': -1, 'test_loss': -1, 'test_acc': -1}, 
        150: {'dict_name': ' ','val_loss': 1, 'val_acc': -1, 'test_loss': -1, 'test_acc': -1}}'''

    epochs = epoch_list[-1]

    model.cuda()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=H['lr'], momentum=0.9, weight_decay=0.0005, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    criterion = nn.CrossEntropyLoss()   

    for epoch in tqdm(range(epochs)):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

        if (epoch+1) in epoch_list:
            # 测试模型
            print('Validate mbnetv2 at epoch {}.'.format( epoch+1))
            val_acc, val_loss = test_model(model, val_loader, len(val_loader)*H['batch_size'], device)

            print('Test mbnetv2 at epoch {}.'.format( epoch+1))
            test_acc, test_loss = test_model(model, test_loader, len(test_loader)*H['batch_size'], device)

            epoch_result[epoch+1] = {'val_loss': val_loss, 'val_acc': val_acc, 'test_loss': test_loss, 'test_acc': test_acc}

            model_save_path = os.path.join(model_save_dir, "test_epoch_{}_int8_cifar10_mbv2.pth".format(epoch+1))
            torch.save(model.state_dict(), model_save_path)
            print("模型已保存在：{}".format(model_save_path))
            model.train()
    return epoch_result


if __name__ == "__main__":
    # Prepare dataset
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, required=True, help='Device on which the models will be run, set 0 or 1')
    
    args = parser.parse_args()

    # 超参数设置
    H0 = {'dataset': 'cifar10','epochs': 200, 'lr': 0.1, 'batch_size': 256}
    H1 = {'dataset': 'cifar10','epochs': 200, 'lr': 0.05, 'batch_size': 256}
    H2 = {'dataset': 'cifar10','epochs': 150, 'lr': 0.01, 'batch_size': 256}
    H3 = {'dataset': 'cifar10','epochs': 150, 'lr': 0.01, 'batch_size': 1024}

    epoch_list = [i*10 for i in range(1,16)]

    yaml_dict = {}

    target_H = H2
    print(args)
    print('Train info: ',target_H)

    train_loader, valid_loader, test_loader = load_data('cifar10', '~/dataset', target_H)
    print('Dataset prepared.')

    yaml_path = '/home/dell/MP-NAS-Bench201/results/mobilenet_v2_int8_cifar10/mobilenet_v2_int8_cifar10.yaml'
    model_save_dir = '/home/dell/MP-NAS-Bench201/results/mobilenet_v2_int8_cifar10'
    # yaml_path = '/home/dell/MP-NAS-Bench201/results/mobilenet_v2_fp32_cifar10/mobilenet_v2_fp32_cifar10.yaml'
    # model_save_dir = '/home/dell/MP-NAS-Bench201/results/mobilenet_v2_fp32_cifar10'
    # 判断路径是否存在
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    model = MobileNetCIFAR()

    print(model)

    convert_conv2d_to_qconv2d(model, w_bit=8, a_bit=8 )
    convert_linear_to_qlinear(model, w_bit=8, a_bit=8)

    # print(model)

    result = train_model_with_epoch_list(model, target_H, model_save_dir, train_loader, valid_loader, test_loader, device = args.device, epoch_list = epoch_list,  not_train = {}, epoch_trained = {}, not_train_epoch_list = [])
    yaml_dict['model'] = 'mobilenet_v2'
    yaml_dict['result'] = result
    yaml_dict['H'] = target_H

    with open(yaml_path, 'w') as f:
        yaml.dump(result, f)

