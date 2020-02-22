import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import resnet
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
from datetime import datetime


def make_sampler(dataset, valid_rate=0.2):
    """
    数据集被划分为训练集、验证集，分别生成训练集采样器、验证集采样器
    :param dataset: 被采样的数据集
    :param valid_rate: 验证集占数据集大小，默认20%
    :return: 训练集采样器、验证集采样器
    """
    assert 0.0 < valid_rate < 1.0  # 划分率在0.0-1.0之间

    dataset_size = len(dataset)  # 数据集大小
    indices = list(range(dataset_size))  # 采样索引
    split = int(np.floor(dataset_size * valid_rate))  # 验证集占多少

    random.shuffle(indices)  # 打乱索引
    train_indices, valid_indices = indices[split:], indices[:split]  # 划分采样索引
    train_sampler = data.SubsetRandomSampler(train_indices)  # 用索引生成训练集采样器
    valid_sampler = data.SubsetRandomSampler(valid_indices)  # 用索引生成验证集采样器
    return train_sampler, valid_sampler


def epoch_timer(func):
    """
    装饰器。epoch计时器，记录一个epoch用时并打印
    :param func: 被装饰函数，是epoch_train
    :return:
    """

    def timer(*args, **kwargs):  # func的所有入参
        begin_time = datetime.now()  # 开始时间
        res = func(*args, **kwargs)  # 执行func，记录func返回值
        end_time = datetime.now()  # 结束时间
        mm, ss = divmod((end_time - begin_time).seconds, 60)  # 秒换算成分、秒
        hh, mm = divmod(mm, 60)  # 分钟换算成时、分
        print('Time: {:02d}:{:02d}:{:02d}'.format(hh, mm, ss))  # HH:mm:ss
        return res  # 返回func返回值

    return timer


@epoch_timer  # 记录一个epoch时间并打印
def epoch_train(net, loss_func, optimizer, train_data, valid_data):
    """
    一个epoch训练过程，分成两个阶段：先训练，再验证
    :param net: 使用的模型
    :param loss_func: loss函数
    :param optimizer: 优化器
    :param train_data: 训练集
    :param valid_data: 验证集
    :return: 一个epoch的训练loss、训练acc、验证loss、验证acc
    """
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')  # 使用GPU，没有GPU就用CPU
    net.to(device)  # 模型装入GPU
    loss_func.to(device)  # loss函数装入CPU

    """训练"""
    train_loss, train_acc = 0., 0.  # 一个epoch训练的loss和正确率acc
    net.train()  # 训练
    for i, (train_image, train_label) in enumerate(train_data):
        train_image = Variable(train_image.to(device))  # 一个训练batch image
        train_label = Variable(train_label.to(device))  # 一个训练batch label

        train_output = net(train_image)  # 前向传播，计算一个训练batch的output
        loss = loss_func(train_output, train_label)  # 计算一个训练batch的loss
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 步进

        train_loss += loss.detach().cpu().numpy()  # 累加训练batch的loss
        train_acc += get_acc(train_output, train_label)  # 累加训练batch的acc
        pass
    train_loss /= len(train_data)  # 求取一个epoch训练的loss
    train_acc /= len(train_data)  # 求取一个epoch训练的acc

    """验证"""
    valid_loss, valid_acc = 0., 0.  # 一个epoch验证的loss和正确率acc
    net.eval()  # 验证
    for i, (valid_image, valid_label) in enumerate(valid_data):
        valid_image = Variable(valid_image.to(device))  # 一个验证batch image
        valid_label = Variable(valid_label.to(device))  # 一个验证batch label

        valid_output = net(valid_image)  # 前项传播，计算一个验证batch的output
        loss = loss_func(valid_output, valid_label)  # 计算一个验证batch的loss
        # 验证的时候不进行反向传播

        valid_loss += loss.detach().cpu().numpy()  # 累加验证batch的loss
        valid_acc += get_acc(valid_output, valid_label)  # 累加验证batch的acc
        pass
    valid_loss /= len(valid_data)  # 求取一个epoch验证的loss
    valid_acc /= len(valid_data)  # 求取一个epoch验证的acc
    return train_loss, train_acc, valid_loss, valid_acc


def save_checkpoint(net, name, epoch):
    """
    保存模型参数
    :param net: 模型
    :param save_dir: 保存的路径
    :param name: 参数文件名
    :param epoch: 训练到第几个epoch的参数
    :return:
    """
    save_dir = os.path.join(os.getcwd(), 'weight')
    save_dir = os.path.join(save_dir, name + '-' + str(epoch) + '.pkl')
    torch.save(net.state_dict(), save_dir)
    pass


def train(net, loss_func, optimizer, train_data, valid_data, name, epochs=5):
    """
    训练
    :param net: 被训练的模型
    :param loss_func: loss函数
    :param optimizer: 优化器
    :param train_data: 训练集
    :param valid_data: 验证集
    :param name: 保存参数文件名
    :param epochs: epoch数，默认是5
    :return:
    """
    for e in range(epochs):
        t_loss, t_acc, v_loss, v_acc = epoch_train(net, loss_func, optimizer,
                                                   train_data, valid_data)  # 一个epoch训练
        epoch_str = ('Epoch: {:d} | '
                     'Train Loss: {:.4f} | Train Acc: {:.4f} | '
                     'Valid Loss: {:.4f} | Valid Acc: {:.4f}')
        print(epoch_str.format(e + 1, t_loss, t_acc, v_loss, v_acc))  # 打印一个epoch的loss和acc
        save_checkpoint(net, name, e)  # 每个epoch的参数都保存
        pass
    pass


pass


def get_acc(output, label):
    """
    计算正确率
    :param output: 模型的输出
    :param label: label
    :return: 正确率
    """
    total = output.shape[0]  # 总数
    pred = torch.argmax(output, dim=1)  # 模型channel方向最大值的索引也就是估值
    num_correct = (pred == label).sum().cpu().numpy()  # 估值与label一致的总数
    return num_correct / total  # 正确率=估值正确总数/总数


if __name__ == '__main__':
    """
    单元测试
    """
    # 数据转换成Tensor并归一化，这个归一化的mean和std是抄来的，不知道为什么是这些数字
    trans = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])

    # cifar10数据集
    # 已经下载好的数据放在'/root/private/torch_datasets'
    # train=True读取训练集
    dataset_cifar10 = CIFAR10('/root/private/torch_datasets', train=True,
                              transform=trans, download=True)

    # 将cifar10的训练集再划分为训练集、验证集，生成两个数据集的采样器
    t_sampler, v_sampler = make_sampler(dataset_cifar10)

    BATCH_SIZE = 10  # 指定batch大小为10

    # 训练集，用训练采样器采样。
    # DataLoader防止一次读取全部数据造成内存不足
    train_dataloader = data.DataLoader(dataset_cifar10, batch_size=BATCH_SIZE,
                                       sampler=t_sampler)
    # 验证集，用验证采样器采样。
    # DataLoader防止一次读取全部数据造成内存不足
    valid_dataloader = data.DataLoader(dataset_cifar10, batch_size=BATCH_SIZE,
                                       sampler=v_sampler)

    resnet_model = resnet.resnet18(n_class=10)  # resnet模型，cifar10分类10种
    print(resnet_model)
    resnet_model_loss_func = nn.CrossEntropyLoss()  # loss函数
    resnet_model_optimizer = torch.optim.Adam(resnet_model.parameters())  # 将模型参数装入优化器

    train(resnet_model, resnet_model_loss_func, resnet_model_optimizer,
          train_dataloader, valid_dataloader, name='resnet18')  # 开始训（炼）练（丹）
