import random
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
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(dataset_size * valid_rate))

    random.shuffle(indices)  # 打乱索引
    train_indices, valid_indices = indices[split:], indices[:split]
    train_sampler = data.SubsetRandomSampler(train_indices)
    valid_sampler = data.SubsetRandomSampler(valid_indices)
    return train_sampler, valid_sampler


def epoch_train(net, loss_func, optimizer, train_data, valid_data, device):
    # TODO
    pass


def train(net, loss_func, optimizer, train_data, valid_data, epochs=10):
    # TODO
    pass


def get_acc(output, label):
    total = output.shape[0]
    pred = torch.argmax(output, dim=1)
    num_correct = (pred == label).sum().cpu().numpy()
    return num_correct / total


if __name__ == '__main__':
    trans = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])
    dataset_cifar10 = CIFAR10('/root/private/torch_datasets', train=True,
                              transform=trans, download=True)
    t_sampler, v_sampler = make_sampler(dataset_cifar10)

    BATCH_SIZE = 10
    EPOCHS = 5

    train_dataloader = data.DataLoader(dataset_cifar10, batch_size=BATCH_SIZE,
                                       sampler=t_sampler)
    valid_dataloader = data.DataLoader(dataset_cifar10, batch_size=BATCH_SIZE,
                                       sampler=v_sampler)

    net = resnet.resnet18(n_class=10)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    loss_func.to(device)

    for e in range(EPOCHS):
        begin_time = datetime.now()

        train_loss, train_acc = 0., 0.
        net.train()
        for i, (train_image, train_label) in enumerate(train_dataloader):
            train_image = Variable(train_image.to(device))
            train_label = Variable(train_label.to(device))

            train_output = net(train_image)
            loss = loss_func(train_output, train_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().cpu().numpy()
            train_acc += get_acc(train_output, train_label)
            pass

        valid_loss, valid_acc = 0., 0.
        net.eval()
        for i, (valid_image, valid_label) in enumerate(valid_dataloader):
            valid_image = Variable(valid_image.to(device))
            valid_label = Variable(valid_label.to(device))

            valid_output = net(valid_image)
            loss = loss_func(valid_output, valid_label)

            valid_loss += loss.detach().cpu().numpy()
            valid_acc += get_acc(valid_output, valid_label)
            pass
        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)
        valid_loss /= len(valid_dataloader)
        valid_acc /= len(valid_dataloader)

        end_time = datetime.now()
        mm, ss = divmod((end_time - begin_time).seconds, 60)
        hh, mm = divmod(mm, 60)

        epoch_str = ('Epoch: {:d} | Time: {:02d}:{:02d}:{:02d} | '
                     'Train Loss: {:.4f} | Train Acc: {:.4f} | '
                     'Valid Loss: {:.4f} | Valid Acc: {:.4f}')
        print(epoch_str.format(e + 1, hh, mm, ss,
                               train_loss, train_acc, valid_loss, valid_acc))
        pass
