import torch
import torch.nn as nn


class _ResNetFactory(nn.Module):
    """
    ResNet工厂类
    """

    def __init__(self, conv2_blocks, conv3_blocks, conv4_blocks, conv5_blocks,
                 out_channels, in_channels=3, n_class=1000):
        """
        通过输入ResNet论文中Conv2_x,Conv3_x,Conv4_x,Conv5_x的blocks来确定是ResNet类型
        连接全连接层要知道输入channel，也就是Residual Block最后输出的channel，也就是入参out_channels
        :param conv2_blocks: 论文Conv2_x中的Residual Block，列表类型
        :param conv3_blocks: 论文Conv3_x中的Residual Block，列表类型
        :param conv4_blocks: 论文Conv4_x中的Residual Block，列表类型
        :param conv5_blocks: 论文Conv5_x中的Residual Block，列表类型
        :param out_channels: 对于ResNet18和ResNet34，out_channels=512
                             对于ResNet50、ResNet101和ResNet152，out_channels=2048
        :param in_channels: 输入图像通道，默认3
        :param n_class: 分类数
        """
        super(_ResNetFactory, self).__init__()

        # conv1 7x7
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3,
                               bias=False)  # 1/2
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, ceil_mode=True)  # 1/4

        # conv2-conv5
        self.conv2 = nn.Sequential(*conv2_blocks)
        self.conv3 = nn.Sequential(*conv3_blocks)
        self.conv4 = nn.Sequential(*conv4_blocks)
        self.conv5 = nn.Sequential(*conv5_blocks)

        # avgpool
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # fc
        self.fc = nn.Linear(out_channels, n_class)
        pass

    def forward(self, x):
        x = self.conv1(x)
        print('conv1', x.shape)
        x = self.bn(x)
        x = self.maxpool(x)
        print('maxpool', x.shape)
        x = self.conv2(x)
        print('conv2', x.shape)
        x = self.conv3(x)
        print('conv3', x.shape)
        x = self.conv4(x)
        print('conv4', x.shape)
        x = self.conv5(x)
        print('conv5', x.shape)
        x = self.avgpool(x)  # 输出是4维张量
        print('avgpool', x.shape)
        x = x.view(1, -1)  # 变成1维向量
        print('avgpool-view', x.shape)
        x = self.fc(x)
        print('fc', x.shape)
        return x

    pass


################################################################################

class _BasicBlockS1(nn.Module):
    """
    Basic Block中的一种

    第一个卷积
    spatial减小一倍，stride=2
    channel增大一倍，out_channels = 2*in_channels

    加shortcut需要1x1卷积调整维度
    因第一个卷积spatial减小一倍，1x1卷积的stride=2

    第二个卷积
    spatial不变
    channel不变
    """

    def __init__(self, in_channels):
        super(_BasicBlockS1, self).__init__()
        out_channels = 2 * in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=2,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        # 调整shortcut的channel
        # 调整shortcut的spatial，stride=2
        self.project = nn.Conv2d(in_channels, out_channels, 1, stride=2)
        pass

    def forward(self, x):
        f = x
        f = self.conv1(f)
        f = self.bn1(f)
        f = self.relu1(f)
        f = self.conv2(f)
        f = f + self.project(x)  # 调整维度才能相加
        f = self.bn2(f)
        f = self.relu2(f)
        return f

    pass


class _BasicBlockS2(nn.Module):
    """
    Basic Block中的一种

    第一个卷积spatial、channel都不变
    加shortcut不需要1x1卷积
    第二个卷积spatial、channel都不变
    """

    def __init__(self, in_channels):
        super(_BasicBlockS2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        pass

    def forward(self, x):
        f = x
        f = self.conv1(f)
        f = self.bn1(f)
        f = self.relu1(f)
        f = self.conv2(f)
        f = f + x
        f = self.bn2(f)
        f = self.relu2(f)
        return f

    pass


def resnet18():
    """
    按照论文实现ResNet18
    :return: ResNet18
    """
    conv2 = [_BasicBlockS2(64), _BasicBlockS2(64)]
    conv3 = [_BasicBlockS1(64), _BasicBlockS2(128)]
    conv4 = [_BasicBlockS1(128), _BasicBlockS2(256)]
    conv5 = [_BasicBlockS1(256), _BasicBlockS2(512)]
    return _ResNetFactory(conv2, conv3, conv4, conv5, 512)


def resnet34():
    """
    按照论文实现ResNet34
    :return: ResNet34
    """
    conv2 = [_BasicBlockS2(64), _BasicBlockS2(64), _BasicBlockS2(64)]
    conv3 = [_BasicBlockS1(64), _BasicBlockS2(128),
             _BasicBlockS2(128), _BasicBlockS2(128)]
    conv4 = [_BasicBlockS1(128), _BasicBlockS2(256), _BasicBlockS2(256),
             _BasicBlockS2(256), _BasicBlockS2(256), _BasicBlockS2(256)]
    conv5 = [_BasicBlockS1(256), _BasicBlockS2(512), _BasicBlockS2(512),
             _BasicBlockS2(512), _BasicBlockS2(512), _BasicBlockS2(512)]
    return _ResNetFactory(conv2, conv3, conv4, conv5, 512)


################################################################################


################################################################################


if __name__ == '__main__':
    in_data = torch.randint(0, 256, (1, 3, 224, 224), dtype=torch.float32)
    print('in_data', in_data.shape)

    # net = resnet18()
    # net = resnet34()
    # out_data = net(in_data)
