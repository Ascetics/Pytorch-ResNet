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
                               bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, ceil_mode=True)

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
        x = self.conv1(x)  # 1/2
        x = self.bn(x)
        x = self.maxpool(x)  # 1/4
        x = self.conv2(x)
        x = self.conv3(x)  # 1/8
        x = self.conv4(x)  # 1/16
        x = self.conv5(x)  # 1/32
        x = self.avgpool(x)  # 输出是4维张量
        x = x.view(1, -1)  # 变成1维向量
        x = self.fc(x)
        return x

    pass


################################################################################

class _BasicBlockDown(nn.Module):
    """
    Basic Block中的一种

    第一个卷积Downsample
    spatial减小一倍，stride=2
    channel增大一倍，out_channels = 2*in_channels

    加shortcut需要1x1卷积调整维度
    因第一个卷积spatial减小一倍，1x1卷积的stride=2

    第二个卷积
    spatial不变
    channel不变
    """

    def __init__(self, in_channels, out_channels):
        super(_BasicBlockDown, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=2,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 调整shortcut的channel
        # 调整shortcut的spatial，stride=2
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu2 = nn.ReLU(inplace=True)

        pass

    def forward(self, x):
        f = x
        f = self.conv1(f)
        f = self.bn1(f)
        f = self.relu1(f)
        f = self.conv2(f)
        f = self.bn2(f)
        f += self.project(x)  # 调整维度才能相加
        f = self.relu2(f)
        return f

    pass


class _BasicBlockSame(nn.Module):
    """
    Basic Block中的一种

    第一个卷积spatial、channel都不变
    加shortcut不需要1x1卷积
    第二个卷积spatial、channel都不变
    """

    def __init__(self, in_channels, out_channels):
        super(_BasicBlockSame, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        pass

    def forward(self, x):
        f = x
        f = self.conv1(f)
        f = self.bn1(f)
        f = self.relu1(f)
        f = self.conv2(f)
        f = self.bn2(f)
        f += x
        f = self.relu2(f)
        return f

    pass


def resnet18():
    """
    按照论文实现ResNet18
    :return: ResNet18
    """
    # conv2不做Downsample
    conv2 = [_BasicBlockSame(64, 64), _BasicBlockSame(64, 64)]

    # conv3-conv5第一个block做Downsample
    conv3 = [_BasicBlockDown(64, 128), _BasicBlockSame(128, 128)]

    conv4 = [_BasicBlockDown(128, 256), _BasicBlockSame(256, 256)]

    conv5 = [_BasicBlockDown(256, 512), _BasicBlockSame(512, 512)]

    return _ResNetFactory(conv2, conv3, conv4, conv5, 512)


def resnet34():
    """
    按照论文实现ResNet34
    :return: ResNet34
    """
    # conv2不做Downsample
    conv2 = [_BasicBlockSame(64, 64), _BasicBlockSame(64, 64),
             _BasicBlockSame(64, 64)]

    # conv3-conv5第一个block做Downsample
    conv3 = [_BasicBlockDown(64, 128), _BasicBlockSame(128, 128),
             _BasicBlockSame(128, 128), _BasicBlockSame(128, 128)]

    conv4 = [_BasicBlockDown(128, 256), _BasicBlockSame(256, 256),
             _BasicBlockSame(256, 256), _BasicBlockSame(256, 256),
             _BasicBlockSame(256, 256), _BasicBlockSame(256, 256)]

    conv5 = [_BasicBlockDown(256, 512), _BasicBlockSame(512, 512),
             _BasicBlockSame(512, 512), _BasicBlockSame(512, 512),
             _BasicBlockSame(512, 512), _BasicBlockSame(512, 512)]

    return _ResNetFactory(conv2, conv3, conv4, conv5, 512)


################################################################################

class _BottleneckBlockForConv2(nn.Module):
    """
    Bottleneck Block的一种，用于Conv2，不做Downsample
    """

    def __init__(self, in_channels=64, out_channels=256):
        super(_BottleneckBlockForConv2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 1,
                               bias=False)  # 第一个卷积不改channel
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1,
                               bias=False)  # 第二个卷积不做Downsample
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        # 调整维度
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu3 = nn.ReLU(inplace=True)

        pass

    def forward(self, x):
        f = x
        f = self.conv1(f)
        f = self.bn1(f)
        f = self.relu1(f)
        f = self.conv2(f)
        f = self.bn2(f)
        f = self.relu2(f)
        f = self.conv3(f)
        f = self.bn3(f)
        f += self.project(x)
        f = self.relu3(f)
        return f

    pass


class _BottleneckBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_BottleneckBlockDown, self).__init__()
        mid_channels = in_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride=2,
                               padding=1, bias=False)  # 做Downsample
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        # 调整维度
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu3 = nn.ReLU(inplace=True)

        pass

    def forward(self, x):
        f = x
        f = self.conv1(f)
        f = self.bn1(f)
        f = self.relu1(f)
        f = self.conv2(f)
        f = self.bn2(f)
        f = self.relu2(f)
        f = self.conv3(f)
        f = self.bn3(f)
        f += self.project(x)
        f = self.relu3(f)
        return f

    pass


class _BottleneckBlockSame(nn.Module):
    """
    Bottleneck Block的一种，每个Conv层里面不做Downsample，重复的block
    """

    def __init__(self, in_channels, out_channel):
        super(_BottleneckBlockSame, self).__init__()
        mid_channels = in_channels // 4
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(mid_channels, out_channel, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)
        self.relu3 = nn.ReLU(inplace=True)
        pass

    def forward(self, x):
        f = x
        f = self.conv1(f)
        f = self.bn1(f)
        f = self.relu1(f)
        f = self.conv2(f)
        f = self.bn2(f)
        f = self.relu2(f)
        f = self.conv3(f)
        f = self.bn3(f)
        f += x
        f = self.relu3(f)
        return f

    pass


def resnet50():
    """
    按照论文实现ResNet50
    :return: ResNet50
    """
    conv2 = [_BottleneckBlockForConv2(64, 256)]
    for i in range(2):
        conv2.append(_BottleneckBlockSame(256, 256))

    conv3 = [_BottleneckBlockDown(256, 512)]
    for i in range(3):
        conv3.append(_BottleneckBlockSame(512, 512))

    conv4 = [_BottleneckBlockDown(512, 1024)]
    for i in range(5):
        conv4.append(_BottleneckBlockSame(1024, 1024))

    conv5 = [_BottleneckBlockDown(1024, 2048)]
    for i in range(2):
        conv5.append(_BottleneckBlockSame(2048, 2048))

    return _ResNetFactory(conv2, conv3, conv4, conv5, 2048)


def resnet101():
    """
    按照论文实现ResNet101
    :return: ResNet101
    """
    conv2 = [_BottleneckBlockForConv2(64, 256)]
    for i in range(2):
        conv2.append(_BottleneckBlockSame(256, 256))

    conv3 = [_BottleneckBlockDown(256, 512)]
    for i in range(3):
        conv3.append(_BottleneckBlockSame(512, 512))

    conv4 = [_BottleneckBlockDown(512, 1024)]
    for i in range(22):
        conv4.append(_BottleneckBlockSame(1024, 1024))

    conv5 = [_BottleneckBlockDown(1024, 2048)]
    for i in range(2):
        conv5.append(_BottleneckBlockSame(2048, 2048))

    return _ResNetFactory(conv2, conv3, conv4, conv5, 2048)


def resnet152():
    """
    按照论文实现ResNet152
    :return: ResNet152
    """
    conv2 = [_BottleneckBlockForConv2(64, 256)]
    for i in range(2):
        conv2.append(_BottleneckBlockSame(256, 256))

    conv3 = [_BottleneckBlockDown(256, 512)]
    for i in range(7):
        conv3.append(_BottleneckBlockSame(512, 512))

    conv4 = [_BottleneckBlockDown(512, 1024)]
    for i in range(35):
        conv4.append(_BottleneckBlockSame(1024, 1024))

    conv5 = [_BottleneckBlockDown(1024, 2048)]
    for i in range(2):
        conv5.append(_BottleneckBlockSame(2048, 2048))

    return _ResNetFactory(conv2, conv3, conv4, conv5, 2048)


################################################################################


if __name__ == '__main__':
    in_data = torch.randint(0, 256, (1, 3, 224, 224), dtype=torch.float32)
    print('in_data', in_data.shape)

    net = resnet18()
    # net = resnet34()
    # net = resnet50()
    # net = resnet50()
    # net = resnet101()
    # net = resnet152()
    out_data = net(in_data)
