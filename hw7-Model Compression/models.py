import torch.nn as nn
import torch.nn.functional as F
import torch
# 一般的Convolution，weight大小 = in_chs * out_chs * kernel_size^2
# nn.Conv2d(in_chs, out_chs, kernel_size, stride, paddding)

# Group Convolution，Group数目可以自行控制，表示要分成几群。其中in_chs和out_chs必要可以被groups整除
# nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, groups)

# Depthwise Convolution，in_chs=out_chs=groups，weight大小 = in_chs * kernel_size^2
# nn.Conv2d(in_chs, out_channels=in_chs, kernel_size, stride, groups=in_chs)

# Pointwise Convolution，也就是one by one Convolution，weight大小 = in_chs * outchs
# nn.Conv2d(in_chs, out_chs, 1)


class StudentNet(nn.Module):
    def __init__(self, base=16, width_mult=1):
        '''
        :param base: 这个model一开始的channel数量，每过一层都为*2，知道base*16为止。
        :param width_mult:  为了之后的Network Pruning使用，在base*8 chs的Layer上会 * width_mult代表剪枝后的channel数量。
        '''
        super(StudentNet, self).__init__()
        multiplier = [1, 2, 4, 8, 16, 16, 16, 16]

        # bandwidth：每一层使用的channel数量
        bandwidth = [base * m for m in multiplier]

        # 只pruning第三层之后的Layer
        for i in range(3, 7):
            bandwidth[i] = int(bandwidth[i] * width_mult)

        self.cnn = nn.Sequential(
            # 第一层通常不会拆解Conv
            nn.Sequential(
                nn.Conv2d(3, bandwidth[0], 3, 1, 1),
                nn.BatchNorm2d(bandwidth[0]),
                nn.ReLU6(),
                nn.MaxPool2d(2, 2, 0),
            ),
            # 接下来每一个Sequential Block都一样
            nn.Sequential(
                # Depthwise Conv
                nn.Conv2d(bandwidth[0], bandwidth[0], 3, 1, 1, groups=bandwidth[0]),
                # Batch Norm
                nn.BatchNorm2d(bandwidth[0]),
                # ReLU6 restrict the size Neuron in range(0, 6)
                nn.ReLU6(),
                # Pointwise Conv
                nn.Conv2d(bandwidth[0], bandwidth[1], 1),
                # after Pointwise Conv, don't need ReLU
                nn.MaxPool2d(2, 2, 0),
            ),
            nn.Sequential(
                nn.Conv2d(bandwidth[1], bandwidth[1], 3, 1, 1, groups=bandwidth[1]),
                nn.BatchNorm2d(bandwidth[1]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[1], bandwidth[2], 1),
                nn.MaxPool2d(2, 2, 0),
            ),
            nn.Sequential(
                nn.Conv2d(bandwidth[2], bandwidth[2], 3, 1, 1, groups=bandwidth[2]),
                nn.BatchNorm2d(bandwidth[2]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[2], bandwidth[3], 1),
                nn.MaxPool2d(2, 2, 0),
            ),
            # until now the img have been down sample many times, don't need MaxPool
            nn.Sequential(
                nn.Conv2d(bandwidth[3], bandwidth[3], 3, 1, 1, groups=bandwidth[3]),
                nn.BatchNorm2d(bandwidth[3]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[3], bandwidth[4], 1),
            ),
            nn.Sequential(
                nn.Conv2d(bandwidth[4], bandwidth[4], 3, 1, 1, groups=bandwidth[4]),
                nn.BatchNorm2d(bandwidth[4]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[4], bandwidth[5], 1),
            ),
            nn.Sequential(
                nn.Conv2d(bandwidth[5], bandwidth[5], 3, 1, 1, groups=bandwidth[5]),
                nn.BatchNorm2d(bandwidth[5]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[5], bandwidth[6], 1),
            ),
            nn.Sequential(
                nn.Conv2d(bandwidth[6], bandwidth[6], 3, 1, 1, groups=bandwidth[6]),
                nn.BatchNorm2d(bandwidth[6]),
                nn.ReLU6(),
                nn.Conv2d(bandwidth[6], bandwidth[7], 1),
            ),
            # 采用Global Average Pooling
            nn.AdaptiveAvgPool2d((1, 1)),
        ),
        self.fc = nn.Sequential(
            # project to 11 dimension
            nn.Linear(bandwidth[7], 11),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)