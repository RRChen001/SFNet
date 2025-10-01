import torch.nn as nn
import torch


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = x.mul(self.sigmoid(out))
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 输入两个通道，一个是maxpool 一个是avgpool的
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)  # 对池化完的数据cat 然后进行卷积
        out = x.mul(self.sigmoid(out))
        return out


class SCA(nn.Module):
    def __init__(self, chanel):
        super(SCA, self).__init__()

        # 通道空间融合
        self.conv1 = nn.Conv2d(chanel * 2, chanel, kernel_size=1)
        self.spatial = SpatialAttention(7)
        self.channel = ChannelAttention(chanel)
        self.conv2 = nn.Sequential(
            nn.Conv2d(chanel * 2, chanel, kernel_size=1),
            nn.ReLU(True), )
        self.conv_gamma = nn.Conv2d(chanel, chanel, kernel_size=3, padding=1)
        self.conv_gamma2 = nn.Conv2d(chanel, chanel, kernel_size=3, padding=1)
        self.conv_beta = nn.Conv2d(chanel, chanel, kernel_size=3, padding=1)
        self.conv_beta2 = nn.Conv2d(chanel, chanel, kernel_size=3, padding=1)
        self.param_free_norm = nn.BatchNorm2d(chanel, affine=True)

    def forward(self, mx, tx):
        x = torch.cat([mx, tx], 1) #沿着通道拼接
        x = self.conv1(x)
        sx = self.spatial(x)
        cx = self.channel(x)
        y = torch.cat([sx, cx], 1)
        y = self.conv2(y)
        gamma = self.conv_gamma(y)
        beta = self.conv_beta(y)
        gamma2 = self.conv_gamma2(y)#
        beta2 = self.conv_beta2(y)#

        normalized_tx = self.param_free_norm(tx)
        normalized_mx = self.param_free_norm(mx)#

        out1 = normalized_tx * (1 + gamma) + beta
        out2 = normalized_mx * (1 + gamma2) + beta2#
        out = torch.cat([normalized_mx, normalized_tx], 1)#
        out = self.conv1(out)#

        return out

