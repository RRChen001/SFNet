import torch.nn as nn
import torch


class MSFA(nn.Module):

    def __init__(self, in_channels, out_channels, dilation_rate_list=[1, 2, 4]):
        super(MSFA, self).__init__()

        self.dilation_rate_list = dilation_rate_list

        for _, dilation_rate in enumerate(dilation_rate_list):
            self.__setattr__('dilated_conv_{:d}'.format(_), nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilation_rate, padding=dilation_rate),
                nn.ReLU(inplace=True))
                             )

        self.weight_calc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, len(dilation_rate_list), 1),
            nn.ReLU(inplace=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):

        weight_map = self.weight_calc(x)

        x_feature_list = []
        for _, dilation_rate in enumerate(self.dilation_rate_list):
            x_feature_list.append(
                self.__getattr__('dilated_conv_{:d}'.format(_))(x)
            )

        output = weight_map[:, 0:1, :, :] * x_feature_list[0] + \
                 weight_map[:, 1:2, :, :] * x_feature_list[1] + \
                 weight_map[:, 2:3, :, :] * x_feature_list[2]

        output = torch.cat([x, output], 1)

        return output


class GFT(nn.Module):
    def __init__(self, chanel):
        super(GFT, self).__init__()

        # 通道空间融合
        self.msfa = MSFA(chanel, chanel)
        self.conv1 = nn.Conv2d(in_channels=chanel * 2, out_channels=chanel, kernel_size=1)

    def forward(self, mx, tx):
        mx_mean = torch.mean(mx, dim=[2, 3])
        mx_squared_sum = torch.mean(mx ** 2, dim=[2, 3])
        mx_std = (mx_squared_sum - mx_mean ** 2) ** 0.5
        tx_mean = torch.mean(tx, dim=[2, 3])
        tx_squared_sum = torch.mean(tx ** 2, dim=[2, 3])
        tx_std = (tx_squared_sum - tx_mean ** 2) ** 0.5
        x = (tx_std.unsqueeze(-1).unsqueeze(-1)).mul((mx - mx_mean.unsqueeze(-1).unsqueeze(-1)) / mx_std.unsqueeze(-1).unsqueeze(-1)) + tx_mean.unsqueeze(-1).unsqueeze(-1)
        x = self.msfa(x)
        out = self.conv1(x)

        return out