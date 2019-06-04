"""
Module containing the different ML Models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FirstNet(nn.Module):
    """
    Net completely based on the pytorch tutorial
    images are 20*20
    """

    def __init__(self):
        super(FirstNet, self).__init__()
        # 5 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv0 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=1)
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=32, kernel_size=5, padding=int(5 / 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=9, padding=int(9 / 2))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=7, padding=int(7 / 2))
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1)
        self.cuda()

    def forward(self, *input_data):
        data_x = input_data[0]
        data_x = F.relu(self.conv0(data_x))
        data_x = F.relu(self.conv1(data_x))
        data_x = F.relu(self.conv2(data_x))
        data_x = F.relu(self.conv3(data_x))
        data_x = F.relu(self.conv4(data_x))

        return data_x

    def forward_single(self, data_x):
        """
        Takes a single element and predicts its value
        """
        data_x = data_x.unsqueeze(0)
        data_x = self.forward(data_x)
        return data_x.squeeze(0)


class SimpleNet(nn.Module):
    """
    Net completely based on the pytorch tutorial
    images are 20*20
    """

    def __init__(self):
        super(SimpleNet, self).__init__()
        # 5 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        layers = []
        layers.append(nn.Conv2d(in_channels=5, out_channels=256, kernel_size=5, padding=2, groups=1, bias=False))
        # layers.append(nn.ReLU(inplace=True))
        # layers.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, groups=1, bias=False))
        layers.append(nn.BatchNorm2d(num_features=256))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=256, out_channels=3, kernel_size=5, padding=2, groups=1))
        self.dncnn = nn.Sequential(*layers)
        self.cuda()

    def forward(self, *input_data):
        data_x = input_data[0]
        out = self.dncnn(data_x * 1)
        return data_x[:, :3, :, ] + out  # + out  # out +

    def forward_single(self, data_x):
        """
        Takes a single element and predicts its value
        """
        data_x = data_x.unsqueeze(0)
        data_x = self.forward(data_x)
        return data_x.squeeze(0)
