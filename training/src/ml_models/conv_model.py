"""
Module containing the different ML Models
"""
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
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=6, kernel_size=5, padding=int(5 / 2))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=int(5 / 2))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=5, padding=int(5 / 2))

    def forward(self, *input_data):
        data_x = input_data[0]
        data_x = F.relu(self.conv1(data_x))
        data_x = F.relu(self.conv2(data_x))
        data_x = F.relu(self.conv3(data_x))

        return data_x

    def num_flat_features(self, data_x):
        """
        number of flat features
        """
        size = data_x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for i in size:
            num_features *= i
        return num_features
