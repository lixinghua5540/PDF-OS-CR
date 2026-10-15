import torch
import torch.nn as nn
import torch.nn.functional as F


class CloudAttentionMap(nn.Module):
    def __init__(self, in_channels):
        super(CloudAttentionMap, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, image):

        output = self.relu(self.bn1(self.conv1(image)))  
        output = self.relu(self.bn2(self.conv2(output)))  

        output_up = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)  

        output = self.relu(self.bn3(self.conv3(output_up)))  

        output_up = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)  

        output = self.sigmoid(self.conv4(output_up)) 

        return output





