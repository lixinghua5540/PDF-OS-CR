import torch
import torch.nn as nn

from models.discriminator.block_branch import BlockBranch
from models.discriminator.edge_branch import EdgeBranch, EdgeDetector


class Discriminator(nn.Module):

    def __init__(self, optical_channels, sar_channels):
        super(Discriminator, self).__init__()

        self.block_branch = BlockBranch(in_channels=optical_channels)
        self.edge_branch = EdgeBranch(in_channels=sar_channels)
        self.edge_detector = EdgeDetector()

    def forward(self, output, sar, sar_edge, is_real):

        if is_real == True:

            block_pred = self.block_branch(output)
            fake_edge = self.edge_detector(output)
            edge_pred = self.edge_branch(torch.cat((sar_edge, sar), dim=1))

        else:

            block_pred = self.block_branch(output)
            fake_edge = self.edge_detector(output)
            edge_pred = self.edge_branch(torch.cat((fake_edge, sar), dim=1))

        return torch.cat((block_pred, edge_pred), dim=1), fake_edge
        
