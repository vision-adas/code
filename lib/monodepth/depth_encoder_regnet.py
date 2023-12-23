from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
from lib.backbone.regnet import regnet_x_800mf
import lib.monodepth.depth_encoder as depth_encoder


class DepthEncoderRegNet(nn.Module):
    def __init__(self, pretrained=False):
        super(DepthEncoderRegNet, self).__init__()
        self.regnet = regnet_x_800mf(pretrained=pretrained)
        self.num_ch_enc = np.array([32, 64, 128, 288, 672])

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        self.features.append(self.regnet.stem(x))
        self.features.append(self.regnet.trunk_output.block1(self.features[-1]))
        self.features.append(self.regnet.trunk_output.block2(self.features[-1]))
        self.features.append(self.regnet.trunk_output.block3(self.features[-1]))
        self.features.append(self.regnet.trunk_output.block4(self.features[-1]))

        return self.features


if __name__ == '__main__':
    regnet = DepthEncoderRegNet(pretrained=True).cuda()
    image = torch.rand((4, 3, 480, 640), device="cuda:0").float()
    result = regnet(image)
    resnet = depth_encoder.DepthEncoder(18).cuda()
    output = resnet(image)
