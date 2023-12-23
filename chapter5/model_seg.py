from __future__ import division

import torch
import torch.nn as nn

from lib.backbone.resnet import resnet18, resnet50
from lib.backbone.regnet import regnet_x_400mf, regnet_x_800mf
from collections import OrderedDict
from lib.seg_head import SegHead


class RegnetBackbone(nn.Module):
    def __init__(self):
        super(RegnetBackbone, self).__init__()
        self.out_channels = [32, 64, 160, 400]  # [64, 128, 288, 672]
        self.regnet = regnet_x_400mf(pretrained=True)  # regnet_x_800mf()

    def forward(self, x):
        features = OrderedDict()
        x = self.regnet.stem(x)
        x = self.regnet.trunk_output.block1(x)
        features['3'] = x
        x = self.regnet.trunk_output.block2(x)
        features['2'] = x
        x = self.regnet.trunk_output.block3(x)
        features['1'] = x
        x = self.regnet.trunk_output.block4(x)
        features['0'] = x
        return features


class ResnetBackbone(nn.Module):
    def __init__(self, pretrained=True):
        super(ResnetBackbone, self).__init__()
        self.resnet = resnet18(pretrained=pretrained)
        # self.resnet = resnet50()
        # del self.resnet.fc
        # self.resnet.load_state_dict(torch.load("densecl_r50_coco_1600ep.pth")["state_dict"])
        self.out_channels = [64, 128, 256, 512]
        # self.out_channels = [64, 512, 1024, 2048]

    def forward(self, x):
        features = OrderedDict()

        x = self.resnet.conv1(x)  # stride 2
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # 64
        features['3'] = x

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)  # stride 2 128
        features['2'] = x
        x = self.resnet.layer3(x)  # stride 2 256
        features['1'] = x
        x = self.resnet.layer4(x)  # stride 2 512
        features['0'] = x
        return features


def build_backbone(backbone_type: str = "resnet"):
    if backbone_type == "resnet":
        return ResnetBackbone()
    elif backbone_type == "regnet":
        return RegnetBackbone()


class SegmentFPN(nn.Module):
    def __init__(self,
                 with_uncertainty=False,
                 num_seg_classes: int = None,
                 backbone: str = "resnet"):
        super(SegmentFPN, self).__init__()
        self.backbone = build_backbone(backbone)
        self.seg_head: SegHead = SegHead(in_channels=self.backbone.out_channels,
                                         num_classes=num_seg_classes,
                                         with_uncertainty=with_uncertainty)

    def train_forward(self, x):
        features = self.backbone(x)
        outputs = dict()
        seg_outputs = self.seg_head(features)
        outputs.update(seg_outputs)

        return outputs

    def forward(self, x):
        """用于部署的前向函数"""
        x = x.permute(0, 3, 1, 2)  # expect channels last input
        x = x/255.0
        features = self.backbone(x)
        seg_outputs = self.seg_head(features)["seg"]["3"]
        return torch.argmax(seg_outputs, dim=1, keepdim=True)

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, weights_path):
        state_dict = torch.load(weights_path)
        self.load_state_dict(state_dict)


if __name__ == '__main__':
    model = SegmentFPN(with_uncertainty=False,
                       num_seg_classes=19,
                       backbone="resnet")
    input = torch.rand((32, 3, 480, 640))
    output = model.train_forward(input)
    pass
