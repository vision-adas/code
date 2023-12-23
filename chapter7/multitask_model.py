from __future__ import division

import torch
import torch.nn as nn

from lib.backbone.resnet import resnet18, resnet50
from lib.backbone.regnet import regnet_x_400mf, regnet_x_800mf
from collections import OrderedDict
import functools
from lib.yolo_head import YoloHead
from lib.seg_head import SegHead


class RegnetBackbone(nn.Module):
    """RegNet主干网络"""

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
    """ResNet主干网络"""

    def __init__(self, pretrained=True):
        super(ResnetBackbone, self).__init__()
        # 若使用ResNet18作为主干网络，记录输出通道数
        self.resnet = resnet18(pretrained=pretrained)
        self.out_channels = [64, 128, 256, 512]
        # 若使用ResNet50作为主干网络，则使用下面的代码
        # self.resnet = resnet50(pretrained=pretrained)
        # self.resnet = resnet50()
        # del self.resnet.fc
        # self.resnet.load_state_dict(torch.load("densecl_r50_coco_1600ep.pth")["state_dict"])
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
    """根据backbone_type构建主干网络"""
    if backbone_type == "resnet":
        return ResnetBackbone()
    elif backbone_type == "regnet":
        return RegnetBackbone()


class MultiTask(nn.Module):
    """多任务模型"""

    def __init__(self,
                 with_seg=False,  # 是否使用语义分割
                 with_det=True,   # 是否使用目标检测
                 with_uncertainty=True,  # 是否使用不确定性加权
                 num_seg_classes: int = None,  # 语义分割类别数
                 backbone: str = "regnet"):    # 主干网络类型
        super(MultiTask, self).__init__()
        self.with_seg = with_seg
        self.with_det = with_det
        self.backbone = build_backbone(backbone)

        if self.with_seg:
            # 若使用语义分割，则构建语义分割头
            self.seg_head: SegHead = SegHead(in_channels=self.backbone.out_channels,
                                             num_classes=num_seg_classes,
                                             with_uncertainty=with_uncertainty)

        if self.with_det:
            # 若使用目标检测，则构建目标检测头
            self.det_head: YoloHead = YoloHead(input_channels=self.backbone.out_channels,
                                               with_uncertainty=with_uncertainty)

    def train_forward(self, x):
        features = self.backbone(x)
        outputs = dict()
        if self.with_det:
            # 若使用目标检测，则计算目标检测输出
            det_outputs = self.det_head(features)
            outputs.update(det_outputs)

        if self.with_seg:
            # 若使用语义分割，则计算语义分割输出
            seg_outputs = self.seg_head(features)
            outputs.update(seg_outputs)

        return outputs

    def forward(self, x):
        # 用于推理的前向传播，输入为HxWxC的uint8图像
        x = x.permute(0, 3, 1, 2)  # expect channels last input
        x = x/255.0
        features = self.backbone(x)
        seg_outputs = self.seg_head(features)["seg"]["3"]
        # 示例代码仅返回语义分割结果
        # det_outputs = self.det_head(features)["det"]
        return torch.argmax(seg_outputs, dim=1, keepdim=True)  # , det_outputs

    def infer_multi(self, x):
        # 用于推理的前向传播，输入为HxWxC的uint8图像，输出为语义分割和目标检测结果
        features = self.backbone(x)
        det_outputs = self.det_head(features)["det"]
        seg_outputs = self.seg_head(features)["seg"]["3"]
        return det_outputs, seg_outputs

    def infer_seg(self, x):
        # 用于推理的前向传播，输入为HxWxC的uint8图像，输出为语义分割结果
        features = self.backbone(x)
        seg_outputs = self.seg_head(features)["seg"]["3"]
        return seg_outputs

    def save_weights(self, path):
        # 保存模型权重
        torch.save(self.state_dict(), path)

    def load_weights(self, weights_path):
        # 加载模型权重
        state_dict = torch.load(weights_path, , map_location="cpu")
        self.load_state_dict(state_dict)


if __name__ == '__main__':
    model = MultiTask(with_seg=True, with_det=True, with_batch_sigma=True, num_seg_classes=19)
    input = torch.rand((32, 3, 480, 640))
    output = model.train_forward(input)
    pass
