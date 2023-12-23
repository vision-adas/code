import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from collections import OrderedDict
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork


class SegHead(nn.Module):
    def __init__(self, in_channels: list, num_classes: int, with_uncertainty: bool):
        super(SegHead, self).__init__()
        self.with_uncertainty = with_uncertainty
        in_channels = in_channels.copy()
        in_channels.reverse()
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(in_channels[0], in_channels[1], 3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(in_channels[1]),
            nn.UpsamplingNearest2d(scale_factor=2.0))

        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(in_channels[1], in_channels[2], 3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(in_channels[2]),
            nn.UpsamplingNearest2d(scale_factor=2.0))

        self.decoder_conv3 = nn.Sequential(
            nn.Conv2d(in_channels[2], in_channels[3], 3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(in_channels[3]),
            nn.UpsamplingNearest2d(scale_factor=2.0))

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels,
            out_channels=num_classes)

        if self.with_uncertainty:
            self.uncertainty_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(num_classes, 64),
                nn.Linear(64, 1))

    def forward(self, features: Dict[str, torch.Tensor]):
        feets = OrderedDict()
        feets["0"] = features["0"]
        feets["1"] = features["1"] + self.decoder_conv1(feets["0"])
        feets["2"] = features["2"] + self.decoder_conv2(feets["1"])
        feets["3"] = features["3"] + self.decoder_conv3(feets["2"])
        seg_outputs = self.fpn(feets)
        outputs = {"seg": seg_outputs}
        if self.with_uncertainty:
            outputs["seg_uncertainty"] = [torch.mean(self.uncertainty_layer(seg_outputs['0']))]
        return outputs

    def compute_loss(self, seg_outputs, segs, weights=[0.125, 0.25, 0.5, 1.0]):
        seg_loss_f = nn.CrossEntropyLoss()
        segs = segs.float()
        loss_scales = []
        scales = [1/32, 1/16, 1/8, 1/4]
        for index_str in seg_outputs:
            index = int(index_str)
            target = F.upsample_nearest(segs, scale_factor=scales[index]).long().squeeze(1)
            loss_scales.append(weights[index]*seg_loss_f(seg_outputs[index_str], target))
        return torch.sum(torch.stack(loss_scales))