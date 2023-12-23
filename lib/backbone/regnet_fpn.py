import warnings

import torch
from torch import nn
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from detection_torchvision.retinanet import RetinaNet
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops import misc as misc_nn_ops
from torchvision.models._utils import IntermediateLayerGetter
from lib.backbone.regnet import regnet_x_800mf, regnet_x_400mf
from collections import OrderedDict


class RegNetWithFPN(nn.Module):
    """
    Adds a FPN on top of a models.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediateLayerGetter apply here.
    Args:
        backbone (nn.Module)
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
        in_channels_list (List[int]): number of channels for each feature map
            that is returned, in the order they are present in the OrderedDict
        out_channels (int): number of channels in the FPN.
    Attributes:
        out_channels (int): the number of channels in the FPN
    """
    def __init__(self, out_channels, pretrained: bool, type="400m"):
        super(RegNetWithFPN, self).__init__()
        if "800m" in type:
            self.regnet = regnet_x_800mf(pretrained)
            self.fpn = FeaturePyramidNetwork(
                in_channels_list=[64, 128, 288, 672],
                out_channels=out_channels,
                extra_blocks=LastLevelMaxPool(),
            )
        elif "400m" in type:
            self.regnet = regnet_x_400mf(pretrained)
            self.fpn = FeaturePyramidNetwork(
                in_channels_list=[32, 64, 160, 400],
                out_channels=out_channels,
                extra_blocks=LastLevelMaxPool(),
            )
        self.out_channels = out_channels

    def forward(self, x):
        features = OrderedDict()
        #x = (x - 0.45) / 0.225
        x = self.regnet.stem(x)
        x = self.regnet.trunk_output.block1(x)
        features['3'] = x
        x = self.regnet.trunk_output.block2(x)
        features['2'] = x
        x = self.regnet.trunk_output.block3(x)
        features['1'] = x
        x = self.regnet.trunk_output.block4(x)
        features['0'] = x

        x = self.fpn(features)

        return x


if __name__ == '__main__':
    image = torch.rand((1, 3, 480, 640), dtype=torch.float32)

    backbone = RegNetWithFPN(out_channels=256, pretrained=True)
    model = FasterRCNN(backbone, num_classes=2, image_mean=[0.4, 0.4, 0.4], image_std=[1, 1, 1])

    backbone = RegNetWithFPN(out_channels=256, pretrained=True)
    model = RetinaNet(backbone, num_classes=2, image_mean=[0.4, 0.4, 0.4], image_std=[1, 1, 1])

    model.eval()
    output = model(image)
    pass