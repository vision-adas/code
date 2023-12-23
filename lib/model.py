from __future__ import division

import torch
import torch.nn as nn

from lib.backbone.resnet import resnet18, resnet50
from lib.backbone.regnet import regnet_x_400mf, regnet_x_800mf
from collections import OrderedDict
import functools
from lib.yolo_head import YoloHead
from lib.seg_head import SegHead


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


class FakeQuantizationWrapper(torch.nn.Module):
    """用于Quantization Aware Training的伪量化模型Wrapper，示例中未使用"""
    def __init__(self, model_fp32):
        super(FakeQuantizationWrapper, self).__init__()
        # QuantStub converts tensors from floating point to quantized
        self.quant = torch.quantization.QuantStub()
        self.model_fp32 = model_fp32
        # DeQuantStub converts tensors from quantized to floating point
        self.dequant = torch.quantization.DeQuantStub()

    def train_forward(self, x):
        x = self.quant(x)
        x = self.model_fp32.train_forward(x)
        output = dict()
        if "det" in x:
            output["det"] = [self.dequant(out) for out in x["det"]]
        if "seg" in x:
            output["seg"] = OrderedDict()
            for key in x["seg"]:
                output["seg"][key] = self.dequant(x["seg"][key])
        return x


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


class MultiTask(nn.Module):
    def __init__(self,
                 with_seg=False,
                 with_det=True,
                 with_uncertainty=True,
                 num_seg_classes: int = None,
                 backbone: str = "regnet"):
        super(MultiTask, self).__init__()
        self.with_seg = with_seg
        self.with_det = with_det
        self.backbone = build_backbone(backbone)

        if self.with_seg:
            self.seg_head: SegHead = SegHead(in_channels=self.backbone.out_channels,
                                             num_classes=num_seg_classes,
                                             with_uncertainty=with_uncertainty)

        if self.with_det:
            self.det_head: YoloHead = YoloHead(input_channels=self.backbone.out_channels,
                                               with_uncertainty=with_uncertainty)

    def train_forward(self, x):
        features = self.backbone(x)
        outputs = dict()
        if self.with_det:
            det_outputs = self.det_head(features)
            outputs.update(det_outputs)

        if self.with_seg:
            seg_outputs = self.seg_head(features)
            outputs.update(seg_outputs)

        return outputs

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # expect channels last input
        x = x/255.0
        features = self.backbone(x)
        seg_outputs = self.seg_head(features)["seg"]["3"]
        # det_outputs = self.det_head(features)["det"]
        return torch.argmax(seg_outputs, dim=1, keepdim=True)  # , det_outputs

    def infer_multi(self, x):
        features = self.backbone(x)
        det_outputs = self.det_head(features)["det"]
        seg_outputs = self.seg_head(features)["seg"]["3"]
        return det_outputs, seg_outputs

    def infer_seg(self, x):
        features = self.backbone(x)
        seg_outputs = self.seg_head(features)["seg"]["3"]
        return seg_outputs

    def save_weights(self, path):
        torch.save(self.state_dict(), path)

    def load_weights(self, weights_path):
        state_dict = torch.load(weights_path)
        state_dict = {key.replace("_orig_mod.", ""): state_dict[key] for key in state_dict}
        self.load_state_dict(state_dict)

    def create_compression_configs(self, threshold: float = 10e-8):
        configs = []
        for name, para in self.named_parameters():
            if "channel_mask" in name:
                curr_config = dict()
                subnetwork, backbone, layer, index, mask = name.split(".")
                curr_config["layer"] = subnetwork+"."+backbone+"."+layer
                curr_config["index"] = int(index)
                curr_config["all"] = para.shape[0]
                curr_config["channels"] = (
                    para < threshold).nonzero().cpu().numpy().flatten().tolist()
                print("{} ouf of {} channels of {} compressed".format(len(curr_config["channels"]),
                                                                      curr_config["all"],
                                                                      name))
                configs.append(curr_config)
        return configs

    def compress(self, compress_config: dict):
        # 遍历压缩方案中的相关网络层
        for compress_targe in compress_config:
            # 获得网络层的名字
            layer_str = compress_targe["layer"]
            # 获得网络层的编号
            layer_index = compress_targe["index"]
            # 获得通道总数
            full_channel = compress_targe["all"]
            # 获得待压缩的通道数
            channel_to_compress = compress_targe["channels"]
            # 建立通道掩膜，待压缩通道对应False，需要留下来的对应True
            mask = [(not i in channel_to_compress)
                    for i in range(full_channel)]
            # 获得本网络层的实例
            layer = rgetattr(self, layer_str)[layer_index]
            # 修改第一个卷积层的权重
            layer.conv1.weight = nn.Parameter(
                layer.conv1.weight[mask, :, :, :])
            # 修改第一个卷积层的输出通道数
            layer.conv1.out_channels = full_channel-len(channel_to_compress)
            # 若卷积层有偏置，也对偏置进行压缩
            if not layer.conv1.bias is None:
                layer.conv1.bias = nn.Parameter(layer.conv1.bias[mask])

            # 对批归一化层进行压缩
            layer.bn1.weight = nn.Parameter(layer.bn1.weight[mask])
            layer.bn1.bias = nn.Parameter(layer.bn1.bias[mask])
            # 批归一化层的移动平均值和移动方差不需要计算梯度
            layer.bn1.running_mean = nn.Parameter(layer.bn1.running_mean[mask],
                                                  requires_grad=False)
            layer.bn1.running_var = nn.Parameter(layer.bn1.running_var[mask],
                                                 requires_grad=False)
            layer.bn1.num_features = full_channel-len(channel_to_compress)

            # 修改第二个卷积层的输入通道数
            layer.conv2.in_channels = full_channel - len(channel_to_compress)
            # 修改第二个卷积层的权重
            conv2_weight = layer.conv2.weight[:, mask, :, :]

            # 若使用通道掩膜，则将通道掩膜与第二个卷积层融合
            if hasattr(layer, "channel_mask"):
                # 获得压缩后的通道掩膜
                channel_mask = layer.channel_mask[mask]
                # 将通道掩膜扩展为和卷积层权重一样的尺寸
                channel_mask = torch.repeat_interleave(
                    channel_mask.unsqueeze(-1), conv2_weight.size(2), -1)
                channel_mask = torch.repeat_interleave(
                    channel_mask.unsqueeze(-1), conv2_weight.size(3), -1)
                channel_mask = torch.repeat_interleave(
                    channel_mask.unsqueeze(0), conv2_weight.size(0), 0)
                # 将通道掩膜与卷积层权重融合
                layer.conv2.weight = nn.Parameter(conv2_weight*channel_mask)
                # 融合结束后，关闭通道掩膜
                layer.with_mask = False
            else:
                # 若不使用通道掩膜，则直接保存修改后的权重
                layer.conv2.weight = nn.Parameter(conv2_weight)


if __name__ == '__main__':
    model = MultiTask(with_seg=True, with_det=True,
                      with_batch_sigma=True, num_seg_classes=19)
    input = torch.rand((32, 3, 480, 640))
    output = model.train_forward(input)
    pass
