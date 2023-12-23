from __future__ import division

import torch
import torch.nn as nn
import functools
from chapter7.multitask_model import MultiTask


def rsetattr(obj, attr, val):
    # 递归设置属性
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    # 递归获取属性
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


class MultiTaskCompress(MultiTask):
    """带可微通道压缩的多任务模型"""

    def __init__(self, **kwargs):
        super(MultiTaskCompress, self).__init__(**kwargs)

    def load_weights(self, weights_path):
        # 加载模型权重
        state_dict = torch.load(weights_path, map_location="cpu")
        state_dict = {key.replace("_orig_mod.", ""): state_dict[key] for key in state_dict}
        self.load_state_dict(state_dict)

    def create_compression_configs(self, threshold: float = 10e-8):
        # 创建压缩方案
        configs = []
        for name, para in self.named_parameters():
            if "channel_mask" in name:
                # 若是通道掩膜，则根据设定的threshold记录需要压缩的通道
                curr_config = dict()
                subnetwork, backbone, layer, index, mask = name.split(".")
                curr_config["layer"] = subnetwork+"."+backbone+"."+layer
                curr_config["index"] = int(index)
                curr_config["all"] = para.shape[0]
                # 筛选出重要性小于threshold的通道编号
                curr_config["channels"] = (para < threshold).nonzero().cpu().numpy().flatten().tolist()
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
            mask = [(not i in channel_to_compress) for i in range(full_channel)]
            # 获得本网络层的实例
            layer = rgetattr(self, layer_str)[layer_index]
            # 修改第一个卷积层的权重
            layer.conv1.weight = nn.Parameter(layer.conv1.weight[mask, :, :, :])
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
                channel_mask = torch.repeat_interleave(channel_mask.unsqueeze(-1), conv2_weight.size(2), -1)
                channel_mask = torch.repeat_interleave(channel_mask.unsqueeze(-1), conv2_weight.size(3), -1)
                channel_mask = torch.repeat_interleave(channel_mask.unsqueeze(0), conv2_weight.size(0), 0)
                # 将通道掩膜与卷积层权重融合
                layer.conv2.weight = nn.Parameter(conv2_weight*channel_mask)
                # 融合结束后，关闭通道掩膜
                layer.with_mask = False
            else:
                # 若不使用通道掩膜，则直接保存修改后的权重
                layer.conv2.weight = nn.Parameter(conv2_weight)


if __name__ == '__main__':
    model = MultiTaskCompress(with_seg=True, with_det=True, with_uncertainty=True, num_seg_classes=19)
    input = torch.rand((32, 3, 480, 640))
    output = model.train_forward(input)
    pass
