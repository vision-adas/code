import torch
import torch.nn as nn


def compute_l2_regularization(model: nn.Module):
    l2_reg_loss = 0
    for name, para in model.named_parameters():
        if "weight" in name:
            l2_reg_loss += torch.norm(para)
    return l2_reg_loss


def compute_l1_regularization(model: nn.Module):
    l1_reg_loss = 0
    for name, para in model.named_parameters():
        #if "conv" in name and not "downsample" in name:
        if "channel_mask" in name:
            l1_reg_loss += torch.sum(torch.abs(para))
    return l1_reg_loss