import os
import json
import torch
from torch import nn
from PIL import Image

from detection_torchvision import utils
from detection_torchvision.engine import train_one_epoch, evaluate
import detection_torchvision.transforms as T

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import faster_rcnn
from torchvision.ops.feature_pyramid_network import \
    FeaturePyramidNetwork, LastLevelMaxPool
from lib.model import ResnetBackbone


def get_transform(train):  # 定义数据增强
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class ResnetWithFPN(nn.Module):
    """一个由Resnet18主干网络和FPN组成的网络"""

    def __init__(self, out_channels=256, pretrained=True):
        super(ResnetWithFPN, self).__init__()
        # 定义一个Resnet18主干网络
        self.backbone = ResnetBackbone(pretrained)
        self.out_channels = out_channels
        # 定义FPN，输入通道数为Resnet18输出通道数
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=self.backbone.out_channels,
            out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool())

    def forward(self, x):
        return self.fpn(self.backbone(x))


def build_faster_rcnn_det(num_classes):
    # 建立带FPN的主干网络，每一层FPN输出64通道特征图
    backbone = ResnetWithFPN(out_channels=64, pretrained=True)
    # 搭建FasterRCNN模型
    model = faster_rcnn.FasterRCNN(backbone, num_classes=num_classes)

    # 获得检测框分类器的输入特征数
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # 根据本例类别数重新定义一个检测框分类器
    model.roi_heads.box_predictor = \
        faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    return model


class MunichDet(Dataset):  # 定义数据集
    def __init__(self, root, from_index=0, to_index=-1,
                 trans=get_transform(False)):
        self.root = root
        self.transforms = trans
        self.img_dir = os.path.join(root, "camera")
        self.det_dir = os.path.join(root, "detections")
        self.img_files = list(sorted(os.listdir(self.img_dir)))  # 读取图片文件名并排序
        self.det_files = list(sorted(os.listdir(self.det_dir)))  # 读取标注文件名并排序
        self.det_files = [file for file in self.det_files if ".json" in file]
        self.img_files = self.img_files[from_index:to_index]
        self.det_files = self.det_files[from_index:to_index]

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        det_path = os.path.join(self.det_dir, self.det_files[idx])
        img = Image.open(img_path).convert("RGB")
        det = json.load(open(det_path))

        boxes = []
        areas = []
        for box in det["boxes"]:
            pos = box["box"]
            xmin = pos[0]
            ymin = pos[1]
            xmax = pos[2]
            ymax = pos[3]
            if box["label"] == 2:
                boxes.append([xmin, ymin, xmax, ymax])
                areas.append((xmax-xmin)*(ymax-ymin))

        # 转换为Torch Tensor
        boxes = torch.as_tensor(boxes)
        num_objs = len(boxes)

        if num_objs == 0:
            boxes = torch.zeros((0, 4))
            area = torch.zeros((0,))
        else:
            boxes = torch.as_tensor(boxes)
            area = torch.as_tensor(areas)

        # 只有一个类别，类别全设为1
        labels = torch.ones((num_objs,)).long()
        image_id = torch.tensor([idx])
        # 将所有目标的iscrowd属性设置为0
        iscrowd = torch.zeros((num_objs,)).long()

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        # 转换为Torch Tensor并进行数据增广
        img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.img_files)


# 检测机器是否支持cuda，如果支持则使用cuda，否则使用cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = MunichDet("./data",
                    from_index=1684, to_index=6389,
                    trans=get_transform(True))

dataset_val = MunichDet("./data", from_index=6390, to_index=7566)

# define training and validation data loaders
data_loader = DataLoader(dataset, batch_size=12,
                         shuffle=True, num_workers=8,
                         collate_fn=utils.collate_fn,
                         drop_last=True)

data_loader_test = torch.utils.data.DataLoader(
    dataset_val, batch_size=1, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn, drop_last=True)

model = build_faster_rcnn_det(num_classes=2)
# move models to the right device
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=0.0001)

# and a learning rate scheduler
lr_scheduler = lr_scheduler.StepLR(optimizer,
                                   step_size=10,
                                   gamma=0.1)

for epoch in range(20):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader,
                    device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    evaluate(model, data_loader_test, device=device)
