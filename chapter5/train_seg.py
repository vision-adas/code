import json
from os.path import join
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
from chapter5.dataset_seg import MunichSegDataset
from chapter5.model_seg import SegmentFPN
from lib.utils import provide_determinism
from lib.loss import compute_l1_regularization
from lib.logger import Logger
from lib.metrics import SegMetrics
from tqdm import tqdm
import albumentations as A
import yaml
from dataclasses import dataclass, asdict


# 定义训练文件文件夹路径，用于保存训练记录
WORK_DIR = "chapter5/test_train"
# 定义训练数据文件夹
DATA_DIR = "data"


@dataclass
class Configs:
    epochs: int = 100                 # 训练的epoch数
    lr: float = 0.001                 # 学习率
    weight_decay: float = 0.0005      # 权重衰减强度
    batch_size: int = 12              # 批次大小
    evaluation_interval: int = 1      # 每隔多少个epoch验证一次
    seed: int = 3407  # 25, #3407,
    l1: float = 0.0                   # L1正则化强度

    def to_dict(self):
        """将配置项转换为dict"""
        config_dict = asdict(self)
        return config_dict


configs = Configs()
provide_determinism(configs.seed)

# 用于向Tensorboard写入训练记录
logger = Logger(log_dir=WORK_DIR)

# 定义数据增广
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Affine(p=0.2),
    A.Cutout(max_w_size=15, max_h_size=15),
    A.RandomBrightnessContrast(p=0.2),
])

# 定义训练数据集，采用数据集第1600帧至6400帧，每隔三帧抽一帧
dataset = MunichSegDataset(img_path=join(DATA_DIR, "camera"),
                           start_index=1600, end_index=6400,
                           interval=3, transform=transform,
                           seg_path=join(DATA_DIR, "seg"))
dataloader = DataLoader(dataset,
                        batch_size=configs.batch_size,
                        num_workers=8,
                        pin_memory=True,
                        drop_last=True,
                        shuffle=True)

# 定义验证数据集，采用数据集第6400帧至7600帧，每隔三帧抽一帧
dataset_val = MunichSegDataset(img_path=join(DATA_DIR, "camera"),
                               start_index=6400,
                               end_index=7600,
                               interval=3,
                               transform=None,
                               seg_path=join(DATA_DIR, "seg"))
dataloader_val = DataLoader(dataset_val,
                            batch_size=configs.batch_size,
                            num_workers=8,
                            pin_memory=True,
                            drop_last=True,
                            shuffle=False)

# 检测当前训练环境的硬件，如果有GPU就是用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义一个基于FPN的语义分割网络
model = SegmentFPN(with_uncertainty=False,
                   num_seg_classes=19,
                   backbone="resnet").to(device)

# 收集需要训练的参数
params = [p for p in model.parameters() if p.requires_grad]

# 定义优化器
optimizer = optim.Adam(
    params,
    lr=configs.lr,
    weight_decay=configs.weight_decay)

# 定义学习率Scheduler
scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                           T_max=configs.epochs)

# 将本次训练的配置保存至配置文件
yaml.dump(configs.to_dict(), open(join(logger.log_dir, "config.yaml"), "w"))

step = 0
for epoch in range(configs.epochs):
    model.train()
    for i, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch}")):
        losses = []
        tensorboard_log = []

        # 将图像张量从BxHxWx3转换为Bx3xHxW并归一化至0~1
        image = batch["image"].to(device).permute(0, 3, 1, 2)/255.0

        # 调用用于训练的forward函数
        outputs = model.train_forward(image)

         # 将语义分割张量从BxHxWx1转换为Bx1xHxW
        segs = batch["seg"].to(device).permute(0, 3, 1, 2)
        loss_seg = model.seg_head.compute_loss(outputs["seg"], segs)

        losses.append(loss_seg)
        tensorboard_log.append(("train/seg_loss", loss_seg.cpu().item()))

        # 若L1正则化强度不为零，计算L1正则化损失并加入到总损失中
        if configs.l1 != 0:
            l1_reg_loss = compute_l1_regularization(model)
            losses.append(configs.l1*l1_reg_loss)
        loss = torch.sum(torch.stack(losses))

        tensorboard_log.append(("train/loss", loss.cpu().item()))
        logger.list_of_scalars_summary(
            tensorboard_log, step=step*configs.batch_size)
        # Log the learning rate
        logger.scalar_summary(
            "train/learning_rate", optimizer.param_groups[0]["lr"], step*configs.batch_size)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step += 1

    # 每一个epoch调整一次学习率
    scheduler.step(epoch=epoch)

    if epoch % configs.evaluation_interval == 0:
        print("\n---- Evaluating Model ----")
        seg_metric = SegMetrics(num_classes=19, device=device)
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader_val, desc=f"Evaluating Epoch {epoch}")):
                image = batch["image"].to(device).permute(0, 3, 1, 2)/255.0
                outputs = model.train_forward(image)
                segs = batch["seg"].to(device).permute(0, 3, 1, 2)
                # 最后一层FPN的语义分割图分辨率最高，为最终输出
                seg_metric.add(torch.argmax(outputs["seg"]["3"], dim=1),
                               F.upsample_nearest(segs.to(device).float(),
                                                  scale_factor=0.25).long().squeeze(1), 0)

        seg_metrics = seg_metric.metrics()
        logger.scalar_summary(
            "validation/mIoU", seg_metrics["mIOU"], step*configs.batch_size)
        print(seg_metrics["mIOU"])

    torch.save(model.state_dict(), join(
        logger.log_dir, "{}.pth".format(configs.epochs)))
