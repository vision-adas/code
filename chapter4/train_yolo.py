from os.path import join
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
from chapter4.dataset_yolo import MunichDetDataset
from chapter4.model_yolo import YoloDetector
from lib.utils import provide_determinism
from lib.loss import compute_l1_regularization
from lib.logger import Logger
from lib.metrics import DetectionMetrics
from tqdm import tqdm
import albumentations as A
import yaml
from dataclasses import dataclass, asdict


WORK_DIR = "chapter4/test_train"
DATA_DIR = "data"


@dataclass
class Configs:
    epochs: int = 100  # 总epoch数
    lr: float = 0.001  # 学习率
    weight_decay: float = 0.0005  # 权重衰减
    batch_size: int = 12          # batch size
    det_weight: float = 1.0       # 目标检测损失权重
    seg_weight: float = 1.0       # 语义分割损失权重
    batch_uw: bool = False        # 是否使用uncertainty weights进行多任务训练
    evaluation_interval: int = 1  # 每隔多少个epoch进行一次评估
    conf_thres: float = 0.1       # 目标检测置信度阈值
    nms_thres: float = 0.5        # 目标检测nms阈值
    iou_thres: float = 0.5        # 目标检测iou阈值
    resume_model: bool = False    # 是否从断点继续训练
    seed: int = 3407  # 25, #3407,  # 随机种子
    enable_QAT: bool = False      # 是否启用Quantization Aware Training
    enable_UW: bool = True        # 是否启用Uncertainty Weighting
    l1: float = 0.0               # 对可微网络压缩所使用的通道掩膜进行L1正则化的系数

    def to_dict(self):
        """将Config类转换为dict"""
        config_dict = asdict(self)
        return config_dict


configs = Configs()

# 设置随机种子
provide_determinism(configs.seed)

# 初始化日志记录器
logger = Logger(log_dir=WORK_DIR)

# 数据增强
transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # 水平翻转
    A.Affine(p=0.2),          # 仿射变换
    A.Cutout(max_w_size=15, max_h_size=15),  # Cutout
    A.RandomBrightnessContrast(p=0.2),       # 随机亮度对比度
], bbox_params=A.BboxParams(format='yolo',   # YOLO格式的bbox
                            label_fields=['class_labels']))

dataset = MunichDetDataset(img_path=join(DATA_DIR, "camera"),
                           start_index=1600, end_index=6400,       # 数据集所用的起始和结束帧
                           interval=3,                             # 每隔3帧抽取一帧
                           transform=transform,                    # 数据增强
                           det_path=join(DATA_DIR, "detections"))  # 目标检测标注文件

dataloader = DataLoader(dataset,                              # 数据集
                        batch_size=configs.batch_size,        # batch size
                        num_workers=8,                        # 定义8个进程读取数据
                        pin_memory=True,                      # 使用pin memory
                        drop_last=True,                       # 丢弃最后一个batch
                        collate_fn=dataset.collate_fn,        # 自定义batch的组合方式
                        shuffle=True)                         # 打乱数据集

dataset_val = MunichDetDataset(img_path=join(DATA_DIR, "camera"),      # 验证集
                               start_index=6400,                       # 从第6400帧开始
                               end_index=7600,                         # 到第7600帧结束
                               interval=3,                             # 每隔3帧抽取一帧
                               transform=None,                         # 不使用数据增强
                               det_path=join(DATA_DIR, "detections"))  # 目标检测标注文件

dataloader_val = DataLoader(dataset_val,                      # 验证集
                            batch_size=configs.batch_size,    # batch size
                            num_workers=8,                    # 定义8个进程读取数据
                            pin_memory=True,                  # 使用pin memory
                            drop_last=True,                   # 丢弃最后一个batch
                            collate_fn=dataset.collate_fn,    # 自定义batch的组合方式
                            shuffle=False)                    # 不打乱数据集

# 检测机器是否支持cuda，如果支持则使用cuda，否则使用cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
model = YoloDetector(backbone="resnet").to(device)

# 获取可训练参数
params = [p for p in model.parameters() if p.requires_grad]

# 定义优化器
optimizer = optim.Adam(
    params,
    lr=configs.lr,
    weight_decay=configs.weight_decay)

# 定义学习率衰减策略
scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                           T_max=configs.epochs)

# 将配置保存到日志目录
yaml.dump(configs.to_dict(), open(join(logger.log_dir, "config.yaml"), "w"))

step = 0
for epoch in range(configs.epochs):
    torch.cuda.empty_cache()  # 清空cuda缓存
    model.train()            # 设置模型为训练模式
    for i, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch}")):
        image = batch["image"].to(device)       # 获得图像
        outputs = model.train_forward(image)    # 前向传播
        losses = []
        tensorboard_log = []
        if "bboxes" in batch and "det" in outputs:
            targets = batch["bboxes"].to(device)  # 获得目标检测标注
            # 32x3x15x20x7, 32x3x30x40x7, 32x3x60x80x7
            loss, loss_components = model.det_head.compute_loss(
                outputs["det"], targets)          # 计算目标检测损失
            losses.append(loss[0])                # 将目标检测损失加入到总损失中
            # Tensorboard logging
            tensorboard_log += [
                ("train/iou_loss", float(loss_components[0])),
                ("train/obj_loss", float(loss_components[1])),
                ("train/class_loss", float(loss_components[2]))]

        if configs.l1 != 0:  # 如果L1正则化系数不为0，则计算L1正则化损失
            l1_reg_loss = compute_l1_regularization(model)
            losses.append(configs.l1*l1_reg_loss)
        loss = torch.sum(torch.stack(losses))

        tensorboard_log.append(("train/loss", loss.cpu().item()))
        logger.list_of_scalars_summary(
            tensorboard_log, step=step*configs.batch_size)
        # Log the learning rate
        logger.scalar_summary(
            "train/learning_rate", optimizer.param_groups[0]["lr"], step*configs.batch_size)

        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        optimizer.zero_grad()  # 清空梯度
        step += 1  # 更新步数
    scheduler.step(epoch=epoch)  # 更新学习率
    del outputs, image, targets, loss  # 释放内存

    # 每隔一定的epoch进行验证
    if epoch % configs.evaluation_interval == 0:
        print("\n---- Evaluating Model ----")
        det_metric = DetectionMetrics()  # 实例化目标检测评价指标计算器
        model.eval()                     # 设置模型为评估模式
        with torch.no_grad():            # 不计算梯度
            for i, batch in enumerate(tqdm(dataloader_val, desc=f"Evaluating Epoch {epoch}")):
                image = batch["image"].to(device)
                outputs = model.train_forward(image)

                if "bboxes" in batch and "det" in outputs:
                    targets = batch["bboxes"]
                    det_metric.compute_batch(yolo_outputs=outputs["det"],
                                             targets=targets,
                                             img_width=640,
                                             img_height=480,
                                             conf_thres=configs.conf_thres,
                                             nms_thres=configs.nms_thres,
                                             iou_thres=configs.iou_thres)
            del outputs, image, targets

        det_metrics = det_metric.metrics()
        logger.list_of_scalars_summary(det_metrics, step*configs.batch_size)
        print(det_metrics)

    torch.save(model.state_dict(), join(
        logger.log_dir, "{}.pth".format(configs.epochs)))
