import json
from os.path import join
from dataclasses import dataclass, asdict
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torch
from chapter7.multitask_dataset import MunichDataset
from chapter7.multitask_model import MultiTask
from chapter7.multitask_model_compress import MultiTaskCompress
from lib.utils import provide_determinism
from lib.loss import compute_l1_regularization
from lib.logger import Logger
from lib.metrics import SegMetrics, DetectionMetrics
from tqdm import tqdm
import albumentations as A
import yaml

WORK_DIR = "chapter7/test_train"
DATA_DIR = "data"


@dataclass
class Configs:
    epochs: int = 50                # 训练的使用的epoch数
    lr: float = 0.001               # 学习率
    weight_decay: float = 0.0005    # 权重衰减
    batch_size: int = 36            # batch size
    det_weight: float = 1.0         # 目标检测损失的权重
    seg_weight: float = 1.0         # 语义分割损失的权重
    evaluation_interval: int = 1    # 每隔多少个epoch进行一次评估
    conf_thres: float = 0.1         # 目标检测的置信度阈值
    nms_thres: float = 0.5          # 目标检测的NMS阈值
    iou_thres: float = 0.5          # 目标检测的IOU阈值
    resume_model: bool = None       # 是否从之前的模型中恢复训练
    seed: int = 3407  # 25, #3407,  # 随机种子
    enable_QAT: bool = False        # 是否启用量化训练
    enable_UW: bool = True          # 是否启用不确定性加权
    l1: float = 0.0001              # L1正则化的权重

    def to_dict(self):
        config_dict = asdict(self)
        return config_dict


configs = Configs()
provide_determinism(configs.seed)

det_weight = configs.det_weight
seg_weight = configs.seg_weight

logger = Logger(log_dir=WORK_DIR)
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Affine(p=0.2),
    A.Cutout(max_w_size=15, max_h_size=15),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='yolo',
                            label_fields=['class_labels']))

dataset = MunichDataset(img_path=join(DATA_DIR, "camera"),
                        start_index=0, end_index=6400,
                        interval=3, transform=transform,
                        det_path=join(DATA_DIR, "detections"),
                        seg_path=join(DATA_DIR, "seg"))

dataloader = DataLoader(dataset,
                        batch_size=configs.batch_size,
                        num_workers=8,
                        pin_memory=True,
                        drop_last=True,
                        collate_fn=dataset.collate_fn,
                        shuffle=True)

dataset_val = MunichDataset(img_path=join(DATA_DIR, "camera"),
                            start_index=6400,
                            end_index=7600,
                            interval=3,
                            transform=None,
                            det_path=join(DATA_DIR, "detections"),
                            seg_path=join(DATA_DIR, "seg"))

dataloader_val = DataLoader(dataset_val,
                            batch_size=configs.batch_size,
                            num_workers=8,
                            pin_memory=True,
                            drop_last=True,
                            collate_fn=dataset.collate_fn,
                            shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiTaskCompress(with_seg=True,
                          with_det=True,
                          with_uncertainty=configs.enable_UW,
                          num_seg_classes=19,
                          backbone="resnet").to(device)

params = [p for p in model.parameters() if p.requires_grad]

optimizer = optim.Adam(
    params,
    lr=configs.lr,
    weight_decay=configs.weight_decay)

scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                           T_max=configs.epochs)

yaml.dump(configs.to_dict(), open(join(logger.log_dir, "config.yaml"), "w"))

step = 0
for epoch in range(configs.epochs):
    model.train()
    for i, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch}")):
        image = batch["image"].to(device)
        outputs = model.train_forward(image)

        if configs.enable_UW and model.with_det and model.with_seg:
            # 如果是多任务学习，且启用了不确定性加权，则计算不确定性加权的权重
            seg_weight = torch.pow(
                torch.exp(-outputs["seg_uncertainty"][0]), 2)
            det_weight = torch.pow(
                torch.exp(-outputs["det_uncertainty"][0]), 2)

        losses = []
        tensorboard_log = []
        if "bboxes" in batch and "det" in outputs:
            # 若batch中包含目标检测的标注，则计算目标检测的损失
            targets = batch["bboxes"].to(device)
            loss, loss_components = model.det_head.compute_loss(
                outputs["det"], targets)
            losses.append(det_weight*loss[0])
            # Tensorboard logging
            tensorboard_log += [
                ("train/iou_loss", float(loss_components[0])),
                ("train/obj_loss", float(loss_components[1])),
                ("train/class_loss", float(loss_components[2]))]

        if "seg" in batch and "seg" in outputs:
            # 若batch中包含语义分割的标注，则计算语义分割的损失
            segs = batch["seg"].to(device)
            loss_seg = model.seg_head.compute_loss(outputs["seg"], segs)
            losses.append(seg_weight*loss_seg)
            tensorboard_log.append(("train/seg_loss", loss_seg.cpu().item()))

        if configs.enable_UW and model.with_det and model.with_seg:
            # 如果是多任务学习，且启用了不确定性加权，则计算不确定性权重的损失
            losses.append(outputs["seg_uncertainty"][0] +
                          outputs["det_uncertainty"][0])

        if configs.l1 != 0:
            # 如果L1正则化系数不为0，则计算L1正则化损失
            l1_reg_loss = compute_l1_regularization(model)
            losses.append(configs.l1*l1_reg_loss)
        loss = torch.sum(torch.stack(losses))

        tensorboard_log.append(("train/loss", loss.cpu().item()))
        if configs.enable_UW and model.with_det and model.with_seg:
            tensorboard_log.append(
                ("general/Seg Weighting", seg_weight.cpu().item()))
            tensorboard_log.append(
                ("general/Det Weighting", det_weight.cpu().item()))

        logger.list_of_scalars_summary(
            tensorboard_log, step=step*configs.batch_size)

        # Log the learning rate
        logger.scalar_summary(
            "train/learning_rate", optimizer.param_groups[0]["lr"], step*configs.batch_size)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step += 1
    scheduler.step(epoch=epoch)

    if epoch % configs.evaluation_interval == 0:
        print("\n---- Evaluating Model ----")
        # 实例化语义分割评价指标计算器
        seg_metric = SegMetrics(num_classes=19, device=device)
        # 实例化目标检测评价指标计算器
        det_metric = DetectionMetrics()
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader_val, desc=f"Evaluating Epoch {epoch}")):
                image = batch["image"].to(device)
                outputs = model.train_forward(image)

                if "bboxes" in batch and "det" in outputs:
                    targets = batch["bboxes"]
                    # 计算目标检测的KPI，内含目标检测的NMS等后处理环节
                    det_metric.compute_batch(yolo_outputs=outputs["det"], targets=targets,
                                             img_width=640, img_height=480,
                                             conf_thres=configs.conf_thres,
                                             nms_thres=configs.nms_thres,
                                             iou_thres=configs.iou_thres)

                if "seg" in batch and "seg" in outputs:
                    segs = batch["seg"].to(device)
                    # 计算语义分割的KPI
                    seg_metric.add(torch.argmax(outputs["seg"]["3"], dim=1),
                                   F.upsample_nearest(segs.to(device).float(),
                                   scale_factor=0.25).long().squeeze(1), 0)

        det_metrics = det_metric.metrics()
        seg_metrics = seg_metric.metrics()
        logger.list_of_scalars_summary(det_metrics, step*configs.batch_size)
        logger.scalar_summary(
            "validation/mIoU", seg_metrics["mIOU"], step*configs.batch_size)
        print(det_metrics)
        print(seg_metrics["mIOU"])
        json.dump({
            "det": det_metrics,
            "seg": ("mIoU", seg_metrics["mIOU"])
        }, open(join(logger.log_dir, "{}.json".format(epoch)), "w"), indent=4)

    torch.save(model.state_dict(), join(
        logger.log_dir, "{}.pth".format(configs.epochs)))
