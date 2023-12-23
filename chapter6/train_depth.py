import os
import shutil
from os.path import join
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from chapter6.folder_dataset import FolderDataset
from chapter6.infer_depth import run_infer_images
from lib.metrics import AverageMeter
from lib.monodepth.net import Baseline
import yaml
from munch import munchify
from tqdm import tqdm
import sys
from lib.utils import (
    clip_grads,
    convert_to_device,
    log_everything,
    provide_determinism
)

# 保存训练记录的文件夹
WORK_DIR = "chapter6/test_train"
# 保存可视化深度图的文件夹
INFER_IMAGE_PATH = "chapter6/test_images"
# 数据文件夹，自监督学习只需要图像文件
datafolder = "data/camera"

provide_determinism(3407, save_path=join(WORK_DIR, "seed.txt"))

epochs = 40

# 设定配置文件
config_file = "configs/monodepth.yaml"
# 将配置文件加载为dict
configs_dict = yaml.safe_load(open(config_file))
# 用munch将dict转换为object
configs = munchify(configs_dict)

# 设置梯度裁剪的最大值，将L2 norm大于35的梯度值裁剪为35
grad_clip = {"max_norm": 35, "norm_type": 2}

# 定义训练数据集
train_dataset = FolderDataset(datafolder,
                              height=480,
                              width=640,
                              neighbor_ids=[0, -1, 1],
                              start_index=100,
                              end_index=7000,
                              is_train=True)
train_loader = DataLoader(train_dataset,
                          batch_size=configs.model.imgs_per_gpu,
                          shuffle=True,
                          drop_last=True)

# 定义测试数据集
val_dataset = FolderDataset(datafolder,
                            height=480,
                            width=640,
                            neighbor_ids=[0, -1, 1],
                            start_index=7000,
                            end_index=9450,
                            is_train=False)
val_loader = DataLoader(val_dataset,
                        batch_size=configs.model.imgs_per_gpu,
                        shuffle=False,
                        drop_last=True)

# 定义模型
model = Baseline(options=configs.model)

# 定义设备，如果有GPU则使用GPU，否则使用CPU
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

# 将模型转移到设备上
model.to(device)

# 定义优化器
optimizer = optim.AdamW(model.parameters(), lr=configs.optim.lr)

checkpoints_folder = join(WORK_DIR, "checkpoints")
vis_folder = join(WORK_DIR, "depth_vis")
os.makedirs(checkpoints_folder, exist_ok=True)
os.makedirs(vis_folder, exist_ok=True)
shutil.copyfile(config_file, join(WORK_DIR, "config.yaml"))

logger = SummaryWriter(log_dir=join(WORK_DIR, "log"))

index = 0
min_val_loss = sys.float_info.max
for epoch in range(epochs):
    model.train()
    model.training = True
    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        index += 1
        if device != torch.device('cpu'):
            # 如果使用GPU，则将batch中的数据转移到显卡上
            batch = convert_to_device(batch, device)
        model_out, losses = model(batch)
        total_loss = sum(_value for _key, _value in losses.items())
        log_everything(logger, losses, index)
        optimizer.zero_grad()
        total_loss.backward()
        clip_grads(model.parameters(), grad_clip)
        optimizer.step()

    with torch.no_grad():
        min_recon_loss = AverageMeter()
        for i, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            if device != torch.device('cpu'):
                batch = convert_to_device(batch, device)
            model_out, losses = model(batch)
            min_recon_loss.update(
                losses[('min_reconstruct_loss', 0)])
        if min_recon_loss.avg < min_val_loss:
            min_val_loss = min_recon_loss.avg
            torch.save(model.state_dict(), os.path.join(
                checkpoints_folder, str(min_val_loss)+"_"+str(epoch)+".pth"))
            print("Best models saved.")

        logger.add_scalar("eval loss", min_recon_loss.avg, index)

    save_folder = join(vis_folder, str(epoch))
    os.makedirs(save_folder, exist_ok=True)
    run_infer_images(model, INFER_IMAGE_PATH, save_folder)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
