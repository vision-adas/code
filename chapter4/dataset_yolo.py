import json
import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import cv2
from os.path import join


def visualize_data(data, width, height):
    for i in range(len(data["bboxes"])):
        centerX = width * data["bboxes"][i][1]
        centerY = height * data["bboxes"][i][2]
        bbWidth = width * data["bboxes"][i][3]
        bbHeight = height * data["bboxes"][i][4]
        x1 = centerX - bbWidth / 2
        y1 = centerY - bbHeight / 2
        x2 = centerX + bbWidth / 2
        y2 = centerY + bbHeight / 2
        cv2.rectangle(data["image"], (int(x1), int(y1)),
                      (int(x2), int(y2)), (0, 255, 0))
    cv2.imshow("image", data["image"])
    cv2.waitKey()


ClassMap = {
    "vehicle": 0
}


class MunichDetDataset(Dataset):
    def __init__(self, img_path: str, start_index: int, end_index: int, interval: int = 1,
                 transform=None, det_path: str = None):
        images = os.listdir(img_path)
        images.sort()
        images = [images[i] for i in range(start_index, end_index, interval)]
        self.image_files = [join(img_path, file) for file in images]
        self.use_det = not det_path is None
        self.det_files = [join(det_path, file.replace(
            ".jpg", ".json")) for file in images]
        self.max_objects = 100
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        data = dict()
        img_path = self.image_files[index]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        data["image"] = img.copy()

        info = json.load(open(self.det_files[index]))
        info["bboxes"] = info["boxes"]
        boxes = []
        class_labels = []
        width = img.shape[1]
        height = img.shape[0]
        for box in info["bboxes"]:
            if box["label"] == 2:
                x1, y1, x2, y2 = box["box"]
                boxes.append(
                    [0.5 * (x1 + x2) / width, 0.5 * (y1 + y2) / height, (x2 - x1) / width, (y2 - y1) / height])
                class_labels.append("vehicle")

        if len(boxes) > 0:
            data["bboxes"] = np.array(boxes)
            data["class_labels"] = class_labels

        if self.transform:
            transform_args = {'image': data["image"]}
            transform_args["bboxes"] = data["bboxes"] if "bboxes" in data else []
            transform_args["class_labels"] = data["class_labels"] if "class_labels" in data else [
            ]

            data = self.transform(**transform_args)

        if "bboxes" in data and len(data["bboxes"]) > 0:
            data["bboxes"] = np.array(data["bboxes"])
            class_col = torch.Tensor([ClassMap[label]
                                     for label in data["class_labels"]])
            class_col = np.expand_dims(class_col, 1)
            data["bboxes"] = np.concatenate(
                [class_col, data["bboxes"]], axis=1)
        elif "bboxes" in data:
            del data["bboxes"]
            # visualize_data(data, width, height)
        return data

    def collate_fn(self, batch):
        batch_output = dict()
        # list of 3x480x640
        imgs = [torch.Tensor(data["image"]).permute((2, 0, 1))
                for data in batch]
        # torch.Size([32, 3, 480, 640])
        batch_output["image"] = torch.stack(imgs, 0) / 255.0

        if "seg" in batch[0]:
            # list of torch.Size([1, 480, 640])
            segs = [torch.Tensor(data["seg"]).permute((2, 0, 1))
                    for data in batch]
            # torch.Size([32, 1, 480, 640])
            batch_output["seg"] = torch.stack(segs, 0)

        boxes = []
        for i, data in enumerate(batch):
            if "bboxes" in batch[i]:
                box = torch.Tensor(batch[i]["bboxes"])
                index_col = i * torch.ones((box.shape[0], 1))
                box = torch.cat([index_col, box], dim=1)
                # list of nx6
                boxes.append(box)
        if len(boxes) > 0:
            # Nx6
            batch_output["bboxes"] = torch.cat(boxes, 0)

        return batch_output


if __name__ == '__main__':

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(p=0.5),
        A.Cutout(max_h_size=15, max_w_size=15),
        A.RandomBrightnessContrast(p=0.2),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    dataset = MunichDetDataset(img_path="./data/camera",
                               start_index=1600, end_index=6400, interval=2, transform=transform,
                               det_path="./data/detections")

    dataloader = DataLoader(dataset, batch_size=32, num_workers=0, pin_memory=True, drop_last=True,
                            collate_fn=dataset.collate_fn)
    for i, batch in enumerate(dataloader):
        print(batch["bboxes"])
        break
