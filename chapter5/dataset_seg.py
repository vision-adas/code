import albumentations as A
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import cv2
from os.path import join


def visualize_data(data):
    cv2.imshow("image", data["image"])
    cv2.imshow("seg", 10 * data["seg"])
    cv2.waitKey()


class MunichSegDataset(Dataset):
    def __init__(self, img_path: str, start_index: int, end_index: int, interval: int = 1,
                 transform=None, seg_path: str = None):
        images = os.listdir(img_path)
        images.sort()
        images = [images[i] for i in range(start_index, end_index, interval)]
        self.image_files = [join(img_path, file) for file in images]
        self.seg_files = [join(seg_path, file.replace(
            ".jpg", ".png")) for file in images]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        data = dict()
        img_path = self.image_files[index]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        data["image"] = img.copy()
        seg_path = self.seg_files[index]
        seg = cv2.imread(seg_path, 0)
        seg = np.expand_dims(seg, 2)
        data["seg"] = seg.copy()

        if self.transform:
            transform_args = {'image': data["image"]}
            transform_args["mask"] = data["seg"]
            data = self.transform(**transform_args)
            data["seg"] = data["mask"]
        # visualize_data(data, width, height)
        return data


if __name__ == '__main__':

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(p=0.5),
        A.Cutout(max_h_size=15, max_w_size=15),
        A.RandomBrightnessContrast(p=0.2)])

    dataset = MunichSegDataset(img_path="./data/camera",
                               start_index=1600, end_index=6400, interval=2, transform=transform,
                               seg_path="./data/seg")

    dataloader = DataLoader(dataset,
                            batch_size=32,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=True)

    for i, batch in enumerate(dataloader):
        print(batch["image"].shape)
