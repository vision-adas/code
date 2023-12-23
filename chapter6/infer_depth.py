

from glob import glob
import os
from os.path import join
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch


def preprocess(cv2_img, height=480, width=640):
    """预处理图像，输出符合尺寸要求且归一化到0-1"""
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    im_tensor = torch.from_numpy(
        cv2_img.astype(np.float32)).unsqueeze(0)
    im_tensor = im_tensor.permute(0, 3, 1, 2).contiguous()
    im_tensor = torch.nn.functional.interpolate(
        im_tensor, [height, width], mode='bilinear', align_corners=False)
    im_tensor /= 255
    return im_tensor


def rescale_depth(normalized_depth, min_depth, max_depth):
    return min_depth + (max_depth-min_depth)*normalized_depth


def predict(cv2_img, model):
    original_height, original_width = cv2_img.shape[:2]
    device = next(model.parameters()).device
    im_tensor = preprocess(cv2_img).to(device)
    input = {}
    input['color_aug', 0, 0] = im_tensor
    outputs = model(input)

    disp = outputs[("disp", 0, 0)]
    disp_resized = torch.nn.functional.interpolate(
        disp, (original_height, original_width), mode="bilinear", align_corners=False)
    min_disp = 1/100.0
    max_disp = 1/0.1
    depth = 1/(disp_resized.squeeze().cpu().numpy()*max_disp + min_disp) * 36
    # depth = rescale_depth(disp_resized.squeeze().cpu().numpy(), 0.1, 100)
    return depth


def run_infer_images(model, image_folder, save_folder):
    model.eval()
    with torch.no_grad():
        images = glob(join(image_folder, "*"))
        for image_file in images:
            cv2_img = cv2.imread(image_file)
            depth = predict(cv2_img, model)

            vmax = np.percentile(depth, 95)
            plt.imsave(os.path.join(save_folder, os.path.basename(image_file)),
                       depth, cmap='magma', vmax=vmax)


def infer_model_images(config, model_file, image_folder, save_folder):
    import yaml
    from munch import munchify
    from lib.monodepth.net import Baseline

    config_file = "configs/monodepth.yaml"
    configs_dict = yaml.safe_load(open(config_file))
    config = munchify(configs_dict)
    model = Baseline(config.model)
    model.load_state_dict(torch.load(model_file))
    model.cuda()
    run_infer_images(model, image_folder, save_folder)


if __name__ == "__main__":
    infer_model_images(config="configs/monodepth.yaml",
                       model_file="best_model.pth",
                       image_folder="chapter6/test_images",
                       save_folder="chapter6/test_images_depth")
