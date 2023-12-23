from __future__ import absolute_import, division, print_function
import torch
import torch.nn.functional as F
import torch.nn as nn

from lib.monodepth.layers import SSIM, Backproject, Project
from lib.monodepth.depth_encoder import DepthEncoder
from lib.monodepth.depth_encoder_regnet import DepthEncoderRegNet
from lib.monodepth.depth_decoder import DepthDecoder
from lib.monodepth.pose_encoder import PoseEncoder
from lib.monodepth.pose_decoder import PoseDecoder


class Baseline(nn.Module):
    def __init__(self, options):
        super(Baseline, self).__init__()
        self.opt = options
        self.num_input_frames = len(self.opt.frame_ids)
        if "mono_fm" in self.opt.name:
            self.DepthEncoder = DepthEncoder(self.opt.depth_num_layers,
                                             self.opt.depth_pretrained_path)
        elif "mono_regnet" in self.opt.name:
            self.DepthEncoder = DepthEncoderRegNet(pretrained=True)

        # 定义深度解码器
        self.DepthDecoder = DepthDecoder(self.DepthEncoder.num_ch_enc)

        # 定义位姿网络编码器
        self.PoseEncoder = PoseEncoder(self.opt.pose_num_layers,
                                       self.opt.pose_pretrained_path,
                                       num_input_images=2)
        # 定义位姿网络解码器
        self.PoseDecoder = PoseDecoder(self.PoseEncoder.num_ch_enc)
        # 定义SSIM损失
        self.ssim = SSIM()
        # 定义反投影层
        self.backproject = Backproject(
            self.opt.imgs_per_gpu, self.opt.height, self.opt.width)
        # 定义投影层
        self.project_3d = Project(
            self.opt.imgs_per_gpu, self.opt.height, self.opt.width)

    def infer(self, input):
        # 用于推理的前向函数
        output = self.DepthDecoder.infer(self.DepthEncoder(input))
        return output

    def forward(self, inputs):
        # 用于训练的前向函数，输入经过数据增广的图像
        outputs = self.DepthDecoder(
            self.DepthEncoder(inputs["color_aug", 0, 0]))
        if self.training:
            # 如为训练模式，计算损失
            outputs.update(self.predict_poses(inputs))
            loss_dict = self.compute_losses(inputs, outputs)
            return outputs, loss_dict

        return outputs

    def robust_l1(self, pred, target):
        # 计算robust L1损失
        eps = 1e-3
        return torch.sqrt(torch.pow(target - pred, 2) + eps ** 2)

    def compute_reprojection_loss(self, pred, target):
        # 计算重投影损失
        photometric_loss = self.robust_l1(pred, target).mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        # 重投影损失为SSIM损失和robust L1损失的加权和
        reprojection_loss = (0.85 * ssim_loss + 0.15 * photometric_loss)
        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        # 计算损失
        loss_dict = {}
        for scale in self.opt.scales:
            # 遍历不同尺度的损失
            """
            初始化损失计算需要的数据
            """
            disp = outputs[("disp", 0, scale)]
            target = inputs[("color", 0, 0)]
            reprojection_losses = []

            """
            根据深度图将邻近帧图像投射到当前帧生成预测图像
            """
            outputs = self.generate_images_pred(inputs, outputs, scale)

            """
            计算当前帧与相邻帧之间的一致性损失，一致的区域损失极小。
            因最终仅选择最小的的损失，对于相邻帧之间一致性高的区域，
            一致性损失会被选定为最小的损失，重投影损失会被舍弃。
            而一致性损失是输入帧之间的区别，和模型无关，故起到了梯度掩膜的作用。
            """
            if self.opt.automask:
                for frame_id in self.opt.frame_ids[1:]:
                    # 将预测帧设置为输入的相邻帧（而非重投影生成的图像）
                    pred = inputs[("color", frame_id, 0)]
                    # 计算当前帧和相邻帧之间的一致性损失
                    identity_reprojection_loss = self.compute_reprojection_loss(
                        pred, target)
                    # 为了防止过度屏蔽，加入随机噪声
                    identity_reprojection_loss += torch.randn(
                        identity_reprojection_loss.shape).to(identity_reprojection_loss.device) * 1e-5
                    reprojection_losses.append(identity_reprojection_loss)

            """
            计算当前帧和从相邻帧重投影生成的图像之间的重投影损失
            """
            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(
                    self.compute_reprojection_loss(pred, target))
            reprojection_loss = torch.cat(reprojection_losses, 1)

            """
            选择最小的重投影损失作为最终的重投影损失，对遮挡区域鲁棒性更强
            """
            min_reconstruct_loss, outputs[("min_index", scale)] = torch.min(
                reprojection_loss, dim=1)
            loss_dict[('min_reconstruct_loss', scale)
                      ] = min_reconstruct_loss.mean()/len(self.opt.scales)

            """
            在计算深度图平滑损失之前，先对深度图进行归一化，使得平滑损失对深度图的尺度不敏感
            """
            if self.opt.disp_norm:
                mean_disp = disp.mean(2, True).mean(3, True)
                disp = disp / (mean_disp + 1e-7)

            """
            计算深度图平滑损失
            """
            smooth_loss = self.get_smooth_loss(disp, target)
            loss_dict[('smooth_loss', scale)] = self.opt.smoothness_weight * \
                smooth_loss / (2 ** scale)/len(self.opt.scales)

        return loss_dict

    def disp_to_depth(self, disp, min_depth, max_depth):
        """将视差图转换为深度图"""
        min_disp = 1 / max_depth  # 0.01
        max_disp = 1 / min_depth  # 10
        scaled_disp = min_disp + (max_disp - min_disp) * \
            disp  # (10-0.01)*disp+0.01
        depth = 1 / scaled_disp
        return scaled_disp, depth

    def rescale_depth(self, normalized_depth, min_depth, max_depth):
        """将归一化的深度图转换为真实深度图"""
        print("mean depth:"+str(normalized_depth.mean().cpu().detach().numpy()))
        return min_depth + (max_depth-min_depth)*normalized_depth

    def predict_poses(self, inputs):
        """预测相邻帧之间的位姿"""
        outputs = {}
        pose_feats = {f_i: F.interpolate(inputs["color_aug", f_i, 0],
                                         [480, 640],
                                         mode="bilinear",
                                         align_corners=False) for f_i in self.opt.frame_ids}
        # 预测相邻帧与当前帧之间的位姿，第一帧为当前帧，故从1开始
        for f_i in self.opt.frame_ids[1:]:
            if not f_i == "s":
                # 计算位姿遵循的顺序为：后一帧->前一帧
                if f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]
                pose_inputs = self.PoseEncoder(torch.cat(pose_inputs, 1))
                # 获得轴角向量和位移向量
                axisangle, translation = self.PoseDecoder(pose_inputs)
                # 将轴角向量和位移向量转换为相机变换矩阵
                outputs[("cam_T_cam", 0, f_i)] = self.axisangle_to_T_matrix(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
        return outputs

    def generate_images_pred(self, inputs, outputs, scale):
        """基于深度图将邻近帧投射至当前帧并生成图像"""
        disp = outputs[("disp", 0, scale)]
        disp = F.interpolate(
            disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)

        # 将输出转换为深度图
        _, depth = self.disp_to_depth(
            disp, self.opt.min_depth, self.opt.max_depth)

        for i, frame_id in enumerate(self.opt.frame_ids[1:]):
            if frame_id == "s":
                T = inputs["stereo_T"]
            else:
                T = outputs[("cam_T_cam", 0, frame_id)]
            # 根据相机内参矩阵和深度图将像素反投影为三维点云
            cam_points = self.backproject(depth, inputs[("inv_K")])
            # 将三维点云坐标变换至当前帧并通过相机内参投影到当前帧获得投影后的像素位置
            pix_coords = self.project_3d(
                cam_points, inputs[("K")], T)  # [b,h,w,2]
            # 根据重投影后的像素位置在原图采样生成投影后的图像
            outputs[("color", frame_id, scale)] = F.grid_sample(
                inputs[("color", frame_id, 0)], pix_coords, padding_mode="border")
        return outputs

    def axisangle_to_T_matrix(self, axisangle, translation, invert=False):
        """将轴角向量和位移向量转换为运动变换矩阵"""
        # 将轴角向量转换为旋转矩阵
        R = self.rot_from_axisangle(axisangle)
        t = translation.clone()
        if invert:
            R = R.transpose(1, 2)
            t *= -1
        T = self.get_translation_matrix(t)
        if invert:
            M = torch.matmul(R, T)
        else:
            M = torch.matmul(T, R)
        return M

    def get_translation_matrix(self, translation_vector):
        """将位移向量转换为位移矩阵"""
        T = torch.zeros(translation_vector.shape[0], 4, 4).to(
            translation_vector.device)
        t = translation_vector.contiguous().view(-1, 3, 1)
        T[:, 0, 0] = 1
        T[:, 1, 1] = 1
        T[:, 2, 2] = 1
        T[:, 3, 3] = 1
        T[:, :3, 3, None] = t
        return T

    def rot_from_axisangle(self, vec):
        """将轴角向量转换为旋转矩阵"""
        angle = torch.norm(vec, 2, 2, True)
        axis = vec / (angle + 1e-7)
        ca = torch.cos(angle)
        sa = torch.sin(angle)
        C = 1 - ca
        x = axis[..., 0].unsqueeze(1)
        y = axis[..., 1].unsqueeze(1)
        z = axis[..., 2].unsqueeze(1)
        xs = x * sa
        ys = y * sa
        zs = z * sa
        xC = x * C
        yC = y * C
        zC = z * C
        xyC = x * yC
        yzC = y * zC
        zxC = z * xC
        rot = torch.zeros((vec.shape[0], 4, 4)).to(vec.device)
        rot[:, 0, 0] = torch.squeeze(x * xC + ca)
        rot[:, 0, 1] = torch.squeeze(xyC - zs)
        rot[:, 0, 2] = torch.squeeze(zxC + ys)
        rot[:, 1, 0] = torch.squeeze(xyC + zs)
        rot[:, 1, 1] = torch.squeeze(y * yC + ca)
        rot[:, 1, 2] = torch.squeeze(yzC - xs)
        rot[:, 2, 0] = torch.squeeze(zxC - ys)
        rot[:, 2, 1] = torch.squeeze(yzC + xs)
        rot[:, 2, 2] = torch.squeeze(z * zC + ca)
        rot[:, 3, 3] = 1
        return rot

    def get_smooth_loss(self, disp, img):
        """计算平滑损失，视差图已经过归一化"""
        b, _, h, w = disp.size()
        a1 = 0.5
        a2 = 0.5
        img = F.interpolate(img, (h, w), mode='area')

        # 计算图像和视差图的一阶导
        disp_dx, disp_dy = self.gradient(disp)
        img_dx, img_dy = self.gradient(img)

        # 计算视差的二阶导
        disp_dxx, disp_dxy = self.gradient(disp_dx)
        disp_dyx, disp_dyy = self.gradient(disp_dy)

        # 计算图像的二阶导
        img_dxx, img_dxy = self.gradient(img_dx)
        img_dyx, img_dyy = self.gradient(img_dy)

        # 计算一阶平滑损失
        smooth1 = torch.mean(disp_dx.abs() * torch.exp(-a1 * img_dx.abs().mean(1, True))) + \
            torch.mean(disp_dy.abs() * torch.exp(-a1 *
                       img_dy.abs().mean(1, True)))

        # 计算二阶平滑损失
        smooth2 = torch.mean(disp_dxx.abs() * torch.exp(-a2 * img_dxx.abs().mean(1, True))) + \
            torch.mean(disp_dxy.abs() * torch.exp(-a2 * img_dxy.abs().mean(1, True))) + \
            torch.mean(disp_dyx.abs() * torch.exp(-a2 * img_dyx.abs().mean(1, True))) + \
            torch.mean(disp_dyy.abs() * torch.exp(-a2 *
                       img_dyy.abs().mean(1, True)))

        return smooth1 + smooth2

    def gradient(self, D):
        D_dy = D[:, :, 1:] - D[:, :, :-1]
        D_dx = D[:, :, :, 1:] - D[:, :, :, :-1]
        return D_dx, D_dy
