#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

segment_loss = nn.CrossEntropyLoss()

def compute_rank_loss(dyn_depth, gt_depth, lambda_depth, sample_nums=1000):
    """
    rank loss 不关心2d位置，只关心前后关系
    dyn_depth：nerf渲染视差
    gt_depth：单目深度估计视差
    """

    pred_depth = dyn_depth.view(1, -1) / dyn_depth.max()
    gt_depth = gt_depth.view(1, -1) / gt_depth.max()

    # 随机取1000个样本
    sample = torch.randint(0, pred_depth.shape[1], (sample_nums,))
    # 采样
    pred_depth = pred_depth[:, sample]
    gt_depth = gt_depth[:, sample]

    # 直接满足前后关系
    mask_rank = torch.where(gt_depth.unsqueeze(-1) - gt_depth.unsqueeze(1) > 0, 1, 0).type(torch.bool)
    rank_loss = (pred_depth.unsqueeze(1) - pred_depth.unsqueeze(-1) + 1e-4)[mask_rank].clamp(0).mean() * lambda_depth

    return rank_loss



def compute_continue_loss(dyn_depth, gt_depth, lambda_depth, sample_nums=100, patch_size=3):
    """
    本质是让在gt上连续的位置(mask)，在pred上也连续
    关键步骤：
    1.找连续位置
        条件：邻居
             连续
    """
    gt_depth = gt_depth / gt_depth.max()
    dyn_depth = dyn_depth / dyn_depth.max()
    # 随机选100个位置，以这100个位置为中心，构建patch
    sample_w = torch.randint(0, gt_depth.shape[1] - patch_size, (sample_nums,))
    sample_h = torch.randint(0, gt_depth.shape[2] - patch_size, (sample_nums,))

    # 初始化一个列表，用于存储 100 个样本点的 6x6 的像素索引
    patchs_gt = []
    patchs_pred = []

    # 遍历每个样本点，获得以该点为中心的 6x6 像素索引
    for i in range(sample_nums):
        w, h = sample_w[i], sample_h[i]

        # 生成以(w, h)为中心的6x6像素索引
        patch_w_indices = torch.arange(w, w + patch_size)
        patch_h_indices = torch.arange(h, h + patch_size)

        # 使用 torch.meshgrid 创建所有可能的组合
        patch_w, patch_h = torch.meshgrid(patch_w_indices, patch_h_indices)

        # 将索引转化为坐标对，形如 (w, h)
        patchs_gt.append(gt_depth[:, patch_w, patch_h])
        patchs_pred.append(dyn_depth[:, patch_w, patch_h])

    gt_depth = torch.cat(patchs_gt).reshape(1, sample_nums, -1).transpose(2, 1)
    pred_depth = torch.cat(patchs_pred).reshape(1, sample_nums, -1).transpose(2, 1)
    condition = (gt_depth[:, (patch_size **2 // 2 + 1), :].unsqueeze(0).permute(1, 0, 2) - gt_depth).abs()
    mask = torch.logical_and(condition <= 1e-3, condition > 0)
    if torch.all(~mask):
        continue_loss = torch.tensor(0).to(dyn_depth.devices)
        return continue_loss
    else:
        continue_loss = ((pred_depth[:, 0, :].unsqueeze(0).permute(1, 0, 2) - pred_depth).abs() - 1e-3)[mask].clamp(0).mean() * lambda_depth
        return continue_loss

def compute_depth_loss(dyn_depth, gt_depth, lambda_depth):

    dyn_depth = dyn_depth.view(1, -1)
    gt_depth = gt_depth.view(1, -1)
    t_d = torch.median(dyn_depth, dim=-1, keepdim=True).values
    s_d = torch.mean(torch.abs(dyn_depth - t_d), dim=-1, keepdim=True)
    dyn_depth_norm = (dyn_depth - t_d) / s_d

    t_gt = torch.median(gt_depth, dim=-1, keepdim=True).values
    s_gt = torch.mean(torch.abs(gt_depth - t_gt), dim=-1, keepdim=True)
    gt_depth_norm = (gt_depth - t_gt) / s_gt
    depth_loss_arr = (dyn_depth_norm - gt_depth_norm) ** 2
    depth_loss_arr[depth_loss_arr > torch.quantile(depth_loss_arr, 0.8, dim=1)[..., None]] = 0
    depth_loss = (depth_loss_arr).mean() * lambda_depth
    return  depth_loss

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

if __name__ == '__main__':
    dyn_depth = torch.Tensor([[1,2,3]])
    gt_depth = torch.Tensor([[1,2,3]])
    # rank_loss(dyn_depth, gt_depth)
    compute_continue_loss(dyn_depth, gt_depth)