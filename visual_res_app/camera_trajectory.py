import numpy as np
from utils.graphics_utils import getProjectionMatrix
from scene.cameras import MiniCam
import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from tqdm import tqdm
from datetime import datetime

THETA = 5
STEP = 0.1
STEP_pcd = 0.01

def rotate_n_z_axis(n=30):
    theta = torch.deg2rad(torch.tensor(n))
    R = torch.tensor([[torch.cos(theta), 0, torch.sin(theta)], [0, 1, 0], [-torch.sin(theta), 0, torch.cos(theta)]])
    return R

def rotate_n_horizontal_axis(n=30):
    theta = torch.deg2rad(torch.tensor(n))
    R = torch.tensor([[torch.cos(theta), -torch.sin(theta), 0], [torch.sin(theta), torch.cos(theta), 0], [0, 0, 1]])
    return R

def rotate_n_vertical_axis(n=30):
    theta = torch.deg2rad(torch.tensor(n))
    R = torch.tensor([[1, 0, 0], [0, torch.cos(theta), -torch.sin(theta)], [0, torch.sin(theta), torch.cos(theta)]])
    return R

def turn_l(camera, theta=THETA):

    R = rotate_n_z_axis(n=theta)
    transform_matrix = camera.world_view_transform.clone()
    transform_matrix[:, :3] = torch.bmm(R.to(transform_matrix.device).unsqueeze(0), transform_matrix.T[:3].unsqueeze(0)).squeeze(0).T

    world_view_transform = transform_matrix
    projection_matrix = getProjectionMatrix(znear=camera.znear, zfar=camera.zfar, fovX=camera.FoVx,
                                                 fovY=camera.FoVy).transpose(0, 1).cuda()
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

    update_camera = MiniCam(camera.image_width, camera.image_height, camera.FoVy, camera.FoVx, camera.znear, camera.zfar, world_view_transform, full_proj_transform)
    return update_camera

def turn_r(camera, theta=-THETA):

    R = rotate_n_z_axis(n=theta)
    transform_matrix = camera.world_view_transform.clone()
    transform_matrix[:, :3] = torch.bmm(R.to(transform_matrix.device).unsqueeze(0), transform_matrix.T[:3].unsqueeze(0)).squeeze(0).T

    world_view_transform = transform_matrix
    projection_matrix = getProjectionMatrix(znear=camera.znear, zfar=camera.zfar, fovX=camera.FoVx,
                                            fovY=camera.FoVy).transpose(0, 1).cuda()
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

    update_camera = MiniCam(camera.image_width, camera.image_height, camera.FoVy, camera.FoVx, camera.znear,
                            camera.zfar, world_view_transform, full_proj_transform)
    return update_camera

def turn_up(camera, theta=-THETA):
    R = rotate_n_vertical_axis(n=theta)
    transform_matrix = camera.world_view_transform.clone()
    transform_matrix[:, :3] = torch.bmm(R.to(transform_matrix.device).unsqueeze(0), transform_matrix.T[:3].unsqueeze(0)).squeeze(0).T

    world_view_transform = transform_matrix
    projection_matrix = getProjectionMatrix(znear=camera.znear, zfar=camera.zfar, fovX=camera.FoVx,
                                            fovY=camera.FoVy).transpose(0, 1).cuda()
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

    update_camera = MiniCam(camera.image_width, camera.image_height, camera.FoVy, camera.FoVx, camera.znear,
                            camera.zfar, world_view_transform, full_proj_transform)
    return update_camera

def turn_down(camera, theta=THETA):

    R = rotate_n_vertical_axis(n=theta)
    transform_matrix = camera.world_view_transform.clone()
    transform_matrix[:, :3] = torch.bmm(R.to(transform_matrix.device).unsqueeze(0), transform_matrix.T[:3].unsqueeze(0)).squeeze(0).T

    world_view_transform = transform_matrix
    projection_matrix = getProjectionMatrix(znear=camera.znear, zfar=camera.zfar, fovX=camera.FoVx,
                                            fovY=camera.FoVy).transpose(0, 1).cuda()
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

    update_camera = MiniCam(camera.image_width, camera.image_height, camera.FoVy, camera.FoVx, camera.znear,
                            camera.zfar, world_view_transform, full_proj_transform)
    return update_camera

def turn_z_axis_l(camera, theta=THETA):

    R = rotate_n_horizontal_axis(n=theta)
    transform_matrix = camera.world_view_transform.clone().clone()
    transform_matrix[:, :3] = torch.bmm(R.to(transform_matrix.device).unsqueeze(0), transform_matrix.T[:3].unsqueeze(0)).squeeze(0).T

    world_view_transform = transform_matrix
    projection_matrix = getProjectionMatrix(znear=camera.znear, zfar=camera.zfar, fovX=camera.FoVx,
                                                 fovY=camera.FoVy).transpose(0, 1).cuda()
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

    update_camera = MiniCam(camera.image_width, camera.image_height, camera.FoVy, camera.FoVx, camera.znear, camera.zfar, world_view_transform, full_proj_transform)
    return update_camera




def turn_z_axis_r(camera, theta=-THETA):

    R = rotate_n_horizontal_axis(n=theta)
    transform_matrix = camera.world_view_transform.clone()
    transform_matrix[:, :3] = torch.bmm(R.to(transform_matrix.device).unsqueeze(0), transform_matrix.T[:3].unsqueeze(0)).squeeze(0).T

    world_view_transform = transform_matrix
    projection_matrix = getProjectionMatrix(znear=camera.znear, zfar=camera.zfar, fovX=camera.FoVx,
                                            fovY=camera.FoVy).transpose(0, 1).cuda()
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    update_camera = MiniCam(camera.image_width, camera.image_height, camera.FoVy, camera.FoVx, camera.znear,
                            camera.zfar, world_view_transform, full_proj_transform)
    return update_camera

def trans_cam_center_to_bbox_center(camera, bbox, base):
    center = torch.mean(bbox, dim=0)
    center_org = center @ base
    inv_trans = torch.inverse(camera.world_view_transform)
    inv_trans[3, :3] = center_org
    transform_matrix = torch.inverse(inv_trans)

    world_view_transform = transform_matrix
    projection_matrix = getProjectionMatrix(znear=camera.znear, zfar=camera.zfar, fovX=camera.FoVx,
                                            fovY=camera.FoVy).transpose(0, 1).cuda()
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    update_camera = MiniCam(camera.image_width, camera.image_height, camera.FoVy, camera.FoVx, camera.znear,
                            camera.zfar, world_view_transform, full_proj_transform)
    return update_camera


def go_forward(camera, step=STEP):
    # 计算w2c
    transform_matrix = camera.world_view_transform.clone()
    transform_matrix[-1, 2] -= step

    world_view_transform = transform_matrix
    projection_matrix = getProjectionMatrix(znear=camera.znear, zfar=camera.zfar, fovX=camera.FoVx,
                                                 fovY=camera.FoVy).transpose(0, 1).cuda()
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

    update_camera = MiniCam(camera.image_width, camera.image_height, camera.FoVy, camera.FoVx, camera.znear, camera.zfar, world_view_transform, full_proj_transform)
    return update_camera

def go_backward(camera, step=STEP):
    # 计算w2c
    transform_matrix = camera.world_view_transform.clone()
    transform_matrix[-1, 2] += step

    world_view_transform = transform_matrix
    projection_matrix = getProjectionMatrix(znear=camera.znear, zfar=camera.zfar, fovX=camera.FoVx,
                                                 fovY=camera.FoVy).transpose(0, 1).cuda()
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

    update_camera = MiniCam(camera.image_width, camera.image_height, camera.FoVy, camera.FoVx, camera.znear, camera.zfar, world_view_transform, full_proj_transform)
    return update_camera

def go_left(camera, step=STEP):
    # 计算w2c
    transform_matrix = camera.world_view_transform.clone()
    transform_matrix[-1, 0] += step

    world_view_transform = transform_matrix
    projection_matrix = getProjectionMatrix(znear=camera.znear, zfar=camera.zfar, fovX=camera.FoVx,
                                                 fovY=camera.FoVy).transpose(0, 1).cuda()
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

    update_camera = MiniCam(camera.image_width, camera.image_height, camera.FoVy, camera.FoVx, camera.znear, camera.zfar, world_view_transform, full_proj_transform)
    return update_camera

def go_right(camera, step=STEP):
    # 计算w2c
    transform_matrix = camera.world_view_transform.clone()
    transform_matrix[-1, 0] -= step

    world_view_transform = transform_matrix
    projection_matrix = getProjectionMatrix(znear=camera.znear, zfar=camera.zfar, fovX=camera.FoVx,
                                                 fovY=camera.FoVy).transpose(0, 1).cuda()
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

    update_camera = MiniCam(camera.image_width, camera.image_height, camera.FoVy, camera.FoVx, camera.znear, camera.zfar, world_view_transform, full_proj_transform)
    return update_camera

def go_up(camera, step=STEP):
    # 计算w2c
    transform_matrix = camera.world_view_transform.clone()
    transform_matrix[-1, 1] += step

    world_view_transform = transform_matrix
    projection_matrix = getProjectionMatrix(znear=camera.znear, zfar=camera.zfar, fovX=camera.FoVx,
                                                 fovY=camera.FoVy).transpose(0, 1).cuda()
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

    update_camera = MiniCam(camera.image_width, camera.image_height, camera.FoVy, camera.FoVx, camera.znear, camera.zfar, world_view_transform, full_proj_transform)
    return update_camera

def go_down(camera, step=STEP):
    # 计算w2c
    transform_matrix = camera.world_view_transform.clone()
    transform_matrix[-1, 1] -= step

    world_view_transform = transform_matrix
    projection_matrix = getProjectionMatrix(znear=camera.znear, zfar=camera.zfar, fovX=camera.FoVx,
                                                 fovY=camera.FoVy).transpose(0, 1).cuda()
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

    update_camera = MiniCam(camera.image_width, camera.image_height, camera.FoVy, camera.FoVx, camera.znear, camera.zfar, world_view_transform, full_proj_transform)
    return update_camera

# 点云移动
def go_forward_pcd(pcd, STEP_pcd=STEP_pcd):
    pcd += torch.tensor([0, 0, STEP_pcd]).to(pcd.device)
    return pcd

def go_backward_pcd(pcd, STEP_pcd=STEP_pcd):
    pcd += torch.tensor([0, 0, -STEP_pcd]).to(pcd.device)
    return pcd

def go_left_pcd(pcd, STEP_pcd=STEP_pcd):
    pcd += torch.tensor([-STEP_pcd, 0, 0]).to(pcd.device)
    return pcd

def go_right_pcd(pcd, STEP_pcd=STEP_pcd):
    pcd += torch.tensor([STEP_pcd, 0, 0]).to(pcd.device)
    return pcd

def go_up_pcd(pcd, STEP_pcd=STEP_pcd):
    pcd += torch.tensor([0, -STEP_pcd, 0]).to(pcd.device)
    return pcd

def go_down_pcd(pcd, STEP_pcd=STEP_pcd):
    pcd += torch.tensor([0, STEP_pcd, 0]).to(pcd.device)
    return pcd

def mouse_con(camera, angle_x, angle_y):
    # 构造绕Y轴的旋转矩阵
    rotation_y = torch.tensor([
        [torch.cos(angle_y), 0, -torch.sin(angle_y)],
        [0, 1, 0],
        [torch.sin(angle_y), 0, torch.cos(angle_y)]
    ])

    # 构造绕X轴的旋转矩阵
    rotation_x = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(angle_x), torch.sin(angle_x)],
        [0, -torch.sin(angle_x), torch.cos(angle_x)]
    ])

    R = torch.mm(rotation_y, rotation_x)
    # print(R)
    # print(R.shape)

    transform_matrix = camera.world_view_transform.clone()
    transform_matrix[:, :3] = torch.bmm(R.to(transform_matrix.device).unsqueeze(0), transform_matrix.T[:3].unsqueeze(0)).squeeze(0).T

    world_view_transform = transform_matrix
    projection_matrix = getProjectionMatrix(znear=camera.znear, zfar=camera.zfar, fovX=camera.FoVx,
                                            fovY=camera.FoVy).transpose(0, 1).cuda()
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

    update_camera = MiniCam(camera.image_width, camera.image_height, camera.FoVy, camera.FoVx, camera.znear,
                            camera.zfar, world_view_transform, full_proj_transform)
    return update_camera

def mouse_con_pcd(pcd, angle_x, angle_y):
    # 计算点云的中心坐标
    center = torch.mean(pcd, dim=0)
    # 将点云中的每个点减去中心坐标
    centered_pcd = pcd - center
    # 构造绕Y轴的旋转矩阵
    rotation_y = torch.tensor([
        [torch.cos(angle_y), 0, -torch.sin(angle_y)],
        [0, 1, 0],
        [torch.sin(angle_y), 0, torch.cos(angle_y)]
    ])

    # 构造绕X轴的旋转矩阵
    rotation_x = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(angle_x), torch.sin(angle_x)],
        [0, -torch.sin(angle_x), torch.cos(angle_x)]
    ])

    R = torch.mm(rotation_y, rotation_x)
    # 将旋转矩阵应用到点云上
    rotated_pcd = torch.mm(centered_pcd, R.T.to(pcd.device))
    # 将旋转后的点云的每个点加上中心坐标
    final_pcd = rotated_pcd + center

    return final_pcd

def slider_con_bbox(pcd, angle_x_, angle_y_, angle_z_):
    # 计算点云的中心坐标
    T = pcd.mean(0)
    center_pcd = pcd - T
    # 将点云中的每个点减去中心坐标
    angle_x, angle_y, angle_z = torch.deg2rad(torch.tensor(angle_x_)), \
                                torch.deg2rad(torch.tensor(angle_y_)), torch.deg2rad(torch.tensor(angle_z_))
    # 构造绕X轴的旋转矩阵
    rotation_x = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(angle_x), torch.sin(angle_x)],
        [0, -torch.sin(angle_x), torch.cos(angle_x)]
    ])
    # 构造绕Y轴的旋转矩阵
    rotation_y = torch.tensor([
        [torch.cos(angle_y), 0, -torch.sin(angle_y)],
        [0, 1, 0],
        [torch.sin(angle_y), 0, torch.cos(angle_y)]
    ])
    # 构造绕 Z 轴的旋转矩阵
    rotation_z = torch.tensor([
        [torch.cos(angle_z), torch.sin(angle_z), 0],
        [-torch.sin(angle_z), torch.cos(angle_z), 0],
        [0, 0, 1]
    ])

    R  = torch.mm(rotation_z, torch.mm(rotation_y, rotation_x))
    # 计算投影坐标
    # 将旋转矩阵应用到点云上
    rotated_pcd = torch.mm(R.to(pcd.device), center_pcd.T).T
    # 将旋转后的点云的每个点加上中心坐标
    final_pcd = rotated_pcd + T

    return final_pcd

def center_r_bbox(pcd, angle_x_, angle_y_, angle_z_):
    """
    先计算bbox坐标系下绕bbox中心的旋转
    然后将该旋转转换到世界坐标系的表示
    bbox 坐标系： 与世界坐标系就相差一个T
    用世界坐标系表示这个函数
    """

    # 将点云中的每个点减去中心坐标
    angle_x, angle_y, angle_z = torch.deg2rad(torch.tensor(angle_x_)), \
                                torch.deg2rad(torch.tensor(angle_y_)), torch.deg2rad(torch.tensor(angle_z_))
    # 构造绕X轴的旋转矩阵
    rotation_x = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(angle_x), torch.sin(angle_x)],
        [0, -torch.sin(angle_x), torch.cos(angle_x)]
    ])
    # 构造绕Y轴的旋转矩阵
    rotation_y = torch.tensor([
        [torch.cos(angle_y), 0, -torch.sin(angle_y)],
        [0, 1, 0],
        [torch.sin(angle_y), 0, torch.cos(angle_y)]
    ])
    # 构造绕 Z 轴的旋转矩阵
    rotation_z = torch.tensor([
        [torch.cos(angle_z), torch.sin(angle_z), 0],
        [-torch.sin(angle_z), torch.cos(angle_z), 0],
        [0, 0, 1]
    ])

    R = torch.mm(rotation_z, torch.mm(rotation_y, rotation_x))
    rotated_pcd = torch.mm(R.to(pcd.device), pcd.T).T

    return rotated_pcd

def slider_con_bbox_inv(pcd, angle_x_, angle_y_, angle_z_):
    # 计算点云的中心坐标
    # center = pcd[0]
    center = torch.mean(pcd, dim=0)
    # 将点云中的每个点减去中心坐标
    centered_pcd = pcd - center
    angle_x, angle_y, angle_z = torch.deg2rad(torch.tensor(angle_x_)), \
                                torch.deg2rad(torch.tensor(angle_y_)), torch.deg2rad(torch.tensor(angle_z_))
    # 构造绕X轴的旋转矩阵
    rotation_x = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(angle_x), torch.sin(angle_x)],
        [0, -torch.sin(angle_x), torch.cos(angle_x)]
    ])
    # 构造绕Y轴的旋转矩阵
    rotation_y = torch.tensor([
        [torch.cos(angle_y), 0, -torch.sin(angle_y)],
        [0, 1, 0],
        [torch.sin(angle_y), 0, torch.cos(angle_y)]
    ])
    # 构造绕 Z 轴的旋转矩阵
    rotation_z = torch.tensor([
        [torch.cos(angle_z), torch.sin(angle_z), 0],
        [-torch.sin(angle_z), torch.cos(angle_z), 0],
        [0, 0, 1]
    ])

    R  = torch.mm(rotation_z, torch.mm(rotation_y, rotation_x)).inverse()
    # 计算投影坐标
    # 将旋转矩阵应用到点云上
    rotated_pcd = torch.mm(R.to(pcd.device), centered_pcd.T).T + center
    # 将旋转后的点云的每个点加上中心坐标
    # final_pcd = rotated_pcd + center

    return rotated_pcd


def rotation_matrix_to_quaternion_torch(
        R: torch.Tensor  # (batch_size, 3, 3)
) -> torch.Tensor:
    q = torch.zeros(R.shape[0], 4, device=R.device, dtype=R.dtype)  # (batch_size, 4) x, y, z, w
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    q0_mask = trace > 0
    q1_mask = (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2]) & ~q0_mask
    q2_mask = (R[..., 1, 1] > R[..., 2, 2]) & ~q0_mask & ~q1_mask
    q3_mask = ~q0_mask & ~q1_mask & ~q2_mask
    if q0_mask.any():
        R_for_q0 = R[q0_mask]
        S_for_q0 = 0.5 / torch.sqrt(1 + trace[q0_mask])
        q[q0_mask, 3] = 0.25 / S_for_q0
        q[q0_mask, 0] = (R_for_q0[..., 2, 1] - R_for_q0[..., 1, 2]) * S_for_q0
        q[q0_mask, 1] = (R_for_q0[..., 0, 2] - R_for_q0[..., 2, 0]) * S_for_q0
        q[q0_mask, 2] = (R_for_q0[..., 1, 0] - R_for_q0[..., 0, 1]) * S_for_q0

    if q1_mask.any():
        R_for_q1 = R[q1_mask]
        S_for_q1 = 2.0 * torch.sqrt(1 + R_for_q1[..., 0, 0] - R_for_q1[..., 1, 1] - R_for_q1[..., 2, 2])
        q[q1_mask, 0] = 0.25 * S_for_q1
        q[q1_mask, 1] = (R_for_q1[..., 0, 1] + R_for_q1[..., 1, 0]) / S_for_q1
        q[q1_mask, 2] = (R_for_q1[..., 0, 2] + R_for_q1[..., 2, 0]) / S_for_q1
        q[q1_mask, 3] = (R_for_q1[..., 2, 1] - R_for_q1[..., 1, 2]) / S_for_q1

    if q2_mask.any():
        R_for_q2 = R[q2_mask]
        S_for_q2 = 2.0 * torch.sqrt(1 + R_for_q2[..., 1, 1] - R_for_q2[..., 0, 0] - R_for_q2[..., 2, 2])
        q[q2_mask, 0] = (R_for_q2[..., 0, 1] + R_for_q2[..., 1, 0]) / S_for_q2
        q[q2_mask, 1] = 0.25 * S_for_q2
        q[q2_mask, 2] = (R_for_q2[..., 1, 2] + R_for_q2[..., 2, 1]) / S_for_q2
        q[q2_mask, 3] = (R_for_q2[..., 0, 2] - R_for_q2[..., 2, 0]) / S_for_q2

    if q3_mask.any():
        R_for_q3 = R[q3_mask]
        S_for_q3 = 2.0 * torch.sqrt(1 + R_for_q3[..., 2, 2] - R_for_q3[..., 0, 0] - R_for_q3[..., 1, 1])
        q[q3_mask, 0] = (R_for_q3[..., 0, 2] + R_for_q3[..., 2, 0]) / S_for_q3
        q[q3_mask, 1] = (R_for_q3[..., 1, 2] + R_for_q3[..., 2, 1]) / S_for_q3
        q[q3_mask, 2] = 0.25 * S_for_q3
        q[q3_mask, 3] = (R_for_q3[..., 1, 0] - R_for_q3[..., 0, 1]) / S_for_q3
    return q

def quaternion_multiply_torch(
    q0: torch.Tensor, # (batch_size, 4)
    q1: torch.Tensor, # (batch_size, 4)
):
    x0, y0, z0, w0 = q0[..., 0], q0[..., 1], q0[..., 2], q0[..., 3]
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
    w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    return torch.stack([x, y, z, w], dim=-1)

def mouse_con_quaternion(quaternion, angle_x, angle_y):
    # 构造绕Y轴的旋转矩阵
    rotation_y = torch.tensor([
        [torch.cos(angle_y), 0, -torch.sin(angle_y)],
        [0, 1, 0],
        [torch.sin(angle_y), 0, torch.cos(angle_y)]
    ])

    # 构造绕X轴的旋转矩阵
    rotation_x = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(angle_x), torch.sin(angle_x)],
        [0, -torch.sin(angle_x), torch.cos(angle_x)]
    ])

    R = torch.mm(rotation_y, rotation_x).T.unsqueeze(0).to(quaternion.device)

    # 将旋转矩阵应用到协方差矩阵上
    quaternion = quaternion_multiply_torch(quaternion, rotation_matrix_to_quaternion_torch(R))

    return quaternion



def inter_two_poses(pose_a, pose_b, alpha):
    # pose c2w 3x4
    ret = np.zeros([3, 4], dtype=np.float32)
    key_rots = R.from_matrix(np.stack([pose_a[:3, :3], pose_b[:3, :3]], 0))
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)
    rot = slerp(1. - alpha)
    ret[:3, :3] = rot.as_matrix()
    ret[:3, 3] = (pose_a * alpha + pose_b * (1. - alpha))[:3, 3]
    return ret

import os

def inter_poses(key_poses, n_out_poses, sigma=1., save_path=None):
    """
    key_poses : MiniCam
    """

    print("平滑插值！～")
    camera = key_poses[0]
    n_key_poses = len(key_poses)
    out_poses = []
    pose_c2w = []
    for i in tqdm(range(n_out_poses), desc="inter progress"):

        w = np.linspace(0, n_key_poses - 1, n_key_poses)
        w = np.exp(-(np.abs(i / n_out_poses * n_key_poses - w) / sigma)**2)
        w = w + 1e-6
        w /= np.sum(w)
        cur_pose = key_poses[0].world_view_transform.transpose(0, 1).inverse()[:3, :].cpu().numpy()
        cur_w = w[0]
        for j in range(0, n_key_poses - 1):
            cur_pose = inter_two_poses(cur_pose, key_poses[j + 1].world_view_transform.transpose(0, 1).inverse()[:3, :].cpu().numpy(), cur_w / (cur_w + w[j + 1]))
            cur_w += w[j + 1]
        # 保存成该形式
        pose_c2w.append(cur_pose)

        world_view_transform = torch.cat([torch.from_numpy(cur_pose), torch.tensor([[0,0,0,1]])]).inverse().transpose(0, 1).cuda()
        projection_matrix = getProjectionMatrix(znear=camera.znear, zfar=camera.zfar, fovX=camera.FoVx,
                                                fovY=camera.FoVy).transpose(0, 1).cuda()
        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

        update_camera = MiniCam(camera.image_width, camera.image_height, camera.FoVy, camera.FoVx, camera.znear,
                                camera.zfar, world_view_transform, full_proj_transform)

        out_poses.append(update_camera)

    if save_path is not None:
        save_pose = np.stack(pose_c2w)
        save = {
            "znear": camera.znear,
            "zfar": camera.zfar,
            "fovX": camera.FoVx,
            "fovY": camera.FoVy,
            "image_width": camera.image_width,
            "image_height": camera.image_height,
            "pose": save_pose
        }
        # current_datetime = datetime.now()
        np.save(os.path.join(save_path, f'poses_render.npy'), save, allow_pickle=True)
        # np.save(os.path.join(save_path, f'{current_datetime.strftime("%Y-%m-%d %H:%M:%S")}_poses_render.npy'), save, allow_pickle=True)

    return out_poses


def read_render_pose_from_npy(poses_path):
    out_poses = []
    camera = np.load(poses_path, allow_pickle=True).item()
    for cur_pose in tqdm(camera["pose"], desc="pose trans format progress"):

        world_view_transform = torch.cat([torch.from_numpy(cur_pose), torch.tensor([[0,0,0,1]])]).inverse().transpose(0, 1).cuda()
        projection_matrix = getProjectionMatrix(znear=camera["znear"], zfar=camera["zfar"], fovX=camera["fovX"],
                                                fovY=camera["fovY"]).transpose(0, 1).cuda()
        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

        update_camera = MiniCam(camera["image_width"], camera["image_height"], camera["fovY"], camera["fovX"], camera["znear"],
                                camera["zfar"], world_view_transform, full_proj_transform)

        out_poses.append(update_camera)
    return out_poses

def cam_vis(H, W, focal_x, focal_y, c2w):

    grid = torch.tensor([[[0, 0],[H-1, 0]],
         [[0, W-1],[H-1, W-1]]], dtype=torch.float)
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = \
        torch.stack([(i-W/2)/focal_x, -(j-H/2)/focal_y, -torch.ones_like(i)], -1) # (H, W, 3)

    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[None, :, 3] # (1, 3)

    rays_d = rays_d.view(-1, 3)
    # 定义五面体表示
    cam_vis_3d = torch.vstack([rays_o, rays_d])
    return cam_vis_3d

import math

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def cam_vis(camera):

    H, W = camera.image_height, \
           camera.image_width

    focal_x, focal_y = fov2focal(camera.FoVx , W), \
                       fov2focal(camera.FoVy , H)
    # c2w = torch.inverse(camera.world_view_transform.T)

    grid = torch.tensor([[[0, 0],[W-1, 0]],
         [[0, H-1],[W-1, H-1]]], dtype=torch.float)
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = \
        torch.stack([(i-W/2)/focal_x, -(j-H/2)/focal_y, -torch.ones_like(i)], -1).to(camera.world_view_transform.device) # (H, W, 3)

    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ camera.world_view_transform.T[:3, :3] # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = camera.camera_center
    rays_d = rays_d.view(-1, 3)
    rays_d = rays_o - rays_d * 0.15

    # 定义五面体表示
    cam_vis_3d = torch.vstack([rays_o, rays_d])

    return cam_vis_3d

def cam_vis_1(H, W, focal_x, focal_y, world_view_transform):

    grid = torch.tensor([[[0, 0],[W-1, 0]],
         [[0, H-1],[W-1, H-1]]], dtype=torch.float)
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = \
        torch.stack([(i-W/2)/focal_x, -(j-H/2)/focal_y, -torch.ones_like(i)], -1).to(world_view_transform.device) # (H, W, 3)

    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ world_view_transform.T[:3, :3] # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = world_view_transform.inverse()[3, :3]
    rays_d = rays_d.view(-1, 3)
    rays_d = rays_o - rays_d * 0.15
    centers  = rays_d.mean(0)
    # 定义五面体表示
    cam_vis_3d = torch.vstack([centers, rays_o, rays_d])

    return cam_vis_3d