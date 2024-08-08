# # 测试昨天想到的坐标变换
# # import torch
# # from visual_res_app.camera_trajectory import *
# #
# #
# # def is_cube_corners(corners):
# #  # 计算每对角点之间的欧几里德距离
# #  distances = []
# #  for i in range(len(corners)):
# #   for j in range(i + 1, len(corners)):
# #    dist = torch.norm(corners[i] - corners[j])
# #    distances.append(dist)
# #
# #  # 判断是否满足边长关系
# #  # 因为一个立方体有12条边，每条边有2个相等的距离
# #  num_equal_distances = distances.count(distances[0])
# #  return num_equal_distances == 12 and len(set(distances)) == 3
# #
# # # 之前修改的值
# # dis = torch.tensor([[1.0, 1.0, 1.0]])
# # # 先来一个世界系的bbox
# # comp_l = torch.tensor([-32.0, -100.0, -1000.0])
# # comp_m = torch.tensor([1.0, 1.0, 1.0])
# # corners = torch.tensor(
# #     [[comp_l[0], comp_l[1], comp_l[2]],
# #      [comp_l[0], comp_l[1], comp_m[2]],
# #      [comp_l[0], comp_m[1], comp_l[2]],
# #      [comp_m[0], comp_l[1], comp_l[2]],
# #      [comp_l[0], comp_m[1], comp_m[2]],
# #      [comp_m[0], comp_l[1], comp_m[2]],
# #      [comp_m[0], comp_m[1], comp_l[2]],
# #      [comp_m[0], comp_m[1], comp_m[2]]]
# # )
# # print(corners)
# # # 先把世界系原点移动到bbox中心
# # # new_w_base = torch.tensor([[1,0,0], [0,1,0], [0,0,1]]) + corners.mean(0)
# # # scale = torch.norm(new_w_base, p=2, dim=1)
# # # new_w_base[0] /= scale[0]
# # # new_w_base[1] /= scale[1]
# # # new_w_base[2] /= scale[2]
# # #
# # # corners_new = (new_w_base @ corners.T).T
# #
# # # 现在旋转
# # corners_r, R = center_r_bbox(corners, 100, 100, 0)
# # # print(corners_r)
# # base = (corners_r[1:4, :] - corners_r[0]).flip(0)
# # scale = torch.norm(base, p=2, dim=1)
# # base[0] /= scale[0]
# # base[1] /= scale[1]
# # base[2] /= scale[2]
# # # # 现在改变位置要沿着基变量的方向改变，之前修改x，现在要修改y,z
# # # # 新表示
# # corners_b = (base @ corners_r.T).T
# #
# # # 现在求原始坐标系下的表示
# #
# # corners_r = corners_b @ base
# #
# # t = corners_r[0]
# #
# # w2b = torch.eye(4)
# # w2b[:3, :3] = R
# # w2b[:3, -1] = t
# #
# # b2w = w2b.inverse()
# # homogeneous_corners = torch.cat((corners_b, torch.ones(corners_b.shape[0], 1)), dim=1)
# # corners_r_new = torch.matmul(b2w, homogeneous_corners.t()).t()
# #
# #
# #
# # c = slider_con_bbox_inv(corners, 100, 100, 0)
# # print(c)
#
# # def scene_select(start_end_array, frist, end):
# #   length_move_scene = end - frist
# #   # 2. 计算子场景满足条件几,只需要计算frist， end元素在数组中的索引
# #   index_f = bisect.bisect_left(start_end_array, frist)
# #   index_e = bisect.bisect_left(start_end_array, end)
# #   if index_f == index_e:  # 完全包含
# #    # 该子场景起点不变，终点减去移除场景的长度，后续子场景如是
# #    b = np.array(start_end_array[index_e:])
# #    b -= length_move_scene
# #    start_end_array[index_e:] = b.tolist()
# #
# #   if abs(index_f - index_e) >= 1:  # 处于两个子场景之间
# #
# #
# #    start_end_array[index_f] = frist
# #    b = np.array(start_end_array[index_e:])
# #    b -= length_move_scene
# #    start_end_array[index_e:] = b.tolist()
# #
# #    for idx in range(index_f + 1, index_e):
# #     start_end_array.pop(idx)
# #
# #   # 去重复
# #   start_end_array = sorted(list(set(start_end_array)))
# #
# #   print(start_end_array)
# #
# # if __name__ == '__main__':
# #     start_end_array  = [0, 100, 1000, 10000]
# #     # 情况1：
# #     frist , end = 200, 800
# #     scene_select(start_end_array, frist, end)
# #
# #     # 情况2：
# #     start_end_array = [0, 100, 1000, 10000]
# #     frist, end = 50, 500
# #     scene_select(start_end_array, frist, end)
# #     # 情况3：
# #     start_end_array = [0, 100, 1000, 10000]
# #     frist, end = 100, 1000
# #     scene_select(start_end_array, frist, end)
# #     # 情况4：
# #     start_end_array = [0, 100, 1000, 10000]
# #     frist, end = 50, 1050
# #     scene_select(start_end_array, frist, end)
# #
# #     # 情况5：
# #     start_end_array = [0, 100, 1000, 10000]
# #     frist, end = 0, 1000
# #     scene_select(start_end_array, frist, end)
# #
# #     start_end_array = [0, 1370505]
# #     frist, end = 13, 1370466
# #     scene_select(start_end_array, frist, end)
#
#
# # COLLAPSED
# import torch
# from kornia import create_meshgrid
#
# def cam_vis(H, W, focal_x, focal_y, c2w):
#     grid1 = create_meshgrid(H, W, normalized_coordinates=False)[0]
#     grid = torch.tensor([[[0, 0],[W-1, 0]],
#          [[0, H-1],[W-1, H-1]]], dtype=torch.float)
#     i, j = grid.unbind(-1)
#     # the direction here is without +0.5 pixel centering as calibration is not so accurate
#     # see https://github.com/bmild/nerf/issues/24
#     directions = \
#         torch.stack([(i-W/2)/focal_x, -(j-H/2)/focal_y, -torch.ones_like(i)], -1) # (H, W, 3)
#
#     # Rotate ray directions from camera coordinate to the world coordinate
#     rays_d = directions @ c2w[:, :3].T # (H, W, 3)
#     rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
#     # The origin of all rays is the camera origin in world coordinate
#     rays_o = c2w[None, :, 3] # (1, 3)
#
#     rays_d = rays_d.view(-1, 3)
#     # 定义五面体表示
#     cam_vis_3d = torch.vstack([rays_o, rays_d])
#     return cam_vis_3d
#
# """logic
# 获取ray_0, ray_d,但是只获取四个角的ray_d就可以了
#
# """
#
# cx = 20.0
# cy = 10.0
# fx = 20.0
# fy = 20.0
# h = 5
# w =10
# c2w = torch.eye(4)[:3, :]
# cam_vis_3d = cam_vis(h, w, fx, fy, c2w)
#
#

# def max_score(a, b, c):
#     # 计算可以形成的 "you" 的数量
#     you_count = min(a, b, c)
#
#     # 计算剩余的 'y' 和 'o' 数量
#     remaining_y = a - you_count
#     remaining_o = b - you_count
#
#     # 计算可以形成的额外的 "00" 的数量
#     extra_00_count = min(remaining_y // 2, remaining_o // 2)
#
#     # 计算总分数
#     total_score = you_count * 2 + extra_00_count
#
#     return total_score
#
#
#
# n = input()
# data = []
# for _ in range(n):
#     data.append(list(map(int, input().split())))
#
# for i in data:
#     result = max_score(i[0], i[1], i[2])
#
#     print(result)

from visual_res_app.camera_trajectory import *
import torch

slider = torch.tensor(
    [[0,0,0],
     [1,1,1]]
,dtype=torch.float32)

center_r_bbox(slider, 0, 0, 0)

