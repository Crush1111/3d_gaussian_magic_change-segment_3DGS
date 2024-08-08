'''
将nerf的transforms.json转换为f2-nerf算法所接受的形式
'''
import json
import os.path
import os

import numpy as np
from os.path import join as pjoin
from pathlib import Path

def nerf2poses(data_dir):
    # 只需要把nerf的transforms.json中的pose读出来然后合成pose_bounds
    with open(f'{data_dir}/transforms.json', 'r') as f:
        pose_meta = json.load(f)

    poses = np.concatenate([np.array(i['transform_matrix'])[:3, :][None] for i in sorted(pose_meta['frames'], key=lambda x: int(x['file_path'].split('/')[-1].split('.')[0]))])

    NOISE = False
    if NOISE:
        # 设置扰动的最大范围
        max_noise = 0.1
        # 生成随机扰动矩阵
        noise = np.random.uniform(-max_noise, max_noise, size=(3, 4))
        for p in poses:
            p[:, -1] += noise[:, -1]

    img_path_vaild = [os.path.join(data_dir, i['file_path']) for i in pose_meta['frames']]
    img_path_all = [os.path.join(data_dir, 'images/', i) for i in os.listdir(os.path.join(data_dir, 'images'))]
    for i in img_path_all:
        if i not in img_path_vaild:
            os.system(f'rm {i}')

    hwf = np.tile(np.array([pose_meta['h'], pose_meta['w'], pose_meta['fl_x'], pose_meta['fl_y'], pose_meta['cx'], pose_meta['cy']]), (len(poses), 1))
    bounds = np.tile(np.array([0.01, 30]), (len(poses), 1))

    # 生成zip-nerf的训练格式
    hwf_zip = hwf[:, :3][:, :, None]
    c2w_mats = poses
    pose_zip = np.concatenate((c2w_mats, hwf_zip), -1)
    # trans nerfstudio[x, y, -z]-> llff [y, -x, -z]
    poses_zip = np.concatenate([pose_zip[:, :, 1:2], -pose_zip[:, :, 0:1], pose_zip[:, :, 2:]], 2).reshape(-1, 15)
    # generate bounding 这里是自定义的方式,colmap中根据重建获得的点云进行计算
    save_arr = np.concatenate((poses_zip, bounds), -1)
    np.save(os.path.join(data_dir, 'poses_bounds.npy'), save_arr)

    # bounds = poses_bounds[:, 15: 17]
    # 生成f2-nerf的训练格式
    n_poses = len(poses)
    intri = np.zeros([n_poses, 3, 3])
    intri[:, :3, :3] = np.eye(3)
    intri[:, 0, 0] = hwf[:, 2]
    intri[:, 1, 1] = hwf[:, 3]
    intri[:, 0, 2] = hwf[:, 4]
    intri[:, 1, 2] = hwf[:, 5]

    data = np.concatenate([
        poses.reshape(n_poses, -1),
        intri.reshape(n_poses, -1),
        np.zeros([n_poses, 4]),
        bounds.reshape(n_poses, -1)
    ], -1)

    data = np.ascontiguousarray(np.array(data).astype(np.float64))
    np.save(pjoin(data_dir, 'cams_meta.npy'), data)


if __name__ == '__main__':
    #hello('/home/guozebin/work_code/LargeScaleNeRFPytorch/data/gym')
    # path = list(filter(lambda x: 'inward' in str(x), list(Path('/data/guozebin_data/f2-nerf_ablation').glob('*'))))
    # for p in path:
    #     nerf2poses(str(p))
    nerf2poses('/home/guozebin/work_code/f2-nerf/data/free_dataset/2f_0731/')