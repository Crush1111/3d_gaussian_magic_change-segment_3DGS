import numpy as np
import os
import json
from pathlib import Path

'''
将从slam导出的pose.txt转换为nerf的格式（适应与：instant-ngp，nerfstudio等算法)
'''
def read_poses(pose):
    pose = np.array(pose, dtype=np.float32).reshape(3, -1)
    pose = np.concatenate([pose, np.array([[0, 0, 0, 1]])])

    # 尝试将colmap坐标系转换到标准nerf坐标系去
    pose[:3, 1:3] *= -1
    # pose = pose[np.array([1, 0, 2, 3]), :]
    # pose[2, :] *= -1

    return pose

# def read_ins():
#     # output: height, width, K(3x3)
#     fx = 1381.34767982
#     fy = 1382.54776057
#     cx = 964.80360438
#     cy = 533.43973592
#
#     h, w = 1080, 1920
#     # distortion parameters
#     k1 = 0.16239136
#     k2 = -0.401863
#     p1 = 0.00082195
#     p2 = 0.00122243
#
#     return h, w, fx, fy, cx, cy, k1, k2, p1, p2

def read_ins():
    # output: height, width, K(3x3)
    fx = 641.158935546875
    fy = 656.1329345703125
    cx = 640.25537109375
    cy = 362.660797119140
    h, w = 720, 1280
    # distortion parameters
    k1 = -0.05613917484879494
    k2 = 0.06781932711601257
    p1 = -7.59519316488877e-05
    p2 = 0.00057106249732896
    return h, w, fx, fy, cx, cy, k1, k2, p1, p2

#转换整个数据集
def save_one(root_dir):
    data = {}

    frames = []
    h, w, fx, fy, cx, cy, k1, k2, p1, p2 = read_ins()  #内参
    # 读取图片名以及对应的pose
    with open(root_dir + "/KeyFramePose.txt", 'r') as f:
        file = f.readlines()
    for i, line in enumerate(file):
        img_id, *pose = line.split()
        pose = read_poses(pose)
        if i == 0:
            data = dict(
                fl_x=fx,
                fl_y=fy,
                k1=k1,
                k2=k2,
                k3=0,
                k4=0,
                p1=p1,
                p2=p2,
                is_fisheye=False,
                cx=cx,
                cy=cy,
                w=w,
                h=h,
                aabb_scale=16,
            )
        frame = {}
        frame["file_path"] = f"images/{img_id}.jpg"
        # Transform matrix to nerfstudio format. Please refer to the documentation for coordinate system conventions.
        frame["transform_matrix"] = [i.tolist() for i in pose]
        frames.append(frame)
    data["frames"] = frames
    with open(root_dir + "transforms.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# 根据划分的block转换数据集
def save_one_block(root_dir):
    block_space_split = [
             [[1, 100]],
             [[50, 250]]
            ]

    h, w, fx, fy, cx, cy, k1, k2, p1, p2 = read_ins()  # 内参
    # 读取图片名以及对应的pose
    with open(root_dir + "/KeyFramePose.txt", 'r') as f:
        file = f.readlines()
    # 构建字典查询
    file_dict = {}
    for i in file:
        img_id, *pose = i.split()
        file_dict[img_id] = pose

    # 将file分块
    for idx in range(len(block_space_split)):
        data = {}
        frames = []
        os.makedirs(f'{root_dir}/block_{idx}/images', exist_ok=True)
        for index in range(len(block_space_split[idx])):
            start, end = block_space_split[idx][index]
            for i in file_dict.keys():
                if int(i) >= start and int(i) <= end:
                    img_id, *pose = str(i), file_dict[i]
                    pose = read_poses(pose)
                    if int(i) == start:
                        data = dict(
                            fl_x=fx,
                            fl_y=fy,
                            k1=k1,
                            k2=k2,
                            k3=0,
                            k4=0,
                            p1=p1,
                            p2=p2,
                            is_fisheye=False,
                            cx=cx,
                            cy=cy,
                            w=w,
                            h=h,
                            aabb_scale=16,
                        )
                    frame = {}
                    frame["file_path"] = f"./images/{img_id}.jpg"
                    os.system(f'cp {root_dir}/images/{img_id}.jpg {root_dir}/block_{idx}/images')
                    # Transform matrix to nerfstudio format. Please refer to the documentation for coordinate system conventions.
                    frame["transform_matrix"] = [i.tolist() for i in pose]
                    frames.append(frame)
            data["frames"] = frames
            with open(f"{root_dir}/block_{idx}/transforms.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)



'''

localrf中的策略：
1. 从首个位置开始，计算位移长度，当位移超过一个超参数T(论文中为2)，则将这个范围内的图像划分为一个local nerf
2. 把下一个相机位置作为下一个nerf的中心，设置一个范围超参数r(论文中为1)，然后继续寻找构建这个nerf需要的pose，以此类推

实现：
一个函数，用于离线计算block_seq,考虑重叠
直接使用T矩阵构建前缀和？
'''
def compute_block_seq(root_dir, K=16):
    # 读取图片名以及对应的pose
    block_seq = []
    with open(root_dir + "/Pose.txt", 'r') as f:
        file = f.readlines()
    file_dict = {}
    for idx, i in enumerate(file):
        # 每次计算首次保存的位置与现在所在位置的位移差，为了简单建模，直接使用当前local的中间pose作为重叠区域的开始，并保持构建序列性
        img_id, *pose = i.split()
        T = read_poses(pose)[:3, -1]
        file_dict[img_id] = T
        # 保存初始值：
        if idx == 0:
            start = [int(img_id), T]

        # 计算当前与初始位置的距离
        distance = np.linalg.norm(T - start[1])
        if distance > K:
            # 保存当前的local，初始化下一个local的开始
            block_seq.append([[start[0], int(img_id)]])
            start = [(int(img_id) + start[0]) // 2, file_dict[img_id]]
    if int(file[-1].split()[0]) not in block_seq[-1]:
        block_seq.append([[(block_seq[-1][0][0] + block_seq[-1][0][1]) // 2, int(file[-1].split()[0])]])

    return block_seq


def split_space_block(root_dir, block_space_split=None):
    # block_space_split = [
    #     [[1, 55], [245, 305], [845, 908]],
    #     [[45, 255]],
    #     [[285, 405], [745, 855]],
    #     [[395, 455], [695, 755]],
    #     [[445, 505], [645, 705]],
    #     [[495, 670]]
    # ]

    # 4.13
    # block_space_split = [
    #     [[1, 100], [210, 290], [470, 600]],
    #     [[100, 210], [290, 470]]
    #     # [[601, 1110]],
    #     # [[1111, 2050]],
    #     # [[2051, 2210]],
    #     # [[2211, 2510]],
    #     # [[2511, 2970]],
    #     # [[2971, 3080]],
    #     # [[3081, 3280]],
    #     # [[3281, 3400]]
    # ]
    # 把效果不太好的block重新训练或者拆分训练。 block2拆成3份，block1训练step提升一倍，block4，block5训练step提升一倍
    # block_space_split = [
    #     # [[1111, 1420]],
    #     # [[1410, 1720]],
    #     # [[1710, 2050]] # block2
    #     # [[600, 860]],
    #     # [[850, 1110]] #block1
    #     [[2500, 2750]],
    #     [[2740, 2970]]
    # ]
    #


    h, w, fx, fy, cx, cy, k1, k2, p1, p2 = read_ins()  # 内参
    # 读取图片名以及对应的pose
    with open(root_dir + "/Pose.txt", 'r') as f:
        file = f.readlines()
    # 构建字典查询
    file_dict = {}
    for i in file:
        img_id, *pose = i.split()
        file_dict[img_id] = pose

    # 将file分块
    for idx in range(len(block_space_split)):
        data = {}
        frames = []
        os.makedirs(f'{root_dir}/block_localrf_{idx}/images', exist_ok=True)
        for index in range(len(block_space_split[idx])):
            start, end = block_space_split[idx][index]
            for i in file_dict.keys():
                if int(i) >= start and int(i) <= end:
                    img_id, *pose = str(i), file_dict[i]
                    pose = read_poses(pose)
                    if int(i) == start:
                        data = dict(
                            fl_x=fx,
                            fl_y=fy,
                            k1=k1,
                            k2=k2,
                            k3=0,
                            k4=0,
                            p1=p1,
                            p2=p2,
                            is_fisheye=False,
                            cx=cx,
                            cy=cy,
                            w=w,
                            h=h,
                            aabb_scale=16,
                        )
                    frame = {}
                    frame["file_path"] = f"./images/{img_id}.jpg"
                    os.system(f'cp {root_dir}/images/{img_id}.jpg {root_dir}/block_localrf_{idx}/images')
                    # Transform matrix to nerfstudio format. Please refer to the documentation for coordinate system conventions.
                    frame["transform_matrix"] = [i.tolist() for i in pose]
                    frames.append(frame)
            data["frames"] = frames
            with open(f"{root_dir}/block_localrf_{idx}/transforms.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)


# 因为由hloc等方法生成的json的file_path未按照从小到达排序，这里将其重新排序
def  rm_transforms(pose_trans):
    with open(pose_trans, 'r') as f:
        pose = json.load(f)
    new_res = []
    for i in sorted(pose['frames'], key=lambda x: int(x['file_path'].split('/')[-1].split('.')[0])):
        # if (int(i['file_path'].split('/')[-1].split('.')[0]) >= 32 and int(i['file_path'].split('/')[-1].split('.')[0]) <= 108) or \
        #         int(i['file_path'].split('/')[-1].split('.')[0]) >=134:
        #     invalid_imgpath = '/data/guozebin_data/bird_filter/' + i['file_path']
        #     os.system(f'rm {invalid_imgpath}')
        #     continue
        # else:
        new_res.append(i)
    pose['frames'] = new_res
    with open(pose_trans, "w", encoding="utf-8") as f:
        json.dump(pose, f, indent=4)


if __name__=="__main__":
    dir = '/data/guozebin/NeRF/liosam/'
    # save_one(dir)
    #block = compute_block_seq(dir)
    # block = [[1, 1953]]
    # split_space_block(dir, block)
    #split_space_block(dir)
    save_one(dir)
    # rm_transforms('/home/guozebin/work_code/f2-nerf/data/free_dataset/meeting_room_5f/transforms.json')