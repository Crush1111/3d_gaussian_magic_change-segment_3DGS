import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# DATA = "360_v2"
# DATASET_NAME ="bicyle_rank_loss"
# STEP = "30000"
#
# PATH = f'/home/guozebin/work_code/3d_gaussian_magic_change/output/{DATA}/{DATASET_NAME}/interpolation/ours_{STEP}'
# video_path = f'/home/guozebin/work_code/3d_gaussian_magic_change/output/{DATA}/{DATASET_NAME}/{DATASET_NAME}-step_{STEP}-test.mp4'
# if __name__ == '__main__':
#     count = 0
#     # for path_scence in os.listdir(PATH):
#     #
#     #     path = os.listdir(os.path.join(PATH, path_scence))
#     #     path.remove('cameras.json')
#     # for img, depth in zip(sorted(os.listdir(PATH), key=lambda x: int(x.split('_')[-1].split('.')[0])),
#     #                       sorted(os.listdir(PATH), key=lambda x: int(x.split('_')[-1].split('.')[0]))):
#     imgs_pred, depths = sorted(list(Path(PATH).joinpath('renders').glob('*'))), \
#                                  sorted(list(Path(PATH).joinpath('depth').glob('*')))
#     for img_pred,  dep in zip(imgs_pred, depths):
#         count += 1
#         im2 = cv2.imread(str(img_pred))
#         d = cv2.imread(str(dep))
#         im = np.concatenate((im2, d), 1)
#         if count == 1:
#             fps, w, h = 30, im.shape[1], im.shape[0]
#             out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
#         out.write(im)
#     print('Done!')


def save_vidio(model_path, NAME, STEP):
    PATH = f'{model_path}/{NAME}/ours_{STEP}'
    video_path = f'{model_path}/{NAME}-step_{STEP}-test.mp4'
    count = 0
    imgs_pred, depths = sorted(list(Path(PATH).joinpath('renders').glob('*'))), \
                                 sorted(list(Path(PATH).joinpath('depth').glob('*')))
    for img_pred,  dep in tqdm(zip(imgs_pred, depths), desc="composite video"):
        count += 1
        im2 = cv2.imread(str(img_pred))
        d = cv2.imread(str(dep))
        im = np.concatenate((im2, d), 1)
        if count == 1:
            fps, w, h = 30, im.shape[1], im.shape[0]
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        out.write(im)
    print('Done!')


def save_vidio_no_depth(model_path, NAME, STEP):
    PATH = f'{model_path}/{NAME}/ours_{STEP}'
    video_path = f'{model_path}/{NAME}-step_{STEP}-test.mp4'
    count = 0
    imgs_pred = sorted(list(Path(PATH).joinpath('renders').glob('*')))
    for img_pred in tqdm(imgs_pred, desc="composite video"):
        count += 1
        im = cv2.imread(str(img_pred))
        if count == 1:
            fps, w, h = 30, im.shape[1], im.shape[0]
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        out.write(im)
    print('Done!')

def save_vidio_folder(PATH):
    video_path = f'test.mp4'
    count = 0
    imgs_pred = sorted(list(Path(PATH).glob('*.jpg')))
    for img_pred in tqdm(imgs_pred, desc="composite video"):
        count += 1
        im = cv2.imread(str(img_pred))
        if count == 1:
            fps, w, h = 30, im.shape[1], im.shape[0]
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        out.write(im)
    print('Done!')


if __name__ == '__main__':
    # save_vidio_no_depth("/home/guozebin/work_code/3d_gaussian_magic_change/output/company_hall_100k", "interpolation", 1)
    PATH = "/home/guozebin/work_code/3d_gaussian_magic_change/data/MVS_Meeting_Room/resized_images"
    save_vidio_folder(PATH)