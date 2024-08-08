'''
由polycam得到的数据生成llf需要的pose_bounding
'''

import os
import json
import cv2

import numpy as np

from pathlib import Path
from typing import List
from enum import Enum


class CameraModel(Enum):
    """Enum for camera types."""

    OPENCV = "OPENCV"
    OPENCV_FISHEYE = "OPENCV_FISHEYE"


CAMERA_MODELS = {
    "perspective": CameraModel.OPENCV,
    "fisheye": CameraModel.OPENCV_FISHEYE,
    "equirectangular": CameraModel.OPENCV,
}


def load_from_json(filename: Path):
    """Load a dictionary from a JSON filename.
    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)

def write_to_json(filename: Path, content: dict):
    """Write data to a JSON file.
    Args:
        filename: The filename to write to.
        content: The dictionary data to write.
    """
    assert filename.suffix == ".json"
    with open(filename, "w", encoding="UnTF-8") as file:
        json.dump(content, file)

def polycam_to_llff(
    image_filenames: List[Path],
    depth_filenames: List[Path],
    cameras_dir: Path,
    base_dir=None,
    min_blur_score: float = 0.0,
    crop_border_pixels: int = 0,
) -> List[str]:
    """Convert Polycam data into a nerfstudio dataset.
    Args:
        image_filenames: List of paths to the original images.
        depth_filenames: List of paths to the original depth maps.
        cameras_dir: Path to the polycam cameras directory.
        output_dir: Path to the output directory.
        min_blur_score: Minimum blur score to use an image. Images below this value will be skipped.
        crop_border_pixels: Number of pixels to crop from each border of the image.
    Returns:
        Summary of the conversion.
    """
    data = {}
    data["camera_model"] = CAMERA_MODELS["perspective"].value
    # Needs to be a string for camera_utils.auto_orient_and_center_poses
    data["orientation_override"] = "none"

    llff = []
    depth_bounding = []
    skipped_frames = 0
    for i, filename in enumerate(zip(image_filenames, depth_filenames)):
        image_filename, depth_filename = filename
        json_filename = cameras_dir / f"{image_filename.stem}.json"
        frame_json = load_from_json(json_filename)
        if "blur_score" in frame_json and frame_json["blur_score"] < min_blur_score:
            skipped_frames += 1
            continue
        frame = {}
        frame["fl_x"] = frame_json["fx"]
        frame["fl_y"] = frame_json["fy"]
        frame["cx"] = frame_json["cx"] - crop_border_pixels
        frame["cy"] = frame_json["cy"] - crop_border_pixels
        frame["w"] = frame_json["width"] - crop_border_pixels * 2
        frame["h"] = frame_json["height"] - crop_border_pixels * 2
        frame["file_path"] = f"./images/frame_{i+1:05d}{image_filename.suffix}"
        frame["transform_matrix"] = [
            [frame_json["t_20"], frame_json["t_21"], frame_json["t_22"], frame_json["t_23"]],
            [frame_json["t_00"], frame_json["t_01"], frame_json["t_02"], frame_json["t_03"]],
            [frame_json["t_10"], frame_json["t_11"], frame_json["t_12"], frame_json["t_13"]],
        ]

        # format llff
        h, w, f = frame["h"], frame["w"], frame["fl_x"]
        hwf = np.array([h, w, f]).reshape([3, 1])
        c2w_mats = np.array(frame["transform_matrix"])
        pose = np.concatenate([c2w_mats, hwf], 1)
        llff.append(pose[:, :, None])

        # 计算深度边界
        depth = cv2.imread(str(depth_filename), 0)
        depth_near, depth_far = depth.min(), depth.max()
        depth_bounding.append(np.array([depth_near, depth_far]))

    # concat
    pose = np.concatenate(llff, -1)

    # trans nerfstudio[x, y, -z]-> llff [y, -x, -z]
    poses = np.concatenate([pose[:, 1:2, :], -pose[:, 0:1, :], pose[:, 2:3, :], pose[:, 3:4, :], pose[:, 4:5, :]], 1)

    # generate bounding 这里是自定义的方式,colmap中根据重建获得的点云进行计算
    save_arr = []
    for i in range(poses.shape[-1]):
        close_depth, inf_depth = 0.01, 1 # 自定义
        save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([close_depth, inf_depth])], 0))


    # 这里使用深度图作为指导,缩放到1
    # save_arr = []
    # for i, (close_depth, inf_depth) in enumerate(depth_bounding):
    #     close_depth, inf_depth = close_depth / inf_depth * 0.5, 0.5
    #     save_arr.append(np.concatenate([poses[..., i].ravel(), np.array([close_depth, inf_depth])], 0))
    save_arr = np.array(save_arr)


    np.save(os.path.join(base_dir, 'poses_bounds.npy'), save_arr)



if __name__ == '__main__':
    base_dir = '/home/guozebin/work_code/f2-nerf'
    cameras_dir = Path('/data/huangqinlong/xbrain-2f/xbrain-poly/keyframes/corrected_cameras')
    depth_filenames = sorted(list(Path('/data/huangqinlong/xbrain-2f/xbrain-poly/keyframes/depth').glob('*png')))
    image_filenames = sorted(list(Path('/data/huangqinlong/xbrain-2f/xbrain-poly/keyframes/images').glob('*jpg')))
    polycam_to_llff(image_filenames, depth_filenames, cameras_dir, base_dir)



