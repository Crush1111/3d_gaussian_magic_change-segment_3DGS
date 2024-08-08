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

import os
import random
import json

import numpy as np

from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from dataclasses import dataclass

class Scene:

    gaussians : GaussianModel

    @dataclass
    class Point_attribute:
        xyz: np.ndarray
        features_dc: np.ndarray
        features_extra: np.ndarray
        opacities: np.ndarray
        scales: np.ndarray
        rots: np.ndarray

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], sub_scene=[None], low_memory=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, using_depth=args.using_depth, using_seg=args.using_seg)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, using_depth=args.using_depth, using_seg=args.using_seg)
        # 尝试下其他数据，比如slam转nerfstudio
        elif os.path.exists(os.path.join(args.source_path, "transforms.json")):
            print("Found transforms_train.json file, assuming NeRFstudio data set!")
            scene_info = sceneLoadTypeCallbacks["NeRFstudio"](args.source_path, args.eval, using_depth=args.using_depth, using_seg=args.using_seg)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            # 在此处添加一个控制，如果low_memory=True,那么训练测试的字典中相机个数都只取一个
            if low_memory:
                print("Loading Cameras in low_memory mode")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras,
                                                                                resolution_scale, args, low_memory=low_memory)
                # self.test_cameras[resolution_scale] = cameraList_from_camInfos([scene_info.test_cameras[0]],
                #                                                                resolution_scale, args)
            else:
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            if sub_scene[0] is None:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                               "point_cloud",
                                                               "iteration_" + str(self.loaded_iter),
                                                               "point_cloud.ply"))
            else:
                sub_scene.insert(0, os.path.join(self.model_path,
                                                               "point_cloud",
                                                               "iteration_" + str(self.loaded_iter),
                                                               "point_cloud.ply"))  # 构造场景点云列表
                self.scene_list = []
                for pcd_path in sub_scene:
                    # print(sub_scene)
                    assert os.path.exists(pcd_path), "pcd path must be exist!"
                    xyz, features_dc, features_extra, opacities, scales, rots =\
                        self.gaussians.load_ply_no_instance(pcd_path)
                    point_attribute = \
                        self.Point_attribute(xyz, features_dc, features_extra, opacities, scales, rots)
                    self.scene_list.append(point_attribute)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def save_clip(self, sub_scene, mask):
        point_cloud_path = os.path.join("sub_scene_lib/{}".format(sub_scene))
        # 去检测文件夹下是否存在文件
        # 1 不存在
        if not os.path.exists(point_cloud_path):
            self.gaussians.save_ply_using_mask(os.path.join(point_cloud_path, "0.ply"), mask)
        else:
            name = len(os.listdir(point_cloud_path))
            self.gaussians.save_ply_using_mask(os.path.join(point_cloud_path, f"{name}.ply"), mask)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]