import cv2

from argparse import ArgumentParser

from utils.graphics_utils import fov2focal
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import Scene
from gaussian_renderer import render
from visual_res_app.camera_trajectory import *
import taichi as ti
import numpy as np
from dataclasses import dataclass
import math
import os
from datetime import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = '3'



## 画图颜色设置
# 选择字体、字体大小、颜色和文本内容
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 0.5
font_color = (255, 255, 0)

def onehot_encoding(logits, dim=0):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot.numpy()


def parse_sub_scene(value):
    if value.lower() == "none":
        return None
    else:
        return value

def parse_sub_scene_all(value):
    """
    输入文件夹路径
    """
    if value.lower() == "none":
        return None

    return [os.path.join(value, i) for i in os.listdir(value)]

def parse_sub_fast_gui(value):
    if value.lower() == "true":
        return True
    return False

@ti.kernel
def torchImage2tiImage(field: ti.template(), data: ti.types.ndarray()):
    for row, col in ti.ndrange(data.shape[0], data.shape[1]):
        field[col, data.shape[0] - row -
              1] = ti.math.vec3(data[row, col, 0], data[row, col, 1], data[row, col, 2])

class Render:

    @dataclass
    class ExtraSceneInfo:
        start_offset: int
        end_offset: int
        center: torch.Tensor
        visible: bool

    def __init__(self, dataset, iteration, pipeline, sub_scene=None, fast_gui=False, low_memory=False):
        print('segment nums: ', dataset.num_class)
        self.gaussians, self.scene, self.background = self.render_init(dataset, iteration, sub_scene, low_memory)
        #添加分割交互logic
        # 获取分割类别以及对应mask的key， value pair
        self.num_class = self.gaussians.get_segment.data.shape[1]
        self.mask_colors = [
            np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            for _ in range(self.num_class + 1)]

        cla = torch.argmax(self.gaussians.get_segment.data, 1) # 这里得到的是类别索引
        self.cla_mask = {
            idx: cla == idx
            for idx in range(self.num_class)
        }
        self.seg_cla = None # 分割模式默认值


        self.pipeline = pipeline
        self.all_view = self.scene.getTrainCameras()
        self.view = self.scene.getTrainCameras()[0]
        self.count = 0
        self.num_view = len(self.all_view)
        self.mode = "render"


        self.key_dict_cam = {
            # 平移
            'a': go_left,
            'd': go_right,
            'w': go_forward,
            's': go_backward,
            'q': go_up,
            'e': go_down,
            # 旋转
            'i': turn_up,
            'k': turn_down,
            'j': turn_l,
            'l': turn_r,
            'u': turn_z_axis_l,
            'o': turn_z_axis_r,
        }
        self.key_dict_pcd = {
            # 平移
            'a': go_left_pcd,
            'd': go_right_pcd,
            'w': go_forward_pcd,
            's': go_backward_pcd,
            'q': go_up_pcd,
            'e': go_down_pcd,
        }
        # 保存设置
        self.scene_name = dataset.model_path.split('/')[-1]
        self.render_path = dataset.model_path
        # 可视化设置
        self.scale = [[1.1, 0.91], [2.0, 0.5]]
        self.radius_scale = 1
        self.selected_scene = 0
        self.mouse_sensitivity = 3
        self.last_mouse_pos = None
        self.img_wh = (self.view.image_width, self.view.image_height)
        self.gui = ti.GUI(
            "Gaussian Point Visualizer",
            self.img_wh,
            fast_gui=fast_gui,
            show_gui=True)
        self.position = [self.view.image_width / 2, self.view.image_height / 2]
        self.image_buffer = ti.Vector.field(3, dtype=ti.f32, shape=(
            self.view.image_width, self.view.image_height))
        # clip设置
        # 初始化的bbox范围应该是覆盖整个场景的
        self.x_min = self.gui.slider('X axis min', 0.0, 1.0, 0.01)
        self.y_min = self.gui.slider('Y axis min', 0.0, 1.0, 0.01)
        self.z_min = self.gui.slider('Z axis min', 0.0, 1.0, 0.01)
        self.x_max = self.gui.slider('X axis max', 0.0, 1.0, 0.01)
        self.y_max = self.gui.slider('Y axis max', 0.0, 1.0, 0.01)
        self.z_max = self.gui.slider('Z axis max', 0.0, 1.0, 0.01)
        self.x_r = self.gui.slider('x rotation angle', 0.0, 360.0, 0.1)
        self.y_r = self.gui.slider('y rotation angle', 0.0, 360.0, 0.1)
        self.z_r = self.gui.slider('z rotation angle', 0.0, 360.0, 0.1)
        self.mask = None
        self.clip = False
        self.rotate = True
        self.projected_2d = None
        # 子场景位置投影
        self.projected_sub_scene = False
        # 初始化bbox范围
        self.pcd_num = self.gaussians.get_xyz.data.shape[0]
        self.comp_l = self.gaussians.get_xyz.data.min(0).values
        self.comp_m = self.gaussians.get_xyz.data.max(0).values
        self.dis = (self.comp_m - self.comp_l).unsqueeze(0)
        self.corners = None
        # 添加关键帧
        self.keyframes_num = self.gui.label('keyframes')
        self.keyframes_num.value = 0
        self.keyframes = []
        # 添加相机范围限制
        # 移动范围-自行选择范围
        self.outbounding_sign = False
        self.cam_limit = False
        self.cam_pan_bbox = None
        self.cam_pan_bbox_base = None
        self.plot_cam_mode = {
            'cam': self.all_view,
            'keyframes': self.keyframes,
        }

    def render_init(self, dataset, iteration, sub_scene=None, low_memory=False):
        with torch.no_grad():
            gaussians = GaussianModel(dataset.sh_degree, dataset.num_class)
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, sub_scene=sub_scene, low_memory=low_memory)

            self.extra_scene_info_dict = {
                0: self.ExtraSceneInfo(
                    start_offset=0,
                    end_offset=int(gaussians.get_xyz.data.shape[0]),
                    center=gaussians.get_xyz.data.mean(0).cpu().numpy(),
                    visible=True
                )}
            # 添加场景合并逻辑
            if sub_scene[0] is not None:
                self._merge_scenes(scene, gaussians)

            bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        return gaussians, scene, background

    def _merge_scenes(self, scene, gaussians):
        # 属性合并
        xyz = np.concatenate([sub_scene.xyz for sub_scene in scene.scene_list])
        features_extra = np.concatenate([sub_scene.features_extra for sub_scene in scene.scene_list])
        features_dc = np.concatenate([sub_scene.features_dc for sub_scene in scene.scene_list])
        opacities = np.concatenate([sub_scene.opacities for sub_scene in scene.scene_list])
        # segments
        segments = np.concatenate([sub_scene.segment for sub_scene in scene.scene_list])
        rots = np.concatenate([sub_scene.rots for sub_scene in scene.scene_list])
        scales = np.concatenate([sub_scene.scales for sub_scene in scene.scene_list])
        # 场景区分
        num_of_points_list = [sub_scene.xyz.shape[0]
                              for sub_scene in scene.scene_list] # 获取每个场景的点数

        start_offset_list = [0] + np.cumsum(num_of_points_list).tolist()[:-1] # 计算合并的场景的初始点位置
        end_offset_list = np.cumsum(num_of_points_list).tolist() # 计算合并的场景的结束点位置
        # print(start_offset_list, end_offset_list)
        self.extra_scene_info_dict = {
            idx: self.ExtraSceneInfo(
                start_offset=start_offset,
                end_offset=end_offset,
                center=scene.scene_list[idx].xyz.mean(0),
                visible=True
            ) for idx, (start_offset, end_offset) in enumerate(zip(start_offset_list, end_offset_list))
        }
        point_object_id = torch.zeros(
            (xyz.shape[0],), dtype=torch.int32, device="cuda")
        for idx, (start_offset, end_offset) in enumerate(zip(start_offset_list, end_offset_list)):
            point_object_id[start_offset:end_offset] = idx

        gaussians.instance_parm(xyz, features_dc, features_extra, opacities, scales, rots, segments)

    def _copy_scenes(self, mask):
        # 场景区分
        self.extra_scene_info_dict.update({
            len(self.extra_scene_info_dict): self.ExtraSceneInfo(
                start_offset=int(self.pcd_num),
                end_offset=int(self.pcd_num + torch.sum(mask==1)),
                center=self.gaussians.get_xyz.data[mask].mean(0).cpu().numpy(),
                visible=True
            )})
        # print(self.extra_scene_info_dict)
        # 属性合并
        self.gaussians._xyz.data = torch.cat([self.gaussians._xyz.data, self.gaussians._xyz.data[mask]])
        self.gaussians._features_dc.data = torch.cat([self.gaussians._features_dc.data, self.gaussians._features_dc.data[mask]])
        self.gaussians._features_rest.data = torch.cat([self.gaussians._features_rest.data, self.gaussians._features_rest.data[mask]])
        self.gaussians._scaling.data = torch.cat([self.gaussians._scaling.data, self.gaussians._scaling.data[mask]])
        self.gaussians._rotation.data = torch.cat([self.gaussians._rotation.data, self.gaussians._rotation.data[mask]])
        self.gaussians._opacity.data = torch.cat([self.gaussians._opacity.data, self.gaussians._opacity.data[mask]])
        # segment feature
        self.gaussians._segment.data = torch.cat([self.gaussians._segment.data, self.gaussians._segment.data[mask]])
        self.num_class = self.gaussians.get_segment.data.shape[1]
        cla = torch.argmax(self.gaussians.get_segment.data, 1) # 这里得到的是类别索引
        self.cla_mask = {
            idx: cla == idx
            for idx in range(self.num_class)
        }

        # mask 扩张
        self.mask = torch.cat([torch.zeros_like(self.mask, dtype=torch.bool).to(self.mask.device),
                               torch.ones(self.mask.sum(0), dtype=torch.bool).to(self.mask.device)])
        self.pcd_num = self.gaussians.get_xyz.data.shape[0]

    @staticmethod
    def scene_select(start_end_array, mask):
        # 只改变场景0
        length_move_scene = mask[:start_end_array[1]].sum(0)
        print("length_move_scene: ", length_move_scene)
        for idx in range(1, len(start_end_array)):
            start_end_array[idx] -= int(length_move_scene)

        return start_end_array, length_move_scene

    def _remove_scenes(self, mask):
        """
        新逻辑：
        rmove scenes时只能在没有子场景时进行
        原因：当根据crop区域获取mask来删除该区域时，mask是无序的，因此无法更新子场景的起始索引
        但是只要在主场景进行操作，更新就不存在问题

        之前的逻辑(存在一些问题)
        # 更新所有现存场景的索引以及中心位置
        # 条件：
        # 被子场景完全包含：该子场景起点不变，终点减去移除场景的长度，后续子场景如是
        # 处于两个子场景之间：前一个子场景的终点变为移除场景的起点，终点减去移除场景的长度，后续子场景如是
        # 跨越子场景： 跟上述逻辑相同，但是需要考虑场景删除，跨越的场景将被删除，对应后续场景的索引递减
        """

        # 1. 获取场景起始数组
        start_end_array = []
        for key in self.extra_scene_info_dict.keys():
            if key == 0:
                start_end_array.extend([self.extra_scene_info_dict[key].start_offset, self.extra_scene_info_dict[key].end_offset])
            else:
                start_end_array.append(self.extra_scene_info_dict[key].end_offset)
        # 只去计算主场景
        mask[start_end_array[1]:] = 0
        # 2. 计算子场景起始数组并根据mask完成更新
        start_end_array, length_move_scene = self.scene_select(start_end_array, mask)
        # 属性减少
        self.gaussians._xyz.data = self.gaussians._xyz.data[~mask]
        self.gaussians._features_dc.data = self.gaussians._features_dc.data[~mask]
        self.gaussians._features_rest.data = self.gaussians._features_rest.data[~mask]
        self.gaussians._scaling.data = self.gaussians._scaling.data[~mask]
        self.gaussians._rotation.data = self.gaussians._rotation.data[~mask]
        self.gaussians._opacity.data = self.gaussians._opacity.data[~mask]
        self.gaussians._segment.data = self.gaussians._segment.data[~mask]

        self.num_class = self.gaussians.get_segment.data.shape[1]
        cla = torch.argmax(self.gaussians.get_segment.data, 1) # 这里得到的是类别索引
        self.cla_mask = {
            idx: cla == idx
            for idx in range(self.num_class)
        }

        # 场景重构
        start_offset_list, end_offset_list = start_end_array[:-1],  start_end_array[1:]
        self.extra_scene_info_dict = {
            idx: self.ExtraSceneInfo(
                start_offset=start_offset,
                end_offset=end_offset,
                center=self.gaussians._xyz.data[start_offset:end_offset].mean(0),
                visible=True
            ) for idx, (start_offset, end_offset) in enumerate(zip(start_offset_list, end_offset_list))
        }
        print(self.extra_scene_info_dict)
        # mask 减少
        self.mask = torch.zeros(mask.shape[0] - length_move_scene, dtype=torch.bool).to(self.mask.device)

        self.pcd_num = self.gaussians.get_xyz.data.shape[0]

    def start(self):
        while self.gui.running:
            events = self.gui.get_events(self.gui.PRESS)
            # 添加场景合并逻辑
            start_offset = 0
            end_offset = self.gaussians._xyz.shape[0]
            object_selected = self.selected_scene != 0
            if object_selected:
                start_offset = self.extra_scene_info_dict[self.selected_scene].start_offset
                end_offset = self.extra_scene_info_dict[self.selected_scene].end_offset
                # if not self.clip:
                #     self.mouse_control_pcd(start_offset, end_offset)

            for event in events:

                if event.key >= "0" and event.key <= "9":
                    scene_index = int(event.key)
                    try:
                        print("场景数量：", len(self.extra_scene_info_dict))
                        print("选择场景：", scene_index)
                    except:
                        break
                    if scene_index <= len(self.extra_scene_info_dict) -1:
                        self.selected_scene = scene_index
                    # 当按下编辑按钮，立马将视角转到距离该点云最近的视角下,只看一眼，多看会爆炸
                    # self.view = self.view_trans(start_offset, end_offset)
                    # 直接画出来

                elif event.key in list(self.key_dict_cam.keys()):
                    if not self.clip and object_selected and (event.key in list(self.key_dict_pcd.keys())):
                        self.key_dict_pcd[event.key](self.gaussians.get_xyz.data[start_offset:end_offset])
                        #  更新中心点
                        self.extra_scene_info_dict[self.selected_scene].center = self.gaussians.get_xyz.data[start_offset:end_offset].mean(0).cpu().numpy()
                    else:
                        # BUG: 行为与预想完全不一致
                        # 需要在这里判断self.view是否在ROT区域
                        # camera_center表示c2w矩阵的T，self.cam_pan_bbox 2x3, 表示bbox在自身坐标系下的self.cam_pan_bbox表示将bbox转换到以自身的一个角点为原点的一组正交基， 将cam_center转换到bbox坐标系下后就可以通过
                        # cam_center_base > self.cam_pan_bbox[0], cam_center_base < self.cam_pan_bbox[1]来判断是否被bbox包含了
                        if self.cam_pan_bbox is not None and event.key in ['w', 'a', 's', 'd', 'q', 'e']:
                            next_view = self.key_dict_cam[event.key](self.view)
                            cam_center = next_view.camera_center
                            cam_center_base = cam_center @ self.cam_pan_bbox_base
                            if torch.all(torch.cat([cam_center_base > self.cam_pan_bbox[0], cam_center_base < self.cam_pan_bbox[1]], -1), -1).item():
                                # print('in bbox')
                                self.outbounding_sign = False
                                self.view = next_view
                            else:
                                self.outbounding_sign = True
                        else:
                            self.view = self.key_dict_cam[event.key](self.view)

                elif  event.key == 'm':
                    self.trans_mode()

                elif  event.key == 'r':
                    self.render_snap()

                elif event.key == '-' :
                    if object_selected:
                        self.gaussians.get_xyz.data[start_offset:end_offset] = \
                            self.center_invariant_scaling(self.gaussians.get_xyz.data[start_offset:end_offset], self.scale[0][1])
                        self.gaussians._scaling.data[start_offset:end_offset] = \
                            (math.log(self.scale[0][1]) + self.gaussians._scaling.data[start_offset:end_offset])

                elif event.key == '=':
                    if object_selected:
                        self.gaussians.get_xyz.data[start_offset:end_offset] = \
                            self.center_invariant_scaling(self.gaussians.get_xyz.data[start_offset:end_offset], self.scale[0][0])
                        self.gaussians._scaling.data[start_offset:end_offset] = \
                            (math.log(self.scale[0][0]) + self.gaussians._scaling.data[start_offset:end_offset])

                # 添加对协方差尺度的控制
                elif event.key == 'z':
                    self.trans_scale()

                elif event.key == 'x':
                    self.trans_segment()

                elif event.key == 'c':
                    self.clip_mode()

                elif event.key == 'p':
                    self.projected_mode()

                # 添加点云导出功能
                elif event.key == 'v':
                    print("save clip scene")
                    self.scene.save_clip(f"{self.scene_name}", self.mask)
                    print("save success")

                elif event.key == 'g':
                    self.sub_scene_projected_mode()

                elif event.key == 'b':
                    # 添加子场景clone
                    # 注：必须保证使用了clip获取了子场景，然后才可以复制
                    if self.mask is not None :
                        print("start clone sub scene")
                        self._copy_scenes(self.mask)
                        print("clone sub scene finish")

                elif event.key == 'n':
                    # 添加子场景clone
                    # 注：必须保证使用了clip获取了子场景，然后才可以复制
                    if self.mask is not None :
                        print("start remove sub scene")
                        self._remove_scenes(self.mask)
                        print("remove sub scene finish")

                elif event.key == ',':
                    self.keyframes_num.value += 1
                    key_frames = MiniCam(self.view.image_width, self.view.image_height, self.view.FoVy, self.view.FoVx, self.view.znear,
                                self.view.zfar, self.view.world_view_transform.clone(), self.view.full_proj_transform.clone())
                    self.keyframes.append(key_frames)

                elif event.key == '.':
                    if len(self.keyframes) >= 1:
                        self.keyframes.pop()
                        self.keyframes_num.value -= 1

                elif event.key == self.gui.SPACE:
                    try:
                        views = inter_poses(self.keyframes, 300, save_path=self.render_path)
                        # 保存一下文件，然后离线渲染视频
                        for view in views:
                            image = self.render_single(view, self.gaussians, self.pipeline, self.background, bbox_mask=self.mask)

                            # segment visualizer
                            # 这里我实现了常规的分割可视化流程，也即将逐层染色
                            if self.mode == 'segment':
                                image = self.visual_segment(image)

                            torchImage2tiImage(self.image_buffer, image)
                            self.gui.set_image(self.image_buffer)
                            self.gui.show()

                    except:
                        print("keyframes number must > 2")

                if event.key == 'f':
                    self.cam_limit_mode()

                # 添加直接保存的功能
                elif event.key == 'y':
                    try:
                        views = inter_poses(self.keyframes, 300, save_path=self.render_path)
                        current_datetime = datetime.now()
                        video_path = f'{self.render_path}/freedom_view_{current_datetime.strftime("%Y-%m-%d-%H-%M-%S")}.mp4'
                        count = 0
                        # 保存一下文件，然后离线渲染视频
                        for view in tqdm(views, desc="render video"):
                            count += 1
                            im = self.render_single(view, self.gaussians, self.pipeline, self.background, bbox_mask=self.mask)
                            # segment visualizer
                            # 这里我实现了常规的分割可视化流程，也即将逐层染色
                            if self.mode == 'segment':
                                image = self.visual_segment(image)

                            img = (im[:, :, ::-1] * 255).clip(0, 255).astype(np.uint8)
                            if count == 1:
                                fps, w, h = 30, img.shape[1], img.shape[0]
                                out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                            out.write(img)
                        print('Done!')
                    except:
                        print("keyframes number must > 2")

                # 添加鼠标滑轮控制FOV的功能
                # elif event.key == self.gui.WHEEL:  # 检测鼠标滚轮事件
                #     # e.delta 表示滚轮的滚动值，正数表示向前滚动，负数表示向后滚动
                #     print('run_here')
                #     print(event.delta)


            if self.clip:
                comp_l, comp_m, base = self.bbox_clip(start_offset, end_offset) # 包含clip + bbox旋转
                if self.cam_limit:
                    # 计算limit
                    self.cam_pan_bbox = torch.vstack([comp_l.clone(), comp_m.clone()])
                    self.cam_pan_bbox_base = base.clone()
                    # print(self.cam_pan_bbox, self.cam_pan_bbox_base)
                    # trans cam center
                    self.view = trans_cam_center_to_bbox_center(self.view, self.cam_pan_bbox, self.cam_pan_bbox_base)
                    print("相机约束完成")
                    self.cam_limit = False
            else:
                # 关闭该模式则不使用mask
                self.mask = None
                self.mouse_control()

            if self.seg_cla is not None:
                self.mask = self.cla_mask[self.seg_cla]
            else:
                self.mask = None

            image = self.render_single(self.view, self.gaussians, self.pipeline, self.background, bbox_mask=self.mask)

            if self.mode == 'render':
                # 越界警告！
                if self.outbounding_sign:
                    self.warning_projected(image)

                if self.clip and self.projected_2d in ['bbox', 'xyz']:
                    self.bbox_projected(image)

                if self.projected_2d in ['cam', 'keyframes']:
                    self.cam_projected(image)

                if self.projected_sub_scene:
                    self.sub_scene_projected(image)

            # 这里我实现了常规的分割可视化流程，也即将逐层染色
            if self.mode == 'segment':
                image = self.visual_segment(image)

            torchImage2tiImage(self.image_buffer, image)
            self.gui.set_image(self.image_buffer)
            self.gui.show()
        self.gui.close()

    def visual_segment(self, image):

        # 需要先进行onehot编码
        image = onehot_encoding(torch.from_numpy(image.transpose(2, 0, 1)))
        segment_img = np.zeros((self.view.image_height, self.view.image_width, 3), dtype=np.uint8)
        for idx, mask_ in enumerate(image):
            color_mask = self.mask_colors[idx]
            mask_ = mask_.astype(bool)
            segment_img[mask_] = color_mask

        return segment_img / 255

    def bbox_projected(self, image, line_color = (0, 255, 0)):
        # 投影画图
        begin, end = self.projected()
        # 画线
        # 遍历每一条线段，绘制到图像上
        for start, end in zip(begin, end):

            cv2.line(image, tuple(start), tuple(end), color=line_color, thickness=1)

    def cam_projected(self, image, line_color = (255, 255, 0)):
        # 投影画图
        begin, end = self.projected()        # 画线
        # 遍历每一条线段，绘制到图像上
        for start, end in zip(begin, end):

            cv2.line(image, tuple(start), tuple(end), color=line_color, thickness=1)

    def sub_scene_projected(self, image, font = cv2.FONT_HERSHEY_SIMPLEX, font_size = 0.5, font_color = (255, 255, 0)):
        try:
            sub_scene_center = torch.stack([torch.tensor(self.extra_scene_info_dict[i].center) for i in range(len(self.extra_scene_info_dict))])
        except:
            sub_scene_center = None

        if sub_scene_center is None:
            return
        # 投影画图
        K = self.view.K
        # w2c
        P = self.view.world_view_transform.T
        # 旋转之后不在满足
        # 将角点坐标从齐次坐标转换为非齐次坐标
        # 3d点需要满足的条件是，对于连线的两个3d点，有两个坐标是相同的
        homogeneous_corners = torch.cat((sub_scene_center.to(self.comp_l.device), torch.ones(sub_scene_center.shape[0], 1).to(self.comp_l.device)), dim=1)

        # 计算投影坐标
        point_cloud_camera = torch.matmul(P, homogeneous_corners.t()).t()
        # Project points onto the image plane
        points_projected = torch.matmul(K, point_cloud_camera[:, :3].t()).t()
        projected_sub_scene = (points_projected[:, :2] / points_projected[:, 2:3]).cpu().numpy().astype(np.int32)

        for text, sub in enumerate(projected_sub_scene):
            if text != 0:
                x, y = sub
                cv2.putText(image, str(text), (x, y), font, font_size, font_color, thickness=2)

    def warning_projected(self, image, font = cv2.FONT_HERSHEY_SIMPLEX, font_size = 0.5, font_color = (255, 0, 0)):
        x, y = 0, 20
        cv2.putText(image, "Warning, reach the scene boundary!", (x, y), font, font_size, font_color, thickness=1)

    @staticmethod
    def world_to_image_projected(homogeneous_pcd, P, K):
        """
        return 
        """
        # 计算投影坐标
        point_cloud_camera = torch.matmul(P, homogeneous_pcd.t()).t()
        # print(point_cloud_camera.shape)
        # print(point_cloud_camera)
        # 使用深度进行过滤
        mask = (point_cloud_camera[0, -2] >= 0)
        # print(mask)
        # Project points onto the image plane
        points_projected = torch.matmul(K, point_cloud_camera[:, :3].t()).t()
        projected_image = (points_projected[:, :2] / points_projected[:, 2:3]).cpu().numpy().astype(np.int32)

        return projected_image, mask

    def projected(self):
        # focal_length_x = fov2focal(self.view.FoVx, self.view.image_width)
        # focal_length_y = fov2focal(self.view.FoVy, self.view.image_height)
        # # c2img
        # K = torch.tensor([
        #     [focal_length_x, 0, self.view.image_width / 2],
        #     [0, focal_length_y, self.view.image_height / 2],
        #     [0, 0, 1]
        # ]).to(self.comp_l.device)
        K = self.view.K
        # w2c
        P = self.view.world_view_transform.T
        # 旋转之后不在满足
        # 将角点坐标从齐次坐标转换为非齐次坐标
        # 3d点需要满足的条件是，对于连线的两个3d点，有两个坐标是相同的
        if self.projected_2d == 'bbox':
            homogeneous_corners = torch.cat((self.corners, torch.ones(8, 1).to(self.comp_l.device)), dim=1)
            projected_coordinate, _ = self.world_to_image_projected(homogeneous_corners, P, K)
            projected_coordinate = projected_coordinate[None]
            # 线的起点和终点索引对应关系
            line_indices = [
                (0, 1), (0, 2), (0, 4),
                (1, 3), (1, 5), (2, 3),
                (2, 6), (3, 7), (4, 5),
                (4, 6), (5, 7), (6, 7)
            ]

        elif self.projected_2d == 'xyz':
            xyz = torch.tensor(
                [[0, 0, 0],
                 [1000, 0, 0],
                 [0, 1000, 0],
                 [0, 0, 1000]], dtype=torch.float
            ).to(self.comp_l.device)
            xyz = center_r_bbox(xyz, self.x_r.value, self.y_r.value, self.z_r.value)
            homogeneous_corners = torch.cat((xyz, torch.ones(4, 1).to(self.comp_l.device)), dim=1)
            projected_coordinate, _ = self.world_to_image_projected(homogeneous_corners, P, K)
            ######
            # 线的起点和终点索引对应关系
            line_indices = [(0, 1), (0, 2), (0, 3)]
            # 将原点移动到图像中心
            img_w, img_h = self.img_wh
            # 计算偏移量
            shift = np.array([img_w / 2, img_h / 2], dtype=np.int32) - projected_coordinate[0]
            projected_coordinate += shift
            projected_coordinate = projected_coordinate[None]

        else:
            # 与其他两种投影模式不同的是这里是多个cam
            # 如何处理没有照相机的特殊情况
            cam_list = []
            mask_list = []
            for cam in self.plot_cam_mode[self.projected_2d]:
                cam_vis_3d = cam_vis(cam)
                homogeneous_cam = torch.cat(
                    (cam_vis_3d, torch.ones(cam_vis_3d.shape[0], 1).to(cam_vis_3d.device)), dim=1)
                # # 使用自带的投影矩阵
                # projected_cam = cam.projection_matrix @  homogeneous_cam
                # projected_cam /= projected_cam[3, :3]
            
                projected_cam, mask = self.world_to_image_projected(homogeneous_cam, P, K)
                cam_list.append(projected_cam[None])
                mask_list.append(mask.cpu().numpy())
            if len(cam_list):
                projected_coordinate = np.concatenate(cam_list)
                # # 过滤下 ：应该在3d空间过滤，判断相机是否在当前视角的视锥内部
                mask = np.array(mask_list)
                # print(mask)
                projected_coordinate = projected_coordinate[mask]
            else:
                projected_coordinate = np.array([])
            # print(projected_coordinate)
            line_indices = [(0, 1), (0, 2), (0, 3), (0, 4),
                            (1, 2), (2, 4), (4, 3), (3, 1)]

        if  projected_coordinate.shape[0]:
            begins = []
            ends = []
            for begin_idx, end_idx in line_indices:
                for projected_object in projected_coordinate:
                    begin_point = projected_object[begin_idx]
                    end_point = projected_object[end_idx]
                    begins.append(begin_point)
                    ends.append(end_point)
                begin = np.vstack(begins)
                end = np.vstack(ends)
        else:
            begin = []
            end = []

        return begin, end

    def bbox_clip(self, start_offset, end_offset):
        """
        添加bbox裁剪:
        1.计算场景点云的范围，然后计算可以将点云放到bbox中的[x_max, y_max, z_max]
        2. 在得到[x_max, y_max, z_max]之后，就可以将不在该范围内的点云通过mask的方式过滤，逻辑是:计算每个点是否在bbox内，不在直接删除
        3. 关键功能，bbox要随着场景旋转来进行旋转，在裁剪时肯定希望地面为xy平往不满足这样的条件，所以最好能把bbox画出来，然后还可以旋转，放大以及缩小，这样就可以很好的框住需要的区域
        """
        # 定局部变量
        comp_l = self.comp_l.clone()
        comp_m = self.comp_m.clone()
        # print('org空间', comp_l, comp_m)
        # 上面添加了缩放，下面添加旋转,以滑块的方式控制
        self.corners = torch.tensor(
            [[comp_l[0], comp_l[1], comp_l[2]],
            [comp_l[0], comp_l[1], comp_m[2]],
            [comp_l[0], comp_m[1], comp_l[2]],
            [comp_l[0], comp_m[1], comp_m[2]],
            [comp_m[0], comp_l[1], comp_l[2]],
            [comp_m[0], comp_l[1], comp_m[2]],
            [comp_m[0], comp_m[1], comp_l[2]],
            [comp_m[0], comp_m[1], comp_m[2]]]
        ).to(self.comp_l.device)
        # print(self.corners)
        # 旋转坐标系
        self.corners = center_r_bbox(self.corners, self.x_r.value, self.y_r.value, self.z_r.value)
        # 使用以bbox角点为原点，正交的三条边为基，来表示bbox
        dis = self.dis.clone()
        base = (torch.vstack([self.corners[1:3, :], self.corners[4]]) - self.corners[0]).flip(0)# 【z, y, x】 - [x, y, z]
        scale = torch.norm(base, p=2, dim=1)

        #变换到欧式空间的尺度
        base[0] /= scale[0]
        base[1] /= scale[1]
        base[2] /= scale[2]

        self.corners = (base @ self.corners.T).T
        # 根据滑动条来修改comp_l以及comp_m
        self.corners[:4, 0] += self.x_min.value * dis[:, 0]
        self.corners[:2, 1] += self.y_min.value * dis[:, 1]
        self.corners[4:6, 1] += self.y_min.value * dis[:, 1]
        self.corners[:, 2][::2] += self.z_min.value * dis[:, 2]

        self.corners[4:, 0] -= self.x_max.value * dis[:, 0]
        self.corners[2:4, 1] -= self.y_max.value * dis[:, 1]
        self.corners[6:, 1] -= self.y_max.value * dis[:, 1]
        self.corners[:, 2][1::2] -= self.z_max.value * dis[:, 2]

        # 0908
        # 思考：转到原始空间的目的只是为了画图，应该在变换空间使用判断得到mask
        # 3. 得到ROI
        pcd_selected = (base @ self.gaussians.get_xyz.data[start_offset:end_offset].T).T
        comp_l = self.corners.min(0).values
        comp_m = self.corners.max(0).values
        # print('新空间', comp_l, comp_m)
        mask_sub = torch.all(torch.cat([pcd_selected > comp_l, pcd_selected < comp_m], -1), -1)

        # 转回原始空间
        self.corners = self.corners @ base

        # # 3. 得到ROI(这里的写法不正确，因为旋转的影响，最大值与最小值组成了新的bbox)
        # comp_l = self.corners.min(0).values
        # comp_m = self.corners.max(0).values
        # # print(comp_l, comp_m)
        # pcd_selected = self.gaussians.get_xyz.data[start_offset:end_offset]
        # mask_sub = torch.all(torch.cat([pcd_selected > comp_l, pcd_selected < comp_m], -1), -1)

        # 4. 构造mask
        # 还需要保证mask的长度永远等于合并场景中点云的个数，因此这里需要进行构造
        # 首先计算场景中点云的总长度
        # 然后构造等长的mask数组
        self.mask = torch.zeros(self.pcd_num, dtype=torch.bool).to(mask_sub.device)
        # 填充
        self.mask[start_offset:end_offset] = mask_sub

        return  comp_l, comp_m, base

    def center_invariant_scaling(self, pcd, scale_factor):
        center = pcd.mean(0)
        # print(center)
        # 将点云中的每个点减去中心坐标
        centered_point_cloud = pcd - center
        # 对点云进行缩放操作
        scaled_point_cloud = centered_point_cloud * scale_factor
        pcd = scaled_point_cloud + center
        return pcd

    def mouse_control_pcd(self, start_offset, end_offset):
        mouse_pos = self.gui.get_cursor_pos()
        if self.gui.is_pressed(self.gui.LMB):
            if self.last_mouse_pos is None:
                self.last_mouse_pos = mouse_pos
            else:
                dy, dx = mouse_pos[0] - self.last_mouse_pos[0], mouse_pos[1] - self.last_mouse_pos[1]
                angle_x = torch.tensor(dx * self.mouse_sensitivity)
                angle_y = torch.tensor(dy * self.mouse_sensitivity)
                # print(start_offset, end_offset)
                self.gaussians.get_xyz.data[start_offset:end_offset] = \
                    mouse_con_pcd(self.gaussians.get_xyz.data[start_offset:end_offset], angle_x, angle_y)
                # 点云旋转，协方差要旋转吗？
                # self.gaussians.get_rotation.data[start_offset:end_offset] = \
                #     mouse_con_quaternion(self.gaussians.get_rotation.data[start_offset:end_offset], angle_x, angle_y)
                self.last_mouse_pos = mouse_pos
        else:
            self.last_mouse_pos = None

    def mouse_control_bbox(self, comp_l, comp_m):
        mouse_pos = self.gui.get_cursor_pos()
        if self.gui.is_pressed(self.gui.LMB):
            if self.last_mouse_pos is None:
                self.last_mouse_pos = mouse_pos
            else:
                dy, dx = mouse_pos[0] - self.last_mouse_pos[0], mouse_pos[1] - self.last_mouse_pos[1]
                angle_x = torch.tensor(dx * self.mouse_sensitivity / 2)
                angle_y = torch.tensor(dy * self.mouse_sensitivity / 2) # 降低鼠标敏感度
                # 中心不变旋转
                pcd = torch.vstack([comp_l, comp_m])
                pcd = mouse_con_pcd(pcd, angle_x, angle_y)
                comp_l, comp_m = pcd[0], pcd[1]
                self.last_mouse_pos = mouse_pos
        else:
            self.last_mouse_pos = None

        return comp_l, comp_m

    def mouse_control(self):
        mouse_pos = self.gui.get_cursor_pos()
        if self.gui.is_pressed(self.gui.LMB):
            if self.last_mouse_pos is None:
                self.last_mouse_pos = mouse_pos
            else:
                dy, dx = mouse_pos[0] - self.last_mouse_pos[0], mouse_pos[1] - self.last_mouse_pos[1]
                angle_x = torch.tensor(dx * self.mouse_sensitivity)
                angle_y = torch.tensor(dy * self.mouse_sensitivity)
                self.view = mouse_con(self.view, angle_x, angle_y)
                self.last_mouse_pos = mouse_pos
        else:
            self.last_mouse_pos = None

    def trans_mode(self):
        mode = ["render", "depth", "segment"]
        idx = mode.index(self.mode)
        self.mode = mode[(idx + 1) % 3]

    def cam_limit_mode(self):
        mode = [True, False]
        idx = mode.index(self.cam_limit)
        self.cam_limit = mode[(idx + 1) % 2]

    def clip_mode(self):
        mode = [True, False]
        idx = mode.index(self.clip)
        self.clip = mode[(idx + 1) % 2]

    def trans_segment(self):
        mode = [i for i in range(self.num_class)] + [None]
        idx = mode.index(self.seg_cla)
        self.seg_cla = mode[(idx + 1) % len(mode)]

    def rotate_mode(self):
        mode = [True, False]
        idx = mode.index(self.rotate)
        self.rotate = mode[(idx + 1) % 2]

    def projected_mode(self):
        mode = ['xyz', 'bbox', 'cam', 'keyframes', None]
        idx = mode.index(self.projected_2d)
        print("投影模式：", mode[(idx + 1) % 5])
        self.projected_2d = mode[(idx + 1) % 5]

    def sub_scene_projected_mode(self):
        mode = [True, False]
        idx = mode.index(self.projected_sub_scene)
        self.projected_sub_scene = mode[(idx + 1) % 2]

    def trans_scale(self):
        scale = [1.0, 0.25, 0.1, 0.01, 0.001]
        idx = scale.index(self.radius_scale)
        self.radius_scale = scale[(idx + 1) % 5]

    def render_snap(self):
        self.view =  self.all_view[self.count]
        self.count += 1
        self.count %= self.num_view

    def render_single(self, view, gaussians, pipeline, background, bbox_mask):
        # if self.mode == "segment":
        # render_res = np.ascontiguousarray(
        #         render(view, gaussians, pipeline, background, self.radius_scale, bbox_mask=bbox_mask)[
        #             self.mode].detach().cpu().numpy())
        # else:
        render_res = np.ascontiguousarray(render(view, gaussians, pipeline, background, self.radius_scale, bbox_mask=bbox_mask)[self.mode].detach().cpu().permute(1, 2, 0).numpy())
        return render_res


if __name__ == '__main__':
    # Set up command line argument parser
    parser = ArgumentParser(description="render script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--sub_scene", default=[None], nargs='+', type=parse_sub_scene)
    parser.add_argument("--sub_scene_all", default=None, type=parse_sub_scene_all)
    parser.add_argument("--fast_gui", action="store_true")
    parser.add_argument("--low_memory", action="store_true")
    args = get_combined_args(parser)
    print("Rendering: " + args.model_path)
    print("sub_scene_list: ", args.sub_scene)
    print("fast_gui ?  ",  args.fast_gui)
    print("low_memory ? ",  args.low_memory)
    # init
    ti.init(arch=ti.cuda, device_memory_GB=1, kernel_profiler=True)

    if args.sub_scene_all is not None:
        render_ = Render(model.extract(args), args.iteration, pipeline, args.sub_scene_all, args.fast_gui, args.low_memory)
    else:
        render_ = Render(model.extract(args), args.iteration, pipeline, args.sub_scene, args.fast_gui, args.low_memory)
    render_.start()