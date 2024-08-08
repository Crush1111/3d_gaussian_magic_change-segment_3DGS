import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import random

import torch
import sys
sys.path.insert(0, "/home/guozebin/work_code/3d_gaussian_magic_change")
import io
from PIL import Image
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import Scene
from gaussian_renderer import render
from camera_trajectory import *
import time
import numpy as np
import asyncio

app = Flask(__name__)
CORS(app)
class Render:
    def __init__(self, dataset, iteration, pipeline):
        self.gaussians, self.scene, self.background = self.render_init(dataset, iteration)
        self.pipeline = pipeline
        self.all_view = self.scene.getTrainCameras()
        self.view = self.scene.getTrainCameras()[0]
        self.init_pose = self.scene.getTrainCameras()[0]
        self.count = 0
        self.key_dict = {
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
            'r': self.render_snap(),
            'x': self.back_to_origin()
        }

    def back_to_origin(self):
        return self.init_pose

    def render_snap(self):
        return self.all_view[self.count]

    def render_init(self, dataset, iteration):
        with torch.no_grad():
            gaussians = GaussianModel(dataset.sh_degree)
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
            bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        return gaussians, scene, background

    def render_single(self, view, gaussians, pipeline, background):
        render_res = render(view, gaussians, pipeline, background)["render"].detach().cpu().permute(1, 2, 0).numpy()
        return render_res

    def render_from_key(self, key):
        self.view = self.key_dict[key](self.view)
        return self.render_single(self.view, self.gaussians, self.pipeline, self.background)

    def render_path(self, view):
        return self.render_single(view, self.gaussians, self.pipeline, self.background)



# 创建一个缓存字典来存储已生成的图像，避免重复计算
image_cache = {}

# 创建一个缓存字典来存储已生成的图像和生成时间
image_cache = {}
image_last_generated = {}

from datetime import datetime, timedelta

# 创建一个缓存字典来存储已生成的图像和生成时间
image_cache = {}
image_last_generated = {}
update_counter = {}


@app.route('/api/generate-image', methods=['GET'])
def generate_image():
    image_type = request.args.get('type', 'a')  # 获取请求参数 type 的值，默认值为 'a'

    now = datetime.now()
    force_refresh = request.args.get('refresh', 'false').lower() == 'true'

    if not force_refresh and image_type in image_cache:
        # 检查是否需要更新缓存
        last_generated = image_last_generated.get(image_type, now)
        if (now - last_generated) < timedelta(seconds=0.01):
            # 增加更新计数器
            counter = update_counter.get(image_type, 0)
            update_counter[image_type] = counter + 1

            if update_counter[image_type] >= 10:
                del image_cache[image_type]  # 删除缓存，以便下次重新生成图像
                update_counter[image_type] = 0

            cached_image = io.BytesIO(image_cache[image_type].getbuffer())
            return send_file(cached_image, mimetype='image/png')

    if image_type in ['a', 's', 'd', 'q', 'w', 'e', 'i', 'j', 'k', 'l', 'u', 'o','r']:
        t1 = time.time()
        image_array = render_.render_from_key(image_type)
        print('推理 time:', time.time() - t1)
        image_array = (image_array * 255).astype(np.uint8)

        # 将 NumPy 数组转换为 PIL.Image 对象
        image = Image.fromarray(image_array)

        # 将图像保存到内存中的字节缓冲区
        buffer = io.BytesIO()
        image.save(buffer, 'PNG')  # 使用 PNG 格式保存图像
        buffer.seek(0)  # 将缓冲区的读取位置重置为开头

        # 将图像缓存起来，以便下次请求直接返回缓存的图像
        image_cache[image_type] = buffer
        image_last_generated[image_type] = now
        update_counter[image_type] = 0

        cached_image = io.BytesIO(buffer.getbuffer())
        return send_file(cached_image, mimetype='image/png')

    else:
        return jsonify({'error': 'Invalid image type'}), 400





if __name__ == '__main__':
    # Set up command line argument parser
    parser = ArgumentParser(description="render script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # init
    render_ = Render(model.extract(args), args.iteration, pipeline)
    # render
    # for key in ['j', 'k', 'l', 'l', 'l', 'l', 'l']:
    #     res = render_.render_from_key(key)
    #     image_array = (res * 255).astype(np.uint8)
    #     cv2.imwrite(f'{key}.jpg', image_array)
    app.run(host='0.0.0.0', port=5000)