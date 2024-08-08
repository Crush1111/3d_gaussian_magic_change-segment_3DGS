# 3D Gaussian Splatting for Real-Time Radiance Field Rendering
## 环境配置
https://github.com/graphdeco-inria/gaussian-splatting
## 新增功能
- [x] 支持nerfstudio的transform.json格式数据
- [x] 提供来自slam的数据转换脚本
- [x] 支持渲染深度图
- [x] 支持来自单目深度估计的深度监督
- [x] 基于taichi的全新可视化界面，支持场景编辑，场景导入等功能

## 数据集准备及模型训练

参考原repo：https://github.com/graphdeco-inria/gaussian-splatting

### nerfstudio格式数据训练
#### 数据准备：

    |- dataset
    |- |- images
    |- |- images_2
    |- |- images_4
    |- |- images_8
    |- |- transforms.json
    |- |- points3d.ply (可选，从slam生成或者从sfm生成)

#### 训练模型：
> python train.py -s <path to NeRF Studio or instant ngp dataset> --eval # Train with train/test split

### slam获取的数据训练
#### 数据准备：

    |- slam dataset
    |- |- images
    |- |- images_2
    |- |- images_4
    |- |- images_8
    |- |- Keyframes.txt（comap格式）
    |- |- points3d.ply (可选，slam生成)


* 首先需要进行数据转换
>python process_data/slam2nerf.py #将路径改为slam得到的数据集路径


此脚本还支持对数据集的一些规则划分方法，如序列划分以及localrf中提到的划分方法

* 得到nerfstudio格式的数据


    |- dataset
    |- |- images
    |- |- images_2
    |- |- images_4
    |- |- images_8
    |- |- transforms.json
    |- |- points3d.ply (可选，从slam生成或者从sfm生成)

### 深度图渲染
根据PR：https://github.com/graphdeco-inria/gaussian-splatting/pull/30，实现了对深度图的渲染
>python render.py -m <path to trained model> # Generate renderings
### 深度监督
在深度渲染的基础上，参考[Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer](https://arxiv.org/abs/1907.01341)
实现了深度监督损失 
##### 如何启动深度监督？
1.运行单目深度估计DPT算法估计数据集的深度图
> python DPT/run_monodepth.py #修改数据集路径
2. 算法将会自行检测数据集路径下是否存在命名为depth的文件夹，如果存在，将会读取图像并进行深度监督
> python train.py -s <path to NeRF Studio or instant ngp dataset> --using_depth(可选) --eval # Train with train/test split 

### 基于taichi的可视化界面
#### 特征
- [x] 背景裁剪(基操勿6)
- [x] 场景融合(融合多个子场景联合渲染)
- [x] 场景编辑(放大，缩小，移动，旋转等)
- [x] 子场景编号投影(可以更好的选择子场景)
- [x] 深度渲染
- [x] 协方差缩放渲染
#### 按键介绍
- 旋转平移

      q(上移) w(前) e(下移)  |  u      i(上旋) o
      a(左)   s(后) d(右)   |  j(左旋) k(下旋) l(右旋) 

- c clip模式切换[True, False]
- v 保存当前裁剪场景
- b 子场景复制，直接复制crop到的区域为子场景表示
- g 子场景编号投影
- p 投影模式切换[xyz, bbox, none]（在crop 模式下执行）
- r 渲染训练路径图像
- 1~9 场景选择按钮，选择对应的子场景