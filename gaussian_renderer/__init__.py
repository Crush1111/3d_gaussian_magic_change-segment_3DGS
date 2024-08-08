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
from scene.cameras import Camera
import torch
import os
from utils.system_utils import mkdir_p
import torch.nn as nn
import tinycudann as tcnn
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.system_utils import searchForMaxIteration
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

# We make an exception on snake case conventions because SO3 != so3.
def exp_map_SO3xR3(tangent_vector):
    """Compute the exponential map of the direct product group `SO(3) x R^3`.

    This can be used for learning pose deltas on SE(3), and is generally faster than `exp_map_SE3`.

    Args:
        tangent_vector: Tangent vector; length-3 translations, followed by an `so(3)` tangent vector.
    Returns:
        [R|t] transformation matrices.
    """
    # code for SO3 map grabbed from pytorch3d and stripped down to bare-bones

    log_rot = tangent_vector[:, 3:]
    nrms = (log_rot * log_rot).sum(1)
    rot_angles = torch.clamp(nrms, 1e-4).sqrt()
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = torch.zeros((log_rot.shape[0], 3, 3), dtype=log_rot.dtype, device=log_rot.device)
    skews[:, 0, 1] = -log_rot[:, 2]
    skews[:, 0, 2] = log_rot[:, 1]
    skews[:, 1, 0] = log_rot[:, 2]
    skews[:, 1, 2] = -log_rot[:, 0]
    skews[:, 2, 0] = -log_rot[:, 1]
    skews[:, 2, 1] = log_rot[:, 0]
    skews_square = torch.bmm(skews, skews)

    ret = torch.zeros(tangent_vector.shape[0], 3, 4, dtype=tangent_vector.dtype, device=tangent_vector.device)
    ret[:, :3, :3] = (
        fac1[:, None, None] * skews
        + fac2[:, None, None] * skews_square
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )

    # Compute the translation
    ret[:, :3, 3] = tangent_vector[:, :3]
    return ret

class CameraOptimizer(nn.Module):
    def __init__(
        self,
        num_cameras,
        device='cuda',
    ):
        super().__init__()
        self.num_cameras = num_cameras
        self.device = device
        print('using pose adjustment!')
        # Initialize learnable parameters.
        self.pose_adjustment = torch.nn.Parameter(torch.zeros((num_cameras, 6), device=device))
        l = [
            {'params': [self.pose_adjustment], 'lr': 0.0001, "name": "pose_adjustment"},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def forward(
            self,
            viewpoint_cam):
        """Indexing into camera adjustments.
        Args:
            indices: indices of Cameras to optimize.
        Returns:
            Transformation matrices from optimized camera coordinates
            to given camera coordinates.
        """
        pose = exp_map_SO3xR3(self.pose_adjustment[viewpoint_cam.uid, :].unsqueeze(0)).squeeze(0)
        R1, t1 = viewpoint_cam.world_view_transform[:3, :3], viewpoint_cam.world_view_transform[3, :3]
        R2, t2 = pose[:3, :3], pose[:3, 3:]
        R = R1.matmul(R2)
        t = t1 + R1.matmul(t2).squeeze()
        viewpoint_cam.world_view_transform[:3, :3] = R
        viewpoint_cam.world_view_transform[3, :3] = t
        viewpoint_cam.R = R.clone().cpu().detach().numpy()
        viewpoint_cam.T = t.clone().cpu().detach().numpy()
        viewpoint_cam.full_proj_transform = (viewpoint_cam.world_view_transform.unsqueeze(0).bmm(viewpoint_cam.projection_matrix.unsqueeze(0))).squeeze(0)
        viewpoint_cam.camera_center = viewpoint_cam.world_view_transform.inverse()[3, :3]
        return viewpoint_cam

class AppearanceOptimizer(nn.Module):
    def __init__(
        self,
        num_cameras,
        device='cuda',
    ):
        super().__init__()
        self.num_cameras = num_cameras
        self.device = device
        print('using appearance embedding!')

        # Initialize learnable parameters.
        self.appearance_emb = torch.nn.Parameter(torch.zeros((num_cameras, 1, 16), device=device))

        self.appearance_embedding_config = {
            "n_input_dims": 32,
            "n_output_dims": 3,
            "encoding_config": {
                "otype": "Frequency",
                "n_frequencies": 4
            },
            "network_config": {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": 32,
                "n_hidden_layers": 2
            },
            "seed": 1337,
        }

        self.init_appearance_embedding_model()
        self.appearance_embedding.to(device)

        l = [
            {'params': [self.appearance_emb], 'lr': 0.0001, "name": "appearance_emb"},
            {'params': list(self.appearance_embedding.parameters()), 'lr': 0.0001, "name": "appearance_embedding"},
        ]

        self.appearance_embedding_optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def init_appearance_embedding_model(self):
        if self.appearance_embedding_config is not None:
            self.appearance_embedding = tcnn.NetworkWithInputEncoding(**self.appearance_embedding_config)

    def forward(self, viewpoint_cam):
        ## 将viewmats作为位置编码
        emb = torch.cat([self.appearance_emb[viewpoint_cam.uid], viewpoint_cam.world_view_transform.reshape(1, -1)], -1)
        appearance_factors = self.appearance_embedding(emb).reshape((-1, 1, 1))
        return appearance_factors

    def save_appearance_embedding(self, path):
        if self.appearance_embedding is not None:
            mkdir_p(os.path.dirname(path))
            torch.save({
                "model_config": self.appearance_embedding_config,
                "model_state_dict": self.appearance_embedding.state_dict(),
                "appearance_emb_state_dict": self.appearance_emb,
            }, path)

    def load_appearance_embedding(self, model_path, load_iteration=None):

        if load_iteration:
            if load_iteration == -1:
                loaded_iter = searchForMaxIteration(os.path.join(model_path, "point_cloud"))
            else:
                loaded_iter = load_iteration
            print("Loading trained appearance embedding model at iteration {}".format(loaded_iter))

        path = os.path.join(model_path,
                          "point_cloud",
                          "iteration_" + str(loaded_iter),
                          "appearance_embedding.ckpt")

        if os.path.exists(path):
            checkpoint = torch.load(path)
            # load config
            self.appearance_embedding_config = checkpoint["model_config"]
            # init model based on config
            self.init_appearance_embedding_model()
            # load model state dict
            self.appearance_embedding.load_state_dict(checkpoint["model_state_dict"])
            self.appearance_emb.load_state_dict(checkpoint["appearance_emb_state_dict"])
            self.appearance_embedding.to(self.device)
            self.appearance_emb.to(self.device)
        else:
            print("disable appearance embedding")
            self.appearance_embedding = None



def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, bbox_mask=None, rgb_factors=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    if bbox_mask is not None:
        screenspace_points = torch.zeros_like(pc.get_xyz[bbox_mask], dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = pc.get_xyz[bbox_mask]
        means2D = screenspace_points # 用来返回2d均值的梯度？
        opacity = pc.get_opacity[bbox_mask]
        segment = pc.get_segment[bbox_mask]

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)[bbox_mask]
            print(cov3D_precomp.shape)
        else:
            scales = pc.get_scaling[bbox_mask]
            rotations = pc.get_rotation[bbox_mask]

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if pipe.convert_SHs_python:
                shs_view = pc.get_features[bbox_mask].transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp = (pc.get_xyz[bbox_mask] - viewpoint_camera.camera_center.repeat(pc.get_features[bbox_mask].shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features[bbox_mask]  # 为什么是16？因为球谐函数的度为3，基函数的项数为16（2^(3+1）)
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, depth, alpha, rendered_segment = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            segments = segment,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

        depth = depth / (depth.max() + 1e-5)

        # Appearance embedding
        if rgb_factors is not None:
            rendered_image = rendered_image * rgb_factors
        else:
            rendered_image = rendered_image

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "depth": depth,
                "alpha": alpha,
                "segment": rendered_segment if rendered_segment is not None else None}

    else:
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity
        segment = pc.get_segment

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if pipe.convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, depth, alpha, rendered_segment = rasterizer(
            means3D=means3D,
            means2D=means2D,
            opacities=opacity,
            shs=shs,
            colors_precomp=colors_precomp,
            segments=segment,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)

        depth = depth / (depth.max() + 1e-5)

        # Appearance embedding
        if rgb_factors is not None:
            rendered_image = rendered_image * rgb_factors
        else:
            rendered_image = rendered_image

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii,
                "depth": depth,
                "alpha": alpha,
                "segment": rendered_segment if rendered_segment is not None else None}