"""
GISC-FMT: 多视角荧光分子3D重建隐式网络

核心架构：共享U-Net + View-specific Adapter + 多视角融合 + 隐式源场
所有配置参数从config对象提取，无硬编码默认值
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..network.encoder import (
    UNet,
    GateFusion,
    PointFeatureSampler,
)
from ..network.fusion import (
    BackgroundGuidanceInteraction,
    CrossViewResidualFusion,
    ScaleFusionNet,
)
from ..utils.cam import project_points_to_camera
from ..config_extractor import ConfigExtractor


# ===== 核心模块 =====


class ResBlock(nn.Module):
    """残差块：LayerNorm + FC层 + 残差连接"""

    def __init__(self, in_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim)
        self.linear1 = nn.Linear(in_dim, in_dim)
        self.linear2 = nn.Linear(in_dim, in_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.layer_norm(x)
        h = self.relu(self.linear1(h))
        h = self.linear2(h)
        return x + h


class ViewSpecificAdapter(nn.Module):
    """View-specific adapter：轻量级残差模块，处理视角结构差异"""

    def __init__(self, in_channels, r_ratio=8):
        super().__init__()
        r = max(4, in_channels // r_ratio)
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, r, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(r, in_channels, kernel_size=1, padding=0),
        )

    def forward(self, x):
        return x + self.adapter(x)


class ViewConditionedAdapter(nn.Module):
    """Single adapter conditioned on view angle (FiLM-style)."""

    def __init__(
        self, in_channels: int, r_ratio: int = 8, cond_dim: int = 4, cond_hidden: int = 32
    ):
        super().__init__()
        r = max(4, in_channels // r_ratio)
        self.conv1 = nn.Conv2d(in_channels, r, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(r, in_channels, kernel_size=1, padding=0)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cond_hidden, 2 * r),
        )
        # Start close to identity.
        nn.init.zeros_(self.cond_mlp[-1].weight)
        nn.init.zeros_(self.cond_mlp[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond: [B, cond_dim]
        h = self.relu(self.conv1(x))
        gb = self.cond_mlp(cond).to(dtype=h.dtype)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        h = h * (1.0 + gamma) + beta
        return x + self.conv2(h)


class ImplicitSourceField(nn.Module):
    """隐式源场网络：坐标支路 + 特征支路 + 输出支路"""

    def __init__(self, coord_dim, feature_dim, hidden_dim, d_x, d_f):
        super().__init__()

        self.mlp_x = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, d_x),
            ResBlock(d_x),
        )

        self.mlp_f = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, d_f),
            ResBlock(d_f),
        )

        self.mlp_out = nn.Sequential(
            nn.Linear(d_x + d_f, hidden_dim),
            nn.ReLU(inplace=True),
            ResBlock(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, gamma_x, fused_features):
        z_x = self.mlp_x(gamma_x)
        z_f = self.mlp_f(fused_features)
        z = torch.cat([z_x, z_f], dim=-1)
        return self.mlp_out(z)


# ===== 主网络 =====


class PointDensityNet(nn.Module):
    """多视角荧光分子3D重建网络

    架构流程：
    1. 共享U-Net特征提取
    2. View-specific adapter处理视角差异
    3. 向量化几何投影和采样
    4. ViewWeightNet多视角融合
    5. ImplicitSourceField隐式密度预测
    """

    def __init__(self, config):
        """
        Args:
            config: 配置对象，包含所有网络参数
        """
        super().__init__()

        if config is None:
            raise ValueError("config 参数为必需项，不能为 None")

        # 提取所有参数
        net_params = ConfigExtractor.extract_network_params(config)
        geo_params = ConfigExtractor.extract_geometry_config(config)
        gisc_params = ConfigExtractor.extract_gisc_fmt_config(config)
        view_angles = ConfigExtractor.extract_view_angles(config)

        # 网络基本参数
        num_views = net_params["num_views"]
        in_channels = net_params["in_channels"]
        feature_dim = net_params["feature_dim"]
        pos_enc_dim = net_params["pos_enc_dim"]

        # GISC-FMT 特定参数
        use_adapter = gisc_params["use_adapter"]
        adapter_strategy = gisc_params["adapter_strategy"]
        adapter_mode = str(gisc_params.get("adapter_mode", "per_view"))
        norm_type = gisc_params["norm_type"]
        adapter_r_ratio = gisc_params["adapter_r_ratio"]
        adapter_cond_dim = int(gisc_params.get("adapter_cond_dim", 4) or 4)
        implicit_hidden = gisc_params["implicit_field_hidden_dim"]
        implicit_d_x = gisc_params["implicit_field_d_x"]
        implicit_d_f = gisc_params["implicit_field_d_f"]
        view_weight_embed = gisc_params["view_weight_embed_dim"]
        view_weight_hidden = gisc_params["view_weight_hidden_dim"]
        ms_params = gisc_params["multiscale"]
        bg_params = gisc_params["background"]

        # 几何参数
        self.camera_distance = geo_params["camera_distance"]
        self.detector_size = geo_params["detector_size"]
        self.global_voxel_shape = geo_params["global_voxel_shape"]
        self.view_list = view_angles[:num_views]
        self.pos_enc_dim = pos_enc_dim
        self.feature_dim = feature_dim
        self.num_views = num_views
        self.config = config

        assert len(self.view_list) == num_views, (
            f"view_list length {len(self.view_list)} != num_views {num_views}"
        )

        # 共享 U-Net
        self.shared_unet = UNet(
            n_channels=in_channels,
            n_features=feature_dim,
            bilinear=True,
            norm_type=norm_type,
        )

        # View adapters
        self.adapter_mode = adapter_mode
        if use_adapter and adapter_strategy == "bottleneck_decoder":
            factor = 2
            bottleneck_ch = feature_dim * 16 // factor
            up1_ch = feature_dim * 8 // factor
            up2_ch = feature_dim * 4 // factor
            up3_ch = feature_dim * 2 // factor
            up4_ch = feature_dim

            if self.adapter_mode == "conditioned":
                self.adapter_bottleneck = ViewConditionedAdapter(
                    bottleneck_ch, adapter_r_ratio, cond_dim=adapter_cond_dim
                )
                self.adapter_up1 = ViewConditionedAdapter(
                    up1_ch, adapter_r_ratio, cond_dim=adapter_cond_dim
                )
                self.adapter_up2 = ViewConditionedAdapter(
                    up2_ch, adapter_r_ratio, cond_dim=adapter_cond_dim
                )
                self.adapter_up3 = ViewConditionedAdapter(
                    up3_ch, adapter_r_ratio, cond_dim=adapter_cond_dim
                )
                self.adapter_up4 = ViewConditionedAdapter(
                    up4_ch, adapter_r_ratio, cond_dim=adapter_cond_dim
                )
            else:
                self.adapter_bottleneck = nn.ModuleList(
                    [ViewSpecificAdapter(bottleneck_ch, adapter_r_ratio) for _ in range(num_views)]
                )
                self.adapter_up1 = nn.ModuleList(
                    [ViewSpecificAdapter(up1_ch, adapter_r_ratio) for _ in range(num_views)]
                )
                self.adapter_up2 = nn.ModuleList(
                    [ViewSpecificAdapter(up2_ch, adapter_r_ratio) for _ in range(num_views)]
                )
                self.adapter_up3 = nn.ModuleList(
                    [ViewSpecificAdapter(up3_ch, adapter_r_ratio) for _ in range(num_views)]
                )
                self.adapter_up4 = nn.ModuleList(
                    [ViewSpecificAdapter(up4_ch, adapter_r_ratio) for _ in range(num_views)]
                )

        elif use_adapter and adapter_strategy == "final":
            # Only apply adapters after UNet feature_fusion.
            if self.adapter_mode == "conditioned":
                # Shared parameters across views (angle-conditioned).
                self.adapter_final = ViewConditionedAdapter(
                    feature_dim, adapter_r_ratio, cond_dim=adapter_cond_dim
                )
            else:
                # Independent parameters per view (no sharing).
                self.adapter_final = nn.ModuleList(
                    [ViewSpecificAdapter(feature_dim, adapter_r_ratio) for _ in range(num_views)]
                )

        elif adapter_strategy not in {"bottleneck_decoder", "final"}:
            raise NotImplementedError(f"adapter_strategy '{adapter_strategy}' 需在config中配置")

        self.use_adapter = use_adapter
        self.adapter_strategy = adapter_strategy

        # ===== Multi-scale sampling projections =====
        # s1(final): C=feature_dim
        # s2(mid):   from UNet up2, C=feature_dim*4//factor
        # s3(coarse):from UNet bottleneck x5, C=feature_dim*16//factor
        factor = 2 if getattr(self.shared_unet, "bilinear", False) else 1
        c_s2 = feature_dim * 4 // factor
        c_s3 = feature_dim * 16 // factor
        self.proj_s2 = nn.Identity() if c_s2 == feature_dim else nn.Linear(c_s2, feature_dim)
        self.proj_s3 = nn.Identity() if c_s3 == feature_dim else nn.Linear(c_s3, feature_dim)

        # ===== Per-scale cross-view residual completion fusion =====
        self.fusion_s1 = CrossViewResidualFusion(
            feature_dim=feature_dim,
            num_views=num_views,
            embed_dim=ms_params["view_embed_dim"],
            depth_dim=ms_params["depth_dim"],
            hidden_dim=ms_params["cross_view_hidden_dim"],
            tau=ms_params["tau"],
            depth_max=ms_params["depth_max"],
        )
        self.fusion_s2 = CrossViewResidualFusion(
            feature_dim=feature_dim,
            num_views=num_views,
            embed_dim=ms_params["view_embed_dim"],
            depth_dim=ms_params["depth_dim"],
            hidden_dim=ms_params["cross_view_hidden_dim"],
            tau=ms_params["tau"],
            depth_max=ms_params["depth_max"],
        )
        self.fusion_s3 = CrossViewResidualFusion(
            feature_dim=feature_dim,
            num_views=num_views,
            embed_dim=ms_params["view_embed_dim"],
            depth_dim=ms_params["depth_dim"],
            hidden_dim=ms_params["cross_view_hidden_dim"],
            tau=ms_params["tau"],
            depth_max=ms_params["depth_max"],
        )

        # ===== Depth-conditioned scale fusion =====
        self.scale_fusion = ScaleFusionNet(
            feature_dim=feature_dim,
            depth_dim=ms_params["depth_dim"],
            hidden_dim=ms_params["scale_hidden_dim"],
            depth_max=ms_params["depth_max"],
        )

        # ===== Background branch (optional, backward compatible) =====
        self.enable_background = bool(bg_params["enable_background"])
        self.bg_guidance_cfg = bg_params["guidance"]
        if self.enable_background:
            self.bg_guidance_interaction = BackgroundGuidanceInteraction(
                feature_dim=feature_dim,
                guidance_dim=self.bg_guidance_cfg["dim"],
                hidden_dim=self.bg_guidance_cfg["hidden_dim"],
                mode=self.bg_guidance_cfg["mode"],
                scale=self.bg_guidance_cfg["scale"],
                gate_init_bias=self.bg_guidance_cfg["gate_init_bias"],
            )
            bg_head_cfg = bg_params["head"]
            self.background_head = ImplicitSourceField(
                coord_dim=pos_enc_dim,
                feature_dim=feature_dim,
                hidden_dim=bg_head_cfg["hidden_dim"],
                d_x=bg_head_cfg["d_x"],
                d_f=bg_head_cfg["d_f"],
            )

        # 门控融合（辅助输出）
        self.gate_fusions = nn.ModuleList([GateFusion(feature_dim) for _ in range(num_views)])

        # 隐式源场
        self.density_head = ImplicitSourceField(
            coord_dim=pos_enc_dim,
            feature_dim=feature_dim,
            hidden_dim=implicit_hidden,
            d_x=implicit_d_x,
            d_f=implicit_d_f,
        )

        # 点采样器
        self.sampler = PointFeatureSampler()

    def _forward_shared_unet(self, multi_view_images):
        """向量化执行共享U-Net"""
        if isinstance(multi_view_images, dict):
            images_list = []
            for view_name in self.view_list:
                if str(view_name) in multi_view_images:
                    proj = multi_view_images[str(view_name)]
                    if proj.dim() == 3:
                        proj = proj.unsqueeze(1)
                    images_list.append(proj)
            images_flat = torch.cat(images_list, dim=0)
        else:
            images_flat = multi_view_images

        B_total, C, H, W = images_flat.shape
        B = B_total // self.num_views

        # 前向通过U-Net各阶段
        x1 = self.shared_unet.inc(images_flat)
        x2 = self.shared_unet.down1(x1)
        x3 = self.shared_unet.down2(x2)
        x4 = self.shared_unet.down3(x3)
        x5 = self.shared_unet.down4(x4)

        # Optional adapters
        if self.use_adapter and self.adapter_strategy == "bottleneck_decoder":
            if self.adapter_mode == "conditioned":
                x5 = self.adapter_bottleneck(x5, self._view_cond(B, x5.device))
            else:
                x5 = self._apply_per_view_adapters(x5, self.adapter_bottleneck, B)

        # coarse-scale feature (bottleneck)
        feat_s3 = x5  # [V*B, C3, H/16, W/16]

        # 解码器
        x = self.shared_unet.up1(x5, x4)
        if self.use_adapter and self.adapter_strategy == "bottleneck_decoder":
            if self.adapter_mode == "conditioned":
                x = self.adapter_up1(x, self._view_cond(B, x.device))
            else:
                x = self._apply_per_view_adapters(x, self.adapter_up1, B)

        x = self.shared_unet.up2(x, x3)
        if self.use_adapter and self.adapter_strategy == "bottleneck_decoder":
            if self.adapter_mode == "conditioned":
                x = self.adapter_up2(x, self._view_cond(B, x.device))
            else:
                x = self._apply_per_view_adapters(x, self.adapter_up2, B)

        # mid-scale feature (decoder intermediate)
        feat_s2 = x  # [V*B, C2, H/4, W/4]

        x = self.shared_unet.up3(x, x2)
        if self.use_adapter and self.adapter_strategy == "bottleneck_decoder":
            if self.adapter_mode == "conditioned":
                x = self.adapter_up3(x, self._view_cond(B, x.device))
            else:
                x = self._apply_per_view_adapters(x, self.adapter_up3, B)

        x = self.shared_unet.up4(x, x1)
        if self.use_adapter and self.adapter_strategy == "bottleneck_decoder":
            if self.adapter_mode == "conditioned":
                x = self.adapter_up4(x, self._view_cond(B, x.device))
            else:
                x = self._apply_per_view_adapters(x, self.adapter_up4, B)

        feature_map = self.shared_unet.feature_fusion(x)

        if self.use_adapter and self.adapter_strategy == "final":
            if self.adapter_mode == "conditioned":
                feature_map = self.adapter_final(
                    feature_map, self._view_cond(B, feature_map.device)
                )
            else:
                feature_map = self._apply_per_view_adapters(feature_map, self.adapter_final, B)

        # 重组为dict（多尺度）
        # s1: final feature_fusion          [B, C1, H, W]
        # s2: decoder intermediate (up2)    [B, C2, H/4, W/4]
        # s3: bottleneck (x5)              [B, C3, H/16, W/16]
        features_s1_dict = {}
        features_s2_dict = {}
        features_s3_dict = {}
        for v_idx, view_name in enumerate(self.view_list):
            sl = slice(v_idx * B, (v_idx + 1) * B)
            features_s1_dict[view_name] = feature_map[sl]
            features_s2_dict[view_name] = feat_s2[sl]
            features_s3_dict[view_name] = feat_s3[sl]

        return features_s1_dict, features_s2_dict, features_s3_dict

    def _view_cond(self, B: int, device: torch.device) -> torch.Tensor:
        """Per-sample angle embedding for the flattened [V*B,...] batch."""
        angles = torch.tensor(
            [float(v) for v in self.view_list], device=device, dtype=torch.float32
        )
        rad = angles * torch.pi / 180.0
        cond = torch.stack(
            [torch.sin(rad), torch.cos(rad), torch.sin(2 * rad), torch.cos(2 * rad)], dim=-1
        )  # [V,4]
        return cond.repeat_interleave(B, dim=0)  # [V*B,4]

    def _apply_per_view_adapters(self, x, adapter_list, B):
        """为每个视角应用对应的adapter"""
        x_adapted_list = []
        for v_idx in range(self.num_views):
            x_v = x[v_idx * B : (v_idx + 1) * B]
            x_v = adapter_list[v_idx](x_v)
            x_adapted_list.append(x_v)
        return torch.cat(x_adapted_list, dim=0)

    def _vectorized_grid_sample(self, view_features, x3d):
        """向量化grid_sample采样所有视角特征"""
        B, N, _ = x3d.shape
        C = next(iter(view_features.values())).shape[1]

        grids_list = []
        feature_maps_list = []

        for view_name in self.view_list:
            if view_name not in view_features:
                continue

            grid = self.sampler.project_points_to_view(
                x3d, view_name, self.camera_distance, self.detector_size, self.global_voxel_shape
            )
            grid = grid.unsqueeze(1)
            grids_list.append(grid)

            feat_map = view_features[view_name]
            feat_map = feat_map.permute(0, 1, 3, 2)
            feature_maps_list.append(feat_map)

        grids_batched = torch.cat(grids_list, dim=0)
        feat_batched = torch.cat(feature_maps_list, dim=0)

        sampled_feats = F.grid_sample(
            feat_batched,
            grids_batched,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        sampled_feats = sampled_feats.squeeze(2)
        sampled_feats = sampled_feats.permute(0, 2, 1)

        f_list = []
        for v_idx in range(self.num_views):
            f_v = sampled_feats[v_idx * B : (v_idx + 1) * B]
            f_list.append(f_v)

        f = torch.stack(f_list, dim=2)
        return f

    def _compute_depth(self, x3d: torch.Tensor) -> torch.Tensor:
        """Compute a view-independent depth cue for each 3D point.

        Returns:
            depth: [B, N, 1] (camera depth for 0° view)
        """
        voxel_shape = torch.as_tensor(
            self.global_voxel_shape, device=x3d.device, dtype=x3d.dtype
        )  # [3]
        points = x3d * (voxel_shape - 1)
        points = points - voxel_shape / 2.0 + 0.5
        _proj, depths = project_points_to_camera(
            points, 0, self.camera_distance, self.detector_size
        )
        return depths[..., 0:1]

    def forward(self, view_projections, x3d, gamma_x=None, bg_guidance=None):
        """
        Args:
            view_projections: dict {view_name: tensor}
            x3d: [B, N, 3] 采样点坐标
            gamma_x: [B, N, pos_enc_dim] 位置编码（可选）

        Returns:
            logits: [B, N, 1] 密度预测
            views_no_projections: dict 辅助输出
        """
        B, N, _ = x3d.shape

        # 特征提取（多尺度）
        features_s1_dict, features_s2_dict, features_s3_dict = self._forward_shared_unet(
            view_projections
        )

        # 门控融合辅助输出（用final尺度特征）
        views_no_projections = {}
        for v_idx, view_name in enumerate(self.view_list):
            if view_name in features_s1_dict:
                feat = features_s1_dict[view_name]  # [B, C, H, W]
                gate_out = self.gate_fusions[v_idx](feat)
                views_no_projections[view_name] = gate_out.squeeze(1)

        # depth: [B, N, 1] (approx 0~camera_distance)
        depth = self._compute_depth(x3d)

        # 多尺度点采样
        f_s1 = self._vectorized_grid_sample(features_s1_dict, x3d)  # [B, N, V, C]
        f_s2 = self._vectorized_grid_sample(features_s2_dict, x3d)  # [B, N, V, C2]
        f_s3 = self._vectorized_grid_sample(features_s3_dict, x3d)  # [B, N, V, C3]

        # 通道统一到 C=feature_dim
        g_s1 = f_s1
        g_s2 = self.proj_s2(f_s2)
        g_s3 = self.proj_s3(f_s3)

        # 每尺度跨视角互补补齐融合
        h1, _w1 = self.fusion_s1(g_s1, depth)  # [B, N, C]
        h2, _w2 = self.fusion_s2(g_s2, depth)  # [B, N, C]
        h3, _w3 = self.fusion_s3(g_s3, depth)  # [B, N, C]

        # 尺度融合（depth-conditioned）
        fused_feat, _a = self.scale_fusion(h1, h2, h3, depth)

        # 位置编码
        if gamma_x is None:
            gamma_x = self._simple_pos_encoding(x3d, self.pos_enc_dim)

        # 隐式密度预测
        logits = self.density_head(gamma_x, fused_feat)

        # Background branch (optional)
        if getattr(self, "enable_background", False):
            extra = {}
            fused_feat_bg = fused_feat
            if bool(self.bg_guidance_cfg["enable"]):
                if bg_guidance is None:
                    raise ValueError(
                        "Background guidance is enabled by config, but bg_guidance is None. "
                        "Please pass bg_guidance=[B,G] (or [B,N,G])."
                    )
                fused_feat_bg, stats = self.bg_guidance_interaction(fused_feat_bg, bg_guidance)
                extra.update({f"bg_guidance/{k}": v for k, v in stats.items()})

            background_logits = self.background_head(gamma_x, fused_feat_bg)  # [B, N, 1]
            extra["background_logits"] = background_logits
            return logits, views_no_projections, extra

        return logits, views_no_projections

    def _simple_pos_encoding(self, x, d_model):
        """Transformer风格的位置编码"""
        B, N, _ = x.shape

        if d_model % 6 != 0:
            d_model = (d_model // 6) * 6

        num_freqs = d_model // 6
        encodings = []

        freqs = torch.linspace(1.0, 10.0, num_freqs, device=x.device)
        freqs = 2 * torch.pi * freqs

        for i in range(3):
            coord = x[..., i]
            for freq in freqs:
                encodings.append(torch.sin(coord * freq))
                encodings.append(torch.cos(coord * freq))

        return torch.stack(encodings, dim=-1)


# Public name (preferred)
GISCFMT = PointDensityNet

# Backward compatible alias
MINRFMT = PointDensityNet
