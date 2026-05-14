"""
GISC-FMT: 多视角荧光分子3D重建隐式网络

核心架构：共享U-Net + View-specific Adapter + 多视角融合 + 隐式源场
所有配置参数从config对象提取，无硬编码默认值
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config_extractor import ConfigExtractor
from ..network.encoder import (
    GateFusion,
    PointFeatureSampler,
    UNet,
)
from ..network.fusion import (
    BackgroundGuidanceInteraction,
    CrossViewResidualFusion,
    ScaleFusionNet,
)
from ..network.feature_refinement import FeatureRefinement
from ..network.ptfa import (
    ptfa_sample_corrected_exit_depth_gaussian,
    ptfa_sample_exit_depth_gaussian,
    ptfa_sample_fixed_gaussian,
)
from ..network.query_aggregation import ConsensusResidualGate, GeometryViewGate, ReliabilityViewGate
from ..utils.cam import project_points_to_camera
from ..utils.fmt_simgen_projection import project_points_mm_to_detector

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


class ResidualScorer(nn.Module):
    """Small zero-initialized residual scorer for query-level logit correction."""

    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, scorer_input: torch.Tensor) -> torch.Tensor:
        return self.net(scorer_input)


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
        ptfa_params = ConfigExtractor.extract_ptfa_config(config)
        residual_scorer_params = ConfigExtractor.extract_residual_scorer_config(config)
        refinement_params = ConfigExtractor.extract_feature_refinement_config(config)
        query_agg_params = ConfigExtractor.extract_query_aggregation_config(config)
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
        ms_params = gisc_params["multiscale"]
        bg_params = gisc_params["background"]

        # 几何参数
        self.camera_distance = geo_params["camera_distance"]
        self.detector_size = geo_params["detector_size"]
        self.detector_resolution = geo_params["detector_resolution"]
        self.fov_mm = geo_params["fov_mm"]
        self.global_voxel_shape = geo_params["global_voxel_shape"]
        self.volume_center_world = geo_params["volume_center_world"]
        self.use_fmt_simgen_projection = geo_params["use_fmt_simgen_projection"]
        self.transpose_feature_map_for_sampling = geo_params["transpose_feature_map_for_sampling"]
        self.ptfa_enabled = bool(ptfa_params["enabled"])
        self.ptfa_scales = set(ptfa_params["scales"])
        self.ptfa_mode = str(ptfa_params["mode"])
        self.ptfa_window = int(ptfa_params["window"])
        self.ptfa_sigma_px = float(ptfa_params["sigma_px"])
        self.ptfa_sigma_min = float(ptfa_params["sigma_min"])
        self.ptfa_sigma_max = float(ptfa_params["sigma_max"])
        self.ptfa_exit_depth_max_mm = float(ptfa_params["exit_depth_max_mm"])
        self.ptfa_invert_depth = bool(ptfa_params["invert_depth"])
        self.aggregation_mode = str(query_agg_params["aggregation_mode"])
        self.residual_scorer_enabled = bool(residual_scorer_params["enabled"])
        self.residual_scorer_lambda = float(residual_scorer_params["lambda_r"])
        self.residual_scorer_input_mode = str(residual_scorer_params["input_mode"])
        self.feature_refinement_enabled = bool(refinement_params["enabled"])
        self.feature_refinement_ptfa_view_aggregation = str(
            refinement_params["ptfa_view_aggregation"]
        )
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

        if self.residual_scorer_enabled:
            if self.residual_scorer_input_mode == "bilinear_s3":
                residual_in_dim = pos_enc_dim + feature_dim
            elif self.residual_scorer_input_mode == "bilinear_s3_plus_s1_ptfa":
                residual_in_dim = pos_enc_dim + feature_dim + feature_dim + 5
            else:
                raise NotImplementedError(
                    f"Unknown residual_scorer.input_mode: {self.residual_scorer_input_mode}"
                )
            self.residual_scorer = ResidualScorer(
                in_dim=residual_in_dim,
                hidden_dim=int(residual_scorer_params["hidden_dim"]),
            )
            self.residual_scorer_input_dim = residual_in_dim

        if self.feature_refinement_enabled:
            if self.aggregation_mode != "legacy_multiscale":
                raise ValueError("feature_refinement currently requires legacy_multiscale fusion")
            if self.residual_scorer_enabled:
                raise ValueError("E8 feature_refinement is defined with residual_scorer.enabled=false")
            if int(refinement_params["geom_dim"]) != 5:
                raise ValueError("E8 feature refinement expects model.feature_refinement.geom_dim=5")
            self.feature_refinement = FeatureRefinement(
                feature_dim=feature_dim,
                geom_dim=int(refinement_params["geom_dim"]),
                hidden_dim=int(refinement_params["hidden_dim"]),
                zero_init=bool(refinement_params["zero_init"]),
            )
            self.feature_refinement_input_dim = feature_dim * 3 + int(
                refinement_params["geom_dim"]
            )
            if self.feature_refinement_ptfa_view_aggregation == "reliability_gate":
                gate_cfg = refinement_params["reliability_gate"]
                self.reliability_gate_geom_set = str(gate_cfg["geom_set"])
                if self.reliability_gate_geom_set == "full":
                    reliability_geom_dim = 12
                elif self.reliability_gate_geom_set == "compact":
                    reliability_geom_dim = 8
                else:
                    raise ValueError(
                        "model.feature_refinement.reliability_gate.geom_set must be "
                        f"'full' or 'compact', got {self.reliability_gate_geom_set}"
                    )
                mix_cfg = gate_cfg["residual_mix"]
                self.reliability_view_gate = ReliabilityViewGate(
                    geom_dim=reliability_geom_dim,
                    hidden_dim=int(gate_cfg["hidden_dim"]),
                    temperature=float(gate_cfg["temperature"]),
                    zero_init=bool(gate_cfg["zero_init"]),
                    norm=str(gate_cfg["norm"]),
                    residual_mix_enabled=bool(mix_cfg["enabled"]),
                    residual_mix_gamma=float(mix_cfg["gamma"]),
                )
                self.reliability_view_gate_input_dim = reliability_geom_dim
            elif self.feature_refinement_ptfa_view_aggregation == "consensus_residual_gate":
                consensus_cfg = refinement_params["consensus_residual_gate"]
                self.consensus_residual_gate_use_evidence_stats = bool(
                    consensus_cfg["use_evidence_stats"]
                )
                consensus_input_dim = 8 + (
                    4 if self.consensus_residual_gate_use_evidence_stats else 0
                )
                self.consensus_residual_gate = ConsensusResidualGate(
                    input_dim=consensus_input_dim,
                    hidden_dim=int(consensus_cfg["hidden_dim"]),
                    gamma=float(consensus_cfg["gamma"]),
                    norm=str(consensus_cfg["norm"]),
                )
                self.consensus_residual_gate_input_dim = consensus_input_dim
            elif self.feature_refinement_ptfa_view_aggregation != "masked_mean":
                raise NotImplementedError(
                    "Unknown feature_refinement.ptfa_view_aggregation: "
                    f"{self.feature_refinement_ptfa_view_aggregation}"
                )

        if self.aggregation_mode == "corrected_exit_ptfa_geom_gate":
            self.query_view_gate = GeometryViewGate(
                geom_dim=9,
                hidden_dim=int(query_agg_params["hidden_dim"]),
                temperature=float(query_agg_params["temperature"]),
                zero_init=bool(query_agg_params["zero_init"]),
            )
        elif self.aggregation_mode != "legacy_multiscale":
            raise NotImplementedError(f"Unknown aggregation_mode: {self.aggregation_mode}")

        self.last_ptfa_stats = {}
        self.last_query_aggregation_weights = None
        self.last_s1_ptfa_evidence_stats = {}
        self.last_feature_refinement_stats = {}
        self.last_reliability_gate_stats = {}
        self.last_reliability_gate_weights = None
        self.last_consensus_residual_gate_stats = {}

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

        grids_batched = torch.cat(grids_list, dim=0).contiguous()
        feat_batched = torch.cat(feature_maps_list, dim=0).contiguous()

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

    def _vectorized_grid_sample_fmt(self, view_features, points_mm: torch.Tensor):
        """Sample view features with FMT-SimGen trunk-local mm projection."""
        return self._vectorized_grid_sample_fmt_pack(
            view_features, self._fmt_projection_pack(points_mm)
        )

    def _vectorized_grid_sample_fmt_pack(
        self,
        view_features,
        projection_pack: dict[str, torch.Tensor],
    ):
        """Sample view features with a precomputed FMT-SimGen projection pack."""
        B = projection_pack["grid"].shape[0]
        grids_list = []
        feature_maps_list = []
        for v_idx, view_name in enumerate(self.view_list):
            if view_name not in view_features:
                continue
            grid = projection_pack["grid"][:, :, v_idx, :]
            grids_list.append(grid.unsqueeze(1))

            feat_map = view_features[view_name]
            if self.transpose_feature_map_for_sampling:
                feat_map = feat_map.permute(0, 1, 3, 2)
            feature_maps_list.append(feat_map)

        grids_batched = torch.cat(grids_list, dim=0)
        feat_batched = torch.cat(feature_maps_list, dim=0)

        sampled_feats = F.grid_sample(
            feat_batched,
            grids_batched,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sampled_feats = sampled_feats.squeeze(2).permute(0, 2, 1)

        f_list = []
        for v_idx in range(self.num_views):
            f_v = sampled_feats[v_idx * B : (v_idx + 1) * B]
            f_list.append(f_v)

        f = torch.stack(f_list, dim=2)

        # Keep invalid FOV samples exactly zero after interpolation.
        valid = projection_pack["valid"].unsqueeze(-1).to(dtype=f.dtype)
        return f * valid

    def _fmt_projection_grids(self, points_mm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return FMT-SimGen projection grids and valid masks in view_list order."""
        pack = self._fmt_projection_pack(points_mm)
        return pack["grid"], pack["valid"]

    def _fmt_projection_pack(self, points_mm: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return FMT-SimGen projection tensors in view_list order."""
        grids = []
        depths = []
        valid_masks = []
        uv_px_list = []
        uv_phys_list = []
        for view_name in self.view_list:
            grid, depth, valid_mask, uv_px, uv_phys = project_points_mm_to_detector(
                points_mm,
                angle_deg=int(view_name),
                camera_distance_mm=self.camera_distance,
                fov_mm=self.fov_mm,
                detector_resolution=self.detector_resolution,
                volume_center_world=self.volume_center_world,
                align_corners=True,
            )
            grids.append(grid)
            depths.append(depth)
            valid_masks.append(valid_mask)
            uv_px_list.append(uv_px)
            uv_phys_list.append(uv_phys)
        return {
            "grid": torch.stack(grids, dim=2),
            "depth": torch.stack(depths, dim=2),
            "valid": torch.stack(valid_masks, dim=2),
            "uv_px": torch.stack(uv_px_list, dim=2),
            "uv_phys": torch.stack(uv_phys_list, dim=2),
        }

    def _view_feature_dict_to_tensor(self, view_features) -> torch.Tensor:
        """Pack view feature dict into [B,V,C,H,W] in view_list order."""
        feature_maps = []
        for view_name in self.view_list:
            feat_map = view_features[view_name]
            if self.transpose_feature_map_for_sampling:
                feat_map = feat_map.permute(0, 1, 3, 2)
            feature_maps.append(feat_map)
        return torch.stack(feature_maps, dim=1)

    def _ptfa_sample_fixed_gaussian(self, view_features, points_mm: torch.Tensor):
        """Sample view features with fixed Gaussian PTFA in detector feature-map space."""
        return self._ptfa_sample_fixed_gaussian_pack(
            view_features, self._fmt_projection_pack(points_mm)
        )

    def _ptfa_sample_fixed_gaussian_pack(
        self,
        view_features,
        projection_pack: dict[str, torch.Tensor],
    ):
        """Sample view features with fixed Gaussian PTFA using a projection pack."""
        center_grid = projection_pack["grid"]
        valid_mask = projection_pack["valid"]
        feature_map = self._view_feature_dict_to_tensor(view_features)
        return ptfa_sample_fixed_gaussian(
            feature_map,
            center_grid,
            valid_mask,
            window=self.ptfa_window,
            sigma_px=self.ptfa_sigma_px,
        )

    def _ptfa_sample_exit_depth_gaussian(
        self,
        view_features,
        points_mm: torch.Tensor,
        depth_maps: torch.Tensor,
        query_depth: torch.Tensor,
    ):
        """Sample view features with exit-depth-dependent Gaussian PTFA."""
        center_grid, valid_mask = self._fmt_projection_grids(points_mm)
        feature_map = self._view_feature_dict_to_tensor(view_features)
        sampled, _stats = ptfa_sample_exit_depth_gaussian(
            feature_map,
            center_grid,
            valid_mask,
            depth_maps,
            query_depth,
            sigma_min=self.ptfa_sigma_min,
            sigma_max=self.ptfa_sigma_max,
            exit_depth_max=self.ptfa_exit_depth_max_mm,
            window=self.ptfa_window,
        )
        return sampled

    def _ptfa_sample_corrected_exit_depth_gaussian(
        self,
        view_features,
        projection_pack: dict[str, torch.Tensor],
        depth_maps: torch.Tensor,
    ):
        """Sample view features with corrected exit-depth-dependent Gaussian PTFA."""
        feature_map = self._view_feature_dict_to_tensor(view_features)
        sampled, stats = ptfa_sample_corrected_exit_depth_gaussian(
            feature_map,
            projection_pack["grid"],
            projection_pack["valid"],
            depth_maps,
            projection_pack["depth"],
            sigma_min=self.ptfa_sigma_min,
            sigma_max=self.ptfa_sigma_max,
            exit_depth_max=self.ptfa_exit_depth_max_mm,
            window=self.ptfa_window,
            invert_depth=self.ptfa_invert_depth,
        )
        return sampled, stats

    def _build_query_geometry_features(
        self,
        projection_pack: dict[str, torch.Tensor],
        ptfa_stats: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Build geometry-only gate features [B,N,V,9]."""
        grid = projection_pack["grid"]
        dtype = grid.dtype
        device = grid.device
        grid_x = grid[..., 0]
        grid_y = grid[..., 1]
        boundary_margin = (1.0 - torch.maximum(grid_x.abs(), grid_y.abs())).clamp(0.0, 1.0)
        depth_eff_norm = (
            ptfa_stats["depth_eff_mm"] / float(self.ptfa_exit_depth_max_mm)
        ).clamp(0.0, 1.0)
        denom = max(self.ptfa_sigma_max - self.ptfa_sigma_min, 1.0e-12)
        sigma_norm = ((ptfa_stats["sigma_px"] - self.ptfa_sigma_min) / denom).clamp(0.0, 1.0)

        angles = torch.as_tensor(
            [float(v) for v in self.view_list], device=device, dtype=dtype
        ) * (torch.pi / 180.0)
        angle_feat = torch.stack(
            [
                torch.sin(angles),
                torch.cos(angles),
                torch.sin(2.0 * angles),
                torch.cos(2.0 * angles),
            ],
            dim=-1,
        ).view(1, 1, self.num_views, 4)
        angle_feat = angle_feat.expand(grid.shape[0], grid.shape[1], -1, -1)
        geom = torch.cat(
            [
                grid_x.unsqueeze(-1),
                grid_y.unsqueeze(-1),
                boundary_margin.unsqueeze(-1),
                depth_eff_norm.unsqueeze(-1),
                sigma_norm.unsqueeze(-1),
                angle_feat,
            ],
            dim=-1,
        )
        return torch.nan_to_num(geom, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_reliability_geometry_features(
        self,
        projection_pack: dict[str, torch.Tensor],
        ptfa_stats: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Build geometry/depth-only reliability features [B,N,V,G]."""
        grid = projection_pack["grid"]
        dtype = grid.dtype
        device = grid.device
        grid_x = grid[..., 0]
        grid_y = grid[..., 1]
        boundary_margin = (1.0 - torch.maximum(grid_x.abs(), grid_y.abs())).clamp(0.0, 1.0)
        center_distance = torch.sqrt((grid_x.square() + grid_y.square()).clamp_min(0.0))
        raw_depth_like_norm = (
            ptfa_stats["raw_depth_like_mm"] / float(self.ptfa_exit_depth_max_mm)
        ).clamp(0.0, 1.0)
        depth_eff_norm = (
            ptfa_stats["depth_eff_mm"] / float(self.ptfa_exit_depth_max_mm)
        ).clamp(0.0, 1.0)
        denom = max(self.ptfa_sigma_max - self.ptfa_sigma_min, 1.0e-12)
        sigma_norm = ((ptfa_stats["sigma_px"] - self.ptfa_sigma_min) / denom).clamp(0.0, 1.0)
        valid_float = projection_pack["valid"].to(dtype=dtype)

        angles = torch.as_tensor(
            [float(v) for v in self.view_list], device=device, dtype=dtype
        ) * (torch.pi / 180.0)
        angle_feat = torch.stack(
            [
                torch.sin(angles),
                torch.cos(angles),
                torch.sin(2.0 * angles),
                torch.cos(2.0 * angles),
            ],
            dim=-1,
        ).view(1, 1, self.num_views, 4)
        angle_feat = angle_feat.expand(grid.shape[0], grid.shape[1], -1, -1)
        if getattr(self, "reliability_gate_geom_set", "full") == "compact":
            geom = torch.cat(
                [
                    grid_x.unsqueeze(-1),
                    grid_y.unsqueeze(-1),
                    boundary_margin.unsqueeze(-1),
                    depth_eff_norm.unsqueeze(-1),
                    sigma_norm.unsqueeze(-1),
                    angle_feat[..., 0:2],
                    valid_float.unsqueeze(-1),
                ],
                dim=-1,
            )
        else:
            geom = torch.cat(
                [
                    grid_x.unsqueeze(-1),
                    grid_y.unsqueeze(-1),
                    boundary_margin.unsqueeze(-1),
                    raw_depth_like_norm.unsqueeze(-1),
                    depth_eff_norm.unsqueeze(-1),
                    sigma_norm.unsqueeze(-1),
                    valid_float.unsqueeze(-1),
                    center_distance.unsqueeze(-1),
                    angle_feat,
                ],
                dim=-1,
            )
        return torch.nan_to_num(geom, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_consensus_residual_gate_input(
        self,
        f_view: torch.Tensor,
        f_mean: torch.Tensor,
        projection_pack: dict[str, torch.Tensor],
        ptfa_stats: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Build [B,N,V,D] compact geometry plus evidence stats for E9c."""
        grid = projection_pack["grid"]
        dtype = f_view.dtype
        device = f_view.device
        grid_x = grid[..., 0].to(device=device, dtype=dtype)
        grid_y = grid[..., 1].to(device=device, dtype=dtype)
        boundary_margin = (1.0 - torch.maximum(grid_x.abs(), grid_y.abs())).clamp(0.0, 1.0)
        depth_eff_norm = (
            ptfa_stats["depth_eff_mm"].to(device=device, dtype=dtype)
            / float(self.ptfa_exit_depth_max_mm)
        ).clamp(0.0, 1.0)
        denom = max(self.ptfa_sigma_max - self.ptfa_sigma_min, 1.0e-12)
        sigma_norm = (
            (
                ptfa_stats["sigma_px"].to(device=device, dtype=dtype)
                - float(self.ptfa_sigma_min)
            )
            / denom
        ).clamp(0.0, 1.0)
        valid_float = projection_pack["valid"].to(device=device, dtype=dtype)
        angles = torch.as_tensor(
            [float(v) for v in self.view_list], device=device, dtype=dtype
        ) * (torch.pi / 180.0)
        angle_feat = torch.stack([torch.sin(angles), torch.cos(angles)], dim=-1).view(
            1, 1, self.num_views, 2
        )
        angle_feat = angle_feat.expand(grid.shape[0], grid.shape[1], -1, -1)
        compact_geom = torch.cat(
            [
                grid_x.unsqueeze(-1),
                grid_y.unsqueeze(-1),
                boundary_margin.unsqueeze(-1),
                depth_eff_norm.unsqueeze(-1),
                sigma_norm.unsqueeze(-1),
                angle_feat,
                valid_float.unsqueeze(-1),
            ],
            dim=-1,
        )

        if not getattr(self, "consensus_residual_gate_use_evidence_stats", True):
            return torch.nan_to_num(compact_geom, nan=0.0, posinf=0.0, neginf=0.0)

        C = max(f_view.shape[-1], 1)
        delta = f_view - f_mean.unsqueeze(2)
        ptfa_l2_norm = f_view.norm(dim=-1) / (float(C) ** 0.5)
        ptfa_abs_mean = f_view.abs().mean(dim=-1)
        ptfa_delta_norm = delta.norm(dim=-1) / (float(C) ** 0.5)
        f_norm = f_view.norm(dim=-1)
        mean_norm = f_mean.norm(dim=-1).unsqueeze(2)
        ptfa_cos_to_mean = (f_view * f_mean.unsqueeze(2)).sum(dim=-1) / (
            f_norm * mean_norm
        ).clamp_min(1.0e-6)
        evidence = torch.stack(
            [ptfa_l2_norm, ptfa_abs_mean, ptfa_delta_norm, ptfa_cos_to_mean],
            dim=-1,
        )
        gate_input = torch.cat([compact_geom, evidence], dim=-1)
        return torch.nan_to_num(gate_input, nan=0.0, posinf=0.0, neginf=0.0)

    def _reliability_gate_stats(
        self,
        weights: torch.Tensor,
        projection_pack: dict[str, torch.Tensor],
        ptfa_stats: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Summarize reliability gate behavior for smoke/debug reporting."""
        valid = projection_pack["valid"].to(dtype=weights.dtype, device=weights.device)
        has_valid = projection_pack["valid"].any(dim=-1)
        valid_weights = weights[projection_pack["valid"]]
        if valid_weights.numel() == 0:
            valid_weights = weights.new_zeros(1)
        valid_count = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
        uniform = torch.where(
            has_valid.unsqueeze(-1),
            valid / valid_count,
            torch.zeros_like(valid),
        )
        uniform_diff = (weights - uniform).abs()[projection_pack["valid"]]
        if uniform_diff.numel() == 0:
            uniform_diff = weights.new_zeros(1)
        entropy = -(weights * torch.log(weights.clamp_min(1.0e-12))).sum(dim=-1)
        entropy_valid = entropy[has_valid]
        if entropy_valid.numel() == 0:
            entropy_valid = entropy.new_zeros(1)
        depth_eff_norm = (
            ptfa_stats["depth_eff_mm"] / float(self.ptfa_exit_depth_max_mm)
        ).clamp(0.0, 1.0)
        sigma_px = ptfa_stats["sigma_px"]
        return {
            "reliability_weights_mean": valid_weights.detach().mean(),
            "reliability_weights_std": valid_weights.detach().std(unbiased=False),
            "reliability_weights_min": valid_weights.detach().min(),
            "reliability_weights_max": valid_weights.detach().max(),
            "reliability_weights_entropy": entropy_valid.detach().mean(),
            "reliability_uniform_absdiff_mean": uniform_diff.detach().mean(),
            "reliability_max_weight_mean": weights.max(dim=-1).values[has_valid].detach().mean(),
            "valid_weight_sum_mean": weights.sum(dim=-1)[has_valid].detach().mean(),
            "invalid_weight_max": (weights * (1.0 - valid)).detach().amax(),
            "valid_view_count_mean": valid.sum(dim=-1).detach().mean(),
            "depth_eff_norm_mean": depth_eff_norm.detach().mean(),
            "depth_eff_norm_std": depth_eff_norm.detach().std(unbiased=False),
            "sigma_px_mean": sigma_px.detach().mean(),
            "sigma_px_std": sigma_px.detach().std(unbiased=False),
            "sigma_px_min": sigma_px.detach().min(),
            "sigma_px_max": sigma_px.detach().max(),
        }

    @staticmethod
    def _valid_masked_mean(f: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Mean over views, excluding invalid projections."""
        weights = valid.unsqueeze(-1).to(dtype=f.dtype, device=f.device)
        denom = weights.sum(dim=2).clamp_min(1.0)
        return (f * weights).sum(dim=2) / denom

    def _geometry_summary(self, projection_pack: dict[str, torch.Tensor]) -> torch.Tensor:
        """Build [B,N,5] geometry summary for residual scorer side evidence."""
        grid = projection_pack["grid"]
        valid = projection_pack["valid"].to(dtype=grid.dtype)
        valid_count = valid.sum(dim=2).clamp_min(1.0)
        valid_ratio = valid.mean(dim=2)

        boundary = (1.0 - torch.maximum(grid[..., 0].abs(), grid[..., 1].abs())).clamp(0.0, 1.0)
        boundary_valid = boundary * valid
        mean_boundary = boundary_valid.sum(dim=2) / valid_count
        min_boundary = torch.where(
            projection_pack["valid"],
            boundary,
            torch.ones_like(boundary),
        ).amin(dim=2)
        min_boundary = torch.where(valid_count > 0, min_boundary, torch.zeros_like(min_boundary))

        depth_norm = (projection_pack["depth"].squeeze(-1) / float(self.camera_distance)).clamp(
            0.0, 1.0
        )
        depth_valid = depth_norm * valid
        mean_depth = depth_valid.sum(dim=2) / valid_count
        var_depth = ((depth_norm - mean_depth.unsqueeze(-1)).square() * valid).sum(
            dim=2
        ) / valid_count
        std_depth = torch.sqrt(var_depth.clamp_min(0.0))

        summary = torch.stack(
            [valid_ratio, mean_boundary, min_boundary, mean_depth, std_depth], dim=-1
        )
        return torch.nan_to_num(summary, nan=0.0, posinf=0.0, neginf=0.0)

    def _s1_ptfa_evidence(
        self,
        features_s1_dict,
        projection_pack: dict[str, torch.Tensor],
        depth_maps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Aggregate s1 PTFA evidence with a valid-view mean for side evidence paths."""
        if self.ptfa_mode == "fixed_gaussian":
            s1_ptfa = self._ptfa_sample_fixed_gaussian_pack(features_s1_dict, projection_pack)
            self.last_ptfa_stats = {}
        elif self.ptfa_mode == "corrected_exit_depth_gaussian":
            if depth_maps is None:
                raise ValueError(
                    "model.ptfa.mode=corrected_exit_depth_gaussian requires batch depth_maps"
                )
            s1_ptfa, ptfa_stats = self._ptfa_sample_corrected_exit_depth_gaussian(
                features_s1_dict, projection_pack, depth_maps
            )
            self.last_ptfa_stats = {k: v.detach() for k, v in ptfa_stats.items()}
        else:
            raise NotImplementedError(f"Unsupported s1 PTFA evidence mode: {self.ptfa_mode}")

        if self.feature_refinement_ptfa_view_aggregation == "masked_mean":
            s1_evidence = self._valid_masked_mean(s1_ptfa, projection_pack["valid"])
            self.last_reliability_gate_stats = {}
            self.last_reliability_gate_weights = None
            self.last_consensus_residual_gate_stats = {}
        elif self.feature_refinement_ptfa_view_aggregation == "reliability_gate":
            if self.ptfa_mode != "corrected_exit_depth_gaussian":
                raise ValueError("reliability_gate aggregation requires corrected_exit_depth_gaussian")
            geom_view = self._build_reliability_geometry_features(projection_pack, ptfa_stats)
            s1_evidence, weights = self.reliability_view_gate(
                s1_ptfa, geom_view, projection_pack["valid"]
            )
            self.last_reliability_gate_weights = weights.detach()
            self.last_reliability_gate_stats = self._reliability_gate_stats(
                weights, projection_pack, ptfa_stats
            )
            self.last_consensus_residual_gate_stats = {}
        elif self.feature_refinement_ptfa_view_aggregation == "consensus_residual_gate":
            if self.ptfa_mode != "corrected_exit_depth_gaussian":
                raise ValueError(
                    "consensus_residual_gate aggregation requires corrected_exit_depth_gaussian"
                )
            f_mean = self._valid_masked_mean(s1_ptfa, projection_pack["valid"])
            gate_input = self._build_consensus_residual_gate_input(
                s1_ptfa, f_mean, projection_pack, ptfa_stats
            )
            s1_evidence, consensus_stats = self.consensus_residual_gate(
                s1_ptfa, gate_input, projection_pack["valid"]
            )
            self.last_consensus_residual_gate_stats = {
                k: v.detach() if torch.is_tensor(v) else v
                for k, v in consensus_stats.items()
                if k != "confidence"
            }
            self.last_consensus_residual_gate_stats["confidence"] = consensus_stats[
                "confidence"
            ].detach()
            self.last_reliability_gate_stats = {}
            self.last_reliability_gate_weights = None
        else:
            raise NotImplementedError(
                "Unknown feature_refinement.ptfa_view_aggregation: "
                f"{self.feature_refinement_ptfa_view_aggregation}"
            )
        self.last_s1_ptfa_evidence_stats = {
            "mean": s1_evidence.detach().mean(),
            "std": s1_evidence.detach().std(),
            "norm": s1_evidence.detach().norm(),
        }
        return s1_evidence

    def _residual_scorer_input(
        self,
        gamma_x: torch.Tensor,
        s3_query_feat: torch.Tensor,
        features_s1_dict,
        projection_pack: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        if self.residual_scorer_input_mode == "bilinear_s3":
            self.last_s1_ptfa_evidence_stats = {}
            return torch.cat([gamma_x, s3_query_feat], dim=-1)

        if self.residual_scorer_input_mode != "bilinear_s3_plus_s1_ptfa":
            raise NotImplementedError(
                f"Unknown residual_scorer.input_mode: {self.residual_scorer_input_mode}"
            )
        if projection_pack is None:
            raise ValueError("bilinear_s3_plus_s1_ptfa requires FMT-SimGen projection_pack")

        s1_evidence = self._s1_ptfa_evidence(features_s1_dict, projection_pack)
        geom_summary = self._geometry_summary(projection_pack).to(dtype=s1_evidence.dtype)
        return torch.cat([gamma_x, s3_query_feat, s1_evidence, geom_summary], dim=-1)

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

    def forward(
        self,
        view_projections,
        x3d,
        gamma_x=None,
        bg_guidance=None,
        points_mm=None,
        depth_maps=None,
    ):
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

        projection_pack = None
        if points_mm is not None and self.use_fmt_simgen_projection:
            projection_pack = self._fmt_projection_pack(points_mm)
            # FMT-SimGen depth cue is view-independent here: use 0° camera depth.
            zero_view_idx = self.view_list.index(0) if 0 in self.view_list else 0
            depth = projection_pack["depth"][:, :, zero_view_idx, :]
            query_depth_all_views = None
            if self.ptfa_enabled and self.ptfa_mode == "exit_depth_gaussian":
                query_depth_all_views = projection_pack["depth"]
        else:
            # depth: [B, N, 1] (approx 0~camera_distance)
            depth = self._compute_depth(x3d)
            query_depth_all_views = None

        # 多尺度点采样
        f_s3_for_residual = None
        if points_mm is not None and self.use_fmt_simgen_projection:
            if self.residual_scorer_enabled:
                # The residual scorer ablation is defined on bilinear s3 features.
                # Keep this input fixed even when the fusion s3 path uses PTFA.
                f_s3_for_residual = self._vectorized_grid_sample_fmt_pack(
                    features_s3_dict, projection_pack
                )

            if self.aggregation_mode == "corrected_exit_ptfa_geom_gate":
                if not (
                    self.ptfa_enabled
                    and self.ptfa_mode == "corrected_exit_depth_gaussian"
                    and "s3" in self.ptfa_scales
                ):
                    raise ValueError(
                        "aggregation_mode=corrected_exit_ptfa_geom_gate requires "
                        "model.ptfa.enabled=true, mode=corrected_exit_depth_gaussian, "
                        "and scales containing 's3'"
                    )
                if depth_maps is None:
                    raise ValueError(
                        "model.ptfa.mode=corrected_exit_depth_gaussian requires batch depth_maps"
                    )
                assert projection_pack is not None
                f_s3, ptfa_stats = self._ptfa_sample_corrected_exit_depth_gaussian(
                    features_s3_dict, projection_pack, depth_maps
                )
                g_s3 = self.proj_s3(f_s3)
                geom = self._build_query_geometry_features(projection_pack, ptfa_stats)
                fused_feat, agg_weights = self.query_view_gate(
                    g_s3, geom, projection_pack["valid"]
                )
                self.last_ptfa_stats = {k: v.detach() for k, v in ptfa_stats.items()}
                self.last_query_aggregation_weights = agg_weights.detach()

                if gamma_x is None:
                    gamma_x = self._simple_pos_encoding(x3d, self.pos_enc_dim)
                logits = self.density_head(gamma_x, fused_feat)
                if self.residual_scorer_enabled:
                    residual_s3 = f_s3_for_residual
                    s3_query_feat = self.proj_s3(residual_s3).mean(dim=2)
                    scorer_input = self._residual_scorer_input(
                        gamma_x,
                        s3_query_feat,
                        features_s1_dict,
                        projection_pack,
                    )
                    residual = self.residual_scorer(scorer_input)
                    logits = logits + self.residual_scorer_lambda * residual

                if getattr(self, "enable_background", False):
                    extra = {"query_view_weights": agg_weights}
                    fused_feat_bg = fused_feat
                    if bool(self.bg_guidance_cfg["enable"]):
                        if bg_guidance is None:
                            raise ValueError(
                                "Background guidance is enabled by config, but "
                                "bg_guidance is None. "
                                "Please pass bg_guidance=[B,G] (or [B,N,G])."
                            )
                        fused_feat_bg, stats = self.bg_guidance_interaction(
                            fused_feat_bg, bg_guidance
                        )
                        extra.update({f"bg_guidance/{k}": v for k, v in stats.items()})

                    background_logits = self.background_head(gamma_x, fused_feat_bg)
                    extra["background_logits"] = background_logits
                    return logits, views_no_projections, extra

                return logits, views_no_projections

            f_s1 = self._vectorized_grid_sample_fmt_pack(features_s1_dict, projection_pack)
            f_s2 = self._vectorized_grid_sample_fmt_pack(features_s2_dict, projection_pack)
            if (
                self.ptfa_enabled
                and self.ptfa_mode == "exit_depth_gaussian"
                and "s3" in self.ptfa_scales
            ):
                if depth_maps is None:
                    raise ValueError(
                        "model.ptfa.mode=exit_depth_gaussian requires batch depth_maps"
                    )
                f_s3 = self._ptfa_sample_exit_depth_gaussian(
                    features_s3_dict, points_mm, depth_maps, query_depth_all_views
                )
            elif (
                self.ptfa_enabled
                and self.ptfa_mode == "fixed_gaussian"
                and "s3" in self.ptfa_scales
            ):
                f_s3 = self._ptfa_sample_fixed_gaussian(features_s3_dict, points_mm)
            else:
                f_s3 = (
                    f_s3_for_residual
                    if f_s3_for_residual is not None
                    else self._vectorized_grid_sample_fmt_pack(features_s3_dict, projection_pack)
                )
        else:
            f_s1 = self._vectorized_grid_sample(features_s1_dict, x3d)  # [B, N, V, C]
            f_s2 = self._vectorized_grid_sample(features_s2_dict, x3d)  # [B, N, V, C2]
            f_s3 = self._vectorized_grid_sample(features_s3_dict, x3d)  # [B, N, V, C3]
            f_s3_for_residual = f_s3

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

        if self.feature_refinement_enabled:
            if projection_pack is None:
                raise ValueError("feature_refinement requires FMT-SimGen points_mm projection_pack")
            if not (
                self.ptfa_enabled
                and self.ptfa_mode in {"fixed_gaussian", "corrected_exit_depth_gaussian"}
                and "s1" in self.ptfa_scales
            ):
                raise ValueError(
                    "feature_refinement requires model.ptfa.enabled=true, "
                    "mode=fixed_gaussian or corrected_exit_depth_gaussian, "
                    "and scales containing 's1'"
                )
            s1_evidence = self._s1_ptfa_evidence(features_s1_dict, projection_pack, depth_maps)
            geom_summary = self._geometry_summary(projection_pack).to(dtype=fused_feat.dtype)
            refined_feat = self.feature_refinement(fused_feat, s1_evidence, geom_summary)
            self.last_feature_refinement_stats = {
                "base_mean": fused_feat.detach().mean(),
                "ptfa_mean": s1_evidence.detach().mean(),
                "delta_norm": (refined_feat.detach() - fused_feat.detach()).norm(),
                "refined_norm": refined_feat.detach().norm(),
            }
            fused_feat = refined_feat
        else:
            self.last_feature_refinement_stats = {}

        # 隐式密度预测
        logits = self.density_head(gamma_x, fused_feat)
        if self.residual_scorer_enabled:
            residual_s3 = f_s3_for_residual if f_s3_for_residual is not None else f_s3
            s3_query_feat = self.proj_s3(residual_s3).mean(dim=2)
            scorer_input = self._residual_scorer_input(
                gamma_x,
                s3_query_feat,
                features_s1_dict,
                projection_pack,
            )
            residual = self.residual_scorer(scorer_input)
            logits = logits + self.residual_scorer_lambda * residual

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
