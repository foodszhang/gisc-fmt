"""
多视图投影数据集 - 所有参数从配置读取

此模块不再包含硬编码的默认值，所有参数都必须从配置中读取。
缺失的配置项在入口处统一报错处理。
"""

import torch
import os
import numpy as np
import json

from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from scipy.ndimage import zoom


class MultiProjDataset(Dataset):
    """
    多视图投影数据集 - 所有参数从配置读取

    支持通过配置限制数据集大小（固定 seed 选择子集），保证同一次训练中数据一致。
    """

    def __init__(
        self,
        data_dir,
        block_dir,
        config=None,
        device="cpu",
        is_training=True,
        split: str = "train",
    ):
        """
        初始化数据集

        Args:
            data_dir: 数据目录
            block_dir: 块数据目录
            config: ConfigHelper对象或配置字典 (不再接受None)
            device: 计算设备
            is_training: 是否为训练模式
        """
        super().__init__()
        self.data_dir = data_dir
        self.block_dir = block_dir
        self.device = device
        self.is_training = is_training
        self.split = split

        # 提取配置参数（必须存在，否则会报错）
        self._extract_config(config)
        self.sample_num = self.config["sample_num"]

        subset_seed = int(self.config.get("subset_seed", 0) or 0)
        split_offset = {"train": 0, "val": 1000, "test": 2000}.get(self.split, 0)
        self._base_seed = subset_seed + split_offset

        # 加载块文件
        self.block_files = [
            os.path.join(block_dir, f) for f in os.listdir(block_dir) if f.endswith(".npz")
        ]
        self.block_files.sort(key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))

        # 加载体素数据 - 从配置读取文件名
        voxel_file = self.config.get("voxel_file")
        voxel_path = os.path.join(data_dir, voxel_file)
        volume_voxel = np.load(voxel_path)
        self.origin_voxel_shape = volume_voxel.shape
        self.global_voxel_shape = volume_voxel.shape

        # 从配置提取体素范围
        voxel_ranges = self.config["voxel_ranges"]
        self.range_x = tuple(voxel_ranges["x"])
        self.range_y = tuple(voxel_ranges["y"])
        self.range_z = tuple(voxel_ranges["z"])

        # 计算可行体素形状
        self.feasible_voxel_shape = (
            self.range_x[1] - self.range_x[0],
            self.range_y[1] - self.range_y[0],
            self.range_z[1] - self.range_z[0],
        )

        # 生成点坐标网格
        points = np.mgrid[
            self.range_x[0] : self.range_x[1],
            self.range_y[0] : self.range_y[1],
            self.range_z[0] : self.range_z[1],
        ]
        points = points.reshape(3, -1)
        self.points = points.transpose(1, 0)  # N, 3

        # 扫描数据目录
        self._scan_data_directory()

    def _extract_config(self, config):
        """从配置对象中提取参数"""
        if config is None:
            raise ValueError(
                "配置不能为None。请通过config参数传递配置对象。\n所有参数都必须从配置文件中读取。"
            )

        from omegaconf import DictConfig, OmegaConf

        # 情况1: OmegaConf DictConfig (Hydra config)
        if isinstance(config, DictConfig):
            # 检查是否包含 'data' 字段（完整Hydra config）
            if "data" in config:
                self.config = OmegaConf.to_container(config.data, resolve=True)
            else:
                # 直接是数据配置
                self.config = OmegaConf.to_container(config, resolve=True)
        # 情况2: 普通字典
        elif isinstance(config, dict):
            if "dataset" in config:
                self.config = config["dataset"]
            elif "data" in config:
                self.config = config["data"]
            else:
                self.config = config
        # 情况3: ConfigHelper对象或其他
        else:
            self.config = self._extract_from_helper(config)

    def _extract_from_helper(self, config_obj):
        """从ConfigHelper对象提取配置"""
        return {
            "max_voxels": config_obj.max_voxels,
            "projection_scale": config_obj.projection_scale,
            "no_projection_scale": config_obj.no_projection_scale,
            "view_angles": config_obj.view_angles,
            "voxel_ranges": config_obj.voxel_ranges,
            "voxel_file": config_obj.dataset_config.get("voxel_file", "volume_brain.npy"),
            "use_importance_sampling": config_obj.dataset_config.get(
                "use_importance_sampling", False
            ),
            "sample_num": config_obj.sample_num,
        }

    def _scan_data_directory(self):
        """扫描数据目录"""
        entries = os.listdir(self.data_dir)

        self.total_num = 0
        self.dirs = []

        for entry in entries:
            entry_path = os.path.join(self.data_dir, entry)

            # 检查是否是文件夹且名称只包含数字
            if os.path.isdir(entry_path) and entry.isdigit():
                self.total_num += 1
                self.dirs.append((entry, entry_path))

        # Ensure deterministic order
        self.dirs.sort(key=lambda x: int(x[0]))

        # Optional dataset size control (deterministic subset)
        max_key = f"{self.split}_max_samples"
        max_samples = self.config.get(max_key, self.config.get("max_samples"))
        if max_samples is not None:
            max_samples = int(max_samples)
            if 0 < max_samples < len(self.dirs):
                policy = str(self.config.get("subset_policy", "first"))
                if policy == "random":
                    rng = np.random.default_rng(self._base_seed)
                    idx = rng.permutation(len(self.dirs))[:max_samples]
                    idx = sorted(int(i) for i in idx)
                    self.dirs = [self.dirs[i] for i in idx]
                else:
                    self.dirs = self.dirs[:max_samples]

        self.total_num = len(self.dirs)

    def __len__(self):
        return self.total_num

    def __getitem__(self, index):
        """获取数据项"""
        # Deterministic per-index RNG (keeps samples consistent across epochs/workers)
        rng = np.random.default_rng(self._base_seed + int(index))

        # 选择一个块（可复现）
        block_idx = int(rng.integers(len(self.block_files)))
        block_path = self.block_files[block_idx]
        data = np.load(block_path)
        coords = data["coords"]

        # 获取数据路径
        entry, entry_path = self.dirs[index]
        proj_path = os.path.join(entry_path, "proj.npz")
        dep_path = os.path.join(entry_path, "dep_proj.npz")
        no_proj_path = os.path.join(entry_path, "no_proj.npz")
        json_path = os.path.join(entry_path, f"{entry}.json")

        # 加载JSON元数据
        json_file = json.load(open(json_path))
        source_pos = json_file["Optode"]["Source"]["Pos"]
        source_pattern = json_file["Optode"]["Source"]["Pattern"]
        source_data = np.fromfile(
            os.path.join(entry_path, source_pattern["Data"]), dtype=np.float32
        )
        source_shape = json_file["Optode"]["Source"]["Param1"]
        source_data = source_data.reshape(source_shape[2], source_shape[1], source_shape[0])
        # source_data = source_data.transpose(2, 1, 0)
        # source_pos = source_pos[::-1]  # 转换为z,y,x顺序

        # 构建源数据体积
        source_in_vol = np.zeros(self.origin_voxel_shape, dtype=np.float32)
        source_in_vol[
            source_pos[0] : source_pos[0] + source_data.shape[0],
            source_pos[1] : source_pos[1] + source_data.shape[1],
            source_pos[2] : source_pos[2] + source_data.shape[2],
        ] = source_data

        # 加载投影数据
        projection_zip = np.load(proj_path)
        no_projection_zip = np.load(no_proj_path)

        # 加载深度数据
        dep_zip = np.load(dep_path)

        # 从配置获取视图角度和缩放比例
        view_angles = self.config["view_angles"]
        projection_scale = self.config["projection_scale"]
        no_projection_scale = self.config["no_projection_scale"]

        # 转换为张量
        view_angles = [str(v) for v in view_angles]

        projections = {
            p: torch.tensor(
                projection_zip[p] / projection_scale,
                dtype=torch.float32,
                device=self.device,
            )
            for p in view_angles
        }
        no_projections = {
            p: torch.tensor(
                no_projection_zip[p] / no_projection_scale,
                dtype=torch.float32,
                device=self.device,
            )
            for p in view_angles
        }

        dep_projections = {
            p: torch.tensor(
                dep_zip[p],
                dtype=torch.float32,
                device=self.device,
            )
            for p in view_angles
        }

        # 获取点密度
        points = self.points.copy()
        point_densities = source_in_vol[
            points[:, 0].astype(int),
            points[:, 1].astype(int),
            points[:, 2].astype(int),
        ]
        point_densities = torch.tensor(point_densities, dtype=torch.float32, device=self.device)
        points = torch.tensor(points, dtype=torch.float32, device=self.device)
        points = points / (
            torch.asarray(self.origin_voxel_shape, device=self.device, dtype=torch.float32) - 1
        )

        projections_packed = torch.stack([projections[v] for v in view_angles], dim=0).unsqueeze(1)
        no_projections_packed = torch.stack(
            [no_projections[v] for v in view_angles], dim=0
        ).unsqueeze(1)
        dep_packed = torch.stack([dep_projections[v] for v in view_angles], dim=0).unsqueeze(1)

        return {
            "sample_id": entry,
            "projections": projections,
            "no_projections": no_projections,
            "dep_projections": dep_projections,
            "projections_packed": projections_packed,  # [V,1,H,W] -> collate => [B,V,1,H,W]
            "no_projections_packed": no_projections_packed,
            "dep_packed": dep_packed,
            "points": points,
            "point_densities": point_densities,
            "global_voxel_shape": self.global_voxel_shape,
            "feasible_voxel_shape": self.feasible_voxel_shape,
            "range_x": self.range_x,
            "range_y": self.range_y,
            "range_z": self.range_z,
        }

    def sample_points(self, points, values, rng=None):
        """采样点"""
        rng = rng or np.random.default_rng(self._base_seed)
        flat_values = values[
            points[:, 0].astype(int),
            points[:, 1].astype(int),
            points[:, 2].astype(int),
        ]
        sum_values = flat_values.sum()
        p = (flat_values + 0.05) / (sum_values + len(flat_values) * 0.05)

        choice = rng.choice(len(points), size=self.sample_num, replace=False)
        points = points[choice]

        if values is not None:
            value = values[
                points[:, 0].astype(int),
                points[:, 1].astype(int),
                points[:, 2].astype(int),
            ]
            return points, value
        else:
            return points
