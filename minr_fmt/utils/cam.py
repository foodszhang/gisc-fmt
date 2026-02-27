import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_projection_view_matrix(
    voxel_data, rotation_deg, camera_distance, detector_size, detector_resolution
):
    """
    使用矩阵运算优化的投影视图生成
    """
    width_pixels, height_pixels = detector_resolution
    width_phys, height_phys = detector_size

    projection = torch.zeros((height_pixels, width_pixels), dtype=torch.float32)
    depth_map = torch.full(
        (height_pixels, width_pixels), torch.inf, dtype=torch.float32
    )

    pixel_to_phys_x = width_phys / width_pixels
    pixel_to_phys_y = height_phys / height_pixels

    nx, ny, nz = voxel_data.shape

    # 收集所有非零体素的位置
    nonzero_indices = []
    values = []

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if voxel_data[i, j, k] != 0:
                    x = i - nx / 2 + 0.5
                    y = j - ny / 2 + 0.5
                    z = k - nz / 2 + 0.5
                    nonzero_indices.append([x, y, z])
                    values.append(voxel_data[i, j, k])

    if len(nonzero_indices) == 0:
        return projection, depth_map

    # 转换为numpy数组进行批量处理
    points = torch.asarray(nonzero_indices, dtype=torch.float32)
    values_array = torch.asarray(values, dtype=torch.float32)

    # 批量投影
    projections, depths = project_points_to_camera(
        points, rotation_deg, camera_distance, detector_size
    )

    # 处理每个投影点
    for idx in range(len(points)):
        u, v = projections[idx, 0], projections[idx, 1]
        depth_val = depths[idx, 0]
        visible = depths[idx, 1] > 0.5

        if visible:
            # 转换到像素坐标
            pixel_u = int((u + width_phys / 2) / pixel_to_phys_x)
            pixel_v = int((v + height_phys / 2) / pixel_to_phys_y)

            # 修正上下颠倒
            # pixel_v = height_pixels - 1 - pixel_v

            if 0 <= pixel_u < width_pixels and 0 <= pixel_v < height_pixels:
                if depth_val < depth_map[pixel_u, pixel_v]:
                    depth_map[pixel_u, pixel_v] = depth_val
                    projection[pixel_u, pixel_v] = values_array[idx]

    return projection, depth_map


def rotation_matrix_y(angle_deg, *, device=None, dtype=torch.float32):
    """绕Y轴旋转矩阵 (torch.compile-safe)"""
    angle = torch.as_tensor(angle_deg, device=device, dtype=dtype)
    angle_rad = torch.deg2rad(angle)
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)

    zero = torch.zeros((), device=angle.device, dtype=angle.dtype)
    one = torch.ones((), device=angle.device, dtype=angle.dtype)

    row0 = torch.stack([cos_a, zero, sin_a])
    row1 = torch.stack([zero, one, zero])
    row2 = torch.stack([-sin_a, zero, cos_a])
    return torch.stack([row0, row1, row2])


def world_to_camera_coords_batch_matrix(points, rotation_deg, camera_distance):
    """
    使用矩阵运算批量将世界坐标系中的点转换到相机坐标系

    参数:
    points: 批量点 [B, N, 3] 或 [N, 3]
    rotation_deg: 旋转角度
    camera_distance: 相机距离

    返回:
    camera_coords: 相机坐标系中的点 [B, N, 3] 或 [N, 3]
    """
    R = rotation_matrix_y(rotation_deg, device=points.device, dtype=points.dtype)

    if points.ndim == 3:
        B, N, _ = points.shape
        # 重塑为 [B*N, 3] 进行矩阵乘法
        points_flat = points.reshape(B * N, 3)
        camera_coords_flat = points_flat @ R.T  # 矩阵乘法
        # 重塑回 [B, N, 3]
        camera_coords = camera_coords_flat.reshape(B, N, 3)
    else:  # points.ndim == 2
        camera_coords = points @ R.T  # 直接矩阵乘法

    # 修正深度计算
    camera_coords[..., 2] = camera_distance - camera_coords[..., 2]

    return camera_coords


def project_points_to_camera(points, rotation_deg, camera_distance, detector_size):
    """
    使用矩阵运算批量投影点

    参数:
    points: 批量点 [B, N, 3] 或 [N, 3]
    rotation_deg: 旋转角度
    camera_distance: 相机距离
    detector_size: 探测器尺寸 (width, height)

    返回:
    projections: 投影坐标 [B, N, 2] 或 [N, 2]
    depths: 深度信息 [B, N, 2] 或 [N, 2] (深度值, 可见性标志)
    """
    # 使用矩阵运算转换坐标

    camera_coords = world_to_camera_coords_batch_matrix(
        points, rotation_deg, camera_distance
    )

    width, height = detector_size

    # 提取UV坐标和深度
    if points.ndim == 3:
        B, N, _ = points.shape
        u = camera_coords[:, :, 0]  # [B, N]
        v = camera_coords[:, :, 1]  # [B, N]
        depth_vals = camera_coords[:, :, 2]  # [B, N]

        # 计算可见性 (使用向量化操作)
        visible = (
            (torch.abs(u) <= width / 2)
            & (torch.abs(v) <= height / 2)
            & (depth_vals > 0)
        )

        # 构建投影坐标 [B, N, 2]
        projections = torch.zeros((B, N, 2), dtype=torch.float32, device=points.device)
        projections[:, :, 0] = u
        projections[:, :, 1] = v

        # 构建深度信息 [B, N, 2]
        depths = torch.zeros((B, N, 2), dtype=torch.float32, device=points.device)
        depths[:, :, 0] = depth_vals
        depths[:, :, 1] = visible.type(torch.float32)

    else:  # points.ndim == 2
        N, _ = points.shape
        u = camera_coords[:, 0]  # [N]
        v = camera_coords[:, 1]  # [N]
        depth_vals = camera_coords[:, 2]  # [N]

        # 计算可见性
        visible = (
            (torch.abs(u) <= width / 2)
            & (torch.abs(v) <= height / 2)
            & (depth_vals > 0)
        )

        # 构建投影坐标 [N, 2]
        projections = torch.zeros((N, 2), dtype=torch.float32, device=points.device)
        projections[:, 0] = u
        projections[:, 1] = v

        # 构建深度信息 [N, 2]
        depths = torch.zeros((N, 2), dtype=torch.float32, device=points.device)
        depths[:, 0] = depth_vals
        depths[:, 1] = visible.type(torch.float32)

    return projections, depths


class VolumeProjector:
    """体积数据投影器 - 使用矩阵运算优化"""

    def __init__(
        self, camera_distance=40, detector_size=(25, 25), detector_resolution=(200, 200)
    ):
        self.camera_distance = camera_distance
        self.detector_size = torch.asarray(detector_size, dtype=torch.float32)
        self.detector_resolution = torch.asarray(detector_resolution, dtype=torch.int32)

    def project_volume(self, volume_data, view_angles=None):
        """使用矩阵运算优化的体积投影"""
        if view_angles is None:
            view_angles = [0, 30, 60, 90, 120, 150, 180]

        projections = []
        depth_maps = []
        angles_list = []

        print(f"开始生成 {len(view_angles)} 个视角的投影 (矩阵运算优化)...")
        for angle in view_angles:
            print(f"  生成 {angle}° 视角投影...")
            print("666666", type(volume_data))
            proj, depth = generate_projection_view_matrix(
                volume_data,
                angle,
                self.camera_distance,
                self.detector_size,
                self.detector_resolution,
            )
            projections.append(proj)
            depth_maps.append(depth)
            angles_list.append(angle)

        return projections, depth_maps, angles_list

    def visualize_projections(self, volume_data, view_angles=None, figsize=(20, 8)):
        """可视化投影结果"""
        if view_angles is None:
            view_angles = [0, 30, 60, 90, 120, 150, 180]

        projections, depth_maps, angles_list = self.project_volume(
            volume_data, view_angles
        )

        n_views = len(view_angles)
        fig, axes = plt.subplots(2, n_views, figsize=figsize)

        if n_views == 1:
            axes = torch.asarray([[axes[0]], [axes[1]]])

        for idx, angle in enumerate(angles_list):
            projection = projections[idx]
            depth_map = depth_maps[idx]

            if n_views == 1:
                ax1 = axes[0, 0]
                ax2 = axes[1, 0]
            else:
                ax1 = axes[0, idx]
                ax2 = axes[1, idx]

            extent = torch.asarray(
                [
                    -self.detector_size[0] / 2,
                    self.detector_size[0] / 2,
                    -self.detector_size[1] / 2,
                    self.detector_size[1] / 2,
                ],
                dtype=np.float32,
            )

            im1 = ax1.imshow(projection, cmap="hot", extent=extent)
            ax1.set_title(f"投影视图 {angle}°")
            plt.colorbar(im1, ax=ax1, fraction=0.046)

            depth_display = depth_map.copy()
            depth_display[depth_display == np.inf] = 0
            im2 = ax2.imshow(depth_display, cmap="viridis", extent=extent)
            ax2.set_title(f"深度图 {angle}°")
            plt.colorbar(im2, ax=ax2, fraction=0.046)

        plt.tight_layout()
        plt.show()

        projections_dict = {}
        depth_maps_dict = {}
        for i, angle in enumerate(angles_list):
            projections_dict[angle] = projections[i]
            depth_maps_dict[angle] = depth_maps[i]

        return projections_dict, depth_maps_dict


def create_test_volume(nx=182, ny=164, nz=210, block_size=3):
    """
    创建含标记点的3D体素数据，在每个标记点周围生成小体素块
    block_size：体素块尺寸（如3表示3x3x3的立方体）
    """
    volume = torch.zeros((nx, ny, nz), dtype=torch.float32)

    # 定义标记点及其体素值（不同值便于区分）
    markers = [
        {"voxel_idx": (91, 82, 105), "value": 100.0, "label": "中心块"},
        {"voxel_idx": (10, 10, 10), "value": 80.0, "label": "左上块"},
        {"voxel_idx": (172, 154, 200), "value": 60.0, "label": "右下块"},
        {"voxel_idx": (91, 102, 105), "value": 40.0, "label": "上移块"},
    ]

    # 在每个标记点周围生成体素块（避免单点插值问题）
    for m in markers:
        i0, j0, k0 = m["voxel_idx"]
        val = m["value"]
        # 生成3x3x3的体素块（确保在网格范围内）
        for di in range(-block_size // 2, block_size // 2 + 1):
            for dj in range(-block_size // 2, block_size // 2 + 1):
                for dk in range(-block_size // 2, block_size // 2 + 1):
                    i = i0 + di
                    j = j0 + dj
                    k = k0 + dk
                    # 确保体素索引在有效范围内
                    if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
                        volume[i, j, k] = val  # 块内所有体素值相同

    # 计算每个标记点中心的世界坐标（用于投影）
    for m in markers:
        i, j, k = m["voxel_idx"]
        x = i - nx / 2 + 0.5  # 原代码世界坐标转换公式
        y = j - ny / 2 + 0.5
        z = k - nz / 2 + 0.5
        m["world_coords"] = (x, y, z)

    return volume, markers


def verify_projection_sampling():
    # 配置参数（与原代码一致）
    nx, ny, nz = 182, 164, 210  # 体素网格尺寸
    rotation_deg = 0  # 验证视角（0°最直观）
    camera_distance = 200  # 相机距离
    detector_size = (256, 256)  # 探测器物理尺寸
    detector_resolution = (256, 256)  # 投影图分辨率 (宽, 高)

    # 生成测试体素和标记点
    volume, markers = create_test_volume(nx, ny, nz)
    print("标记点世界坐标：")
    for m in markers:
        print(f"{m['label']} → 世界坐标: {m['world_coords']}")

    # 生成投影图（使用原代码的投影器）
    projector = VolumeProjector(
        camera_distance=camera_distance,
        detector_size=detector_size,
        detector_resolution=detector_resolution,
    )
    projections, _, _ = projector.project_volume(volume, view_angles=[rotation_deg])
    projection_img = projections[0]  # 0°视角的投影图

    # 提取标记点的3D世界坐标
    points_3d = torch.asarray([m["world_coords"] for m in markers], dtype=torch.float32)

    # 步骤1：计算3D点的投影物理坐标
    projections_phys, depths = project_points_to_camera(
        points_3d, rotation_deg, camera_distance, detector_size
    )
    print("\n标记块深度值（越小越近）：")
    for i, m in enumerate(markers):
        depth_val = depths[i, 0]
        print(f"{m['label']} → 深度: {depth_val:.1f}")
    visible = depths[:, 1] > 0.5  # 可见性判断

    # 步骤2：物理坐标→像素坐标→归一化坐标（适配grid_sample）
    W_phys, H_phys = detector_size
    W_pix, H_pix = detector_resolution

    # 物理坐标转像素坐标
    x_pix = (projections_phys[:, 0] + W_phys / 2) / W_phys * W_pix
    y_pix = (projections_phys[:, 1] + H_phys / 2) / H_phys * H_pix

    # 修正上下颠倒（若原投影图已修正，需同步开启）
    # y_pix = H_pix - 1 - y_pix  # 与原代码保持一致

    # 像素坐标转归一化坐标（[-1, 1]）
    x_norm = (x_pix / (W_pix - 1)) * 2 - 1
    y_norm = (y_pix / (H_pix - 1)) * 2 - 1

    # 步骤3：用grid_sample采样投影值
    grid = torch.stack([x_norm, y_norm], axis=1)
    grid = grid.unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]

    projection_tensor = (
        torch.tensor(projection_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    )
    projection_tensor = projection_tensor.transpose(2, 3)
    with torch.no_grad():
        sampled_values = (
            torch.nn.functional.grid_sample(
                input=projection_tensor,
                grid=grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            .squeeze()
            .numpy()
        )

    # --------------------------
    # 3. 可视化验证结果
    # --------------------------
    plt.figure(figsize=(10, 8))
    plt.imshow(projection_img, cmap="hot", origin="lower")  # origin='lower'使Y轴上正

    # 绘制标记点的投影位置
    for i, m in enumerate(markers):
        if visible[i]:
            # 绘制像素坐标点
            plt.scatter(
                x_pix[i],
                y_pix[i],
                s=100,
                marker="x",
                c="blue",
                label=f"{m['label']} (可见)",
            )
            # 标注采样值
            plt.text(
                x_pix[i] + 5,
                y_pix[i] + 5,
                f"采样值: {sampled_values[i]:.1f}",
                color="blue",
                fontsize=9,
            )
        else:
            plt.scatter(
                x_pix[i],
                y_pix[i],
                s=100,
                marker="o",
                c="red",
                label=f"{m['label']} (不可见)",
            )

    plt.title(f"3D点投影验证 (视角 {rotation_deg}°)")
    plt.xlabel("像素X坐标")
    plt.ylabel("像素Y坐标")
    plt.legend()
    plt.colorbar(label="投影值")
    plt.show()
    #
    # 打印数值验证结果
    print("\n验证结果：")
    for i, m in enumerate(markers):
        status = "可见" if visible[i] else "不可见"
        val = sampled_values[i] if visible[i] else "N/A"
        print(
            f"{m['label']} → 投影位置: ({x_pix[i]:.1f}, {y_pix[i]:.1f}) → {status} → 采样值: {val}"
        )

    # 运行验证


if __name__ == "__main__":

    verify_projection_sampling()
