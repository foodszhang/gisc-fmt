import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxProjectionLightLoss(nn.Module):
    def __init__(
        self,
        init_scatter_weight=1.0,
        target_scatter_weight=0.5,
        start_decay_epoch=200,
        decay_epochs=100,
        pos_weight=150.0,
        sparse_weight=0.01,
        lambda_dice=0.3,
        use_tversky: bool = True,
        tversky_alpha: float = 0.6,
        tversky_beta: float = 0.4,
        tversky_gamma: float = 1.33,
        tversky_weight: float | None = None,
        light_weight: float = 1.0,
    ):
        super().__init__()
        self.init_aux_projection_weight = init_scatter_weight
        self.target_aux_projection_weight = target_scatter_weight
        self.start_decay_epoch = start_decay_epoch
        self.decay_epochs = decay_epochs
        self.current_epoch = 0
        self.light_weight = float(light_weight)
        self.sparse_light_loss = SparseLightLoss(
            pos_weight,
            sparse_weight,
            lambda_dice=lambda_dice,
            use_tversky=use_tversky,
            alpha=tversky_alpha,
            beta=tversky_beta,
            gamma=tversky_gamma,
            lambda_tv=lambda_dice if tversky_weight is None else tversky_weight,
        )
        self.l1_loss = nn.L1Loss()

    def update_epoch(self, epoch):
        self.current_epoch = epoch

    def get_dynamic_aux_projection_weight(self):
        if self.current_epoch < self.start_decay_epoch:
            return self.init_aux_projection_weight
        if self.current_epoch < self.start_decay_epoch + self.decay_epochs:
            decay_ratio = (self.current_epoch - self.start_decay_epoch) / self.decay_epochs
            return self.init_aux_projection_weight - decay_ratio * (
                self.init_aux_projection_weight - self.target_aux_projection_weight
            )
        return self.target_aux_projection_weight

    def get_dynamic_scatter_weight(self):
        return self.get_dynamic_aux_projection_weight()

    def aux_projection_loss(self, pred_aux: dict | None, target_aux: dict | None, device):
        """Supervise the auxiliary projection branch when targets are available."""
        if not pred_aux or target_aux is None:
            return torch.zeros((), dtype=torch.float32, device=device)

        total_aux_loss = torch.zeros((), dtype=torch.float32, device=device)
        matched = 0
        for angle, pred in pred_aux.items():
            key = str(angle)
            if key not in target_aux:
                continue
            total_aux_loss = total_aux_loss + self.l1_loss(pred, target_aux[key])
            matched += 1
        if matched == 0:
            return torch.zeros((), dtype=torch.float32, device=device)
        return total_aux_loss / matched

    def scatter_correction_loss(self, pred_scatter: dict, gt_scatter: dict):
        device = next(iter(pred_scatter.values())).device
        return self.aux_projection_loss(pred_scatter, gt_scatter, device)

    def forward(self, pred_aux, target_aux, pred_density, gt_density):
        aux_weight = self.get_dynamic_aux_projection_weight()
        aux_loss = self.aux_projection_loss(pred_aux, target_aux, pred_density.device)
        if self.light_weight == 0.0:
            light_loss = torch.zeros((), dtype=pred_density.dtype, device=pred_density.device)
        else:
            light_loss = self.sparse_light_loss(pred_density, gt_density)
        total_loss = aux_weight * aux_loss + self.light_weight * light_loss
        return {
            "total_loss": total_loss,
            "aux_projection_loss": aux_loss,
            "light_loss": light_loss,
            "aux_projection_weight": aux_weight,
            "light_weight": torch.as_tensor(
                self.light_weight, dtype=torch.float32, device=pred_density.device
            ),
        }


ScatterLightLoss = AuxProjectionLightLoss


class SparseLightLoss(nn.Module):
    """Sparse-source density loss with weighted BCE and focal Tversky terms."""

    def __init__(
        self,
        pos_weight=150.0,
        sparse_weight=0.01,
        attn_weight=0.1,
        lambda_dice=0.3,
        use_tversky: bool = True,
        alpha: float = 0.6,
        beta: float = 0.4,
        gamma: float = 1.33,
        lambda_tv: float = 0.3,
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.sparse_weight = sparse_weight
        self.attn_weight = attn_weight
        self.use_tversky = bool(use_tversky)
        self.lambda_dice = lambda_dice
        self.lambda_tv = lambda_tv if not (lambda_tv == 0.3 and lambda_dice != 0.3) else lambda_dice
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)

    def forward(self, pred_density, gt_density):
        if pred_density.dim() == 3 and pred_density.size(-1) == 1:
            pred_density = pred_density.squeeze(-1)
        if gt_density.dim() == 3 and gt_density.size(-1) == 1:
            gt_density = gt_density.squeeze(-1)
        gt_density = gt_density.to(device=pred_density.device, dtype=pred_density.dtype)

        bce_loss = F.binary_cross_entropy_with_logits(
            pred_density,
            gt_density,
            weight=gt_density * (self.pos_weight - 1) + 1,
        )
        probability = torch.sigmoid(pred_density)
        sparse_loss = self.sparse_weight * torch.mean(probability * (1.0 - gt_density))

        focal_tversky_loss = torch.zeros(
            (), dtype=pred_density.dtype, device=pred_density.device
        )
        if self.use_tversky and self.lambda_tv > 0:
            eps = 1e-6
            tp = (probability * gt_density).sum(dim=1)
            fp = (probability * (1.0 - gt_density)).sum(dim=1)
            fn = ((1.0 - probability) * gt_density).sum(dim=1)
            tversky = (tp + eps) / (tp + self.alpha * fp + self.beta * fn + eps)
            focal_tversky_loss = torch.clamp(1.0 - tversky, 0.0, 1.0).pow(self.gamma).mean()

        return bce_loss + sparse_loss + self.lambda_tv * focal_tversky_loss


class VoxelReconstructionLoss(nn.Module):
    """Voxel reconstruction loss with optional native-grid supervision.

    A model may expose ``aux_outputs['native_pred_voxel']`` and set
    ``train_on_native_grid=True``. In that case, the continuous fluorescence
    density target is trilinearly resampled to the native grid before loss
    computation. The fixed full-grid output remains available for common
    validation and test metrics.
    """

    def __init__(
        self,
        pos_weight=2.0,
        sparse_weight=0.05,
        lambda_dice=0.5,
        use_tversky: bool = True,
        tversky_alpha: float = 0.6,
        tversky_beta: float = 0.4,
        tversky_gamma: float = 1.33,
        tversky_weight: float | None = None,
    ):
        super().__init__()
        self.sparse_light_loss = SparseLightLoss(
            pos_weight=pos_weight,
            sparse_weight=sparse_weight,
            lambda_dice=lambda_dice,
            use_tversky=use_tversky,
            alpha=tversky_alpha,
            beta=tversky_beta,
            gamma=tversky_gamma,
            lambda_tv=lambda_dice if tversky_weight is None else tversky_weight,
        )
        self.l1_loss = nn.L1Loss()

    @staticmethod
    def _remove_singleton_channel(volume: torch.Tensor) -> torch.Tensor:
        if volume.dim() == 5 and volume.size(1) == 1:
            return volume[:, 0]
        return volume

    @staticmethod
    def _resample_continuous_target(
        target: torch.Tensor,
        output_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        if tuple(target.shape[1:]) == output_shape:
            return target
        return F.interpolate(
            target.unsqueeze(1),
            size=output_shape,
            mode="trilinear",
            align_corners=False,
        )[:, 0]

    def forward(self, pred_voxel, target_voxel, aux_outputs=None):
        pred_source = pred_voxel
        train_on_native = False
        if isinstance(aux_outputs, dict):
            native_pred = aux_outputs.get("native_pred_voxel")
            train_on_native = bool(aux_outputs.get("train_on_native_grid", False))
            if train_on_native and torch.is_tensor(native_pred):
                pred_source = native_pred

        pred = self._remove_singleton_channel(pred_source)
        target = self._remove_singleton_channel(target_voxel)
        target = target.to(device=pred.device, dtype=pred.dtype)
        if train_on_native:
            target = self._resample_continuous_target(target, tuple(pred.shape[1:]))

        if pred.shape[1:] != target.shape[1:]:
            raise ValueError(
                "Voxel prediction and target shapes differ. "
                f"pred={tuple(pred.shape)}, target={tuple(target.shape)}. "
                "Declare an explicit native grid or map the prediction to the common reference grid."
            )

        flat_pred = pred.reshape(pred.shape[0], -1, 1)
        flat_target = target.reshape(target.shape[0], -1, 1)
        voxel_weight = 1.0
        loss_type = "default"
        dice_weight = None
        if isinstance(aux_outputs, dict):
            voxel_weight = float(aux_outputs.get("voxel_loss_weight", 1.0))
            loss_type = str(aux_outputs.get("voxel_loss_type", "default")).lower()
            if "dice_weight" in aux_outputs:
                dice_weight = float(aux_outputs["dice_weight"])

        if loss_type == "mse":
            probability = torch.sigmoid(pred)
            mse_loss = F.mse_loss(probability, target)
            if dice_weight is None:
                dice_weight = 0.0
            probability_flat = probability.reshape(probability.shape[0], -1)
            target_flat = target.reshape(target.shape[0], -1)
            intersection = (probability_flat * target_flat).sum(dim=1)
            dice = (2.0 * intersection + 1e-6) / (
                probability_flat.sum(dim=1) + target_flat.sum(dim=1) + 1e-6
            )
            dice_loss = 1.0 - dice.mean()
            light_loss = dice_loss
            rec_l1 = mse_loss
            total = voxel_weight * (mse_loss + float(dice_weight) * dice_loss)
        else:
            light_loss = self.sparse_light_loss(flat_pred, flat_target)
            rec_l1 = self.l1_loss(torch.sigmoid(pred), target)
            total = voxel_weight * (light_loss + rec_l1)

        extra_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
        if isinstance(aux_outputs, dict):
            for key in ("projection_loss", "perceptual_loss", "distribution_loss", "stn_loss"):
                value = aux_outputs.get(key)
                if torch.is_tensor(value):
                    extra_loss = extra_loss + value
        total = total + extra_loss
        return {
            "total_loss": total,
            "voxel_light_loss": light_loss,
            "voxel_l1_loss": rec_l1,
            "voxel_aux_loss": extra_loss,
        }


class MorphologyAwareDensityLoss(nn.Module):
    """Probability-domain SSQ-FMT density loss with optional SDF supervision."""

    def __init__(
        self,
        lambda_sdf: float = 0.0,
        tau_s: float = 3.0,
        boundary_weight: float = 1.0,
        pos_weight: float = 1.0,
        dice_weight: float = 0.0,
        sparse_weight: float = 0.0,
    ):
        super().__init__()
        self.lambda_sdf = float(lambda_sdf)
        self.tau_s = float(tau_s)
        self.boundary_weight = float(boundary_weight)
        self.pos_weight = float(pos_weight)
        self.dice_weight = float(dice_weight)
        self.sparse_weight = float(sparse_weight)

    @staticmethod
    def _sample_gt(gt_voxels: torch.Tensor, points_ijk: torch.Tensor) -> torch.Tensor:
        idx = points_ijk.round().long()
        x = idx[..., 0].clamp(0, gt_voxels.shape[1] - 1)
        y = idx[..., 1].clamp(0, gt_voxels.shape[2] - 1)
        z = idx[..., 2].clamp(0, gt_voxels.shape[3] - 1)
        batch = torch.arange(gt_voxels.shape[0], device=gt_voxels.device)[:, None]
        return gt_voxels[batch, x, y, z].unsqueeze(-1)

    def forward(
        self,
        pred_density: torch.Tensor,
        target_density: torch.Tensor,
        aux_outputs: dict | None = None,
        *,
        gt_voxels: torch.Tensor | None = None,
        points_ijk: torch.Tensor | None = None,
        sdf_targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        target_density = target_density.to(device=pred_density.device, dtype=pred_density.dtype)
        pred = pred_density.clamp(0.0, 1.0)
        density_err = F.smooth_l1_loss(pred, target_density, reduction="none")
        density_weight = 1.0 + target_density.clamp(0.0, 1.0) * (self.pos_weight - 1.0)
        density_loss = (density_err * density_weight).sum() / density_weight.sum().clamp_min(1e-8)

        eps = 1e-6
        pred_flat = pred.squeeze(-1)
        target_flat = target_density.squeeze(-1).clamp(0.0, 1.0)
        intersection = (pred_flat * target_flat).sum(dim=1)
        dice = (2.0 * intersection + eps) / (
            pred_flat.sum(dim=1) + target_flat.sum(dim=1) + eps
        )
        dice_loss = 1.0 - dice.mean()
        sparse_loss = (pred * (1.0 - target_density.clamp(0.0, 1.0))).mean()
        total = density_loss + self.dice_weight * dice_loss + self.sparse_weight * sparse_loss

        sdf_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
        if (
            self.lambda_sdf > 0.0
            and aux_outputs is not None
            and "sdf" in aux_outputs
            and sdf_targets is not None
        ):
            sdf_target = sdf_targets.to(device=pred.device, dtype=pred.dtype)
            sdf_pred = aux_outputs["sdf"].to(dtype=pred.dtype)
            support = aux_outputs.get("measurement_supported")
            if torch.is_tensor(support):
                support = support.to(device=pred.device, dtype=pred.dtype).unsqueeze(-1)
            else:
                support = torch.ones_like(sdf_target)
            weight = 1.0 + self.boundary_weight * (
                sdf_target.abs() < 0.25
            ).to(dtype=pred.dtype)
            err = F.smooth_l1_loss(
                sdf_pred, sdf_target, reduction="none"
            ) * weight * support
            sdf_loss = err.sum() / support.sum().clamp_min(1e-8)
            total = total + self.lambda_sdf * sdf_loss

        return {
            "total_loss": total,
            "density_loss": density_loss,
            "dice_loss": dice_loss,
            "sparse_loss": sparse_loss,
            "sdf_loss": sdf_loss,
        }


def dice_coefficient(pred, target, threshold=0.5, eps=1e-8):
    pred_bin = (pred >= threshold).float()
    intersection = (pred_bin * target).sum(dim=(1, 2, 3))
    pred_sum = pred_bin.sum(dim=(1, 2, 3))
    target_sum = target.sum(dim=(1, 2, 3))
    dice_per_sample = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    return dice_per_sample.mean()


def compute_dice(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    if union == 0:
        return 1.0
    return 2 * intersection / (union + 1e-8)
