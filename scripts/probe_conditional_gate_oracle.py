"""Probe whether frozen PHSA diagnostics contain the oracle routing direction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import hydra
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minr_fmt.datamodule import TrainingDataModule  # noqa: E402
from minr_fmt.module import TrainingLightningModule  # noqa: E402


def to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_device(item, device) for item in value)
    return value


def weighted_candidate_stat(stat: torch.Tensor, applicability: torch.Tensor) -> torch.Tensor:
    weight = applicability / applicability.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    return (weight * stat[:, None]).sum(dim=-1)


@torch.inference_mode()
def extract(module, loader, device, max_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    features, labels = [], []
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_samples:
            break
        batch = to_device(batch, device)
        output = module._call_ssq_model(batch, return_diagnostics=True)
        diagnostics = output["diagnostics"]
        target = (batch["point_densities"] > 0).float()
        positive = diagnostics["routing_positive_density"].float().squeeze(-1)
        negative = diagnostics["routing_negative_density"].float().squeeze(-1)
        positive = positive.clamp(1.0e-6, 1.0 - 1.0e-6)
        negative = negative.clamp(1.0e-6, 1.0 - 1.0e-6)
        positive_error = -(target * positive.log() + (1 - target) * (1 - positive).log())
        negative_error = -(target * negative.log() + (1 - target) * (1 - negative).log())
        oracle_margin = (positive_error - negative_error).abs()
        oracle_positive = positive_error < negative_error

        applicability = diagnostics["applicability"].float()
        separability = diagnostics["separability"].float().transpose(1, 2)
        support = diagnostics["candidate_view_support"].float()
        valid = diagnostics["candidate_valid_mask"].bool()
        valid_view = valid[..., None].expand_as(separability)
        sep_masked = separability.masked_fill(~valid_view, 0.0)
        sep_count = valid_view.sum(dim=-1).clamp_min(1)
        sep_mean = sep_masked.sum(dim=-1) / sep_count
        sep_std = (
            ((separability - sep_mean[..., None]).square() * valid_view).sum(dim=-1)
            / sep_count
        ).sqrt()
        sep_max = separability.masked_fill(~valid_view, -torch.inf).amax(dim=-1)
        sep_min = separability.masked_fill(~valid_view, torch.inf).amin(dim=-1)
        sep_range = torch.nan_to_num(sep_max - sep_min, nan=0.0, posinf=0.0)
        support_std = support.std(dim=-1, unbiased=False)

        base_weight = diagnostics["base_view_weights"].float().transpose(1, 2)
        base_entropy = -(base_weight.clamp_min(1.0e-8) * base_weight.clamp_min(1.0e-8).log())
        base_entropy = base_entropy.sum(dim=-1)
        base_entropy = base_entropy / torch.log(base_entropy.new_tensor(base_weight.shape[-1]))
        shared_margin = (diagnostics["shared_density"].float().squeeze(-1) - 0.5).abs()
        candidate_count = valid.sum(dim=-1).float()[:, None].expand_as(shared_margin)
        enriched_features = torch.stack(
            [
                base_entropy,
                weighted_candidate_stat(sep_range, applicability),
                weighted_candidate_stat(sep_std, applicability),
                applicability.amax(dim=-1),
                candidate_count,
                diagnostics["projected_overlap"].float(),
                shared_margin,
                weighted_candidate_stat(support_std, applicability),
                diagnostics["hypothesis_gate"].float(),
            ],
            dim=-1,
        )
        query_features = torch.cat(
            [diagnostics["gain_features"].float(), enriched_features], dim=-1
        )
        eligible = (diagnostics["hypothesis_gate"] > 0.05) & (oracle_margin > 1.0e-5)
        features.append(query_features[eligible].cpu())
        labels.append(oracle_positive[eligible].float().cpu())
    return torch.cat(features), torch.cat(labels)


def balanced_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    prediction = logits >= 0
    positive = labels > 0.5
    sensitivity = prediction[positive].float().mean()
    specificity = (~prediction[~positive]).float().mean()
    return float((sensitivity + specificity) * 0.5)


def fit_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    hidden: int | None,
    shuffle_labels: bool = False,
) -> dict[str, float]:
    torch.manual_seed(42)
    if shuffle_labels:
        train_y = train_y[torch.randperm(len(train_y))]
    model = (
        nn.Linear(train_x.shape[1], 1)
        if hidden is None
        else nn.Sequential(nn.Linear(train_x.shape[1], hidden), nn.SiLU(), nn.Linear(hidden, 1))
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1.0e-3)
    positive_weight = (train_y == 0).sum() / (train_y == 1).sum().clamp_min(1)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(train_x).squeeze(-1), train_y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = model(val_x).squeeze(-1)
    return {
        "balanced_accuracy": balanced_accuracy(logits, val_y),
        "accuracy": float(((logits >= 0) == (val_y > 0.5)).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-samples", type=int, default=100)
    parser.add_argument("--val-samples", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    overrides = [
        "model=ssq_fmt",
        "exp=fmt_simgen_v2_phsa_view_gating",
        "data.dataset_type=fmt_simgen",
        "data.train_dir=/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k",
        "data.val_dir=/home/foods/pro/FMT-SimGen/data/fmt_simgen_v2_3k_20k",
        f"data.train_max_samples={args.train_samples}",
        f"data.val_max_samples={args.val_samples}",
        "data.num_workers=0",
        "data.persistent_workers=false",
    ]
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "configs")):
        cfg = hydra.compose(config_name="config", overrides=overrides)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = TrainingLightningModule(cfg)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    module.load_state_dict(state, strict=True)
    module.eval().to(device)
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    data = TrainingDataModule(cfg)
    data.setup("fit")
    train_x, train_y = extract(module, data.train_dataloader(), device, args.train_samples)
    val_x, val_y = extract(module, data.val_dataloader(), device, args.val_samples)
    train_x, train_y = train_x.clone(), train_y.clone()
    val_x, val_y = val_x.clone(), val_y.clone()
    mean, std = train_x.mean(dim=0), train_x.std(dim=0).clamp_min(1.0e-6)
    train_x, val_x = (train_x - mean) / std, (val_x - mean) / std
    majority = max(float(val_y.mean()), 1.0 - float(val_y.mean()))
    current_feature_names = [
        "gate_hypothesis_gate",
        "routing_abs_mean",
        "routing_rms",
        "routing_max",
        "routing_min",
    ]
    enriched_feature_names = [
            "base_view_weight_entropy",
            "separability_range",
            "separability_std",
            "nearest_applicability",
            "effective_candidate_count",
            "projected_overlap",
            "shared_prediction_margin",
            "candidate_support_dispersion",
            "hypothesis_gate",
    ]
    result = {
        "current_feature_names": current_feature_names,
        "enriched_feature_names": enriched_feature_names,
        "train_eligible": len(train_y),
        "val_eligible": len(val_y),
        "train_positive_ratio": float(train_y.mean()),
        "val_positive_ratio": float(val_y.mean()),
        "majority_accuracy": majority,
        "majority_balanced_accuracy": 0.5,
        "current_linear": fit_probe(
            train_x[:, :5], train_y, val_x[:, :5], val_y, hidden=None
        ),
        "current_mlp": fit_probe(
            train_x[:, :5], train_y, val_x[:, :5], val_y, hidden=32
        ),
        "enriched_linear": fit_probe(
            train_x[:, 5:], train_y, val_x[:, 5:], val_y, hidden=None
        ),
        "enriched_mlp": fit_probe(
            train_x[:, 5:], train_y, val_x[:, 5:], val_y, hidden=32
        ),
        "combined_mlp": fit_probe(train_x, train_y, val_x, val_y, hidden=32),
        "shuffled_label_mlp": fit_probe(
            train_x, train_y, val_x, val_y, hidden=32, shuffle_labels=True
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
