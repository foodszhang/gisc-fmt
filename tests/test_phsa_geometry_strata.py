import importlib.util
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "phsa_strata", Path(__file__).parents[1] / "scripts" / "build_phsa_geometry_strata.py"
)
STRATA = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(STRATA)


def _focus(center, radius=1.0, intensity=1.0):
    return {
        "center": center,
        "radius": radius,
        "rx": None,
        "ry": None,
        "rz": None,
        "params": {"intensity": intensity},
    }


def test_projected_separability_changes_across_views():
    score = STRATA.projected_pair_separability(
        [_focus([10, 10, 10]), _focus([20, 10, 10])], [0, 90]
    )
    assert score.shape == (2, 1)
    assert score[0, 0] > 4.9
    assert np.isclose(score[1, 0], 0.0, atol=1.0e-6)


def test_assignment_is_fixed_by_train_thresholds():
    thresholds = {
        "mixed_sep_low": 0.5,
        "mixed_sep_high": 1.5,
        "range_high": 1.0,
        "depth_imbalance": 1.5,
        "intensity_imbalance": 1.5,
    }
    row = {
        "num_sources": 2,
        "view_sep_min": 0.2,
        "view_sep_max": 2.0,
        "view_sep_range": 1.8,
        "depth_ratio": 1.0,
        "intensity_ratio": 2.0,
    }
    assert STRATA.assign(row, thresholds) == ("multi_view_complementary", True)
