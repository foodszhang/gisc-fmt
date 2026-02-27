from __future__ import annotations

from typing import Any

from .hash_encoder import Frequency, HashEncoder


def get_encoder(name: str, **kwargs: Any):
    name = name.lower()
    if name in {"hashgrid", "hash", "hash_encoder"}:
        return HashEncoder(**kwargs)
    if name in {"frequency", "posenc", "positional"}:
        dim = int(kwargs.pop("dim", 3))
        n_levels = int(kwargs.pop("n_levels", 10))
        return Frequency(dim=dim, n_levels=n_levels)
    raise ValueError(f"Unknown encoder: {name}")


__all__ = ["HashEncoder", "Frequency", "get_encoder"]
