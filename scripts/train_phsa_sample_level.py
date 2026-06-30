#!/usr/bin/env python3
"""Training entrypoint that activates the sample-level PHSA hypothesis path."""

from __future__ import annotations

import sys

from minr_fmt.phsa_sample_level import activate_phsa_sample_level_hypotheses

activate_phsa_sample_level_hypotheses()

from train import _rewrite_positional_task, hydra_main  # noqa: E402


if __name__ == "__main__":
    _rewrite_positional_task(sys.argv)
    hydra_main()
