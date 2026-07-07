from __future__ import annotations

import sys

import train as base_train

if __name__ == "__main__":
    base_train._rewrite_positional_task(sys.argv)
    base_train.hydra_main()
