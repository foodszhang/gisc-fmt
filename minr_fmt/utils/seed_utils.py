"""Seed and reproducibility utilities"""

import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True, benchmark: bool = False):
    """
    Set random seed for reproducibility across all libraries
    
    Args:
        seed: Random seed value
        deterministic: If True, set torch.backends.cudnn.deterministic=True
        benchmark: If True, set torch.backends.cudnn.benchmark=True (may be faster but less deterministic)
        
    Note:
        deterministic and benchmark are mutually exclusive in terms of performance trade-off.
        Set deterministic=True for reproducibility (slower).
        Set benchmark=True for speed (non-deterministic).
    """
    # Python
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # CuDNN settings for reproducibility
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark
