"""Hydra-specific utilities for config handling and metadata"""

import os
import subprocess
from typing import Dict, Optional


def resolve_paths(base_dir: str, paths: Dict[str, str]) -> Dict[str, str]:
    """
    Resolve relative paths to absolute paths
    
    Args:
        base_dir: Base directory for relative paths
        paths: Dictionary of paths to resolve
        
    Returns:
        Dictionary with resolved paths
    """
    resolved = {}
    for key, path in paths.items():
        if path is None:
            resolved[key] = None
        elif os.path.isabs(path):
            resolved[key] = path
        else:
            resolved[key] = os.path.join(base_dir, path)
    return resolved


def get_git_info() -> Dict[str, str]:
    """
    Get current git commit hash, branch, and dirty status
    
    Returns:
        Dictionary with git info: commit, branch, dirty
        
    Raises:
        Exception: If git command fails or repo not found
    """
    try:
        # Get commit hash
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        
        # Get branch name
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        
        # Check if dirty
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).strip())
        
        return {
            "commit": commit,
            "branch": branch,
            "dirty": dirty,
        }
    except Exception as e:
        raise Exception(f"Could not get git info: {e}")


def log_config(cfg_dict: Dict, output_file: str) -> None:
    """
    Save config dictionary to file (as YAML)
    
    Args:
        cfg_dict: Configuration dictionary
        output_file: Output file path
    """
    from omegaconf import OmegaConf
    
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        OmegaConf.save(cfg_dict, f)
