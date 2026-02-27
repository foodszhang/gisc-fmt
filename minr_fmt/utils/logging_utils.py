"""Logging utilities with rank-aware logging for distributed training"""

import logging
import os
from typing import Literal

import torch
import torch.distributed as dist


def setup_logger(
    name: str,
    level: int = logging.INFO,
    format_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> logging.Logger:
    """
    Setup a logger with rank-aware behavior (only logs on rank 0 in DDP)
    
    Args:
        name: Logger name
        level: Logging level
        format_str: Format string for log messages
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Only setup handlers on rank 0
    if get_rank() == 0:
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter(format_str)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger


def rank_zero_log(
    logger: logging.Logger,
    level: Literal["debug", "info", "warning", "error", "critical"],
    msg: str,
) -> None:
    """
    Log only on rank 0 in distributed training
    
    Args:
        logger: Logger instance
        level: Log level as string
        msg: Message to log
    """
    if get_rank() == 0:
        getattr(logger, level)(msg)


def get_rank() -> int:
    """
    Get the rank of current process in distributed training
    
    Returns:
        Process rank (0 if not in distributed mode)
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """
    Get the total number of processes in distributed training
    
    Returns:
        World size (1 if not in distributed mode)
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1
