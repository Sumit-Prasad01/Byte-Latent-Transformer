"""Deterministic random number generator seeding utilities across Python, NumPy, and PyTorch.
"""

import os
import random
from typing import Optional
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def seed_everything(seed: int = 42, deterministic_cudnn: bool = True) -> int:
    """Seed all pseudo-random number generators for reproducible runs.

    Args:
        seed: Integer seed value.
        deterministic_cudnn: If True and CUDA is available, set CuDNN to deterministic mode.

    Returns:
        The seed integer applied.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if deterministic_cudnn:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    return seed


def seed_worker(worker_id: int) -> None:
    """Worker init function for PyTorch DataLoader to ensure worker-level randomness uniqueness."""
    worker_seed = (torch.initial_seed() % (2**32)) + worker_id if HAS_TORCH else 42 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
