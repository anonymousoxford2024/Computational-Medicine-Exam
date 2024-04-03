import random

import numpy as np
import torch


def set_random_seeds(seed: int = 42) -> None:
    """
    Sets the seed for generating random numbers in torch, numpy, and random to ensures
    reproducibility of results by making PyTorch's use of CUDA deterministic.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
