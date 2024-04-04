import numpy as np
import torch

from utils import set_random_seeds


def test_set_random_seeds():
    seed = 0
    set_random_seeds(seed)
    assert np.random.get_state()[1][0][0] == seed

    if torch.cuda.is_available():
        assert torch.cuda.initial_seed() == seed
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
