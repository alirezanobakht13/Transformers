import random

import torch
import torch.nn as nn


def set_seed(seed: int):
    """set seed for reproducible result

    Args:
        seed (int): seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
