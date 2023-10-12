import random

import torch
import torch.nn as nn

__all__ = ["set_seed", "LookupEmbedding"]


def set_seed(seed: int):
    """set seed for a   reproducible result

    Args:
        seed (int): seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LookupEmbedding(nn.Module):
    def __init__(self, token_size: int, embedding_size: int, *args, **kwargs) -> None:
        """Use this module when you want to have lookup table for your input embedding

        Args:
            token_size (int): number of tokens
            embedding_size (int): output embedding size
        """
        super().__init__(*args, **kwargs)
        self.embedding_table = nn.Embedding(token_size, embedding_size)

    def forward(self, input_idx: torch.Tensor):
        logits = self.embedding_table(input_idx)
        return logits
