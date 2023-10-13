""" A module containing simple implementation of Transformer architecture


References:
https://www.youtube.com/watch?v=kCc8FmEb1nY&ab_channel=AndrejKarpathy
https://github.com/karpathy/nanoGPT/blob/master/model.py
https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
https://lilianweng.github.io/posts/2018-06-24-attention/
"""

import random
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

__all__ = ["set_seed"]


def set_seed(seed: int):
    """set seed for a   reproducible result

    Args:
        seed (int): seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def num_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters())


class CausalSelfAttentionHead(nn.Module):
    def __init__(
        self,
        input_emb_size: int,
        query_key_emb_size: int,
        output_ebm_size: int,
        context_size: int,
        dropout_p: float = 0.0,
        bias: bool = True,
        mechanism: str = "scaled_dot_product",
        activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        *args,
        **kwargs,
    ) -> None:
        """Casual self-attention head. it is causal because information from the future isn't used for each token

        Args:
            input_emb_size (int): size of input embedding
            query_key_emb_size (int): query and key embedding size
            output_ebm_size (int): embedding size of output (size of value)
            context_size (int): context length (how many of previous token are used to generate a new token)
            dropout_p (float, optional): dropout probability. Defaults to 0.0.
            bias (bool, optional): to use bias or not. Defaults to True.
            mechanism (str, optional): type of mechanism to calculate attention scores (weights). Defaults to "scaled_dot_product".
            activation (Callable[[torch.Tensor], torch.Tensor], optional): activation function. Defaults to F.relu.
        """
        super().__init__(*args, **kwargs)

        all_mechanism = [
            "scaled_dot_product",
            "linear",
            "mlp",
        ]  # pull request to add other attention mechanisms

        assert (
            mechanism in all_mechanism
        ), f"attention mechanism should be one of {all_mechanism}"

        self.mechanism = mechanism
        self.activation = activation
        self.dropout_p = dropout_p

        self.key = nn.Linear(input_emb_size, query_key_emb_size, bias=bias)
        self.query = nn.Linear(input_emb_size, query_key_emb_size, bias=bias)
        self.value = nn.Linear(input_emb_size, output_ebm_size, bias=bias)

        self.attn_dropout = nn.Dropout(dropout_p)
        self.resid_dropout = nn.Dropout(dropout_p)

        self.register_buffer(
            "tril_mask", torch.tril(torch.ones(context_size, context_size))
        )

        if mechanism == "scaled_dot_product":
            pass
        elif mechanism == "linear":
            self.lnr = nn.Linear(2 * query_key_emb_size, 1, bias=bias)
        elif mechanism == "mlp":
            self.lnr1 = nn.Linear(2 * query_key_emb_size, query_key_emb_size, bias=bias)
            self.lnr2 = nn.Linear(query_key_emb_size, 1, bias=bias)

    def forward(self, x: torch.Tensor):
        batch_size, context_size, embed_size = x.shape[-3:]

        k = self.key(x)  # (batch_size, context_length, query_key_emb_size)
        q = self.query(x)  # (batch_size, context_length, query_key_emb_size)
        v = self.value(x)  # (batch_size, context_length, output_emb_size)

        if self.mechanism == "scaled_dot_product" and hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        ):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                dropout_p=self.dropout_p if self.training else 0,
            )

        else:
            weights: torch.Tensor  # will be (batch_size, context_length, context_size)

            if self.mechanism == "scaled_dot_product":
                weights = (
                    q @ k.transpose(-2, -1) / (k.size(-1) ** 0.5)
                )  # (batch_size, context_length, context_length)
            elif self.mechanism == "linear":
                k, q = self.activation(k), self.activation(q)
                temp_q = q.repeat_interleave(q.size(-2), dim=-2)
                temp_k = (
                    k.unsqueeze(dim=-3)
                    .repeat_interleave(k.size(-2), dim=-3)
                    .view(*k.shape[:-2], k.size(-2) * k.size(-2), k.size(-1))
                )
                """ To understand this part better, look at the following example:
                >>> q = torch.arange(4).view(2,2)
                >>> q
                tensor([[0, 1],
                        [2, 3]])
                >>> k = torch.arange(4).view(2,2)
                >>> k
                tensor([[0, 1],
                        [2, 3]])
                >>> # to match each embedding of q to k then we have 
                >>> temp_q = q.repeat_interleave(2,-2)
                >>> temp_q
                tensor([[0, 1],
                        [0, 1],
                        [2, 3],
                        [2, 3]])
                >>> temp_k = (
                    k.unsqueeze(dim=-3)
                    .repeat_interleave(k.size(-2), dim=-3)
                    .view(*k.shape[:-2], k.size(-2) * k.size(-2), k.size(-1))
                )
                >>> temp_k
                tensor([[0, 1],
                        [2, 3],
                        [0, 1],
                        [2, 3]])
                >>> torch.cat([temp_q, temp_k], dim=-1)
                tensor([[0, 1, 0, 1],
                        [0, 1, 2, 3],
                        [2, 3, 0, 1],
                        [2, 3, 2, 3]])
                """

                weights = self.lnr(torch.cat([temp_q, temp_k], dim=-1)).view(
                    *x.shape[:-2], x.size(-2), x.size(-2)
                )  # output is of size (batch_size, context_size ** 2, 1). then view it as (batch_size, context_size, context_size)

            elif self.mechanism == "mlp":
                k, q = self.activation(k), self.activation(q)
                temp_q = q.repeat_interleave(q.size(-2), dim=-2)
                temp_k = (
                    k.unsqueeze(dim=-3)
                    .repeat_interleave(k.size(-2), dim=-3)
                    .view(*k.shape[:-2], k.size(-2) * k.size(-2), k.size(-1))
                )
                ln1 = self.lnr1(torch.cat([temp_q, temp_k], dim=-1))
                ln1 = self.attn_dropout(self.activation(ln1))
                ln2 = self.lnr2(ln1)
                weights = ln2.view(*x.shape[:-2], x.size(-2), x.size(-2))

            weights = weights.masked_fill(
                self.tril_mask[:context_size, :context_size] == 0, float("-inf")
            )
            weights = F.softmax(weights, dim=-1)
            weights = self.attn_dropout(weights)
            out = weights @ v

        out = self.resid_dropout(out)
        return out
