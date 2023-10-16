""" A module containing simple implementation of Transformer architecture


References:
https://www.youtube.com/watch?v=kCc8FmEb1nY&ab_channel=AndrejKarpathy
https://github.com/karpathy/nanoGPT/blob/master/model.py
https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
https://lilianweng.github.io/posts/2018-06-24-attention/
"""

import random
from typing import Callable, List

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
        input_size: int,
        query_key_emb_size: int,
        output_size: int,
        context_size: int,
        dropout_p: float = 0.0,
        bias: bool = True,
        mechanism: str = "scaled_dot_product",  # seems to work better than others
        activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        *args,
        **kwargs,
    ) -> None:
        """Casual self-attention head. it is causal because information from the future isn't used for each token

        Args:
            input_size (int): size of input embedding
            query_key_emb_size (int): query and key embedding size
            output_size (int): embedding size of output (size of value)
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

        self.key = nn.Linear(input_size, query_key_emb_size, bias=bias)
        self.query = nn.Linear(input_size, query_key_emb_size, bias=bias)
        self.value = nn.Linear(input_size, output_size, bias=bias)

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
        ):  # calculated faster on gpu
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


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        query_key_emb_size: int,
        value_emb_size: int,
        num_head: int,
        context_size: int,
        dropout_p: float = 0.0,
        head_class: type = CausalSelfAttentionHead,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        bias: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Multi-head attention. different attention classes can be used.

        Args:
            input_size (int): input size
            output_size (int): output size
            query_key_emb_size (int): query and key embedding size
            value_emb_size (int): value embedding size
            num_head (int): number of heads
            context_size (int): context length
            dropout_p (float, optional): dropout probability. Defaults to 0.0.
            head_class (type, optional): attention class to use for heads. Defaults to CausalSelfAttentionHead.
            activation (Callable[[torch.Tensor], torch.Tensor], optional): activation function. Defaults to F.relu.
            bias (bool, optional): whether to use bias or not. Defaults to True.
        """
        super().__init__(*args, **kwargs)

        self.activation = activation

        self.heads = nn.ModuleList(
            [
                head_class(
                    input_size=input_size,
                    output_size=value_emb_size,
                    query_key_emb_size=query_key_emb_size,
                    context_size=context_size,
                    dropout_p=dropout_p,
                    activation=activation,
                    bias=bias,
                )
                for _ in range(num_head)
            ]
        )
        self.proj = nn.Linear(value_emb_size * num_head, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.activation(out))
        out = self.proj(out)
        out = self.dropout(out)

        return out


# TODO
# Also implement vectorized multi-head attention
# - It would be faster
# - but it would not be as customizable as above implementation (only one type of head is allowed then)


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout_p: float = 0.0,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        bias: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Multilayer perceptron

        Args:
            input_size (int): input size
            hidden_size (int): hidden layer size
            output_size (int): output
            dropout_p (float, optional): dropout probability. Defaults to 0.0.
            activation (Callable[[torch.Tensor], torch.Tensor], optional): activation function. Defaults to F.relu.
            bias (bool, optional): whether to use bias. Defaults to True.
        """
        super().__init__(*args, **kwargs)

        self.activation = activation

        self.ln1 = nn.Linear(input_size, hidden_size, bias=bias)
        self.ln2 = nn.Linear(hidden_size, output_size, bias=bias)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor):
        x = self.ln1(x)
        x = self.activation(x)
        x = self.ln2(x)
        x = self.dropout(x)

        return x


class Block(nn.Module):
    def __init__(
        self,
        input_size: int,
        atten_kqv_size: int,
        mlp_hidden_size: int,
        output_size: int,
        context_size: int,
        num_head: int,
        residual: bool = True,
        atten_class: type = CausalSelfAttentionHead,
        dropout_p: float = 0.0,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        bias: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Transformer block. consist of multi-head attention (for communication) followed by
        multilayer perceptron (for computation)

        Args:
            input_size (int): input size
            atten_kqv_size (int): attention key, query and value size
            mlp_hidden_size (int): multilayer perceptron hidden layer size
            output_size (int): output size
            context_size (int): context length (the windows moving on the sequence)
            num_head (int): number of head
            residual (bool, optional): whether to have residual connection or not. Defaults to True.
            atten_class (type, optional): attention class. Defaults to CausalSelfAttentionHead.
            dropout_p (float, optional): dropout probability. Defaults to 0.0.
            activation (Callable[[torch.Tensor], torch.Tensor], optional): activation function. Defaults to F.relu.
            bias (bool, optional): whether to use bias. Defaults to True.
        """
        super().__init__(*args, **kwargs)

        self.residual = residual

        self.mha = MultiHeadAttention(
            input_size=input_size,
            output_size=input_size,
            query_key_emb_size=atten_kqv_size,
            value_emb_size=atten_kqv_size,
            num_head=num_head,
            context_size=context_size,
            dropout_p=dropout_p,
            head_class=atten_class,
            activation=activation,
            bias=bias,
        )

        self.mlp = MLP(
            input_size=input_size,
            hidden_size=mlp_hidden_size,
            output_size=output_size,
            dropout_p=dropout_p,
            activation=activation,
            bias=bias,
        )

        self.l_norm1 = nn.LayerNorm(input_size)
        self.l_norm2 = nn.LayerNorm(input_size)

    def forward(self, x: torch.Tensor):
        if self.residual:
            x = x + self.mha(self.l_norm1(x))
            x = x + self.mlp(self.l_norm2(x))
        else:
            x = self.mha(self.l_norm1(x))
            x = self.mlp(self.l_norm2(x))
        return x


class MyTransformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        context_size: int,
        num_layers: int,
        num_head: int | List[int],
        atten_kqv_size: int | List[int],
        atten_output: int | List[int],
        mlp_hidden_size: int | List[int],
        residual: bool | List[bool] = True,
        atten_head_class: type | List[type] = CausalSelfAttentionHead,
        dropout_p: float | List[float] = 0.0,
        bias: bool | List[bool] = True,
        activation: Callable[[torch.Tensor], torch.Tensor]
        | List[Callable[[torch.Tensor], torch.Tensor]] = F.relu,
        *args,
        **kwargs,
    ) -> None:
        """Transformer architecture. Note that embedding is not included in this module and
        input tokens should be embedded beforehand.
        Also note that some function's arguments can be in list type, which length of this list
        should be equal to number of layers, and the ith element indicates that property for ith
        layer's block

        Args:
            input_size (int): input size
            output_size (int): output size
            context_size (int): context length
            num_layers (int): number of layers
            num_head (int | List[int]): number of heads
            atten_kqv_size (int | List[int]): attention head key, query and value embedding size
            atten_output (int | List[int]): attention head output
            mlp_hidden_size (int | List[int]): block multilayer perceptron hidden layer
            residual (bool | List[bool], optional): whether to use residual connection. Defaults to True.
            atten_head_class (type | List[type], optional): block attention class. Defaults to CausalSelfAttentionHead.
            dropout_p (float | List[float], optional): block dropout probability. Defaults to 0.0.
            bias (bool | List[bool], optional): whether to use bias of block. Defaults to True.
            activation (Callable[[torch.Tensor], torch.Tensor] | List[Callable[[torch.Tensor], torch.Tensor]], optional): block activation function. Defaults to F.relu.
        """
        super().__init__(*args, **kwargs)

        if isinstance(num_head, list):
            assert (
                len(num_head) == num_layers
            ), f"length of num_head ({len(num_head)}) is not equal to num_layers ({len(num_layers)})"
        else:
            num_head = [num_head for _ in range(num_layers)]

        if isinstance(atten_kqv_size, list):
            assert (
                len(atten_kqv_size) == num_layers
            ), f"length of atten_kqv_size ({len(atten_kqv_size)}) is not equal to num_layers ({len(num_layers)})"
        else:
            atten_kqv_size = [atten_kqv_size for _ in range(num_layers)]

        if isinstance(atten_output, list):
            assert (
                len(atten_output) == num_layers
            ), f"length of atten_output ({len(atten_output)}) is not equal to num_layers ({len(num_layers)})"
        else:
            atten_output = [atten_output for _ in range(num_layers)]

        if isinstance(mlp_hidden_size, list):
            assert (
                len(mlp_hidden_size) == num_layers
            ), f"length of mlp_hidden_size ({len(mlp_hidden_size)}) is not equal to num_layers ({len(num_layers)})"
        else:
            mlp_hidden_size = [mlp_hidden_size for _ in range(num_layers)]

        if isinstance(residual, list):
            assert (
                len(residual) == num_layers
            ), f"length of residual ({len(residual)}) is not equal to num_layers ({len(num_layers)})"
        else:
            residual = [residual for _ in range(num_layers)]

        if isinstance(atten_head_class, list):
            assert (
                len(atten_head_class) == num_layers
            ), f"length of atten_head_class ({len(atten_head_class)}) is not equal to num_layers ({len(num_layers)})"
        else:
            atten_head_class = [atten_head_class for _ in range(num_layers)]

        if isinstance(dropout_p, list):
            assert (
                len(dropout_p) == num_layers
            ), f"length of dropout_p ({len(dropout_p)}) is not equal to num_layers ({len(num_layers)})"
        else:
            dropout_p = [dropout_p for _ in range(num_layers)]

        if isinstance(bias, list):
            assert (
                len(bias) == num_layers
            ), f"length of bias ({len(bias)}) is not equal to num_layers ({len(num_layers)})"
        else:
            bias = [bias for _ in range(num_layers)]

        if isinstance(activation, list):
            assert (
                len(activation) == num_layers
            ), f"length of activation ({len(activation)}) is not equal to num_layers ({len(num_layers)})"
        else:
            activation = [activation for _ in range(num_layers)]

        self.blocks = nn.ModuleList(
            [
                Block(
                    input_size=input_size if i == 0 else atten_output[i - 1],
                    context_size=context_size,
                    output_size=atten_output[i],
                    num_head=num_head[i],
                    atten_kqv_size=atten_kqv_size[i],
                    mlp_hidden_size=mlp_hidden_size[i],
                    dropout_p=dropout_p[i],
                    residual=residual[i],
                    atten_class=atten_head_class[i],
                    activation=activation[i],
                    bias=bias[i],
                )
                for i in range(num_layers)
            ]
        )

        self.l_norm = nn.LayerNorm(atten_output[-1])
        self.proj = nn.Linear(atten_output[-1], output_size)
        self.activation = activation[-1]

    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x = block(x)

        x = self.l_norm(self.activation(x))

        if self.training:
            return self.proj(x)
        return self.proj(x[:, [-1], :])  # it's just a little optimization:)
