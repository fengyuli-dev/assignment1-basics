from math import sqrt

import torch
from einops import einsum, rearrange
from torch import nn


def softmax(x: torch.Tensor, dim: int = -1):
    vector_max = torch.amax(x, dim, keepdim=True)
    return torch.exp(x - vector_max) / torch.exp(x - vector_max).sum(dim=dim, keepdim=True)


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.dtype = dtype

        init_std = sqrt(2 / (in_features + out_features))
        self.weights = (
            nn.Parameter(
                nn.init.trunc_normal_(
                    torch.empty((out_features, in_features)), 0, init_std, -3 * init_std, 3 * init_std
                )
            )
            .to(device)
            .to(dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weights, "b ... t in_dim, out_dim in_dim -> b ... t out_dim")


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weights = nn.Parameter(
            nn.init.trunc_normal_(torch.empty((num_embeddings, embedding_dim)), 0, 1, -3, 3).to(device).to(dtype)
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weights = nn.Parameter(torch.ones((d_model)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.square(x).sum(-1, keepdim=True) / self.d_model + self.eps)
        result = x / rms * self.weights
        return result.to(in_dtype)


class SiLU(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(x)


# Transformer feed-forward layer
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w_1 = Linear(d_ff, d_model, device, dtype)
        self.w_2 = Linear(d_model, d_ff, device, dtype)
        self.w_3 = Linear(d_ff, d_model, device, dtype)
        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.silu(self.w_1(x)) * self.w_3(x)
        return self.w_2(x)


class RoPE(nn.Module):
    def __init__(self, rope_base: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        inv_freqs = 1.0 / (rope_base ** (torch.arange(0, d_k, 2) / d_k))
        self.thetas = einsum(torch.arange(max_seq_len), inv_freqs, "t, d -> t d")
        self.register_buffer("rope_thetas", self.thetas, persistent=False)

    def apply_rope(self, x: torch.Tensor, token_positions: torch.Tensor):
        thetas = self.thetas[token_positions]
        cos_thetas = torch.cos(thetas)
        sin_thetas = torch.sin(thetas)
        # odd and even here are 1-indexed
        x_odd = x[..., ::2]
        x_even = x[..., 1::2]

        rotated_odd = cos_thetas * x_odd - sin_thetas * x_even
        rotated_even = cos_thetas * x_even + sin_thetas * x_odd
        # Interleave odd and even parts
        rotated_x = torch.stack((rotated_odd, rotated_even), dim=-1)
        rotated_x = rearrange(rotated_x, "b ... t d n -> b ... t (d n)")
        return rotated_x

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        assert x.shape[-2] <= self.max_seq_len
        return self.apply_rope(x, token_positions)


def self_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None):
    d = q.shape[-1]
    a = einsum(q, k, "b ... i d, b ... j d -> b ... i j")
    a = a / sqrt(d)
    if mask is not None:
        a.masked_fill_(~mask, -torch.inf)
    a = softmax(a)
    return einsum(a, v, "b ... i j, b ... j d -> b ... i d")


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope: RoPE | None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_head = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device, dtype)
        self.k_proj = Linear(d_model, d_model, device, dtype)
        self.v_proj = Linear(d_model, d_model, device, dtype)
        self.out_proj = Linear(d_model, d_model, device, dtype)
        self.rope = rope
        self.device = device

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(seq_len)
        q = rearrange(self.q_proj(x), "... t (n d) -> ... n t d", n=self.num_heads)
        k = rearrange(self.k_proj(x), "... t (n d) -> ... n t d", n=self.num_heads)
        v = rearrange(self.v_proj(x), "... t (n d) -> ... n t d", n=self.num_heads)
        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        mask = torch.tril(torch.ones((seq_len, seq_len), device=self.device, dtype=torch.bool))
        out = self_attention(q, k, v, mask)
        out = rearrange(out, "... n t d -> ... t (n d)")
        return self.out_proj(out)


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, rope: RoPE | None, device=None, dtype=None):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, rope, device, dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)
        self.rms_norm_mha = RMSNorm(d_model, device=device, dtype=dtype)
        self.rms_norm_ffn = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mha(self.rms_norm_mha(x))
        x = x + self.ffn(self.rms_norm_ffn(x))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, vocab_size, ctx_len, num_layers, rope_base, device=None, dtype=None):
        super().__init__()
        self.embeddings = Embedding(vocab_size, d_model, device, dtype)
        rope = RoPE(rope_base, d_model // num_heads, ctx_len, device)
        self.blocks = nn.ModuleList(
            [TransformerDecoderBlock(d_model, num_heads, d_ff, rope, device, dtype) for _ in range(num_layers)]
        )
        self.rms_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.out = Linear(d_model, vocab_size)
        self.device = device

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x.to(self.device)
        x = self.embeddings(x)
        for block in self.blocks:
            x = block(x)
        x = self.rms_norm(x)
        return self.out(x)
