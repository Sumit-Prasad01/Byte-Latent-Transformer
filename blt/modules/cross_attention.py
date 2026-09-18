"""Encoder and Decoder Cross-Attention mechanisms for BLT (Paper §3.2.2 & §3.3.1, Figure 5).

- Encoder Cross-Attention: Patch queries attend to constituent byte tokens (patch-membership masked).
- Decoder Cross-Attention: Byte queries attend to latent patch representations (causally masked up to current patch).
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from blt.modules.transformer_layers import RMSNorm


def create_patch_membership_mask(
    patch_indices: torch.Tensor,
    num_patches: int,
) -> torch.Tensor:
    """Construct boolean mask of shape (batch, 1, num_patches, seq_len).

    True where byte t belongs to patch j.
    """
    # patch_indices: (batch, seq_len)
    p_ids = torch.arange(num_patches, device=patch_indices.device).view(1, num_patches, 1)
    membership = (patch_indices.unsqueeze(1) == p_ids)  # (batch, num_patches, seq_len)
    return membership.unsqueeze(1)  # (batch, 1, num_patches, seq_len)


def create_decoder_causal_patch_mask(
    patch_indices: torch.Tensor,
    num_patches: int,
) -> torch.Tensor:
    """Construct causal mask for decoder cross-attention of shape (batch, 1, seq_len, num_patches).

    True where patch p <= current patch of byte t.
    """
    # patch_indices: (batch, seq_len)
    p_ids = torch.arange(num_patches, device=patch_indices.device).view(1, 1, 1, num_patches)
    mask = (patch_indices.unsqueeze(1).unsqueeze(-1) >= p_ids)  # (batch, 1, seq_len, num_patches)
    return mask


class EncoderCrossAttention(nn.Module):
    """Encoder Cross-Attention (Paper §3.2.2, Equations 5-8).

    Patch representations (queries, initialized via mean-pooling of constituent bytes)
    attend to byte hidden states restricted to their patch membership.
    """

    def __init__(
        self,
        byte_dim: int = 256,
        patch_dim: int = 512,
        n_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.byte_dim = byte_dim
        self.patch_dim = patch_dim
        self.n_heads = n_heads
        self.head_dim = patch_dim // n_heads
        self.dropout = dropout

        self.query_norm = RMSNorm(byte_dim)
        self.kv_norm = RMSNorm(byte_dim)

        self.q_proj = nn.Linear(byte_dim, patch_dim, bias=False)
        self.k_proj = nn.Linear(byte_dim, patch_dim, bias=False)
        self.v_proj = nn.Linear(byte_dim, patch_dim, bias=False)
        self.out_proj = nn.Linear(patch_dim, patch_dim, bias=False)

    def forward(
        self,
        byte_hidden: torch.Tensor,
        patch_indices: torch.Tensor,
        num_patches: int,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            byte_hidden: (batch, seq_len, byte_dim) representations from local encoder.
            patch_indices: (batch, seq_len) integer patch IDs for each byte.
            num_patches: Total number of patches in this sequence.

        Returns:
            patch_hidden: (batch, num_patches, patch_dim) latent patch representations.
        """
        bsz, seq_len, _ = byte_hidden.shape

        # 1. Pool initial patch representations from constituent bytes
        # mask: (batch, num_patches, seq_len)
        membership = (patch_indices.unsqueeze(1) == torch.arange(num_patches, device=byte_hidden.device).view(1, num_patches, 1)).float()
        counts = membership.sum(dim=-1, keepdim=True).clamp(min=1.0)
        pooled_bytes = torch.bmm(membership, byte_hidden) / counts  # (batch, num_patches, byte_dim)

        # 2. Queries from pooled bytes, Keys and Values from all bytes
        q = self.q_proj(self.query_norm(pooled_bytes)).view(bsz, num_patches, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(self.kv_norm(byte_hidden)).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(self.kv_norm(byte_hidden)).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # 3. Patch membership attention mask: (batch, 1, num_patches, seq_len)
        attn_mask = membership.unsqueeze(1).bool()

        # 4. Attention
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(bsz, num_patches, self.patch_dim)
        patch_repr = self.out_proj(out)

        return patch_repr


class DecoderCrossAttention(nn.Module):
    """Decoder Cross-Attention (Paper §3.3.1, Equations 9-12).

    Byte representations (queries) attend causally to patch representations
    from the Latent Transformer up to the current patch.
    """

    def __init__(
        self,
        byte_dim: int = 256,
        patch_dim: int = 512,
        n_heads: int = 4,
        k_ratio: int = 2,  # patch_dim / byte_dim
        dropout: float = 0.0,
    ):
        super().__init__()
        self.byte_dim = byte_dim
        self.patch_dim = patch_dim
        self.n_heads = n_heads
        self.head_dim = byte_dim // n_heads
        self.dropout = dropout

        self.query_norm = RMSNorm(byte_dim)
        self.patch_norm = RMSNorm(patch_dim)

        self.q_proj = nn.Linear(byte_dim, byte_dim, bias=False)
        self.k_proj = nn.Linear(patch_dim, byte_dim, bias=False)
        self.v_proj = nn.Linear(patch_dim, byte_dim, bias=False)
        self.out_proj = nn.Linear(byte_dim, byte_dim, bias=False)

    def forward(
        self,
        byte_hidden: torch.Tensor,
        patch_hidden: torch.Tensor,
        patch_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            byte_hidden: (batch, seq_len, byte_dim) byte hidden states.
            patch_hidden: (batch, num_patches, patch_dim) latent patch representations.
            patch_indices: (batch, seq_len) patch IDs for each byte.

        Returns:
            output: (batch, seq_len, byte_dim) updated byte hidden states.
        """
        bsz, seq_len, _ = byte_hidden.shape
        num_patches = patch_hidden.shape[1]

        q = self.q_proj(self.query_norm(byte_hidden)).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(self.patch_norm(patch_hidden)).view(bsz, num_patches, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(self.patch_norm(patch_hidden)).view(bsz, num_patches, self.n_heads, self.head_dim).transpose(1, 2)

        # Causal attention mask: byte t attends to patches p <= patch_id(t)
        # mask shape: (batch, 1, seq_len, num_patches)
        attn_mask = create_decoder_causal_patch_mask(patch_indices, num_patches)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.byte_dim)

        # Residual connection
        return byte_hidden + self.out_proj(out)
