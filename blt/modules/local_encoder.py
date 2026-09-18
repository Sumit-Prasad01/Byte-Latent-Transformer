"""Local Encoder for Byte Latent Transformer (BLT Paper §3.2).

Processes raw byte representations with a local sliding window causal transformer,
and pools them into latent patch representations via EncoderCrossAttention.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from blt.modules.transformer_layers import RMSNorm, TransformerBlock
from blt.modules.cross_attention import EncoderCrossAttention


class LocalEncoder(nn.Module):
    """Local byte encoder consisting of n_enc sliding-window transformer layers

    and an encoder cross-attention layer mapping byte states to patch states.
    """

    def __init__(
        self,
        dim: int = 256,
        n_layers: int = 1,
        n_heads: int = 4,
        window_size: int = 256,
        patch_dim: int = 512,
        cross_attn_heads: int = 4,
        max_seq_len: int = 4096,
        rope_theta: float = 500000.0,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.window_size = window_size
        self.patch_dim = patch_dim

        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                n_heads=n_heads,
                max_seq_len=max_seq_len,
                rope_theta=rope_theta,
                sliding_window=window_size,
                dropout=dropout,
                norm_eps=norm_eps,
            )
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim, eps=norm_eps)

        self.cross_attn = EncoderCrossAttention(
            byte_dim=dim,
            patch_dim=patch_dim,
            n_heads=cross_attn_heads,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        patch_indices: Optional[torch.Tensor] = None,
        num_patches: Optional[int] = None,
        mask: Optional[torch.Tensor] = None,
        use_checkpoint: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through local encoder.

        Args:
            x: (batch, seq_len, dim) byte embeddings.
            patch_indices: Optional (batch, seq_len) integer patch IDs.
            num_patches: Optional total patch count in the sequence.
            mask: Optional explicit attention mask.
            use_checkpoint: Whether to use gradient checkpointing.

        Returns:
            If patch_indices & num_patches provided:
                (byte_hidden, patch_hidden)
                byte_hidden: (batch, seq_len, dim)
                patch_hidden: (batch, num_patches, patch_dim)
            Else:
                byte_hidden: (batch, seq_len, dim)
        """
        h = x
        for layer in self.layers:
            h = layer(h, mask=mask, use_checkpoint=use_checkpoint)
        byte_hidden = self.norm(h)

        if patch_indices is not None and num_patches is not None:
            patch_hidden = self.cross_attn(byte_hidden, patch_indices, num_patches)
            return byte_hidden, patch_hidden

        return byte_hidden
