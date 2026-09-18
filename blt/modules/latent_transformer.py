"""Global Latent Transformer for Byte Latent Transformer (BLT Paper §3.3).

Processes variable-length patch representations using a deep causal transformer stack
with RoPE, SwiGLU, and gradient checkpointing for 4GB VRAM execution.
"""

from typing import Optional
import torch
import torch.nn as nn
from blt.modules.transformer_layers import RMSNorm, TransformerBlock


class LatentTransformer(nn.Module):
    """Deep latent transformer operating on pooled patch representations.

    Performs block-causal global reasoning across patches.
    """

    def __init__(
        self,
        dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        max_seq_len: int = 2048,
        rope_theta: float = 500000.0,
        grad_checkpointing: bool = True,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.grad_checkpointing = grad_checkpointing

        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                n_heads=n_heads,
                max_seq_len=max_seq_len,
                rope_theta=rope_theta,
                sliding_window=None,  # Global causal attention across all patches
                dropout=dropout,
                norm_eps=norm_eps,
            )
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim, eps=norm_eps)

    def forward(
        self,
        patch_repr: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_checkpoint: Optional[bool] = None,
    ) -> torch.Tensor:
        """Forward pass through latent transformer.

        Args:
            patch_repr: (batch, num_patches, dim) patch representations.
            mask: Optional attention mask across patches.
            use_checkpoint: Optional override for gradient checkpointing.

        Returns:
            latent_patches: (batch, num_patches, dim) contextualized patch representations.
        """
        use_cp = self.grad_checkpointing if use_checkpoint is None else use_checkpoint
        h = patch_repr
        for layer in self.layers:
            h = layer(h, mask=mask, use_checkpoint=use_cp)
        return self.norm(h)
