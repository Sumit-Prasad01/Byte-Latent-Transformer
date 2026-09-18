"""Local Decoder for Byte Latent Transformer (BLT Paper §3.3.1).

Integrates latent patch representations via DecoderCrossAttention, processes the byte states
with local sliding-window causal transformer layers, and projects to next-byte vocabulary logits.
"""

from typing import Optional
import torch
import torch.nn as nn
from blt.data.dataset import TOTAL_VOCAB_SIZE
from blt.modules.transformer_layers import RMSNorm, TransformerBlock
from blt.modules.cross_attention import DecoderCrossAttention


class LocalDecoder(nn.Module):
    """Local byte decoder consisting of decoder cross-attention, n_dec sliding-window

    transformer layers, and an LM projection head.
    """

    def __init__(
        self,
        dim: int = 256,
        patch_dim: int = 512,
        n_layers: int = 4,
        n_heads: int = 4,
        cross_attn_heads: int = 4,
        window_size: int = 256,
        vocab_size: int = TOTAL_VOCAB_SIZE,  # 260
        max_seq_len: int = 4096,
        rope_theta: float = 500000.0,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.patch_dim = patch_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.window_size = window_size
        self.vocab_size = vocab_size

        # Cross-attention: bytes attend to patch representations causally
        self.cross_attn = DecoderCrossAttention(
            byte_dim=dim,
            patch_dim=patch_dim,
            n_heads=cross_attn_heads,
            dropout=dropout,
        )

        # Local sliding-window causal transformer layers
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

        # Output projection head: dim -> vocab_size
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(
        self,
        byte_hidden: torch.Tensor,
        patch_hidden: torch.Tensor,
        patch_indices: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_checkpoint: bool = False,
    ) -> torch.Tensor:
        """Forward pass through local decoder.

        Args:
            byte_hidden: (batch, seq_len, dim) contextual byte states from local encoder.
            patch_hidden: (batch, num_patches, patch_dim) latent patch states from latent transformer.
            patch_indices: (batch, seq_len) integer patch IDs.
            mask: Optional attention mask for self-attention.
            use_checkpoint: Whether to use gradient checkpointing.

        Returns:
            logits: (batch, seq_len, vocab_size) next-byte predictions.
        """
        # 1. Decoder cross-attention (residual connection internal to module)
        h = self.cross_attn(byte_hidden, patch_hidden, patch_indices)

        # 2. Local causal transformer blocks
        for layer in self.layers:
            h = layer(h, mask=mask, use_checkpoint=use_checkpoint)

        # 3. Final normalization and projection
        h = self.norm(h)
        logits = self.lm_head(h)
        return logits
