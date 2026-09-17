"""Baseline Decoder-Only Transformer (MegaByte Stride-1 and BPE Control Baseline).

Standard causal language model architecture with:
- Token / Byte Embedding
- Pre-RMSNorm TransformerBlocks with RoPE and SwiGLU
- Final RMSNorm and linear projection head to vocabulary
- Autoregressive generation with nucleus sampling
"""

import math
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from blt.modules.transformer_layers import TransformerBlock, RMSNorm


class BaselineTransformer(nn.Module):
    """Decoder-only autoregressive transformer model.

    Used in 'stride-1 byte mode' (vocab=260) as a direct character-level baseline,
    and in 'BPE mode' (vocab=32k-100k) as the subword tokenized control group.
    """

    def __init__(
        self,
        vocab_size: int = 260,
        dim: int = 256,
        n_layers: int = 6,
        n_heads: int = 4,
        head_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        max_seq_len: int = 2048,
        rope_theta: float = 500000.0,
        sliding_window: Optional[int] = None,
        dropout: float = 0.0,
        tie_weights: bool = True,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.grad_checkpointing = grad_checkpointing

        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                n_heads=n_heads,
                head_dim=head_dim,
                hidden_dim=hidden_dim,
                max_seq_len=max_seq_len,
                rope_theta=rope_theta,
                sliding_window=sliding_window,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

        if tie_weights:
            self.output.weight = self.tok_embeddings.weight

        # Weight initialization (standard LLaMA/GPT normal)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            tokens: (batch_size, seq_len) token IDs.
            targets: Optional (batch_size, seq_len) target token IDs for loss calculation.
            mask: Optional custom attention mask.

        Returns:
            Tuple of (logits, loss).
        """
        bsz, seq_len = tokens.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        h = self.tok_embeddings(tokens)

        for layer in self.layers:
            h = layer(h, mask=mask, use_checkpoint=self.grad_checkpointing)

        h = self.norm(h)
        logits = self.output(h)

        loss = None
        if targets is not None:
            # Shift targets for next-token prediction if not already shifted
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Autoregressive generation with top-p (nucleus) and temperature sampling."""
        self.eval()
        tokens = prompt_tokens.clone()

        for _ in range(max_new_tokens):
            context = tokens[:, -self.max_seq_len:]
            logits, _ = self(context)
            next_token_logits = logits[:, -1, :] / max(1e-5, temperature)

            # Nucleus (top-p) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right to keep at least 1 token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("Inf"))

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens
