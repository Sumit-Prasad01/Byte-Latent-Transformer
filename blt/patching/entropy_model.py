"""Byte-level Entropy Model for dynamic patch boundary creation (Equation 1).

A lightweight next-byte transformer with sliding-window attention (~1.5M parameters)
used to estimate Shannon entropy H(x_i) at every byte position.
"""

import math
from typing import Optional, Union, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from blt.modules.transformer_layers import TransformerBlock, RMSNorm
from blt.data.dataset import DOC_BOUNDARY_TOKEN, TOTAL_VOCAB_SIZE


class ByteEntropyModel(nn.Module):
    """Small autoregressive byte-level transformer predicting next-byte probabilities and entropy."""

    def __init__(
        self,
        vocab_size: int = TOTAL_VOCAB_SIZE,  # 260
        dim: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        head_dim: Optional[int] = None,
        sliding_window: int = 256,
        max_seq_len: int = 4096,
        rope_theta: float = 500000.0,
        dropout: float = 0.0,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.sliding_window = sliding_window
        self.max_seq_len = max_seq_len

        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                n_heads=n_heads,
                head_dim=head_dim,
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.

        Args:
            tokens: (batch_size, seq_len) token IDs.
            targets: Optional (batch_size, seq_len) next-token IDs for training.

        Returns:
            Tuple of (logits, loss).
        """
        bsz, seq_len = tokens.shape
        h = self.tok_embeddings(tokens)

        for layer in self.layers:
            h = layer(h)

        h = self.norm(h)
        logits = self.output(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def compute_entropy_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute Shannon entropy in bits (base 2) from logits tensor.

        H = -sum_v p(v) * log2(p(v))
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1) / math.log(2.0)  # convert to base 2
        entropy = -torch.sum(probs * log_probs, dim=-1)  # (batch, seq_len)
        return entropy

    @torch.no_grad()
    def compute_entropy(
        self,
        byte_sequence: Union[torch.Tensor, np.ndarray, bytes, str],
        prior_entropy: float = 8.0,
    ) -> torch.Tensor:
        """Compute per-byte Shannon entropy for an input sequence (Equation 1).

        Args:
            byte_sequence: Input bytes, string, numpy array, or 1D/2D torch Tensor.
            prior_entropy: Default entropy for position 0 (unconditional, default: 8.0 bits).

        Returns:
            torch.Tensor of shape (seq_len,) containing per-byte entropy H(x_i) in bits.
        """
        self.eval()
        device = next(self.parameters()).device

        if isinstance(byte_sequence, str):
            byte_arr = np.frombuffer(byte_sequence.encode("utf-8"), dtype=np.uint8)
            tokens = torch.from_numpy(byte_arr.astype(np.int64)).unsqueeze(0).to(device)
        elif isinstance(byte_sequence, bytes):
            byte_arr = np.frombuffer(byte_sequence, dtype=np.uint8)
            tokens = torch.from_numpy(byte_arr.astype(np.int64)).unsqueeze(0).to(device)
        elif isinstance(byte_sequence, np.ndarray):
            tokens = torch.from_numpy(byte_sequence.astype(np.int64)).to(device)
            if tokens.ndim == 1:
                tokens = tokens.unsqueeze(0)
        elif isinstance(byte_sequence, torch.Tensor):
            tokens = byte_sequence.to(device)
            if tokens.ndim == 1:
                tokens = tokens.unsqueeze(0)
        else:
            raise TypeError(f"Unsupported input type for compute_entropy: {type(byte_sequence)}")

        seq_len = tokens.shape[1]
        if seq_len == 0:
            return torch.empty(0, device=device, dtype=torch.float32)

        # Logits at position t predict byte t+1
        logits, _ = self(tokens)

        # Compute entropy for predictions at positions 0 .. seq_len - 2 (predicting 1 .. seq_len - 1)
        if seq_len > 1:
            step_entropy = self.compute_entropy_from_logits(logits[:, :-1, :])  # shape: (batch, seq_len - 1)
            # Position 0 has no preceding context -> assign prior_entropy
            first_pos_entropy = torch.full(
                (tokens.shape[0], 1),
                prior_entropy,
                device=device,
                dtype=torch.float32,
            )
            full_entropy = torch.cat([first_pos_entropy, step_entropy], dim=1)
        else:
            full_entropy = torch.full(
                (tokens.shape[0], 1),
                prior_entropy,
                device=device,
                dtype=torch.float32,
            )

        return full_entropy.squeeze(0)
