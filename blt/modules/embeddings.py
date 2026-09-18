"""Byte and Hash N-Gram Embeddings (BLT Paper §3.1, Appendix C, Equations 2-4).

Combines direct byte embeddings with multiple rolling polynomial hash n-gram embedding tables
(n=3, 4, 5) accelerated via the native C++ RollPolyHash dynamic library.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from blt.csrc.bridge import compute_ngram_hashes_native
from blt.data.dataset import TOTAL_VOCAB_SIZE


class ByteHashNgramEmbedding(nn.Module):
    """Composite embedding layer combining byte embeddings with hash n-gram embeddings."""

    def __init__(
        self,
        dim: int = 256,
        byte_vocab_size: int = TOTAL_VOCAB_SIZE,  # 260
        ngram_sizes: Tuple[int, ...] = (3, 4, 5),
        vocab_per_ngram: int = 20000,
        hash_prime: int = 31337,
    ):
        super().__init__()
        self.dim = dim
        self.ngram_sizes = list(ngram_sizes)
        self.vocab_per_ngram = vocab_per_ngram
        self.hash_prime = hash_prime
        self.num_ngrams = len(self.ngram_sizes)

        # 1. Base byte embedding table: (vocab_size, dim)
        self.byte_embedding = nn.Embedding(byte_vocab_size, dim)

        # 2. Hash n-gram embedding tables: E_hash_n for each n in ngram_sizes
        self.ngram_embeddings = nn.ModuleList([
            nn.Embedding(vocab_per_ngram, dim) for _ in self.ngram_sizes
        ])

        # Normalization factor: 1 / (|N| + 1) per Equation 4
        self.norm_factor = 1.0 / (self.num_ngrams + 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.byte_embedding.weight, mean=0.0, std=0.02)
        for emb in self.ngram_embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def compute_hash_indices(self, byte_tokens: torch.Tensor) -> torch.Tensor:
        """Compute rolling hash bucket indices in 64-bit integer space outside autocast.

        Args:
            byte_tokens: (batch_size, seq_len) LongTensor of token IDs.

        Returns:
            LongTensor of shape (batch_size, num_ngrams, seq_len) with bucket indices.
        """
        device = byte_tokens.device
        bsz, seq_len = byte_tokens.shape
        byte_tokens_np = byte_tokens.detach().cpu().numpy().astype(np.uint8)

        vocab_sizes = [self.vocab_per_ngram] * self.num_ngrams
        batch_hashes = []

        for b in range(bsz):
            # C++ accelerated RollPolyHash calculation
            hashes_2d = compute_ngram_hashes_native(
                byte_tokens_np[b],
                ngram_sizes=self.ngram_sizes,
                vocab_sizes=vocab_sizes,
                prime=self.hash_prime,
            )  # shape: (num_ngrams, seq_len)
            batch_hashes.append(hashes_2d)

        stacked_hashes = np.stack(batch_hashes, axis=0)  # (batch_size, num_ngrams, seq_len)
        return torch.from_numpy(stacked_hashes).to(device=device, dtype=torch.long)

    def forward(
        self,
        byte_tokens: torch.Tensor,
        precomputed_hash_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute composite embeddings: e_i = 1/(|N|+1) * (E_byte[x_i] + sum_n E_hash_n[h_{i,n}]).

        Args:
            byte_tokens: (batch_size, seq_len) LongTensor.
            precomputed_hash_indices: Optional precomputed hash indices of shape (batch_size, num_ngrams, seq_len).

        Returns:
            Tensor of shape (batch_size, seq_len, dim).
        """
        # Base byte embedding: (batch, seq_len, dim)
        e = self.byte_embedding(byte_tokens)

        # Hash n-gram embeddings
        if precomputed_hash_indices is None:
            hash_indices = self.compute_hash_indices(byte_tokens)
        else:
            hash_indices = precomputed_hash_indices

        # Sum hash embeddings across all n-gram sizes
        for idx in range(self.num_ngrams):
            # hash_indices[:, idx, :]: (batch, seq_len)
            e = e + self.ngram_embeddings[idx](hash_indices[:, idx, :])

        # Normalize by (|N| + 1)
        return e * self.norm_factor
