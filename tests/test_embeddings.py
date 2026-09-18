"""Unit tests for ByteHashNgramEmbedding layer."""

import pytest
import torch
from blt.modules.embeddings import ByteHashNgramEmbedding


def test_embedding_shapes_and_forward():
    bsz, seq_len = 2, 16
    dim = 64
    ngram_sizes = (3, 4, 5)
    vocab_per_ngram = 500

    emb_layer = ByteHashNgramEmbedding(
        dim=dim,
        byte_vocab_size=260,
        ngram_sizes=ngram_sizes,
        vocab_per_ngram=vocab_per_ngram,
    )

    # Random byte tokens in [0, 255]
    tokens = torch.randint(0, 256, (bsz, seq_len), dtype=torch.long)

    # Hash indices computation
    hash_idx = emb_layer.compute_hash_indices(tokens)
    assert hash_idx.shape == (bsz, len(ngram_sizes), seq_len)
    assert hash_idx.min() >= 0
    assert hash_idx.max() < vocab_per_ngram

    # Forward pass
    out = emb_layer(tokens)
    assert out.shape == (bsz, seq_len, dim)
    assert not torch.isnan(out).any()

    # Forward with precomputed hash indices
    out_pre = emb_layer(tokens, precomputed_hash_indices=hash_idx)
    assert torch.allclose(out, out_pre, atol=1e-6)


def test_embedding_gradients():
    bsz, seq_len = 2, 8
    dim = 32
    emb_layer = ByteHashNgramEmbedding(
        dim=dim,
        byte_vocab_size=260,
        ngram_sizes=(3,),
        vocab_per_ngram=100,
    )

    tokens = torch.randint(0, 256, (bsz, seq_len), dtype=torch.long)
    out = emb_layer(tokens)
    loss = out.sum()
    loss.backward()

    assert emb_layer.byte_embedding.weight.grad is not None
    assert emb_layer.ngram_embeddings[0].weight.grad is not None
