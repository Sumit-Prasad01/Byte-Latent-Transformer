"""FLOPs Accounting and Complexity Metrics (BLT Paper Appendix B).

Implements exact analytical formulas for floating-point operations (FLOPs)
across Local Encoder, Latent Transformer, Local Decoder, and Cross-Attentions,
allowing precise comparison of inference and training efficiency.
"""

from typing import Dict, Any, Union, Optional
import math


def count_linear_flops(in_features: int, out_features: int, num_tokens: int) -> int:
    """FLOPs for a linear projection y = xW (1 multiply + 1 add per MAC = 2 FLOPs)."""
    return 2 * num_tokens * in_features * out_features


def count_swiglu_flops(dim: int, hidden_dim: int, num_tokens: int) -> int:
    """FLOPs for SwiGLU feed-forward network (gate, up, down projections + silu/elementwise)."""
    # 3 linear projections: dim -> hidden, dim -> hidden, hidden -> dim
    proj_flops = 3 * (2 * num_tokens * dim * hidden_dim)
    # Elementwise activation (silu + mult)
    elem_flops = 2 * num_tokens * hidden_dim
    return proj_flops + elem_flops


def count_attention_flops(
    dim: int,
    num_queries: int,
    num_keys: Optional[int] = None,
    sliding_window: Optional[int] = None,
) -> int:
    """FLOPs for multi-head attention (Q, K, V, Out projections + SDPA)."""
    if num_keys is None:
        num_keys = num_queries

    # Projections: Q (dim -> dim), K (dim -> dim), V (dim -> dim), Out (dim -> dim)
    proj_flops = 4 * (2 * num_queries * dim * dim)

    # SDPA: QK^T and Softmax(QK^T) * V
    if sliding_window is not None:
        effective_keys = min(num_keys, sliding_window)
    else:
        effective_keys = num_keys

    # QK^T: 2 * Q * K_eff * dim
    # Attn * V: 2 * Q * K_eff * dim
    sdpa_flops = 4 * num_queries * effective_keys * dim

    return proj_flops + sdpa_flops


def compute_blt_flops(
    seq_len_bytes: int,
    avg_patch_size: float = 4.5,
    byte_dim: int = 256,
    patch_dim: int = 512,
    encoder_layers: int = 1,
    latent_layers: int = 8,
    decoder_layers: int = 4,
    encoder_window: int = 256,
    decoder_window: int = 256,
    vocab_size: int = 260,
) -> Dict[str, float]:
    """Calculate exact forward pass FLOPs for Byte Latent Transformer.

    Args:
        seq_len_bytes: Total number of bytes in the sequence T.
        avg_patch_size: Average bytes per patch (default: 4.5).
        byte_dim: Hidden dimension of local encoder and decoder (d_E).
        patch_dim: Hidden dimension of latent transformer (d_P).
        encoder_layers: Number of local encoder layers n_enc.
        latent_layers: Number of latent transformer layers n_lat.
        decoder_layers: Number of local decoder layers n_dec.
        encoder_window: Local attention window for encoder w_E.
        decoder_window: Local attention window for decoder w_D.
        vocab_size: Byte output vocabulary size (260).

    Returns:
        Dictionary breakdown of FLOPs by component and total.
    """
    T = seq_len_bytes
    M = max(1, int(math.ceil(T / avg_patch_size)))
    enc_ffn_dim = int(2 * (4 * byte_dim) / 3)
    lat_ffn_dim = int(2 * (4 * patch_dim) / 3)

    # 1. Local Encoder
    # n_enc layers of sliding window attention + SwiGLU
    enc_layer_flops = (
        count_attention_flops(byte_dim, T, sliding_window=encoder_window)
        + count_swiglu_flops(byte_dim, enc_ffn_dim, T)
    )
    enc_total_flops = encoder_layers * enc_layer_flops

    # 2. Encoder Cross-Attention (pooling + Q, K, V, Out + SDPA)
    # Q: M x byte_dim -> patch_dim
    # K, V: T x byte_dim -> patch_dim
    # Out: M x patch_dim -> patch_dim
    enc_cross_proj = (
        count_linear_flops(byte_dim, patch_dim, M)
        + 2 * count_linear_flops(byte_dim, patch_dim, T)
        + count_linear_flops(patch_dim, patch_dim, M)
    )
    # SDPA: each patch queries only its ~avg_patch_size bytes
    enc_cross_sdpa = 4 * M * int(avg_patch_size) * patch_dim
    enc_cross_total = enc_cross_proj + enc_cross_sdpa

    # 3. Latent Transformer
    # n_lat layers of causal attention over M patches + SwiGLU
    lat_layer_flops = (
        count_attention_flops(patch_dim, M)
        + count_swiglu_flops(patch_dim, lat_ffn_dim, M)
    )
    lat_total_flops = latent_layers * lat_layer_flops

    # 4. Decoder Cross-Attention
    # Q: T x byte_dim -> byte_dim
    # K, V: M x patch_dim -> byte_dim
    # Out: T x byte_dim -> byte_dim
    dec_cross_proj = (
        count_linear_flops(byte_dim, byte_dim, T)
        + 2 * count_linear_flops(patch_dim, byte_dim, M)
        + count_linear_flops(byte_dim, byte_dim, T)
    )
    # SDPA: byte queries attend to average M/2 preceding patches
    dec_cross_sdpa = 4 * T * (M // 2) * byte_dim
    dec_cross_total = dec_cross_proj + dec_cross_sdpa

    # 5. Local Decoder
    # n_dec layers of sliding window attention + SwiGLU + LM head
    dec_layer_flops = (
        count_attention_flops(byte_dim, T, sliding_window=decoder_window)
        + count_swiglu_flops(byte_dim, enc_ffn_dim, T)
    )
    dec_total_flops = decoder_layers * dec_layer_flops
    lm_head_flops = count_linear_flops(byte_dim, vocab_size, T)

    total_forward_flops = (
        enc_total_flops
        + enc_cross_total
        + lat_total_flops
        + dec_cross_total
        + dec_total_flops
        + lm_head_flops
    )

    return {
        "local_encoder_flops": enc_total_flops,
        "encoder_cross_attn_flops": enc_cross_total,
        "latent_transformer_flops": lat_total_flops,
        "decoder_cross_attn_flops": dec_cross_total,
        "local_decoder_flops": dec_total_flops,
        "lm_head_flops": lm_head_flops,
        "total_forward_flops": total_forward_flops,
        "flops_per_byte": total_forward_flops / max(1, T),
        "patches_count": M,
    }


def compute_baseline_flops(
    seq_len_bytes: int,
    dim: int = 512,
    n_layers: int = 8,
    vocab_size: int = 260,
) -> Dict[str, float]:
    """Calculate forward pass FLOPs for an equivalent stride-1 byte baseline transformer."""
    T = seq_len_bytes
    ffn_dim = int(2 * (4 * dim) / 3)

    layer_flops = (
        count_attention_flops(dim, T)
        + count_swiglu_flops(dim, ffn_dim, T)
    )
    total_layer_flops = n_layers * layer_flops
    lm_head_flops = count_linear_flops(dim, vocab_size, T)

    total = total_layer_flops + lm_head_flops
    return {
        "layers_flops": total_layer_flops,
        "lm_head_flops": lm_head_flops,
        "total_forward_flops": total,
        "flops_per_byte": total / max(1, T),
    }
