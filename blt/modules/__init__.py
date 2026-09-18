"""Modular neural network blocks and layers for BLT."""

from blt.modules.transformer_layers import (
    RMSNorm,
    SwiGLU,
    CausalSelfAttention,
    TransformerBlock,
    apply_rope,
    precompute_rope_freqs_cis,
)
from blt.modules.embeddings import ByteHashNgramEmbedding
from blt.modules.cross_attention import (
    EncoderCrossAttention,
    DecoderCrossAttention,
    create_patch_membership_mask,
    create_decoder_causal_patch_mask,
)
from blt.modules.local_encoder import LocalEncoder
from blt.modules.latent_transformer import LatentTransformer
from blt.modules.local_decoder import LocalDecoder

__all__ = [
    "RMSNorm",
    "SwiGLU",
    "CausalSelfAttention",
    "TransformerBlock",
    "apply_rope",
    "precompute_rope_freqs_cis",
    "ByteHashNgramEmbedding",
    "EncoderCrossAttention",
    "DecoderCrossAttention",
    "create_patch_membership_mask",
    "create_decoder_causal_patch_mask",
    "LocalEncoder",
    "LatentTransformer",
    "LocalDecoder",
]
