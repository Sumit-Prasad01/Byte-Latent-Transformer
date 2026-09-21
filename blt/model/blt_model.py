"""Byte Latent Transformer (BLT) Architecture Assembly (Meta FAIR, Pagnoni et al., Dec 2024).

Wires the full 4-stage pipeline:
1. Byte and Hash n-gram Embeddings (§3.1)
2. Local Encoder with sliding-window causal self-attention + Encoder Cross-Attention (§3.2)
3. Latent Transformer with block-causal global reasoning & gradient checkpointing (§3.3)
4. Local Decoder with causal Decoder Cross-Attention + next-byte projection head (§3.3.1)
"""

from typing import Optional, Tuple, Union, Dict, Any, List
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from blt.data.dataset import TOTAL_VOCAB_SIZE
from blt.modules.embeddings import ByteHashNgramEmbedding
from blt.modules.local_encoder import LocalEncoder
from blt.modules.latent_transformer import LatentTransformer
from blt.modules.local_decoder import LocalDecoder
from blt.patching.patchers import StreamingPatcher, entropy_patcher_monotonic


def compute_patch_indices_from_boundaries(patch_boundaries: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Convert binary/boolean patch start boundaries to integer patch IDs.

    Args:
        patch_boundaries: (batch, seq_len) with 1/True at patch start bytes.

    Returns:
        patch_indices: (batch, seq_len) with integer patch IDs in [0, num_patches - 1].
        num_patches: Total number of patches.
    """
    bounds = patch_boundaries.clone().long()
    bounds[:, 0] = 1  # First byte is always start of patch 0
    indices = torch.cumsum(bounds, dim=-1) - 1
    num_patches = int(indices.max().item()) + 1
    return indices, num_patches


def create_strided_patch_indices(
    batch_size: int,
    seq_len: int,
    stride: int = 4,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, int]:
    """Vectorized creation of uniform strided patch indices (t // stride)."""
    t = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    indices = torch.div(t, stride, rounding_mode="floor")
    num_patches = int(indices.max().item()) + 1
    return indices, num_patches


class ByteLatentTransformer(nn.Module):
    """Complete Byte Latent Transformer (BLT) model.

    Scalable byte-level language model operating over dynamically discovered patches
    in latent space, enabling subword-free LLM pretraining and inference.
    """

    def __init__(
        self,
        vocab_size: int = TOTAL_VOCAB_SIZE,  # 260
        byte_dim: int = 256,
        patch_dim: int = 512,
        encoder_layers: int = 1,
        encoder_heads: int = 4,
        encoder_window: int = 256,
        latent_layers: int = 8,
        latent_heads: int = 8,
        latent_max_patches: int = 2048,
        decoder_layers: int = 4,
        decoder_heads: int = 4,
        decoder_window: int = 256,
        cross_attn_heads: int = 4,
        ngram_sizes: Tuple[int, ...] = (3, 4, 5),
        vocab_per_ngram: int = 20000,
        hash_prime: int = 31337,
        max_seq_len: int = 4096,
        rope_theta: float = 500000.0,
        grad_checkpointing: bool = True,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
        default_patch_size: int = 4,
        entropy_model: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.byte_dim = byte_dim
        self.patch_dim = patch_dim
        self.max_seq_len = max_seq_len
        self.grad_checkpointing = grad_checkpointing
        self.default_patch_size = default_patch_size
        self.entropy_model = entropy_model

        # 1. Byte + Hash n-gram Embedding layer
        self.embeddings = ByteHashNgramEmbedding(
            dim=byte_dim,
            byte_vocab_size=vocab_size,
            ngram_sizes=ngram_sizes,
            vocab_per_ngram=vocab_per_ngram,
            hash_prime=hash_prime,
        )

        # 2. Local Encoder with EncoderCrossAttention
        self.local_encoder = LocalEncoder(
            dim=byte_dim,
            n_layers=encoder_layers,
            n_heads=encoder_heads,
            window_size=encoder_window,
            patch_dim=patch_dim,
            cross_attn_heads=cross_attn_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            dropout=dropout,
            norm_eps=norm_eps,
        )

        # 3. Latent Transformer
        self.latent_transformer = LatentTransformer(
            dim=patch_dim,
            n_layers=latent_layers,
            n_heads=latent_heads,
            max_seq_len=latent_max_patches,
            rope_theta=rope_theta,
            grad_checkpointing=grad_checkpointing,
            dropout=dropout,
            norm_eps=norm_eps,
        )

        # 4. Local Decoder with DecoderCrossAttention and LM Head
        self.local_decoder = LocalDecoder(
            dim=byte_dim,
            patch_dim=patch_dim,
            n_layers=decoder_layers,
            n_heads=decoder_heads,
            cross_attn_heads=cross_attn_heads,
            window_size=decoder_window,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            dropout=dropout,
            norm_eps=norm_eps,
        )

    @classmethod
    def from_config(cls, config_source: Union[str, Dict[str, Any]], entropy_model: Optional[nn.Module] = None):
        """Construct ByteLatentTransformer from a YAML config file or dictionary."""
        if isinstance(config_source, str):
            with open(config_source, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = config_source

        m_cfg = cfg["model"]
        return cls(
            vocab_size=m_cfg.get("vocab_size", TOTAL_VOCAB_SIZE),
            byte_dim=m_cfg["local_encoder"]["hidden"],
            patch_dim=m_cfg["latent_transformer"]["hidden"],
            encoder_layers=m_cfg["local_encoder"]["layers"],
            encoder_heads=m_cfg["local_encoder"]["heads"],
            encoder_window=m_cfg["local_encoder"].get("window_size", 256),
            latent_layers=m_cfg["latent_transformer"]["layers"],
            latent_heads=m_cfg["latent_transformer"]["heads"],
            latent_max_patches=m_cfg["latent_transformer"].get("max_patches", 2048),
            decoder_layers=m_cfg["local_decoder"]["layers"],
            decoder_heads=m_cfg["local_decoder"]["heads"],
            decoder_window=m_cfg["local_decoder"].get("window_size", 256),
            cross_attn_heads=m_cfg["cross_attention"]["heads"],
            ngram_sizes=tuple(m_cfg["hash_ngrams"]["sizes"]),
            vocab_per_ngram=m_cfg["hash_ngrams"]["vocab_per_size"],
            grad_checkpointing=m_cfg["latent_transformer"].get("grad_checkpointing", True),
            dropout=cfg.get("training", {}).get("dropout", 0.0),
            default_patch_size=int(cfg.get("entropy_model", {}).get("target_avg_patch_size", 4.5)),
            entropy_model=entropy_model,
        )

    def count_parameters(self) -> Dict[str, int]:
        """Return breakdown of trainable parameter counts by module."""
        n_emb = sum(p.numel() for p in self.embeddings.parameters() if p.requires_grad)
        n_enc = sum(p.numel() for p in self.local_encoder.parameters() if p.requires_grad)
        n_lat = sum(p.numel() for p in self.latent_transformer.parameters() if p.requires_grad)
        n_dec = sum(p.numel() for p in self.local_decoder.parameters() if p.requires_grad)
        total = n_emb + n_enc + n_lat + n_dec
        return {
            "embeddings": n_emb,
            "local_encoder": n_enc,
            "latent_transformer": n_lat,
            "local_decoder": n_dec,
            "total": total,
        }

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        patch_boundaries: Optional[torch.Tensor] = None,
        patch_indices: Optional[torch.Tensor] = None,
        precomputed_hash_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through full BLT architecture.

        Args:
            tokens: (batch_size, seq_len) byte token IDs.
            targets: Optional (batch_size, seq_len) target token IDs.
            patch_boundaries: Optional (batch_size, seq_len) boolean/binary patch start flags.
            patch_indices: Optional (batch_size, seq_len) precomputed patch IDs.
            precomputed_hash_indices: Optional precomputed hash indices.

        Returns:
            Tuple of (logits, loss).
            logits: (batch_size, seq_len, vocab_size)
            loss: scalar CrossEntropyLoss or None
        """
        bsz, seq_len = tokens.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        # 1. Resolve patch indices
        if patch_indices is not None:
            p_indices = patch_indices
            num_patches = int(p_indices.max().item()) + 1
        elif patch_boundaries is not None:
            p_indices, num_patches = compute_patch_indices_from_boundaries(patch_boundaries)
        elif self.entropy_model is not None:
            # Dynamically compute patch boundaries via entropy model
            with torch.no_grad():
                entropy = self.entropy_model.compute_entropy(tokens)
                bounds_np = entropy_patcher_monotonic(
                    entropy=entropy.cpu().numpy(),
                    bytes_seq=tokens.cpu().numpy(),
                )
                bounds = torch.from_numpy(bounds_np).to(device=tokens.device)
            p_indices, num_patches = compute_patch_indices_from_boundaries(bounds)
        else:
            # Fallback uniform strided patch indices
            p_indices, num_patches = create_strided_patch_indices(
                batch_size=bsz,
                seq_len=seq_len,
                stride=self.default_patch_size,
                device=tokens.device,
            )

        # 2. Embedding layer: byte + hash n-grams -> (batch, seq_len, byte_dim)
        byte_embed = self.embeddings(tokens, precomputed_hash_indices=precomputed_hash_indices)

        # 3. Local Encoder -> (batch, seq_len, byte_dim) and (batch, num_patches, patch_dim)
        byte_hidden, patch_repr = self.local_encoder(
            byte_embed,
            patch_indices=p_indices,
            num_patches=num_patches,
            use_checkpoint=self.grad_checkpointing,
        )

        # 4. Latent Transformer -> (batch, num_patches, patch_dim)
        latent_patches = self.latent_transformer(
            patch_repr,
            use_checkpoint=self.grad_checkpointing,
        )

        # 5. Local Decoder -> (batch, seq_len, vocab_size)
        logits = self.local_decoder(
            byte_hidden=byte_hidden,
            patch_hidden=latent_patches,
            patch_indices=p_indices,
            use_checkpoint=self.grad_checkpointing,
        )

        # 6. Loss calculation
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
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
        patch_boundaries: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Autoregressive generation of byte tokens with nucleus sampling."""
        self.eval()
        tokens = prompt_tokens.clone()

        for _ in range(max_new_tokens):
            context = tokens[:, -self.max_seq_len:]
            logits, _ = self(context, patch_boundaries=patch_boundaries)
            next_token_logits = logits[:, -1, :] / max(1e-5, temperature)

            # Nucleus (top-p) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least 1 token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("Inf"))

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens
