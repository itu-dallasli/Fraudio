"""Shared SSL-based spoof detection classifier.

Same head architecture for both WavLM Base+ and Wav2Vec2 Base. Padding frames
are excluded from pooling using the encoder's downsampled attention mask.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


def _build_attention_mask(waveform: torch.Tensor, lengths: Optional[torch.Tensor]) -> torch.Tensor:
    """Build [B, T] waveform-level attention mask. 1 = real sample, 0 = padding."""
    B, T = waveform.shape
    if lengths is None:
        return torch.ones(B, T, dtype=torch.long, device=waveform.device)
    idx = torch.arange(T, device=waveform.device).unsqueeze(0)  # [1, T]
    return (idx < lengths.unsqueeze(1)).long()


def _downsampled_mask(model: nn.Module, attn_mask: torch.Tensor, out_len: int) -> torch.Tensor:
    """Project the waveform-level attention mask onto encoder frame timesteps."""
    # transformers' Wav2Vec2/WavLM expose _get_feat_extract_output_lengths.
    raw_lengths = attn_mask.sum(-1)  # [B]
    feat_lengths_fn = getattr(model, "_get_feat_extract_output_lengths", None)
    if feat_lengths_fn is not None:
        out_lengths = feat_lengths_fn(raw_lengths).to(attn_mask.device)
    else:
        # Fallback: assume full mask.
        out_lengths = torch.full_like(raw_lengths, out_len)
    out_lengths = out_lengths.clamp(max=out_len)
    idx = torch.arange(out_len, device=attn_mask.device).unsqueeze(0)
    return (idx < out_lengths.unsqueeze(1)).to(attn_mask.dtype)  # [B, out_len]


def masked_mean_std_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pool [B, T, H] frame features into [B, 2H] with masked mean+std stats."""
    mask = mask.to(hidden.dtype).unsqueeze(-1)        # [B, T, 1]
    denom = mask.sum(dim=1).clamp(min=1.0)            # [B, 1]
    mean = (hidden * mask).sum(dim=1) / denom         # [B, H]
    var = ((hidden - mean.unsqueeze(1)) ** 2 * mask).sum(dim=1) / denom
    std = torch.sqrt(var.clamp(min=1e-8))             # [B, H]
    return torch.cat([mean, std], dim=-1)             # [B, 2H]


class SSLForSpoofDetection(nn.Module):
    """Common WavLM / Wav2Vec2 head for binary bonafide-vs-spoof classification."""

    def __init__(
        self,
        encoder_name: str = "microsoft/wavlm-base-plus",
        hidden_size: int = 768,
        proj_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
        freeze_encoder: bool = False,
        unfreeze_last_n_layers: int = 2,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        config = AutoConfig.from_pretrained(encoder_name)
        # Disable mask_time_indices noise so eval/inference is deterministic.
        config.mask_time_prob = 0.0
        config.mask_feature_prob = 0.0
        self.encoder = AutoModel.from_pretrained(encoder_name, config=config)
        actual_hidden = getattr(self.encoder.config, "hidden_size", hidden_size)
        self.head = nn.Sequential(
            nn.Linear(2 * actual_hidden, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes),
        )
        self.apply_freezing(freeze_encoder, unfreeze_last_n_layers)

    # --------------------------------------------------------------------- #
    # parameter group helpers
    # --------------------------------------------------------------------- #
    def apply_freezing(self, freeze_encoder: bool, unfreeze_last_n_layers: int) -> None:
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            return
        # Partial fine-tune: freeze feature extractor + early transformer layers.
        if hasattr(self.encoder, "feature_extractor"):
            for p in self.encoder.feature_extractor.parameters():
                p.requires_grad = False
        if hasattr(self.encoder, "feature_projection"):
            for p in self.encoder.feature_projection.parameters():
                p.requires_grad = False
        layers = getattr(self.encoder.encoder, "layers", None)
        if layers is not None and unfreeze_last_n_layers > 0:
            n = len(layers)
            cutoff = max(0, n - unfreeze_last_n_layers)
            for i, layer in enumerate(layers):
                req = i >= cutoff
                for p in layer.parameters():
                    p.requires_grad = req

    def trainable_parameter_groups(self, encoder_lr: float, head_lr: float, weight_decay: float):
        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        head_params = list(self.head.parameters())
        groups = []
        if encoder_params:
            groups.append({"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay})
        groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
        return groups

    # --------------------------------------------------------------------- #
    # forward
    # --------------------------------------------------------------------- #
    def forward(
        self,
        waveform: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> dict:
        """waveform: [B, T] float, lengths: [B] long (optional)."""
        attn_mask = _build_attention_mask(waveform, lengths)
        outputs = self.encoder(
            input_values=waveform,
            attention_mask=attn_mask,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state                       # [B, T', H]
        frame_mask = _downsampled_mask(self.encoder, attn_mask, hidden.size(1))
        pooled = masked_mean_std_pool(hidden, frame_mask)        # [B, 2H]
        logits = self.head(pooled)                                # [B, num_classes]
        return {"logits": logits, "pooled": pooled}


def build_model_from_cfg(cfg_model: dict) -> SSLForSpoofDetection:
    return SSLForSpoofDetection(
        encoder_name=cfg_model["encoder_name"],
        hidden_size=cfg_model.get("hidden_size", 768),
        proj_dim=cfg_model.get("proj_dim", 256),
        num_classes=cfg_model.get("num_classes", 2),
        dropout=cfg_model.get("dropout", 0.3),
        freeze_encoder=cfg_model.get("freeze_encoder", False),
        unfreeze_last_n_layers=cfg_model.get("unfreeze_last_n_layers", 2),
    )
