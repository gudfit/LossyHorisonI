from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..bits.bit_cost import binary_entropy_bits


@dataclass
class VQConfig:
    codebook_k_k: int = 512
    codebook_k_v: int = 512
    decay: float = 0.99
    beta_k: float = 0.25
    beta_v: float = 0.25
    gamma_k: float = 0.25
    gamma_v: float = 0.25
    util_kl: float = 1e-3
    schedule_tau0: float = 10_000.0
    schedule_k: float = 2_000.0
    anchors_every: int = 0
    refinement_steps: int = 2
    refinement_topm: int = 8


class EMAQuantiser(nn.Module):

    def __init__(self, dim: int, k: int, decay: float):
        super().__init__()
        self.dim = dim
        self.k = k
        self.decay = decay
        self.register_buffer("codebook", torch.randn(k, dim) * 0.02)
        self.register_buffer("ema_count", torch.zeros(k))
        self.register_buffer("ema_sum", torch.zeros(k, dim))

    @torch.no_grad()
    def _ema_update(self, z: torch.Tensor, indices: torch.Tensor):
        device = z.device
        counts = torch.bincount(indices, minlength=self.k).to(self.ema_count.dtype)
        sums = torch.zeros(self.k, self.dim, device=device, dtype=z.dtype)
        if z.numel() > 0:
            sums.index_add_(0, indices, z)
        self.ema_count.mul_(self.decay).add_(
            counts.to(self.ema_count.device) * (1.0 - self.decay)
        )
        self.ema_sum.mul_(self.decay).add_(sums * (1.0 - self.decay))
        denom = torch.clamp(self.ema_count.unsqueeze(-1), min=1.0)
        self.codebook.copy_(self.ema_sum / denom)

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        z_flat = z.reshape(-1, z.shape[-1])
        # ||z - c||^2 = ||z||^2 + ||c||^2 - 2 z c^T
        z2 = (z_flat**2).sum(dim=1, keepdim=True)
        c2 = (self.codebook**2).sum(dim=1)
        logits = z_flat @ self.codebook.t()  # [N, K]
        dist = z2 + c2.unsqueeze(0) - 2 * logits
        indices = torch.argmin(dist, dim=1)
        z_q = self.codebook[indices].view_as(z)
        z_q_st = z + (z_q - z).detach()
        # Commit: ||sg[z] - z_q||^2 ; Embed: ||z - sg[z_q]||^2
        loss_commit = F.mse_loss(z_q.detach(), z, reduction="mean")
        loss_embed = F.mse_loss(z_q, z.detach(), reduction="mean")
        with torch.no_grad():
            usage = torch.bincount(indices, minlength=self.k).to(z.dtype)
            if usage.sum() > 0:
                usage = usage / usage.sum()

        if self.training:
            self._ema_update(z_flat.detach(), indices.detach())

        return (
            z_q_st,
            indices.view(*z.shape[:-1]),
            {"commit": loss_commit, "embed": loss_embed},
            usage,
        )


def inverse_sigmoid_alpha(step: int, tau0: float, k: float) -> float:
    return float(1.0 / (1.0 + math.exp((tau0 - float(step)) / float(k))))


class HeadVQ(nn.Module):

    def __init__(self, d_h: int, cfg: VQConfig):
        super().__init__()
        self.cfg = cfg
        self.qk = EMAQuantiser(d_h, cfg.codebook_k_k, cfg.decay)
        self.qv = EMAQuantiser(d_h, cfg.codebook_k_v, cfg.decay)

    def forward(
        self, K: torch.Tensor, V: torch.Tensor, step: int
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]
    ]:
        # K,V: [B, T, d_h]
        K_q, idx_k, losses_k, usage_k = self.qk(K)
        V_q, idx_v, losses_v, usage_v = self.qv(V)

        alpha = inverse_sigmoid_alpha(step, self.cfg.schedule_tau0, self.cfg.schedule_k)
        if self.training:
            gates = torch.bernoulli(
                torch.full(K[..., :1].shape, alpha, device=K.device)
            )
        else:
            gates = torch.ones_like(K[..., :1])

        K_mix = (1.0 - gates) * K + gates * K_q
        V_mix = (1.0 - gates) * V + gates * V_q

        losses = {
            "commit_k": self.cfg.beta_k * losses_k["commit"],
            "commit_v": self.cfg.beta_v * losses_v["commit"],
            "embed_k": self.cfg.gamma_k * losses_k["embed"],
            "embed_v": self.cfg.gamma_v * losses_v["embed"],
        }
        util = {"usage_k": usage_k, "usage_v": usage_v}
        return K_mix, V_mix, losses, util


def utilisation_kl(usage: torch.Tensor, k: int) -> torch.Tensor:
    if usage.sum() <= 0:
        return torch.tensor(0.0, device=usage.device)
    p = usage.clamp_min(1e-12)
    u = 1.0 / float(k)
    return torch.sum(p * torch.log(p / u))


class VQREModel:

    def __init__(
        self,
        base_model: Any,
        d_h: int,
        num_layers: int,
        num_heads: int,
        cfg: Optional[VQConfig] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.cfg = cfg or VQConfig()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.q_heads = nn.ModuleList(
            [HeadVQ(d_h, self.cfg) for _ in range(num_layers * num_heads)]
        )
        self._last_losses: Dict[str, torch.Tensor] = {}
        self._vq_patched: bool = False
        self._global_step: int = 0

    def head(self, layer: int, head: int) -> HeadVQ:
        return self.q_heads[layer * self.num_heads + head]

    def attention_kv_hook(
        self, layer: int, head: int, K: torch.Tensor, V: torch.Tensor, step: int
    ):
        K_mix, V_mix, losses, util = self.head(layer, head)(K, V, step)
        for k, v in losses.items():
            self._last_losses[k] = self._last_losses.get(k, 0.0) + v
        for tag, usage in util.items():
            k_size = (
                self.cfg.codebook_k_k if tag.endswith("k") else self.cfg.codebook_k_v
            )
            kl = utilisation_kl(usage, k_size) * self.cfg.util_kl
            self._last_losses[f"util_{tag}"] = (
                self._last_losses.get(f"util_{tag}", 0.0) + kl
            )
        return K_mix, V_mix

    def _ensure_vq_hooks(self):
        if self._vq_patched:
            return

        try:
            from ..models.bert_vq import (
                apply_vq_to_bert,
            )
        except Exception:
            return

        def step_provider() -> int:
            return self._global_step

        try:
            apply_vq_to_bert(self.base_model, self.cfg, step_provider)
            self._vq_patched = True
        except Exception:
            self._vq_patched = False

    def forward_quantised(
        self, input_ids=None, attention_mask=None, step: Optional[int] = None, **kwargs
    ):
        self._ensure_vq_hooks()
        if step is not None:
            self._global_step = int(step)
        else:
            self._global_step += 1

        self._last_losses = {}
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

        try:
            from ..models.bert_vq import collect_vq_losses_from_bert

            losses = collect_vq_losses_from_bert(self.base_model)
            for k, v in losses.items():
                self._last_losses[k] = self._last_losses.get(k, 0.0) + v
        except Exception:
            pass

        return outputs

    def losses(self) -> Dict[str, torch.Tensor]:
        return self._last_losses


def select_anchors(T: int, every: int) -> torch.Tensor:
    if every <= 0:
        return torch.zeros(T, dtype=torch.bool)
    mask = torch.zeros(T, dtype=torch.bool)
    mask[::every] = True
    return mask


def anchor_rate_bits(
    T: int, f_a: float, anchor_count: int, avg_token_bits: float
) -> float:
    return T * binary_entropy_bits(f_a) + anchor_count * avg_token_bits


@torch.no_grad()
def refine_decode(
    tokenizer, model, input_ids: torch.Tensor, steps: int = 2, topm: int = 8
) -> torch.Tensor:
    ids = input_ids.clone()
    for _ in range(max(0, steps)):
        logits = model(input_ids=ids).logits  # [B, T, V]
        probs = F.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)  # [B, T]
        # Uncertainty u = 1 - conf; pick top-M positions per batch
        B, T = conf.shape
        for b in range(B):
            u = 1.0 - conf[b]
            m = min(topm, T)
            idx = torch.topk(u, k=m).indices
            ids[b, idx] = pred[b, idx]
    return ids
