from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..codecs.vqre import EMAQuantiser, inverse_sigmoid_alpha, utilisation_kl, VQConfig


def _ensure_vq_state(attn: nn.Module, n_heads: int, head_dim: int, cfg: VQConfig):
    if not hasattr(attn, "_vq_state"):
        qk = nn.ModuleList(
            [
                EMAQuantiser(head_dim, cfg.codebook_k_k, cfg.decay)
                for _ in range(n_heads)
            ]
        )
        qv = nn.ModuleList(
            [
                EMAQuantiser(head_dim, cfg.codebook_k_v, cfg.decay)
                for _ in range(n_heads)
            ]
        )
        try:
            dev = next(attn.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")
        qk.to(dev)
        qv.to(dev)
        attn._vq_state = {"qk": qk, "qv": qv, "cfg": cfg, "losses": {}}
        attn.add_module("_vq_qk", qk)
        attn.add_module("_vq_qv", qv)


def apply_vq_to_distilbert(model: nn.Module, cfg: VQConfig, step_provider):
    if not hasattr(model, "distilbert") or not hasattr(model.distilbert, "transformer"):
        raise ValueError("apply_vq_to_distilbert expects .distilbert.transformer")
    blocks = model.distilbert.transformer.layer
    for block in blocks:
        attn = block.attention
        n_heads = int(attn.n_heads)
        head_dim = int(attn.dim // attn.n_heads)
        _ensure_vq_state(attn, n_heads, head_dim, cfg)
        orig_forward = attn.forward

        def patched_forward(
            self_attn: nn.Module,
            query,
            key,
            value,
            mask,
            head_mask=None,
            output_attentions=False,
        ):
            bs, q_len, dim = query.size()
            k_len = key.size(1)

            q = self_attn.q_lin(query)
            k = self_attn.k_lin(key)
            v = self_attn.v_lin(value)

            def split_heads(x):
                return x.view(
                    bs, -1, self_attn.n_heads, self_attn.dim // self_attn.n_heads
                ).permute(0, 2, 1, 3)

            def merge_heads(x):
                return (
                    x.permute(0, 2, 1, 3)
                    .contiguous()
                    .view(
                        bs, -1, self_attn.n_heads * (self_attn.dim // self_attn.n_heads)
                    )
                )

            q = split_heads(q)
            k = split_heads(k)
            v = split_heads(v)

            cfg_local: VQConfig = self_attn._vq_state["cfg"]
            alpha = inverse_sigmoid_alpha(
                step_provider(), cfg_local.schedule_tau0, cfg_local.schedule_k
            )

            key_mixed, value_mixed = [], []
            for h in range(self_attn.n_heads):
                K = k[:, h]
                V = v[:, h]
                qk = self_attn._vq_state["qk"][h]
                qv = self_attn._vq_state["qv"][h]
                K_q, _, losses_k, usage_k = qk(K)
                V_q, _, losses_v, usage_v = qv(V)
                if self_attn.training:
                    gates = torch.bernoulli(
                        torch.full(K[..., :1].shape, alpha, device=K.device)
                    )
                else:
                    gates = torch.ones_like(K[..., :1])
                K_mix = (1.0 - gates) * K + gates * K_q
                V_mix = (1.0 - gates) * V + gates * V_q

                util_k = (
                    utilisation_kl(usage_k, cfg_local.codebook_k_k) * cfg_local.util_kl
                )
                util_v = (
                    utilisation_kl(usage_v, cfg_local.codebook_k_v) * cfg_local.util_kl
                )
                for pref, (losses, util) in (
                    (f"key_h{h}_", (losses_k, util_k)),
                    (f"value_h{h}_", (losses_v, util_v)),
                ):
                    for name, val in (
                        ("commit", losses["commit"]),
                        ("embed", losses["embed"]),
                        ("util", util),
                    ):
                        prev = self_attn._vq_state["losses"].get(pref + name, 0.0)
                        self_attn._vq_state["losses"][pref + name] = prev + val

                key_mixed.append(K_mix)
                value_mixed.append(V_mix)

            k_q = torch.stack(key_mixed, dim=1)
            v_q = torch.stack(value_mixed, dim=1)

            scale = math.sqrt(self_attn.dim // self_attn.n_heads)
            scores = torch.matmul(q / scale, k_q.transpose(-1, -2))

            if mask is not None:
                mask_ = (mask == 0).view(bs, 1, 1, k_len) if mask.dim() == 2 else mask
                scores = scores.masked_fill(mask_, torch.finfo(scores.dtype).min)

            attn_weights = torch.softmax(scores, dim=-1)
            if hasattr(self_attn, "dropout_attn") and callable(
                getattr(self_attn, "dropout_attn", None)
            ):
                attn_weights = self_attn.dropout_attn(attn_weights)
            elif hasattr(self_attn, "dropout_prob"):
                attn_weights = F.dropout(
                    attn_weights,
                    p=float(self_attn.dropout_prob),
                    training=self_attn.training,
                )
            else:
                drop = getattr(self_attn, "dropout", None)
                if isinstance(drop, nn.Dropout):
                    attn_weights = drop(attn_weights)

            context = torch.matmul(attn_weights, v_q)
            context = merge_heads(context)
            context = self_attn.out_lin(context)
            if hasattr(self_attn, "dropout") and isinstance(
                getattr(self_attn, "dropout", None), nn.Dropout
            ):
                context = self_attn.dropout(context)
            elif hasattr(self_attn, "dropout_prob"):
                context = F.dropout(
                    context,
                    p=float(self_attn.dropout_prob),
                    training=self_attn.training,
                )

            if output_attentions:
                return (context, attn_weights)
            else:
                return (context,)

        attn.forward = patched_forward.__get__(attn, type(attn))
