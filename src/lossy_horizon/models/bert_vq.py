from __future__ import annotations

from typing import Callable, Dict, Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..codecs.vqre import EMAQuantiser, inverse_sigmoid_alpha, utilisation_kl, VQConfig


def _ensure_vq_state(
    attn_self: nn.Module, num_heads: int, head_dim: int, cfg: VQConfig
):
    if not hasattr(attn_self, "_vq_state"):
        qk = [
            EMAQuantiser(head_dim, cfg.codebook_k_k, cfg.decay)
            for _ in range(num_heads)
        ]
        qv = [
            EMAQuantiser(head_dim, cfg.codebook_k_v, cfg.decay)
            for _ in range(num_heads)
        ]
        attn_self._vq_state = {
            "qk": nn.ModuleList(qk),
            "qv": nn.ModuleList(qv),
            "cfg": cfg,
            "losses": {},
        }
        attn_self.add_module("_vq_qk", attn_self._vq_state["qk"])
        attn_self.add_module("_vq_qv", attn_self._vq_state["qv"])


def apply_vq_to_bert(
    model: nn.Module, cfg: VQConfig, step_provider: Callable[[], int]
) -> None:
    if not hasattr(model, "bert") or not hasattr(model.bert, "encoder"):
        raise ValueError(
            "apply_vq_to_bert expects a BertModel-like object with .bert.encoder"
        )
    layers = model.bert.encoder.layer
    for layer_idx, layer in enumerate(layers):
        attn = layer.attention.self
        if hasattr(attn, "num_attention_heads"):
            num_heads = int(attn.num_attention_heads)
        else:
            num_heads = int(getattr(model.config, "num_attention_heads"))
        if hasattr(attn, "attention_head_size"):
            head_dim = int(attn.attention_head_size)
        else:
            hidden_size = int(getattr(model.config, "hidden_size"))
            head_dim = hidden_size // num_heads
        _ensure_vq_state(attn, num_heads, head_dim, cfg)
        state = attn._vq_state

        orig_forward = attn.forward

        def patched_forward(
            self_attn: nn.Module,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            output_attentions: bool = False,
            past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            **kwargs,
        ):
            mixed_query_layer = self_attn.query(hidden_states)  # [B, T, H*D]

            def transpose_for_scores(x: torch.Tensor) -> torch.Tensor:
                new_x_shape = x.size()[:-1] + (
                    self_attn.num_attention_heads,
                    self_attn.attention_head_size,
                )
                x = x.view(*new_x_shape)
                return x.permute(0, 2, 1, 3)  # [B, H, T, D]

            query_layer = transpose_for_scores(mixed_query_layer)

            if encoder_hidden_states is not None:
                pkv = past_key_value if past_key_value is not None else past_key_values
                fw_kwargs = dict(kwargs)
                if "past_key_values" in fw_kwargs:
                    fw_kwargs.pop("past_key_values")
                return orig_forward(
                    hidden_states,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    pkv,
                    output_attentions,
                    **fw_kwargs,
                )

            # Self-attention
            mixed_key_layer = self_attn.key(hidden_states)
            mixed_value_layer = self_attn.value(hidden_states)
            key_layer = transpose_for_scores(mixed_key_layer)  # [B, H, T, D]
            value_layer = transpose_for_scores(mixed_value_layer)  # [B, H, T, D]

            # Quantise K,V per head
            cfg_local: VQConfig = self_attn._vq_state["cfg"]
            alpha = inverse_sigmoid_alpha(
                step_provider(), cfg_local.schedule_tau0, cfg_local.schedule_k
            )

            B, H, T, D = key_layer.shape
            key_mixed = []
            value_mixed = []
            for h in range(H):
                K = key_layer[:, h]  # [B, T, D]
                V = value_layer[:, h]  # [B, T, D]
                quant_k: EMAQuantiser = self_attn._vq_state["qk"][h]
                quant_v: EMAQuantiser = self_attn._vq_state["qv"][h]
                K_q, idx_k, losses_k, usage_k = quant_k(K)
                V_q, idx_v, losses_v, usage_v = quant_v(V)

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
                key_pref_k = f"key_h{h}_"
                key_pref_v = f"value_h{h}_"
                for name, val in (
                    ("commit", losses_k["commit"]),
                    ("embed", losses_k["embed"]),
                    ("util", util_k),
                ):
                    prev = self_attn._vq_state["losses"].get(key_pref_k + name, 0.0)
                    self_attn._vq_state["losses"][key_pref_k + name] = prev + val
                for name, val in (
                    ("commit", losses_v["commit"]),
                    ("embed", losses_v["embed"]),
                    ("util", util_v),
                ):
                    prev = self_attn._vq_state["losses"].get(key_pref_v + name, 0.0)
                    self_attn._vq_state["losses"][key_pref_v + name] = prev + val

                key_mixed.append(K_mix)
                value_mixed.append(V_mix)

            key_layer_q = torch.stack(key_mixed, dim=1)  # [B, H, T, D]
            value_layer_q = torch.stack(value_mixed, dim=1)  # [B, H, T, D]

            # Compute attention logits with quantised K
            attention_scores = torch.matmul(query_layer, key_layer_q.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(
                self_attn.attention_head_size
            )

            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask

            attention_probs = nn.Softmax(dim=-1)(attention_scores)

            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            attention_probs = self_attn.dropout(attention_probs)

            context_layer = torch.matmul(attention_probs, value_layer_q)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (
                self_attn.all_head_size,
            )
            context_layer = context_layer.view(*new_context_layer_shape)

            outputs = (context_layer,)
            if output_attentions:
                outputs = outputs + (attention_probs,)
            return outputs

        attn.forward = patched_forward.__get__(attn, type(attn))


def collect_vq_losses_from_bert(model: nn.Module) -> Dict[str, torch.Tensor]:
    losses: Dict[str, torch.Tensor] = {}
    layers = model.bert.encoder.layer if hasattr(model, "bert") else []
    for layer in layers:
        attn = layer.attention.self
        if hasattr(attn, "_vq_state"):
            for k, v in attn._vq_state["losses"].items():
                losses[k] = losses.get(k, 0.0) + v
            attn._vq_state["losses"] = {}
    return losses
