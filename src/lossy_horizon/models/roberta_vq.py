from __future__ import annotations

import torch
import torch.nn as nn

from ..codecs.vqre import EMAQuantiser, inverse_sigmoid_alpha, utilisation_kl, VQConfig


def _ensure_vq_state(
    attn_self: nn.Module, num_heads: int, head_dim: int, cfg: VQConfig
):
    if not hasattr(attn_self, "_vq_state"):
        qk = nn.ModuleList(
            [
                EMAQuantiser(head_dim, cfg.codebook_k_k, cfg.decay)
                for _ in range(num_heads)
            ]
        )
        qv = nn.ModuleList(
            [
                EMAQuantiser(head_dim, cfg.codebook_k_v, cfg.decay)
                for _ in range(num_heads)
            ]
        )
        try:
            dev = next(attn_self.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")
        qk.to(dev)
        qv.to(dev)
        attn_self._vq_state = {"qk": qk, "qv": qv, "cfg": cfg, "losses": {}}
        attn_self.add_module("_vq_qk", qk)
        attn_self.add_module("_vq_qv", qv)


def apply_vq_to_roberta(model: nn.Module, cfg: VQConfig, step_provider):
    if not hasattr(model, "roberta") or not hasattr(model.roberta, "encoder"):
        raise ValueError(
            "apply_vq_to_roberta expects a Roberta* model with .roberta.encoder"
        )
    layers = model.roberta.encoder.layer
    for layer in layers:
        attn = layer.attention.self
        num_heads = int(
            getattr(
                attn,
                "num_attention_heads",
                getattr(model.config, "num_attention_heads"),
            )
        )
        head_dim = int(
            getattr(
                attn,
                "attention_head_size",
                getattr(model.config, "hidden_size") // num_heads,
            )
        )
        _ensure_vq_state(attn, num_heads, head_dim, cfg)
        orig_forward = attn.forward

        def patched_forward(
            self_attn: nn.Module,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            head_mask: torch.Tensor | None = None,
            encoder_hidden_states: torch.Tensor | None = None,
            encoder_attention_mask: torch.Tensor | None = None,
            past_key_value=None,
            output_attentions: bool = False,
            **kwargs,
        ):
            mixed_query_layer = self_attn.query(hidden_states)

            def transpose_for_scores(x: torch.Tensor) -> torch.Tensor:
                new_x_shape = x.size()[:-1] + (
                    self_attn.num_attention_heads,
                    self_attn.attention_head_size,
                )
                x = x.view(*new_x_shape)
                return x.permute(0, 2, 1, 3)

            query_layer = transpose_for_scores(mixed_query_layer)

            if encoder_hidden_states is not None:
                return orig_forward(
                    hidden_states,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    **kwargs,
                )

            mixed_key_layer = self_attn.key(hidden_states)
            mixed_value_layer = self_attn.value(hidden_states)
            key_layer = transpose_for_scores(mixed_key_layer)
            value_layer = transpose_for_scores(mixed_value_layer)

            cfg_local = self_attn._vq_state["cfg"]
            alpha = inverse_sigmoid_alpha(
                step_provider(), cfg_local.schedule_tau0, cfg_local.schedule_k
            )

            B, H, T, D = key_layer.shape
            key_mixed, value_mixed = [], []
            for h in range(H):
                K = key_layer[:, h]
                V = value_layer[:, h]
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

            key_layer_q = torch.stack(key_mixed, dim=1)
            value_layer_q = torch.stack(value_mixed, dim=1)

            attention_scores = torch.matmul(query_layer, key_layer_q.transpose(-1, -2))
            attention_scores = attention_scores / (self_attn.attention_head_size**0.5)
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask

            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            if head_mask is not None:
                attention_probs = attention_probs * head_mask
            attention_probs = self_attn.dropout(attention_probs)

            context_layer = torch.matmul(attention_probs, value_layer_q)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            context_layer = context_layer.view(
                *context_layer.size()[:-2], self_attn.all_head_size
            )

            outputs = (context_layer,)
            if output_attentions:
                outputs = outputs + (attention_probs,)
            return outputs

        attn.forward = patched_forward.__get__(attn, type(attn))
