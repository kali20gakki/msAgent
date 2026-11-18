"""Name normalisation and combined-projection splitting for UWC comparisons.

Bridge exports are usually HF-named, but depending on the Megatron-Bridge
version they may carry wrapper prefixes (``module.``, ``model.model.``) or use
fused projections (``qkv_proj`` / ``gate_up_proj``).  This module maps both
sides to one canonical HF dictionary.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_WRAPPER_PREFIXES = (
    # ``model.model.`` / ``model.module.`` are wrapped HF models: keep ONE
    # ``model.`` so the canonical name matches ``model.layers...`` exports.
    ("model.model.", "model."),
    ("model.module.", "model."),
    ("module.", ""),
    ("_orig_mod.", ""),
    ("language_model.", "model."),
)

_QKV_RE = re.compile(r"^(?P<prefix>.*\.)qkv_proj\.(?P<kind>weight|bias)$")
_GATE_UP_RE = re.compile(r"^(?P<prefix>.*\.)gate_up_proj\.(?P<kind>weight|bias)$")


def strip_wrapper_prefix(name: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix, replacement in _WRAPPER_PREFIXES:
            if name.startswith(prefix):
                name = replacement + name[len(prefix) :]
                changed = True
                break
    return name


def canonical_name(name: str) -> str:
    return strip_wrapper_prefix(name)


def load_config(config_path: str | Path | None) -> dict:
    """Load an HF ``config.json`` (or a directory containing one)."""
    if not config_path:
        return {}
    path = Path(config_path)
    if path.is_dir():
        path = path / "config.json"
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    # Multimodal configs keep the text model settings under "text_config".
    if isinstance(config.get("text_config"), dict):
        merged = dict(config)
        merged.update(config["text_config"])
        return merged
    return config


def _shape_rows(tensor, dim: int = 0) -> int | None:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return None
    try:
        return int(shape[dim])
    except (IndexError, TypeError, ValueError):
        return None


def _split_qkv(tensor, config: dict, layout: str = "contiguous"):
    """Split a fused QKV projection into (q, k, v) along dim 0."""
    hidden = int(config.get("hidden_size") or config.get("n_embd") or 0)
    num_heads = int(config.get("num_attention_heads") or config.get("n_head") or 0)
    num_kv_heads = int(config.get("num_key_value_heads") or num_heads or 0)
    head_dim = int(config.get("head_dim") or (hidden // num_heads if num_heads else 0))
    if not (num_heads and head_dim and num_kv_heads):
        return None

    q_rows = num_heads * head_dim
    kv_rows = num_kv_heads * head_dim
    rows = _shape_rows(tensor)
    if rows != q_rows + 2 * kv_rows:
        return None

    import torch

    if not isinstance(tensor, torch.Tensor):
        return None

    if layout == "interleaved" and num_heads % num_kv_heads == 0:
        group = num_heads // num_kv_heads
        rest = list(tensor.shape[1:])
        grouped = tensor.view(num_kv_heads, group + 2, head_dim, *rest)
        q = grouped[:, :group].reshape(q_rows, *rest).contiguous()
        k = grouped[:, group].reshape(kv_rows, *rest).contiguous()
        v = grouped[:, group + 1].reshape(kv_rows, *rest).contiguous()
        return q, k, v

    q = tensor[:q_rows].contiguous()
    k = tensor[q_rows : q_rows + kv_rows].contiguous()
    v = tensor[q_rows + kv_rows :].contiguous()
    return q, k, v


def _split_gate_up(tensor, layout: str = "contiguous"):
    """Split a fused gate/up projection into two equal halves."""
    rows = _shape_rows(tensor)
    if rows is None or rows % 2 != 0:
        return None
    import torch

    if not isinstance(tensor, torch.Tensor):
        return None
    half = rows // 2
    if layout == "interleaved":
        # [gate0, up0, gate1, up1, ...] layout.
        rest = list(tensor.shape[1:])
        paired = tensor.view(half, 2, *rest)
        gate = paired[:, 0].contiguous()
        up = paired[:, 1].contiguous()
        return gate, up
    return tensor[:half].contiguous(), tensor[half:].contiguous()


def normalize_weights(
    weights,
    model_config: dict | str | Path | None = None,
    qkv_layout: str = "contiguous",
    gate_up_layout: str = "contiguous",
) -> tuple[dict, dict]:
    """Return ``(canonical_dict, meta)``.

    Combined projections are split when a usable model config is supplied;
    otherwise they are kept under their original name and reported in
    ``meta["unresolved_combined"]`` so the caller can surface a clear failure.
    """
    if isinstance(weights, dict):
        items = list(weights.items())
    elif hasattr(weights, "items"):
        items = list(weights.items())
    else:
        items = list(weights)

    config = load_config(model_config) if not isinstance(model_config, dict) else model_config
    result: dict = {}
    meta = {
        "input_keys": len(items),
        "unresolved_combined": [],
        "split_qkv": [],
        "split_gate_up": [],
        "renamed": [],
    }

    # First pass: canonical names for non-combined keys.  Separate q/k/v keys
    # win over a fused key if both are present.
    fused = []
    for name, tensor in items:
        canonical = canonical_name(str(name))
        if canonical != name:
            meta["renamed"].append([str(name), canonical])
        if _QKV_RE.match(canonical):
            fused.append((canonical, tensor, "qkv"))
        elif _GATE_UP_RE.match(canonical):
            fused.append((canonical, tensor, "gate_up"))
        else:
            result[canonical] = tensor

    for canonical, tensor, kind in fused:
        if kind == "qkv":
            match = _QKV_RE.match(canonical)
            prefix, field = match.group("prefix"), match.group("kind")
            split = _split_qkv(tensor, config, qkv_layout)
            if split is None:
                meta["unresolved_combined"].append(canonical)
                result[canonical] = tensor
                continue
            for suffix, value in zip(("q_proj", "k_proj", "v_proj"), split):
                target = f"{prefix}{suffix}.{field}"
                if target not in result:  # explicit separate keys win
                    result[target] = value
            meta["split_qkv"].append(canonical)
        else:
            match = _GATE_UP_RE.match(canonical)
            prefix, field = match.group("prefix"), match.group("kind")
            split = _split_gate_up(tensor, gate_up_layout)
            if split is None:
                meta["unresolved_combined"].append(canonical)
                result[canonical] = tensor
                continue
            for suffix, value in zip(("gate_proj", "up_proj"), split):
                target = f"{prefix}{suffix}.{field}"
                if target not in result:
                    result[target] = value
            meta["split_gate_up"].append(canonical)

    return result, meta


def normalize_weights_only(
    weights,
    model_config: dict | str | Path | None = None,
    qkv_layout: str = "contiguous",
    gate_up_layout: str = "contiguous",
) -> dict:
    """Convenience wrapper returning only the canonical dictionary."""
    return normalize_weights(
        weights,
        model_config=model_config,
        qkv_layout=qkv_layout,
        gate_up_layout=gate_up_layout,
    )[0]
