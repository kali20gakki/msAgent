"""Logits comparison for the G0 vs G1 forward verification."""

from __future__ import annotations

import json
from pathlib import Path


def _load_tensor(path: str | Path):
    import torch

    path = Path(path)
    if path.suffix == ".npy":
        import numpy as np

        return torch.from_numpy(np.load(path))
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            for key in ("logits", "G0", "G1", "output"):
                if key in payload:
                    payload = payload[key]
                    break
        return torch.tensor(payload)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict):
        for key in ("logits", "G0", "G1", "output"):
            if key in payload:
                return payload[key]
        raise KeyError(f"no logits key found in {path}: {list(payload)[:10]}")
    return payload


def compare_logits(
    g0,
    g1,
    avg_cos_threshold: float = 0.99,
    min_cos_threshold: float = 0.95,
    max_cos_threshold: float | None = None,
) -> dict:
    """Compare two logits tensors.

    Defaults follow the UWC document: average cosine > 0.99 and min (and max)
    cosine >= 0.95.  Cosine is computed per sequence position over the vocab
    dimension when shapes are 2-D/3-D; otherwise the flattened tensor is used.
    """
    import torch
    import torch.nn.functional as F

    if not isinstance(g0, torch.Tensor):
        g0 = torch.as_tensor(g0)
    if not isinstance(g1, torch.Tensor):
        g1 = torch.as_tensor(g1)

    result = {
        "shape_g0": list(g0.shape),
        "shape_g1": list(g1.shape),
        "dtype_g0": str(g0.dtype),
        "dtype_g1": str(g1.dtype),
        "avg_cos": None,
        "min_cos": None,
        "max_cos": None,
        "max_abs_diff": None,
        "mean_abs_diff": None,
        "num_positions": 0,
        "thresholds": {
            "avg_cos": avg_cos_threshold,
            "min_cos": min_cos_threshold,
            "max_cos": max_cos_threshold,
        },
        "status": "fail",
        "reason": "",
    }
    if list(g0.shape) != list(g1.shape):
        result["reason"] = f"shape mismatch: {list(g0.shape)} vs {list(g1.shape)}"
        return result
    if g0.numel() == 0:
        result.update({"status": "pass", "avg_cos": 1.0, "min_cos": 1.0, "max_cos": 1.0})
        return result

    a = g0.detach().float()
    b = g1.detach().float()
    if a.dim() >= 2:
        a = a.reshape(-1, a.shape[-1])
        b = b.reshape(-1, b.shape[-1])
    else:
        a = a.reshape(1, -1)
        b = b.reshape(1, -1)

    cos = F.cosine_similarity(a, b, dim=-1, eps=1e-12)
    cos = torch.nan_to_num(cos, nan=0.0, posinf=1.0, neginf=-1.0)
    diff = (a - b).abs()

    avg_cos = float(cos.mean().item())
    min_cos = float(cos.min().item())
    max_cos = float(cos.max().item())
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())

    passed = avg_cos > avg_cos_threshold and min_cos >= min_cos_threshold
    if max_cos_threshold is not None:
        passed = passed and max_cos >= max_cos_threshold

    result.update(
        {
            "avg_cos": avg_cos,
            "min_cos": min_cos,
            "max_cos": max_cos,
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
            "num_positions": int(cos.numel()),
            "status": "pass" if passed else "fail",
            "reason": "" if passed else "cosine similarity below threshold",
        }
    )
    return result


def compare_logits_files(g0_path, g1_path, **kwargs) -> dict:
    return compare_logits(_load_tensor(g0_path), _load_tensor(g1_path), **kwargs)
