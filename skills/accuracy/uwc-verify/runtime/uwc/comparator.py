"""Numeric comparison utilities for UWC weight dumps."""

from __future__ import annotations

import math
from typing import Iterable

NON_WEIGHT_SUFFIXES = (
    "rotary_emb.inv_freq",
    "rope_freqs",
    "_extra_state",
)


def is_non_weight_key(name: str) -> bool:
    return any(name.endswith(suffix) or suffix in name for suffix in NON_WEIGHT_SUFFIXES)


def _as_flat_float(tensor):
    import torch

    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    return tensor.detach().float().reshape(-1)


def tensor_stats(tensor) -> dict:
    """Return cheap statistics used in reports (works for empty tensors too)."""
    import torch

    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    flat = tensor.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "numel": 0}
    finite = torch.isfinite(flat)
    if finite.any():
        values = flat[finite]
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": int(flat.numel()),
            "mean": float(values.mean().item()),
            "std": float(values.std().item()) if values.numel() > 1 else 0.0,
            "abs_max": float(values.abs().max().item()),
        }
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": int(flat.numel()),
        "mean": "nan",
        "std": "nan",
        "abs_max": "nan",
    }


def compare_tensors(a, b, atol: float = 1e-3, rtol: float = 0.0, exact: bool = False) -> dict:
    """Compare two tensors and return a JSON-serialisable result."""
    import torch

    if not isinstance(a, torch.Tensor):
        a = torch.as_tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.as_tensor(b)

    result = {
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
        "max_abs_diff": None,
        "mean_abs_diff": None,
        "mismatch": 0,
        "numel": int(a.numel()),
        "passed": False,
        "reason": "",
    }
    if list(a.shape) != list(b.shape):
        result["reason"] = f"shape mismatch: {list(a.shape)} vs {list(b.shape)}"
        return result

    af = _as_flat_float(a)
    bf = _as_flat_float(b)
    if af.numel() == 0:
        result["passed"] = True
        result["max_abs_diff"] = 0.0
        result["mean_abs_diff"] = 0.0
        return result

    nan_a = torch.isnan(af)
    nan_b = torch.isnan(bf)
    nan_mismatch = int((nan_a != nan_b).sum().item())

    finite = ~(nan_a | nan_b)
    if finite.any():
        diff = (af[finite] - bf[finite]).abs()
        max_abs = float(diff.max().item())
        mean_abs = float(diff.mean().item())
    else:
        diff = af.new_empty(0)
        max_abs = 0.0 if nan_mismatch == 0 else float("nan")
        mean_abs = 0.0 if nan_mismatch == 0 else float("nan")

    if exact:
        # Bitwise equality for the original dtype.  NaN == NaN is treated as equal.
        same = a.detach().cpu() == b.detach().cpu()
        same = same | (torch.isnan(a.detach().cpu()) & torch.isnan(b.detach().cpu()))
        mismatch = int((~same).sum().item())
        threshold_desc = "exact"
    else:
        tol = atol + rtol * bf[finite].abs() if finite.any() else af.new_empty(0)
        mismatch = int((diff > tol).sum().item()) if finite.any() else 0
        mismatch += nan_mismatch
        threshold_desc = f"atol={atol:g}" + (f", rtol={rtol:g}" if rtol else "")

    result.update(
        {
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
            "mismatch": mismatch,
            "nan_mismatch": nan_mismatch,
            "threshold": threshold_desc,
            "passed": mismatch == 0,
        }
    )
    if mismatch:
        result["reason"] = f"{mismatch}/{a.numel()} elements exceed {threshold_desc}"
    return result


def _normalise_container(weights):
    """Accept dict[str, Tensor], list[(name, Tensor)] or a generator."""
    if isinstance(weights, dict):
        return dict(weights)
    if hasattr(weights, "items"):
        return dict(weights.items())
    return {str(name): tensor for name, tensor in weights}


def compare_weight_dicts(
    reference,
    candidate,
    atol: float = 1e-3,
    rtol: float = 0.0,
    exact: bool = False,
    ignore_non_weight: bool = True,
    name_normalizer=None,
    worst_k: int = 20,
) -> dict:
    """Compare two name -> tensor mappings.

    ``name_normalizer`` is an optional callable applied to both key sets, e.g.
    ``uwc.name_normalizer.normalize_weights``.  It must return either a dict or
    a ``(dict, meta)`` tuple.
    """
    ref = _normalise_container(reference)
    cand = _normalise_container(candidate)

    if name_normalizer is not None:
        ref = name_normalizer(ref)
        cand = name_normalizer(cand)
        if isinstance(ref, tuple):
            ref = ref[0]
        if isinstance(cand, tuple):
            cand = cand[0]

    if ignore_non_weight:
        ref = {k: v for k, v in ref.items() if not is_non_weight_key(k)}
        cand = {k: v for k, v in cand.items() if not is_non_weight_key(k)}

    missing = sorted(set(ref) - set(cand))
    extra = sorted(set(cand) - set(ref))
    common = sorted(set(ref) & set(cand))

    details = []
    mismatch_count = 0
    num_weights = 0
    global_max = 0.0
    global_sum = 0.0
    for key in common:
        item = compare_tensors(ref[key], cand[key], atol=atol, rtol=rtol, exact=exact)
        num_weights += 1
        if item["mismatch"]:
            mismatch_count += 1
        if item["max_abs_diff"] is not None and not math.isnan(item["max_abs_diff"]):
            global_max = max(global_max, item["max_abs_diff"])
            global_sum += item["max_abs_diff"] * max(item["numel"], 1)
        details.append({"name": key, **item})

    details.sort(key=lambda item: (item["max_abs_diff"] is None, -(item["max_abs_diff"] or 0.0)))
    worst = details[:worst_k]

    passed = not missing and not extra and mismatch_count == 0
    reasons = []
    if missing:
        reasons.append(f"{len(missing)} keys missing in candidate")
    if extra:
        reasons.append(f"{len(extra)} unexpected keys in candidate")
    if mismatch_count:
        reasons.append(f"{mismatch_count}/{num_weights} weights mismatch")

    return {
        "status": "pass" if passed else "fail",
        "reference_keys": len(ref),
        "candidate_keys": len(cand),
        "compared_keys": num_weights,
        "mismatched_keys": mismatch_count,
        "missing_keys": missing[:50],
        "extra_keys": extra[:50],
        "missing_count": len(missing),
        "extra_count": len(extra),
        "max_abs_diff": global_max,
        "mean_abs_diff": (global_sum / sum(max(item["numel"], 1) for item in details)) if details else 0.0,
        "threshold": f"exact" if exact else f"atol={atol:g}" + (f", rtol={rtol:g}" if rtol else ""),
        "worst": worst,
        "reasons": reasons,
    }


def filter_weights(weights, keep: Iterable[str] | None = None, drop: Iterable[str] | None = None):
    keep_set = set(keep) if keep is not None else None
    drop_set = set(drop) if drop is not None else None
    result = {}
    for name, tensor in _normalise_container(weights).items():
        if keep_set is not None and name not in keep_set:
            continue
        if drop_set is not None and name in drop_set:
            continue
        result[name] = tensor
    return result
