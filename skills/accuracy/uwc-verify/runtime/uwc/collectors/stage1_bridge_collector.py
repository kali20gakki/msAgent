"""Stage 1 collectors: P0 (original HF weights) and P2 (exported HF weights)."""

from __future__ import annotations

import inspect
import os
from pathlib import Path

from ..common import (
    ensure_dir,
    get_dump_dir,
    get_rank,
    get_step,
    is_main_rank,
    log,
    log_warning,
    save_pt,
    sync_device,
    to_cpu,
    write_json,
)

_P0_CACHE: dict[tuple[str, float], dict] = {}


def _torch():
    import torch

    return torch


def _load_safetensors(path: Path) -> dict:
    from safetensors.torch import load_file

    return dict(load_file(str(path), device="cpu"))


def _load_torch_file(path: Path) -> dict:
    torch = _torch()
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        # Legacy checkpoints may contain optimizer state or other non-tensor
        # objects; only then fall back to the full pickle loader.
        log_warning(
            f"{path}: weights_only=True load failed ({exc}); "
            "falling back to weights_only=False. Only load checkpoints from a trusted source."
        )
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        # Some checkpoints wrap the state dict.
        for key in ("state_dict", "model", "module"):
            if key in payload and isinstance(payload[key], dict):
                return dict(payload[key])
        return dict(payload)
    raise TypeError(f"unsupported checkpoint payload in {path}: {type(payload)}")


def _load_weight_file(path: Path) -> dict:
    if path.suffix == ".safetensors":
        return _load_safetensors(path)
    return _load_torch_file(path)


def load_hf_weights(local_path: str | Path) -> dict:
    """Load an HF checkpoint from a directory or a single weight file.

    Supports safetensors shards, ``pytorch_model.bin`` shards and plain
    ``.pt``/``.bin`` state dicts.  Returns CPU tensors.
    """
    path = Path(local_path)
    if not path.exists():
        raise FileNotFoundError(f"model path does not exist: {path}")
    if path.is_file():
        return _load_weight_file(path)

    weights: dict = {}
    safetensor_files = sorted(path.glob("*.safetensors"))
    if safetensor_files:
        for file in safetensor_files:
            weights.update(_load_safetensors(file))
        return weights

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = path / index_name
        if index_path.is_file():
            import json

            with open(index_path, "r", encoding="utf-8") as handle:
                index = json.load(handle)
            shard_files = sorted(set(index.get("weight_map", {}).values()))
            for shard in shard_files:
                weights.update(_load_weight_file(path / shard))
            return weights

    for bin_name in ("pytorch_model.bin", "model.bin"):
        bin_path = path / bin_name
        if bin_path.is_file():
            return _load_torch_file(bin_path)

    torch_files = sorted(path.glob("*.pt")) + sorted(path.glob("*.pth"))
    if len(torch_files) == 1:
        return _load_torch_file(torch_files[0])
    if torch_files:
        for file in torch_files:
            payload = _load_torch_file(file)
            if isinstance(payload, dict):
                weights.update(payload)
        return weights

    raise FileNotFoundError(f"no HF weight files found under {path}")


def load_p0_weights(local_path: str | Path) -> dict:
    """Load P0 with a small path+mtime cache (used by the sender diagnostics)."""
    path = Path(local_path)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    key = (str(path.resolve()), mtime)
    if key not in _P0_CACHE:
        _P0_CACHE[key] = load_hf_weights(path)
    return _P0_CACHE[key]


def collect_p0(bridge, local_path, dump_dir=None, step=None) -> str | None:
    """Dump P0: original HF weights before ``load_weights``.  Rank 0 only."""
    if not is_main_rank():
        return None
    step = get_step() if step is None else step
    dump_dir = Path(dump_dir) if dump_dir else get_dump_dir()
    if not local_path:
        log_warning("stage1 P0: local_path is empty, skip")
        return None
    weights = load_hf_weights(local_path)
    filtered = {
        name: to_cpu(tensor)
        for name, tensor in weights.items()
        if "rope_freqs" not in name and "rotary_emb.inv_freq" not in name
    }
    out_dir = ensure_dir(dump_dir / "stage1_bridge")
    out_path = out_dir / f"P0_hf_original_step{step}_rank{get_rank()}.pt"
    save_pt(out_path, filtered, _torch())
    log(f"stage1 P0 saved to {out_path}, keys={len(filtered)}")
    return str(out_path)


def _find_export_fn(bridge):
    for name in ("export_weights", "export_hf_weights"):
        fn = getattr(bridge, name, None)
        if callable(fn):
            return name, fn
    return None, None


def export_hf_weights(bridge, module, keep: bool = True) -> dict:
    """Export mcore weights back to HF names, version-tolerantly.

    Handles both ``export_weights(models)`` (Megatron-Bridge) and
    ``export_hf_weights(models, conversion_tasks=...)`` (newer verl/mbridge).
    ``keep=False`` still consumes the generator (required for collectives) but
    does not retain CPU copies on non-zero ranks.
    """
    models = module if isinstance(module, (list, tuple)) else [module]
    name, fn = _find_export_fn(bridge)
    if fn is None:
        raise AttributeError("bridge has neither export_weights nor export_hf_weights")

    kwargs = {}
    try:
        parameters = inspect.signature(fn).parameters
        if "conversion_tasks" in parameters:
            get_tasks = getattr(bridge, "get_conversion_tasks", None)
            if callable(get_tasks):
                kwargs["conversion_tasks"] = get_tasks(models)
        if "merge_adapter_weights" in parameters:
            kwargs["merge_adapter_weights"] = False
    except (TypeError, ValueError):
        parameters = {}

    try:
        generator = fn(models, **kwargs)
    except TypeError:
        generator = fn(models)

    weights = {}
    for item in generator:
        try:
            key, tensor = item
        except (TypeError, ValueError):
            raise TypeError(f"unexpected export item from {name}: {type(item)}")
        if keep:
            weights[str(key)] = to_cpu(tensor)
    return weights


def collect_p2(bridge, module, dump_dir=None, step=None) -> str | None:
    """Dump P2: HF weights exported after ``load_weights``.

    ``export_weights`` performs collectives (all_gather/broadcast), so *every*
    rank must consume the generator even though only rank 0 writes to disk.
    """
    step = get_step() if step is None else step
    dump_dir = Path(dump_dir) if dump_dir else get_dump_dir()
    keep = is_main_rank()
    try:
        weights = export_hf_weights(bridge, module, keep=keep)
    except Exception as exc:  # best-effort: never break training
        log_warning(f"stage1 P2 export failed: {exc}")
        return None
    if not keep:
        return None
    out_dir = ensure_dir(dump_dir / "stage1_bridge")
    out_path = out_dir / f"P2_hf_exported_step{step}_rank{get_rank()}.pt"
    save_pt(out_path, weights, _torch())
    log(f"stage1 P2 saved to {out_path}, keys={len(weights)}")
    return str(out_path)


def diag_export_weights(weights, local_path, dump_dir=None, step=None) -> list[dict]:
    """Print the first three exported tensors and compare them against P0.

    Mirrors the document's ``[UWC export diag]`` line.  Returns the diagnostics
    as a list so callers can include them in a report.
    """
    if not is_main_rank():
        return []
    step = get_step() if step is None else step
    try:
        p0 = load_p0_weights(local_path) if local_path else None
    except Exception as exc:
        log_warning(f"export diag: P0 load failed: {exc}")
        p0 = None

    diagnostics = []
    for index, (name, tensor) in enumerate(weights):
        if index >= 3:
            break
        try:
            import torch

            tf = tensor.float()
            line = (
                f"stage2 export diag {name}: device={tensor.device}, dtype={tensor.dtype}, "
                f"shape={list(tensor.shape)}, mean={tf.mean().item():.6f}, std={tf.std().item():.6f}"
            )
            entry = {
                "name": name,
                "device": str(tensor.device),
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "mean": float(tf.mean().item()),
                "std": float(tf.std().item()),
            }
            if p0 is not None:
                candidates = [name, name.replace("model.model.", "model."), name.replace("model.", "model.model.", 1)]
                key = next((candidate for candidate in candidates if candidate in p0), None)
                if key is not None:
                    max_diff = (tf.cpu() - p0[key].float()).abs().max().item()
                    line += f", vs_P0_max_diff={max_diff:.6f}"
                    entry["vs_P0_max_diff"] = float(max_diff)
                else:
                    line += ", vs_P0=key_not_found"
                    entry["vs_P0_max_diff"] = None
            log(line)
            diagnostics.append(entry)
        except Exception as exc:
            log_warning(f"export diag failed for {name}: {exc}")
    if dump_dir is not None:
        try:
            write_json(Path(dump_dir) / "stage1_bridge" / f"export_diag_step{step}.json", diagnostics)
        except Exception:
            pass
    return diagnostics


def collect_p2_sender_diag(weights, local_path, dump_dir=None, step=None) -> None:
    """Materialise a generator, run the export diagnostic and keep tensors."""
    materialised = [(str(name), tensor.clone() if hasattr(tensor, "clone") else tensor) for name, tensor in weights]
    diag_export_weights(materialised, local_path, dump_dir=dump_dir, step=step)
    return materialised
