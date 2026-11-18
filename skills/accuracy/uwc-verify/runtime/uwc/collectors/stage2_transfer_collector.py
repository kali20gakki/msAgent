"""Stage 2 collectors: P2_sender (sender-side HF weights) and P5 (receiver-side HF weights)."""

from __future__ import annotations

from pathlib import Path

from ..common import (
    ensure_dir,
    get_dump_dir,
    get_rank,
    get_step,
    is_main_rank,
    log,
    save_pt,
    sync_device,
    to_cpu,
    write_json,
)


def _torch():
    import torch

    return torch


def _materialise(weights):
    if isinstance(weights, dict):
        items = list(weights.items())
    elif hasattr(weights, "items"):
        items = list(weights.items())
    else:
        items = list(weights)
    materialised = []
    for name, tensor in items:
        try:
            materialised.append((str(name), to_cpu(tensor)))
        except Exception:
            materialised.append((str(name), tensor))
    return materialised


def collect_p2_sender(weights, dump_dir=None, step=None) -> str | None:
    """Dump the HF weights actually handed to the transport at sync time."""
    if not is_main_rank():
        return None
    step = get_step() if step is None else step
    dump_dir = Path(dump_dir) if dump_dir else get_dump_dir()
    sync_device()
    materialised = _materialise(weights)
    out_dir = ensure_dir(dump_dir / "stage2_transfer")
    out_path = out_dir / f"P2_sender_step{step}_rank{get_rank()}.pt"
    save_pt(out_path, materialised, _torch())
    log(f"stage2 P2_sender saved to {out_path}, keys={len(materialised)}")
    return str(out_path)


def collect_p5(weights, dump_dir=None, step=None) -> str | None:
    """Dump the HF weights received by the rollout worker (``_update_weights``)."""
    if not is_main_rank():
        return None
    step = get_step() if step is None else step
    dump_dir = Path(dump_dir) if dump_dir else get_dump_dir()
    sync_device()
    materialised = _materialise(weights)
    out_dir = ensure_dir(dump_dir / "stage2_transfer")
    out_path = out_dir / f"P5_receiver_step{step}_rank{get_rank()}.pt"
    save_pt(out_path, materialised, _torch())
    log(f"stage2 P5 saved to {out_path}, keys={len(materialised)}")
    return str(out_path)


def collect_transport_metadata(weights, dump_dir=None, step=None) -> str | None:
    """Write a lightweight JSON manifest (names/shapes/dtypes) for triage."""
    if not is_main_rank():
        return None
    step = get_step() if step is None else step
    dump_dir = Path(dump_dir) if dump_dir else get_dump_dir()
    items = []
    for name, tensor in _materialise(weights):
        items.append(
            {
                "name": name,
                "shape": list(getattr(tensor, "shape", [])),
                "dtype": str(getattr(tensor, "dtype", type(tensor).__name__)),
                "nbytes": int(getattr(tensor, "nbytes", 0) or 0),
            }
        )
    payload = {"step": step, "count": len(items), "items": items}
    out = ensure_dir(dump_dir / "stage2_transfer") / f"transport_metadata_step{step}.json"
    write_json(out, payload)
    return str(out)
