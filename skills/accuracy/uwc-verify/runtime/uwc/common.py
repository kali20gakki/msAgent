"""Shared helpers for UWC collectors and offline tools."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

LOG_PREFIX = "[UWC]"


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_dump_dir() -> Path:
    return Path(os.environ.get("UWC_DUMP_DIR", "/tmp/uwc_dump"))


def get_step() -> int:
    try:
        return int(os.environ.get("UWC_STEP", "0"))
    except ValueError:
        return 0


def get_rank() -> int:
    """Return global rank, preferring torch.distributed when initialised."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:
        pass
    try:
        return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    except ValueError:
        return 0


def is_main_rank() -> bool:
    return get_rank() == 0


def stage_enabled(stage: int) -> bool:
    """UWC_ENABLE is the master switch; UWC_STAGEn defaults to on when enabled."""
    if not env_bool("UWC_ENABLE"):
        return False
    return env_bool(f"UWC_STAGE{stage}", True)


def log(message: str) -> None:
    print(f"{LOG_PREFIX} {message}", flush=True)


def log_warning(message: str) -> None:
    print(f"{LOG_PREFIX} WARNING: {message}", flush=True)


def sync_device() -> None:
    """Synchronise the active accelerator before copying device memory.

    NPU IPC / shared-memory transfers are asynchronous; copying without a
    synchronise may read a partially written buffer.
    """
    try:
        import torch

        npu = getattr(torch, "npu", None)
        if npu is not None and hasattr(npu, "synchronize"):
            try:
                if npu.is_available():
                    npu.synchronize()
                    return
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def to_cpu(tensor):
    """Detach + copy a tensor to CPU, materialising lazy / GPU-backed views."""
    try:
        return tensor.detach().to("cpu", copy=True).contiguous()
    except Exception:
        return tensor.cpu().contiguous()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_pt(path: Path, payload, torch_module=None) -> str:
    """Atomically save a torch payload.

    ``run`` treats the appearance of a dump file as "stage finished" and kills
    the process group, so a half-written ``.pt`` would look complete and then
    fail to load. Write a sibling ``.tmp`` file first and rename into place.
    """
    if torch_module is None:
        import torch as torch_module
    path = Path(path)
    ensure_dir(path.parent)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as handle:
        torch_module.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return str(path)


def write_json(path: Path, payload) -> str:
    ensure_dir(Path(path).parent)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)
    return str(path)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(value):
    """Best-effort JSON conversion for torch tensors / numpy types."""
    if hasattr(value, "detach"):
        try:
            return value.detach().cpu().tolist()
        except Exception:
            return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def python_version() -> str:
    return sys.version.split()[0]
