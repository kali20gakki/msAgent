"""Stage 3 collectors: data_ptr coverage check and dummy/safetensors output dump."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from ..common import (
    ensure_dir,
    get_dump_dir,
    get_rank,
    get_step,
    log,
    log_warning,
    now_iso,
    sync_device,
    write_json,
)

_DUMPED_TAGS: set[str] = set()
_DUMP_LOCK = threading.Lock()


def _torch():
    import torch

    return torch


_SAMPLE_SIZE = 32


def _content_sample(param) -> list[float]:
    """Return a small deterministic sample of a parameter's values.

    vLLM's ``model.load_weights`` writes with ``copy_`` into the existing
    buffer, so ``data_ptr`` stays constant even when a layer *was* synced.
    Sampling a few strided elements lets us see the in-place write without
    pulling the whole tensor to CPU.
    """
    flat = param.detach().reshape(-1)
    numel = int(flat.numel())
    if numel == 0:
        return []
    stride = max(1, numel // _SAMPLE_SIZE)
    return flat[::stride][:_SAMPLE_SIZE].to("cpu", _torch().float32).tolist()


def snapshot_params(model) -> dict | None:
    """Snapshot pointer + content fingerprint for every parameter of ``model``."""
    if model is None:
        log_warning("stage3 ptr snapshot: model is None")
        return None
    snapshot = {}
    try:
        for name, param in model.named_parameters():
            try:
                snapshot[name] = {
                    "ptr": int(param.data_ptr()),
                    "sample": _content_sample(param),
                    "numel": int(param.numel()),
                }
            except Exception as exc:
                snapshot[name] = {"error": str(exc)}
    except Exception as exc:
        log_warning(f"stage3 ptr snapshot failed: {exc}")
        return None
    if not snapshot:
        log_warning("stage3 ptr snapshot: model has no named parameters")
        return None
    return snapshot


def compare_ptr_snapshots(model, snapshot, dump_dir=None, step=None) -> dict:
    """Compare a round-start snapshot with the model state after the round.

    A layer counts as synced when either its pointer changed (parameters were
    re-assigned) or its sampled values changed (in-place copy).  This is the
    NPU-safe reading of the UWC ``data_ptr`` check.
    """
    step = get_step() if step is None else step
    dump_dir = Path(dump_dir) if dump_dir else get_dump_dir()
    report = {
        "status": "pass",
        "step": step,
        "rank": get_rank(),
        "method": "ptr_or_content_sample",
        "checked_layers": len(snapshot or {}),
        "synced_layers": 0,
        "not_synced_layers": 0,
        "missing_layers": 0,
        "ptr_changed_layers": 0,
        "content_changed_layers": 0,
        "details": [],
        "not_synced": [],
    }

    if not snapshot:
        report["status"] = "error"
        report["error"] = "snapshot is empty (collection failed or model has no parameters)"
        log_warning(f"stage3 ptr compare skipped: {report['error']}")
    else:
        try:
            sync_device()
            current = {name: param for name, param in model.named_parameters()}
        except Exception as exc:
            report["status"] = "error"
            report["error"] = f"post-round collection failed: {exc}"
            current = {}

        for name, before in snapshot.items():
            after = current.get(name)
            if before is None or "error" in (before or {}) or after is None:
                report["missing_layers"] += 1
                report["details"].append(
                    {
                        "name": name,
                        "ptr_before": (before or {}).get("ptr"),
                        "ptr_after": None,
                        "synced": False,
                        "reason": "missing_after" if after is None else "snapshot_error",
                    }
                )
                continue
            try:
                ptr_after = int(after.data_ptr())
                sample_after = _content_sample(after)
            except Exception as exc:
                report["missing_layers"] += 1
                report["details"].append({"name": name, "synced": False, "reason": f"read_error: {exc}"})
                continue

            ptr_changed = ptr_after != before["ptr"]
            content_changed = sample_after != before["sample"]
            if ptr_changed:
                report["ptr_changed_layers"] += 1
            if content_changed:
                report["content_changed_layers"] += 1
            synced = ptr_changed or content_changed
            if synced:
                report["synced_layers"] += 1
            else:
                report["not_synced_layers"] += 1
                report["not_synced"].append(name)
            report["details"].append(
                {
                    "name": name,
                    "ptr_before": before["ptr"],
                    "ptr_after": ptr_after,
                    "ptr_changed": ptr_changed,
                    "content_changed": content_changed,
                    "synced": synced,
                }
            )

        if report["not_synced_layers"] or report["missing_layers"]:
            report["status"] = "fail"
        if report["status"] == "fail":
            log_warning(
                f"stage3 ptr compare: {report['not_synced_layers']} layers not synced, "
                f"{report['missing_layers']} layers missing after load"
            )

    out_path = ensure_dir(dump_dir / "stage3_vllm") / f"ptr_before_after_step{step}_rank{get_rank()}.json"
    write_json(out_path, report)
    return report


def collect_ptr_before(model) -> dict | None:
    """Record ``data_ptr`` for every named parameter before ``load_weights``.

    Returns ``None`` on failure (never an empty dict) so the caller can
    distinguish "collection failed" from "model has no parameters" and avoid a
    false pass.
    """
    if model is None:
        log_warning("stage3 ptr_before: model is None")
        return None
    ptr_before = {}
    try:
        for name, param in model.named_parameters():
            try:
                ptr_before[name] = int(param.data_ptr())
            except Exception:
                ptr_before[name] = None
    except Exception as exc:
        log_warning(f"stage3 ptr_before collection failed: {exc}")
        return None
    if not ptr_before:
        log_warning("stage3 ptr_before: model has no named parameters")
        return None
    return ptr_before


def collect_ptr_after(model, ptr_before, dump_dir=None, step=None) -> dict:
    """Compare ``data_ptr`` after ``load_weights``; unchanged => layer not synced."""
    step = get_step() if step is None else step
    dump_dir = Path(dump_dir) if dump_dir else get_dump_dir()
    report = {
        "status": "pass",
        "step": step,
        "rank": get_rank(),
        "checked_layers": len(ptr_before or {}),
        "synced_layers": 0,
        "not_synced_layers": 0,
        "missing_layers": 0,
        "details": [],
        "not_synced": [],
    }

    if not ptr_before:
        report["status"] = "error"
        report["error"] = "ptr_before is empty (collection failed or model has no parameters)"
        log_warning(f"stage3 ptr_after skipped: {report['error']}")
    else:
        try:
            current = {name: int(param.data_ptr()) for name, param in model.named_parameters()}
        except Exception as exc:
            report["status"] = "error"
            report["error"] = f"ptr_after collection failed: {exc}"
            current = {}

        for name, before in ptr_before.items():
            if name not in current:
                report["missing_layers"] += 1
                report["details"].append(
                    {"name": name, "ptr_before": before, "ptr_after": None, "synced": False, "reason": "missing_after"}
                )
                continue
            after = current[name]
            # A pointer must change to prove the layer was written.  A None
            # "before" (non-tensor param) cannot be judged.
            synced = before is None or after != before
            if synced:
                report["synced_layers"] += 1
            else:
                report["not_synced_layers"] += 1
                report["not_synced"].append(name)
            report["details"].append(
                {"name": name, "ptr_before": before, "ptr_after": after, "synced": synced}
            )

        if report["not_synced_layers"] or report["missing_layers"]:
            report["status"] = "fail"
        if report["status"] == "fail":
            log_warning(
                f"stage3 ptr_after: {report['not_synced_layers']} layers not synced, "
                f"{report['missing_layers']} layers missing after load"
            )

    out_path = ensure_dir(dump_dir / "stage3_vllm") / f"ptr_before_after_step{step}_rank{get_rank()}.json"
    write_json(out_path, report)
    return report


def _tensor_to_list(value):
    try:
        if hasattr(value, "detach"):
            return value.detach().cpu().tolist()
        if hasattr(value, "tolist"):
            return value.tolist()
    except Exception:
        pass
    return value


def _extract_payload(output) -> dict:
    """Extract responses / input_ids / scores from DataProto, TensorDict or dict."""
    payload: dict = {}

    batch = getattr(output, "batch", None)
    if batch is not None:
        for key in (
            "responses",
            "input_ids",
            "attention_mask",
            "rm_scores",
            "token_level_scores",
            "response_mask",
        ):
            try:
                payload[key] = _tensor_to_list(batch[key])
            except Exception:
                continue
        if payload:
            return payload

    if isinstance(output, dict):
        for key in ("responses", "input_ids", "text", "texts", "output", "outputs"):
            if key in output:
                payload[key] = _tensor_to_list(output[key])
        if payload:
            return payload

    if hasattr(output, "responses"):
        payload["responses"] = _tensor_to_list(output.responses)

    # veRL 0.9 async rollout returns ``TokenOutput`` (token ids + logprobs).
    token_ids = getattr(output, "token_ids", None)
    if token_ids is not None:
        payload["responses"] = [_tensor_to_list(token_ids)]
        log_probs = getattr(output, "log_probs", None)
        if log_probs is not None:
            payload["log_probs"] = [_tensor_to_list(log_probs)]
        payload["stop_reason"] = getattr(output, "stop_reason", None)
    return payload


def dump_generation_output(output, dump_dir=None, tag: str | None = None, step=None, tokenizer=None) -> str | None:
    """Dump the first RL-step generation output for dummy vs safetensors diff.

    The tag decides the filename: ``dummy_rl_first_step_output.json`` or
    ``safetensors_rl_first_step_output.json`` (matching the UWC document).
    """
    step = get_step() if step is None else step
    dump_dir = Path(dump_dir) if dump_dir else get_dump_dir()
    tag = tag or os.environ.get("UWC_DIFF_TAG") or os.environ.get("UWC_LOAD_FORMAT") or "run"
    tag = tag.strip().lower().replace("-", "_")

    with _DUMP_LOCK:
        if tag in _DUMPED_TAGS:
            return None
        _DUMPED_TAGS.add(tag)

    payload = _extract_payload(output)
    texts = payload.get("texts") or payload.get("text") if isinstance(payload, dict) else None
    if isinstance(texts, str):
        texts = [texts]
    responses = payload.get("responses")
    if tokenizer is not None and responses is not None:
        try:
            flat = responses
            if flat and isinstance(flat[0], list):
                texts = tokenizer.batch_decode(flat, skip_special_tokens=True)
            else:
                texts = tokenizer.batch_decode([flat], skip_special_tokens=True)
        except Exception as exc:
            log_warning(f"stage3 diff: tokenizer decode failed: {exc}")

    record = {
        "tag": tag,
        "step": step,
        "generated_at": now_iso(),
        "source": "uwc.stage3.dump_generation_output",
        "load_format": os.environ.get("UWC_LOAD_FORMAT") or os.environ.get("LOAD_FORMAT"),
        "texts": texts,
        "payload": payload,
    }
    out_path = ensure_dir(dump_dir / "stage3_vllm") / f"{tag}_rl_first_step_output.json"
    write_json(out_path, record)
    log(f"stage3 diff output saved to {out_path}")
    return str(out_path)


def maybe_dump_first_generation(output, tokenizer=None) -> str | None:
    """Hook entry: dump only when UWC stage3 diff dumping is enabled."""
    if os.environ.get("UWC_ENABLE", "0") != "1":
        return None
    if os.environ.get("UWC_STAGE3", "1") != "1":
        return None
    if os.environ.get("UWC_DIFF_DUMP", "1") != "1":
        return None
    try:
        return dump_generation_output(output, tag=None, tokenizer=tokenizer)
    except Exception as exc:
        log_warning(f"stage3 diff dump failed: {exc}")
        return None
