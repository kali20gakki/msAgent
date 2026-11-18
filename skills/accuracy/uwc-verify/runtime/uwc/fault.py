"""Opt-in runtime fault injection used to validate the uwc-verify hooks.

Every fault is disabled unless its ``UWC_FAULT_*`` environment variable is set,
so normal runs are unaffected.  The faults implement the R1-R5
negative-validation experiments of the uwc-verify design notes:

===================  ======================================================
Variable             Effect
===================  ======================================================
UWC_FAULT_STAGE1     After the Bridge load: perturb the first mcore parameter
                     whose name contains ``<substr>``.  P2 then disagrees
                     with P0 (conversion / load defect).
UWC_FAULT_STAGE2     After P2_sender is dumped: perturb the first tensor
                     handed to the transport.  P5 then disagrees with
                     P2_sender (transport defect).
UWC_FAULT_STAGE3_SKIP  Drop matching tensors before vLLM ``_update_weights``,
                     after P5 was recorded: the receiver never applies them
                     (missed-sync defect).
UWC_FAULT_STAGE4     Before the mcore forward: shift every element of the
                     first parameter whose name contains ``<substr>``
                     (forward defect; a single element is too small to move
                     the G0/G1 cosine).
UWC_FAULT_STAGE5_SAME_FORMAT
                     Force both stage-5 runs to record ``load_format=dummy``
                     (A/B override defect).
===================  ======================================================

Spec syntax is ``<substr>:<delta>[,<substr>:<delta>]``; ``delta`` defaults to
0.01.  All helpers are best-effort: a fault that cannot be applied only logs a
``[UWC] fault`` line.
"""

from __future__ import annotations

import os

from .common import env_bool, log


def _spec(name: str):
    raw = os.environ.get(name)
    if not raw:
        return []
    specs = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            substr, delta = chunk.rsplit(":", 1)
            try:
                delta = float(delta)
            except ValueError:
                substr, delta = chunk, 0.01
        else:
            substr, delta = chunk, 0.01
        if substr:
            specs.append((substr, delta))
    return specs


def stage1_spec():
    return _spec("UWC_FAULT_STAGE1")


def stage2_spec():
    return _spec("UWC_FAULT_STAGE2")


def stage3_skip_patterns():
    raw = os.environ.get("UWC_FAULT_STAGE3_SKIP")
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def stage4_spec():
    return _spec("UWC_FAULT_STAGE4")


def stage5_same_format() -> bool:
    return env_bool("UWC_FAULT_STAGE5_SAME_FORMAT")


def _bump(tensor, delta) -> bool:
    """Change one element in place; return True when the value really changed."""
    import torch

    # Model parameters are leaves that require grad; in-place writes to a view
    # of such a leaf are rejected unless autograd tracking is disabled.
    with torch.no_grad():
        flat = tensor.reshape(-1) if hasattr(tensor, "reshape") else tensor.view(-1)
        if flat.numel() == 0:
            return False
        old = flat[0].item()
        for candidate in (old + delta, old * 1.01 + delta, old + 1.0):
            try:
                flat[0] = candidate
            except Exception:
                continue
            if flat[0].item() != old:
                return True
            flat[0] = old
    return False


def _bump_full(tensor, delta) -> bool:
    """Shift every element in place (stage 4).

    A single element is below the cosine threshold on a 4B model: the 2026-11-17
    single-chip R4 attempt changed one ``linear_proj`` weight by 0.5 and the
    G0/G1 cosine stayed at 0.99988, so the forward check still passed.  Shifting
    the whole tensor models a wrong conversion / wrong tensor, which is what the
    forward check is supposed to catch.
    """
    import torch

    with torch.no_grad():
        try:
            before = tensor.detach().clone()
            tensor.add_(delta)
        except Exception:
            return False
        try:
            return not bool(torch.equal(tensor.detach(), before))
        except Exception:
            return True


def perturb_named(named, specs, stage, full: bool = False) -> list[str]:
    """Perturb the first parameter per spec; ``named`` is a callable/list."""
    if not specs:
        return []
    bump = _bump_full if full else _bump
    items = list(named() if callable(named) else named)
    changed = []
    for substr, delta in specs:
        for name, tensor in items:
            if substr in str(name):
                try:
                    if bump(tensor, delta):
                        changed.append(str(name))
                        log(f"fault stage{stage}: perturbed {name} (+{delta})")
                    else:
                        log(f"fault stage{stage}: could not change {name}")
                except Exception as exc:
                    log(f"fault stage{stage}: perturb failed for {name}: {exc}")
                break
        else:
            log(f"fault stage{stage}: no parameter matched {substr!r}")
    return changed


def perturb_models(models, specs, stage, full: bool = False) -> list[str]:
    """Perturb parameters of one model or a list of pipeline chunks."""
    if not specs or models is None:
        return []
    chunks = models if isinstance(models, (list, tuple)) else [models]
    changed = []
    for chunk in chunks:
        named = getattr(chunk, "named_parameters", None)
        if callable(named):
            changed.extend(perturb_named(named, specs, stage, full=full))
    return changed


def perturb_weights(weights, specs, stage) -> list[str]:
    """Corrupt the tensors handed to the transport (``[(name, tensor), ...]``).

    The matched entry is replaced by a perturbed copy so the fault hits what is
    sent without writing through to the training model (stage-2 tensors may
    alias model parameters).  Only matched tensors are copied, so a one-layer
    fault costs one tensor instead of duplicating the whole export set.
    """
    if not specs or weights is None:
        return []
    is_dict = hasattr(weights, "items")
    items = list(weights.items()) if is_dict else list(weights)
    changed = []
    for substr, delta in specs:
        for index, (name, tensor) in enumerate(items):
            if substr in str(name):
                try:
                    target = tensor.detach().clone()
                    if _bump(target, delta):
                        if is_dict:
                            weights[name] = target
                        else:
                            weights[index] = (items[index][0], target)
                        items[index] = (items[index][0], target)
                        changed.append(str(name))
                        log(f"fault stage{stage}: perturbed {name} (+{delta})")
                    else:
                        log(f"fault stage{stage}: could not change {name}")
                except Exception as exc:
                    log(f"fault stage{stage}: perturb failed for {name}: {exc}")
                break
        else:
            log(f"fault stage{stage}: no tensor matched {substr!r}")
    return changed


def drop_weights(weights, patterns, stage=3) -> list[str]:
    """Remove matching entries from a list/dict in place, after P5 is recorded."""
    if not patterns or weights is None:
        return []
    dropped = []

    def matches(name) -> bool:
        return name is not None and any(pattern in str(name) for pattern in patterns)

    if isinstance(weights, dict):
        for name in list(weights):
            if matches(name):
                weights.pop(name)
                dropped.append(str(name))
    elif isinstance(weights, list):
        keep = []
        for item in weights:
            name = item[0] if isinstance(item, (list, tuple)) and item else None
            if matches(name):
                dropped.append(str(name))
            else:
                keep.append(item)
        if dropped:
            weights[:] = keep
    else:
        log(
            f"fault stage{stage}: weights is {type(weights).__name__}, "
            "not a list/dict; skip fault not applied"
        )
        return dropped

    if dropped:
        log(f"fault stage{stage}: dropped {len(dropped)} tensor(s), e.g. {dropped[0]}")
    else:
        log(f"fault stage{stage}: no tensor matched the skip patterns")
    return dropped
