"""G0 vs G1 forward verification: HF transformers logits vs mcore logits."""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..common import ensure_dir, get_dump_dir, log, now_iso, write_json
from .. import fault
from ..logits_comparator import compare_logits, compare_logits_files


def _torch():
    import torch

    return torch


def _resolve_device(device: str | None):
    import torch

    if device:
        return torch.device(device)
    npu = getattr(torch, "npu", None)
    try:
        if npu is not None and npu.is_available():
            return torch.device("npu:0")
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _dtype(name: str):
    import torch

    return {
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }.get(name.lower(), torch.float32)


def _extract_logits(output):
    import torch

    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            logits = _extract_logits(item)
            if logits is not None:
                return logits
        return None
    logits = getattr(output, "logits", None)
    if logits is not None:
        return logits
    return None


def run_hf(
    local_path: str,
    input_ids: list[int],
    device: str | None = None,
    dtype: str = "fp32",
    trust_remote_code: bool = True,
) -> dict:
    import torch
    from transformers import AutoModelForCausalLM

    target = _resolve_device(device)
    if str(target).startswith("npu"):
        try:
            torch.npu.set_device(target)
        except Exception:
            pass

    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        torch_dtype=_dtype(dtype),
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.to(target)
    model.eval()

    ids = torch.tensor([input_ids], dtype=torch.long, device=target)
    attention_mask = torch.ones_like(ids)
    with torch.no_grad():
        output = model(input_ids=ids, attention_mask=attention_mask)
    logits = _extract_logits(output)
    if logits is None:
        raise RuntimeError("HF model output does not contain logits")
    return {
        "backend": "hf",
        "model": str(local_path),
        "device": str(target),
        "dtype": dtype,
        "input_ids": list(input_ids),
        "logits": logits.detach().float().cpu(),
        "generated_at": now_iso(),
    }


def _model_chunk(model):
    if isinstance(model, (list, tuple)):
        return model[0]
    return model


def _unwrap_model(model):
    chunk = _model_chunk(model)
    return getattr(chunk, "module", chunk)


def _load_bridge():
    """Return an ``AutoBridge`` class from either Bridge generation.

    ``megatron.bridge`` is Megatron-Bridge >= 0.4 (used by veRL 0.9+); the
    standalone ``mbridge`` package is the legacy name veRL <= 0.8 pinned.
    """
    try:
        from megatron.bridge import AutoBridge

        return AutoBridge
    except ImportError:
        from mbridge import AutoBridge

        return AutoBridge


def _bridge_build_and_load(bridge_cls, local_path, dtype, trust_remote_code):
    """Build an mcore model and load HF weights, tolerating both APIs."""
    bridge = bridge_cls.from_hf_pretrained(local_path, trust_remote_code=trust_remote_code)
    bf16 = dtype.lower() in {"bf16", "bfloat16"}
    fp16 = dtype.lower() == "fp16"

    if hasattr(bridge, "to_megatron_model"):
        # Megatron-Bridge >= 0.4: the HF load happens inside to_megatron_model
        # (or can be replayed explicitly through load_hf_weights).
        model = bridge.to_megatron_model(
            load_weights=False,
            hf_path=local_path,
            fp16=fp16,
            bf16=bf16,
            # Inference needs a bare module; DDP wrapping would require a
            # DistributedDataParallelConfig (Bridge's own export path does the
            # same).
            wrap_with_ddp=False,
        )
        bridge.load_hf_weights(model, local_path)
        return bridge, model

    # Legacy mbridge API.
    model = bridge.get_model(fp16=fp16, bf16=bf16)
    bridge.load_weights(model, local_path)
    return bridge, model


def _prepare_npu_megatron():
    """Make Megatron's CUDA-only RNG tracker usable on Ascend.

    ``model_parallel_cuda_manual_seed`` calls ``torch.cuda.get_rng_state()``,
    which raises "Torch not compiled with CUDA enabled" on the NPU build.
    veRL's training path avoids this because MindSpeed's adaptor patches
    Megatron; a standalone process has to load the adaptor itself, with a
    direct torch.cuda -> torch_npu RNG shim as fallback.
    """
    import torch

    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        return
    try:
        import mindspeed.megatron_adaptor  # noqa: F401

        return
    except Exception:
        pass
    if not torch.cuda.is_available():
        try:
            import torch_npu

            for name in (
                "get_rng_state",
                "set_rng_state",
                "get_rng_state_all",
                "set_rng_state_all",
                "manual_seed",
                "manual_seed_all",
            ):
                target = getattr(torch_npu.npu, name, None)
                if target is not None:
                    setattr(torch.cuda, name, target)
        except Exception:
            pass


def run_mcore(
    local_path: str,
    input_ids: list[int],
    device: str | None = None,
    dtype: str = "fp32",
    trust_remote_code: bool = True,
    tp: int = 1,
    pp: int = 1,
) -> dict:
    """Single-card mcore forward via Megatron-Bridge.

    This path is best-effort and version sensitive.  It intentionally keeps
    TP=PP=1 (as required by the UWC document) and initialises a one-rank
    process group.
    """
    import torch

    _prepare_npu_megatron()

    target = _resolve_device(device)
    backend = "hccl" if str(target).startswith("npu") else ("nccl" if torch.cuda.is_available() else "gloo")
    if str(target).startswith("npu"):
        try:
            torch.npu.set_device(target)
        except Exception:
            pass

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29577")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend=backend,
            rank=0,
            world_size=1,
            init_method="env://",
        )

    from megatron.core import parallel_state as mpu
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    if not mpu.is_initialized():
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            virtual_pipeline_model_parallel_size=None,
            context_parallel_size=1,
            expert_model_parallel_size=1,
        )
    model_parallel_cuda_manual_seed(0)

    bridge_cls = _load_bridge()
    bridge, model = _bridge_build_and_load(
        bridge_cls, local_path, dtype, trust_remote_code
    )
    model = _unwrap_model(model)
    model.to(target)

    # Opt-in fault R4: perturb mcore weights after loading, before the forward
    # pass, to prove G0/G1 cosine detects a forward-side weight defect.
    specs = fault.stage4_spec()
    if specs:
        try:
            fault.perturb_models(model, specs, 4, full=True)
        except Exception as exc:
            log(f"fault stage4 injection failed: {exc}")

    model.eval()

    ids = torch.tensor([input_ids], dtype=torch.long, device=target)
    attention_mask = torch.ones_like(ids)
    position_ids = torch.arange(ids.shape[1], dtype=torch.long, device=target).unsqueeze(0)
    with torch.no_grad():
        output = model(ids, position_ids, attention_mask)
    logits = _extract_logits(output)
    if logits is None:
        raise RuntimeError("mcore model output does not contain logits")
    return {
        "backend": "mcore",
        "model": str(local_path),
        "device": str(target),
        "dtype": dtype,
        "input_ids": list(input_ids),
        "logits": logits.detach().float().cpu(),
        "generated_at": now_iso(),
    }


def _save(payload: dict, out_path: str | Path) -> str:
    torch = _torch()
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    from ..common import save_pt

    return save_pt(out_path, payload, torch)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="UWC G0/G1 forward verification")
    parser.add_argument("--backend", choices=["hf", "mcore", "compare"], required=True)
    parser.add_argument("--local_path", default=None, help="HF model directory")
    parser.add_argument("--input_ids", default="1,2,3,4,5", help="comma-separated token ids")
    parser.add_argument("--device", default=None, help="e.g. npu:0 / cuda:0 / cpu")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow model repositories to execute custom code while loading",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--dump_dir", default=None)
    parser.add_argument("--g0", default=None)
    parser.add_argument("--g1", default=None)
    parser.add_argument("--avg_cos", type=float, default=0.99)
    parser.add_argument("--min_cos", type=float, default=0.95)
    args = parser.parse_args(argv)

    dump_dir = Path(args.dump_dir) if args.dump_dir else get_dump_dir()

    if args.backend == "compare":
        g0 = args.g0 or (dump_dir / "stage1_bridge" / "G0_logits_gt.pt")
        g1 = args.g1 or (dump_dir / "stage1_bridge" / "G1_logits_mcore.pt")
        report = compare_logits_files(g0, g1, avg_cos_threshold=args.avg_cos, min_cos_threshold=args.min_cos)
        out = ensure_dir(dump_dir / "stage1_bridge") / "forward_report.json"
        write_json(out, report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["status"] == "pass" else 1

    if not args.local_path:
        parser.error("--local_path is required for hf/mcore backends")
    ids = [int(item) for item in str(args.input_ids).split(",") if item.strip()]
    if args.backend == "hf":
        payload = run_hf(
            args.local_path,
            ids,
            device=args.device,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        out = args.out or (dump_dir / "stage1_bridge" / "G0_logits_gt.pt")
    else:
        payload = run_mcore(
            args.local_path,
            ids,
            device=args.device,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        out = args.out or (dump_dir / "stage1_bridge" / "G1_logits_mcore.pt")
    path = _save(payload, out)
    log(f"{args.backend} logits saved to {path}")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
