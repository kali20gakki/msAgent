"""One-click offline comparison for UWC stage 1 / 2 / 3 dumps."""

from __future__ import annotations

import argparse
import functools
import json
import re
from pathlib import Path

from ..common import get_dump_dir, log, now_iso, read_json, write_json
from ..comparator import compare_weight_dicts
from ..logits_comparator import compare_logits_files
from ..name_normalizer import normalize_weights
from . import stage3_diff_test


def _load_pt(path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def _find(dump_dir: Path, pattern: str) -> list[Path]:
    return sorted(Path(dump_dir).glob(pattern))


def _step_of(path: Path) -> int:
    match = re.search(r"step(\d+)", path.name)
    return int(match.group(1)) if match else -1


def _pick_latest(paths: list[Path], step: int | None = None) -> Path | None:
    if not paths:
        return None
    if step is not None:
        for path in paths:
            if _step_of(path) == step:
                return path
    return max(paths, key=lambda path: (_step_of(path), path.stat().st_mtime))


def _to_dict(weights) -> tuple[dict, int]:
    if isinstance(weights, dict):
        return dict(weights), 0
    duplicates = 0
    result = {}
    for name, tensor in weights:
        if name in result:
            duplicates += 1
        result[str(name)] = tensor
    return result, duplicates


def compare_stage1(
    dump_dir: str | Path,
    model_config: str | Path | dict | None = None,
    atol: float = 1e-3,
    rtol: float = 0.0,
    exact: bool = False,
    step: int | None = None,
    qkv_layout: str = "contiguous",
    gate_up_layout: str = "contiguous",
    avg_cos: float = 0.99,
    min_cos: float = 0.95,
    write: bool = True,
) -> dict:
    dump_dir = Path(dump_dir)
    report = {
        "stage": 1,
        "generated_at": now_iso(),
        "dump_dir": str(dump_dir),
        "status": "skip",
        "export": {"status": "skip", "reason": "P0/P2 dump not found"},
        "forward": {"status": "skip", "reason": "G0/G1 dump not found"},
    }

    p0_path = _pick_latest(_find(dump_dir, "stage1_bridge/P0_hf_original_step*_rank*.pt"), step)
    p2_path = _pick_latest(_find(dump_dir, "stage1_bridge/P2_hf_exported_step*_rank*.pt"), step)
    if p0_path and p2_path:
        try:
            p0, p0_dupes = _to_dict(_load_pt(p0_path))
            p2, p2_dupes = _to_dict(_load_pt(p2_path))
            normalizer = functools.partial(
                normalize_weights,
                model_config=model_config,
                qkv_layout=qkv_layout,
                gate_up_layout=gate_up_layout,
            )
            comparison = compare_weight_dicts(
                p0,
                p2,
                atol=atol,
                rtol=rtol,
                exact=exact,
                name_normalizer=normalizer,
            )
            comparison["P0_file"] = str(p0_path)
            comparison["P2_file"] = str(p2_path)
            comparison["P0_duplicate_keys"] = p0_dupes
            comparison["P2_duplicate_keys"] = p2_dupes
            report["export"] = comparison
        except Exception as exc:
            report["export"] = {"status": "error", "error": str(exc)}
    else:
        report["export"] = {
            "status": "skip",
            "reason": "P0/P2 dump not found",
            "P0_file": str(p0_path) if p0_path else None,
            "P2_file": str(p2_path) if p2_path else None,
        }

    g0_path = _pick_latest(_find(dump_dir, "stage1_bridge/G0_logits_gt.pt"))
    g1_path = _pick_latest(_find(dump_dir, "stage1_bridge/G1_logits_mcore.pt"))
    if g0_path and g1_path:
        try:
            forward = compare_logits_files(
                g0_path,
                g1_path,
                avg_cos_threshold=avg_cos,
                min_cos_threshold=min_cos,
            )
            forward["G0_file"] = str(g0_path)
            forward["G1_file"] = str(g1_path)
            report["forward"] = forward
        except Exception as exc:
            report["forward"] = {"status": "error", "error": str(exc)}
    else:
        report["forward"] = {
            "status": "skip",
            "reason": "G0/G1 logits not found",
            "G0_file": str(g0_path) if g0_path else None,
            "G1_file": str(g1_path) if g1_path else None,
        }

    statuses = [report["export"].get("status"), report["forward"].get("status")]
    if "fail" in statuses or "error" in statuses:
        report["status"] = "fail"
    elif "pass" in statuses:
        report["status"] = "pass"
    else:
        report["status"] = "skip"

    if write:
        write_json(dump_dir / "stage1" / "report.json", report)
    return report


def compare_stage2(dump_dir: str | Path, exact: bool = True, step: int | None = None, write: bool = True) -> dict:
    dump_dir = Path(dump_dir)
    report = {"stage": 2, "generated_at": now_iso(), "dump_dir": str(dump_dir), "status": "skip"}

    sender_path = _pick_latest(_find(dump_dir, "stage2_transfer/P2_sender_step*_rank*.pt"), step)
    receiver_path = _pick_latest(_find(dump_dir, "stage2_transfer/P5_receiver_step*_rank*.pt"), step)
    if not (sender_path and receiver_path):
        report["transfer"] = {
            "status": "skip",
            "reason": "P2_sender or P5 dump not found",
            "P2_sender_file": str(sender_path) if sender_path else None,
            "P5_file": str(receiver_path) if receiver_path else None,
        }
    else:
        try:
            sender, sender_dupes = _to_dict(_load_pt(sender_path))
            receiver, receiver_dupes = _to_dict(_load_pt(receiver_path))
            comparison = compare_weight_dicts(sender, receiver, exact=exact)
            comparison["P2_sender_file"] = str(sender_path)
            comparison["P5_file"] = str(receiver_path)
            comparison["P2_sender_duplicate_keys"] = sender_dupes
            comparison["P5_duplicate_keys"] = receiver_dupes
            report["transfer"] = comparison
        except Exception as exc:
            report["transfer"] = {"status": "error", "error": str(exc)}

    report["status"] = report["transfer"].get("status", "skip")
    if write:
        write_json(dump_dir / "stage2" / "report.json", report)
    return report


def compare_stage3(
    dump_dir: str | Path,
    threshold: float = 0.5,
    write: bool = True,
    tokenizer_path: str | None = None,
    trust_remote_code: bool = True,
) -> dict:
    dump_dir = Path(dump_dir)
    report = {"stage": 3, "generated_at": now_iso(), "dump_dir": str(dump_dir), "status": "skip"}

    ptr_files = _find(dump_dir, "stage3_vllm/ptr_before_after_*.json")
    if ptr_files:
        ptr_report = {
            "status": "pass",
            "files": [],
            "checked_layers": 0,
            "synced_layers": 0,
            "not_synced_layers": 0,
            "missing_layers": 0,
            "not_synced": [],
            "errors": [],
        }
        for path in ptr_files:
            try:
                item = read_json(path)
            except Exception as exc:
                ptr_report["errors"].append({"file": str(path), "error": str(exc)})
                continue
            ptr_report["files"].append(str(path))
            ptr_report["checked_layers"] += int(item.get("checked_layers", 0) or 0)
            ptr_report["synced_layers"] += int(item.get("synced_layers", 0) or 0)
            ptr_report["not_synced_layers"] += int(item.get("not_synced_layers", 0) or 0)
            ptr_report["missing_layers"] += int(item.get("missing_layers", 0) or 0)
            for name in item.get("not_synced", []) or []:
                if name not in ptr_report["not_synced"]:
                    ptr_report["not_synced"].append(name)
            status = item.get("status")
            if status in {"fail", "error"}:
                ptr_report["status"] = "fail"
            elif status == "pass" and ptr_report["status"] != "fail":
                ptr_report["status"] = "pass"
            if item.get("error"):
                ptr_report["errors"].append({"file": str(path), "error": item["error"]})
        if ptr_report["errors"] and ptr_report["status"] == "pass":
            ptr_report["status"] = "error"
        report["ptr"] = ptr_report
    else:
        report["ptr"] = {"status": "skip", "reason": "ptr_before_after_*.json not found"}

    dummy_path = _pick_latest(_find(dump_dir, "stage3_vllm/dummy_rl_first_step_output.json"))
    safe_path = _pick_latest(_find(dump_dir, "stage3_vllm/safetensors_rl_first_step_output.json"))
    diff_report_path = dump_dir / "stage3_vllm" / "diff_test_report.json"
    if dummy_path and safe_path:
        try:
            diff = stage3_diff_test.run(
                dummy_path,
                safe_path,
                out_path=diff_report_path if write else None,
                threshold=threshold,
                tokenizer_path=tokenizer_path,
                trust_remote_code=trust_remote_code,
            )
            report["diff_test"] = diff
        except Exception as exc:
            report["diff_test"] = {"status": "error", "error": str(exc)}
    else:
        report["diff_test"] = {
            "status": "skip",
            "reason": "dummy / safetensors RL first-step output not found",
            "dummy_file": str(dummy_path) if dummy_path else None,
            "safetensors_file": str(safe_path) if safe_path else None,
        }

    statuses = [report["ptr"].get("status"), report["diff_test"].get("status")]
    if "fail" in statuses or "error" in statuses:
        report["status"] = "fail"
    elif "pass" in statuses:
        report["status"] = "pass"
    else:
        report["status"] = "skip"

    if write:
        write_json(dump_dir / "stage3" / "report.json", report)
    return report


def compare_all(
    dump_dir: str | Path | None = None,
    model_config=None,
    stage1_atol: float = 1e-3,
    stage1_rtol: float = 0.0,
    stage1_exact: bool = False,
    stage2_exact: bool = True,
    step: int | None = None,
    qkv_layout: str = "contiguous",
    gate_up_layout: str = "contiguous",
    avg_cos: float = 0.99,
    min_cos: float = 0.95,
    diff_threshold: float = 0.5,
    tokenizer_path: str | None = None,
    trust_remote_code: bool = True,
) -> dict:
    dump_dir = Path(dump_dir) if dump_dir else get_dump_dir()
    stage1 = compare_stage1(
        dump_dir,
        model_config=model_config,
        atol=stage1_atol,
        rtol=stage1_rtol,
        exact=stage1_exact,
        step=step,
        qkv_layout=qkv_layout,
        gate_up_layout=gate_up_layout,
        avg_cos=avg_cos,
        min_cos=min_cos,
    )
    stage2 = compare_stage2(dump_dir, exact=stage2_exact, step=step)
    stage3 = compare_stage3(
        dump_dir,
        threshold=diff_threshold,
        tokenizer_path=tokenizer_path,
        trust_remote_code=trust_remote_code,
    )
    summary = {"generated_at": now_iso(), "dump_dir": str(dump_dir), "stage1": stage1, "stage2": stage2, "stage3": stage3}
    write_json(dump_dir / "summary.json", summary)
    log(f"summary written to {dump_dir / 'summary.json'}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="UWC offline comparison (stage 1/2/3)")
    parser.add_argument("--dump_dir", default=None)
    parser.add_argument("--stage", choices=["1", "2", "3", "all"], default="all")
    parser.add_argument("--model_config", default=None, help="HF config.json for fused-projection splitting")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--exact", action="store_true", help="bitwise comparison for stage 1")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--qkv_layout", choices=["contiguous", "interleaved"], default="contiguous")
    parser.add_argument("--gate_up_layout", choices=["contiguous", "interleaved"], default="contiguous")
    parser.add_argument("--avg_cos", type=float, default=0.99)
    parser.add_argument("--min_cos", type=float, default=0.95)
    parser.add_argument("--diff_threshold", type=float, default=0.5)
    parser.add_argument("--tokenizer", default=None, help="HF model dir/tokenizer to decode stage-3 token ids")
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow the tokenizer repository to execute custom code while loading",
    )
    args = parser.parse_args(argv)

    dump_dir = Path(args.dump_dir) if args.dump_dir else get_dump_dir()
    # Only stage 1 needs the HF config (for fused qkv/gate_up splitting); the
    # other stages compare like-for-like key dumps.
    comparison_kwargs = dict(dump_dir=dump_dir, step=args.step)
    if args.stage == "1":
        report = compare_stage1(
            **comparison_kwargs,
            model_config=args.model_config,
            atol=args.atol,
            rtol=args.rtol,
            exact=args.exact,
            qkv_layout=args.qkv_layout,
            gate_up_layout=args.gate_up_layout,
            avg_cos=args.avg_cos,
            min_cos=args.min_cos,
        )
    elif args.stage == "2":
        # A transport check must be bit-exact: P2_sender and P5 are the same
        # buffers on either side of the IPC hop, so any drift is a real defect.
        report = compare_stage2(**comparison_kwargs, exact=True)
    elif args.stage == "3":
        report = compare_stage3(
            dump_dir,
            threshold=args.diff_threshold,
            tokenizer_path=args.tokenizer,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        report = compare_all(
            dump_dir=dump_dir,
            model_config=args.model_config,
            stage1_atol=args.atol,
            stage1_rtol=args.rtol,
            stage1_exact=args.exact,
            step=args.step,
            qkv_layout=args.qkv_layout,
            gate_up_layout=args.gate_up_layout,
            avg_cos=args.avg_cos,
            min_cos=args.min_cos,
            diff_threshold=args.diff_threshold,
            tokenizer_path=args.tokenizer,
            trust_remote_code=args.trust_remote_code,
        )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
