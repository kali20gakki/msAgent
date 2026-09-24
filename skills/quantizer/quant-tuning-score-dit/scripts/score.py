# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score DiT inference outputs via AISBench-VBench.

Pipeline (single in-process call — no nested subprocess):

1.  ``patch_config()`` rewrites the upstream ``eval_vbench_standard.py``
    template with three runtime variables (``DATA_PATH``, ``VBENCH_CACHE_DIR``,
    ``full_json_dir``).
2.  ``subprocess.run(['ais_bench', <patched>, '--mode', 'eval', ...])`` runs
    the scorer; stderr is captured and classified against known env-error
    patterns (see ``STDERR_PATTERNS``) so the user gets a stable error code
    + a remediation hint + the official AISBench docs URL.
3.  Per-dimension scores are read from
    ``{work_dir}/results/{model_abbr}/vbench_<dim>.json``; the Quality /
    Semantic / Total aggregates come straight from the upstream
    ``VBenchSummarizer`` summary file — never recomputed locally.
4.  Optional ``--baseline-outputs`` re-runs steps 1-3 on an FP baseline dir
    and computes ``loss_vs_baseline`` + ``is_satisfied`` (loss within
    ``--baseline-tolerance`` of zero).
5.  Result is wrapped in the ``msagent-io v1`` envelope.

Precheck contract (mandatory)
-----------------------------
The orchestrator MUST run ``scripts/check_vbench_cache.py`` BEFORE invoking
this script and surface the returned ``candidates`` to the user for explicit
confirmation. This skill never auto-selects a cache path and never downloads
weights.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from patch_config import TemplateError, map_template_err, patch_config


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


_OFFICIAL_VBENCH_DOC = (
    "https://github.com/AISBench/benchmark/blob/master/docs/"
    "source_zh_cn/extended_benchmark/lmm_generate/vbench.md"
)

_KNOWN_DIMENSIONS = frozenset({
    "subject_consistency", "background_consistency", "temporal_flickering",
    "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality",
    "object_class", "multiple_objects", "human_action", "color",
    "spatial_relationship", "scene", "appearance_style", "temporal_style",
    "overall_consistency",
})

_KNOWN_AGGREGATES = frozenset({"vbench_quality", "vbench_semantic", "vbench_total"})

DEFAULT_BASELINE_TOLERANCE = 0.05

# (regex, error_code, remediation_hint). Specific patterns first; order matters.
_STDERR_PATTERNS: List[Tuple[str, str, str]] = [
    (r"ModuleNotFoundError.*['\"]decord['\"]",
     "ENV_DECORD_MISSING",
     "`pip install decord` (x86_64) or build from source on ARM."),
    (r"ModuleNotFoundError.*['\"]detectron2['\"]",
     "ENV_DETECTRON2_MISSING",
     "`pip install -e ais_bench/third_party/detectron2 --no-build-isolation`."),
    (r"ModuleNotFoundError.*['\"]torch['\"]",
     "ENV_TORCH_MISSING",
     "Install PyTorch matching the local CUDA / Ascend toolkit."),
    (r"ModuleNotFoundError.*['\"]torchvision['\"]",
     "ENV_TORCHVISION_MISSING",
     "Install torchvision matching PyTorch; Ascend: https://gitcode.com/Ascend/vision"),
    (r"ModuleNotFoundError.*['\"]huggingface_hub['\"]",
     "ENV_HUGGINGFACE_HUB_MISSING",
     "`pip install huggingface_hub`."),
    (r"(?i)ffmpeg.*not found|No such file.*['\"]?ffmpeg['\"]?",
     "ENV_FFMPEG_MISSING",
     "`apt install ffmpeg` or `conda install -c conda-forge ffmpeg`."),
    (r"(?i)libtorch_cuda|cuda.*not available|CUDA driver version is insufficient",
     "ENV_CUDA_MISMATCH",
     "Check CUDA toolkit / driver / PyTorch triplet."),
    (r"(?i)CUDA out of memory|cuda\.OOM|out of memory.*GPU",
     "ENV_CUDA_OOM",
     "Reduce `--max-num-workers 1` and/or shrink video resolution."),
    (r"(?i)huggingface.*401|hf_token|HF_TOKEN.*required|RepositoryNotFound.*401",
     "ENV_HF_TOKEN_MISSING",
     "Set `$HF_TOKEN` or `huggingface-cli login`."),
    (r"KeyError.*['\"]infer_cfg['\"]",
     "AISBENCH_INCOMPATIBLE_VERSION",
     "pip ais_bench too old: `pip install -U ais_bench`."),
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ScoreError(Exception):
    """Score-pipeline failure. ``code`` is forwarded verbatim into the envelope."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _err(code: str, message: str) -> Dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message}}


def _classify_stderr(stderr: str) -> Optional[Tuple[str, str]]:
    """Return ``(code, hint)`` on the first pattern match, else ``None``."""
    for pattern, code, hint in _STDERR_PATTERNS:
        if re.search(pattern, stderr, re.IGNORECASE):
            return code, hint
    return None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_args(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """Return a ready-to-emit ``error`` dict if invalid, else ``None``."""
    if not args.full_json_dir:
        return {"code": "FULL_JSON_REQUIRED", "message": "--full-json-dir required (path to VBench_kmeans_info*.json)."}
    full = Path(args.full_json_dir).expanduser()
    if not full.is_file():
        return {"code": "FULL_JSON_NOT_FOUND", "message": f"--full-json-dir not found: {args.full_json_dir!r}"}
    if full.suffix.lower() != ".json":
        return {"code": "FULL_JSON_SCHEMA_MISMATCH", "message": f"--full-json-dir must be .json: suffix={full.suffix!r}"}
    try:
        payload = json.loads(full.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"code": "FULL_JSON_SCHEMA_MISMATCH", "message": f"--full-json-dir not valid JSON: {exc}"}
    if not isinstance(payload, list) or not payload:
        return {"code": "FULL_JSON_SCHEMA_MISMATCH", "message": "must be a non-empty JSON list."}
    first = payload[0]
    if not isinstance(first, dict) or "prompt_en" not in first or "dimension" not in first:
        return {"code": "FULL_JSON_SCHEMA_MISMATCH", "message": "first item must contain 'prompt_en' and 'dimension'."}

    if not args.vbench_cache_dir:
        return {"code": "CACHE_DIR_MISSING", "message":
            "--vbench-cache-dir required. User must supply a pre-populated path; "
            "this skill does not download. Run scripts/check_vbench_cache.py first."}
    if not Path(args.vbench_cache_dir).expanduser().is_dir():
        return {"code": "CACHE_DIR_NOT_DIR", "message":
            f"--vbench-cache-dir not a dir: {args.vbench_cache_dir!r}"}

    if not args.infer_outputs:
        return {"code": "DATA_PATH_REQUIRED", "message": "--infer-outputs required."}
    if not Path(args.infer_outputs).expanduser().is_dir():
        return {"code": "DATA_PATH_NOT_DIR", "message": f"--infer-outputs not a dir: {args.infer_outputs!r}"}
    return None


# ---------------------------------------------------------------------------
# ais_bench invocation
# ---------------------------------------------------------------------------


def _run_aisbench(
    *, patched_config: Path, work_dir: Path, max_num_workers: int, timeout_sec: int
) -> Dict[str, Any]:
    cmd = ["ais_bench", str(patched_config), "--mode", "eval",
           "--max-num-workers", str(max_num_workers)]
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, cwd=str(work_dir), timeout=timeout_sec, check=False,
            stdout=None, stderr=subprocess.PIPE, text=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise ScoreError("SUBSCORER_TIMEOUT", f"ais_bench timeout after {timeout_sec}s") from exc

    stderr = proc.stderr or ""
    if stderr:
        sys.stderr.write(stderr)
        sys.stderr.flush()

    if proc.returncode != 0:
        classified = _classify_stderr(stderr)
        if classified:
            code, hint = classified
            raise ScoreError(code, f"{hint} Official docs: {_OFFICIAL_VBENCH_DOC}")
        tail = stderr[-500:] if stderr else "(no stderr)"
        raise ScoreError(
            "AISBENCH_EXIT_NONZERO",
            f"ais_bench exit={proc.returncode}. Docs: {_OFFICIAL_VBENCH_DOC}. Tail: {tail}",
        )
    return {"cmd": cmd, "duration_sec": time.perf_counter() - start}


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------


def _find_model_abbr(results_dir: Path) -> str:
    if not results_dir.is_dir():
        raise ScoreError("AISBENCH_NO_OUTPUT", f"Expected ais_bench output missing: {results_dir}")
    candidates = [p for p in results_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise ScoreError("AISBENCH_NO_OUTPUT", f"No model dir under {results_dir}")
    if len(candidates) > 1:
        names = sorted(p.name for p in candidates)
        raise ScoreError("AISBENCH_MULTIPLE_MODELS", f"Multiple models found: {names}")
    return candidates[0].name


def _resolve_output_dirs(work_dir: Path) -> Tuple[Path, Path, str]:
    """Locate AISBench output dirs regardless of version-specific layout.

    Two layouts supported (whichever exists first wins):

    1. **Conventional** — ``{work_dir}/results/`` + ``{work_dir}/summary/``
       Used when AISBench's config maps ``outputs`` to ``work_dir`` root.
    2. **Default timestamped** — ``{work_dir}/outputs/default/<latest-ts>/results/``
       + ``.../summary/``. AISBench's stock config writes here; multiple
       runs accumulate under different timestamps — we pick the latest.

    Returns ``(results_dir, summary_dir, source)``. Raises ``ScoreError`` if
    neither layout yields a usable results directory.
    """
    # Convention 1: work_dir/results/ and work_dir/summary/
    conv_results = work_dir / "results"
    conv_summary = work_dir / "summary"
    if conv_results.is_dir():
        return conv_results, conv_summary, "conventional"

    # Convention 2: work_dir/outputs/default/<ts>/{results,summary}/
    outputs_root = work_dir / "outputs" / "default"
    if outputs_root.is_dir():
        ts_dirs = sorted(p for p in outputs_root.iterdir() if p.is_dir())
        if ts_dirs:
            latest = ts_dirs[-1]
            ts_results = latest / "results"
            ts_summary = latest / "summary"
            if ts_results.is_dir():
                return ts_results, ts_summary, f"outputs/default/{latest.name}"

    raise ScoreError(
        "AISBENCH_NO_OUTPUT",
        f"No AISBench output found under {work_dir}. "
        f"Tried: results/, outputs/default/*/results/",
    )


def _load_per_dim_scores(results_dir: Path, requested: Optional[Sequence[str]]) -> Dict[str, float]:
    """Walk ``{work_dir}/results/{model_abbr}/vbench_*.json`` and collect scores."""
    model_abbr = _find_model_abbr(results_dir)
    scores: Dict[str, float] = {}
    for json_path in sorted((results_dir / model_abbr).glob("vbench_*.json")):
        dim = json_path.stem[len("vbench_"):]  # "vbench_<dim>.json" → "<dim>"
        if dim not in _KNOWN_DIMENSIONS:
            continue
        if requested and dim not in requested:
            continue
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ScoreError("AISBENCH_BAD_OUTPUT", f"Bad JSON {json_path}: {exc}") from exc
        score = payload.get("accuracy")
        if score is None:
            details = payload.get("details", {})
            dim_detail = details.get(dim) or details.get(dim.replace("_", " ")) or {}
            raw = dim_detail.get("score")
            if raw is None:
                continue
            score = raw * 100
        scores[dim] = float(score)
    return scores


_SUMMARY_LINE = re.compile(r"^(vbench_\w+):\s*\{\s*'accuracy':\s*([\d.eE+-]+)\s*\}")


def _load_aggregates(summary_dir: Path) -> Dict[str, float]:
    """Read Quality / Semantic / Total from AISBench's summary file (verbatim)."""
    if not summary_dir.is_dir():
        raise ScoreError("AISBENCH_NO_SUMMARY", f"No summary dir: {summary_dir}")
    txt_files = sorted(summary_dir.glob("summary_*.txt"))
    if not txt_files:
        raise ScoreError("AISBENCH_NO_SUMMARY", f"No summary_*.txt under {summary_dir}")

    found: Dict[str, float] = {}
    for line in txt_files[-1].read_text(encoding="utf-8").splitlines():
        m = _SUMMARY_LINE.match(line.strip())
        if m and m.group(1) in _KNOWN_AGGREGATES:
            found[m.group(1)] = float(m.group(2))
    missing = _KNOWN_AGGREGATES - set(found)
    if missing:
        raise ScoreError("AISBENCH_INCOMPLETE_SUMMARY", f"summary missing: {sorted(missing)}")
    return found


# ---------------------------------------------------------------------------
# run_manifest.json sidecar (inference parameter attribution)
# ---------------------------------------------------------------------------


def _load_run_manifest(infer_outputs: Path) -> Optional[Dict[str, Any]]:
    """Read ``{infer_outputs}/run_manifest.json`` if present; missing → None."""
    p = infer_outputs / "run_manifest.json"
    if not p.is_file():
        print(f"[score] INFO: no run_manifest.json at {p}; inference_params omitted.", file=sys.stderr)
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[score] WARN: failed to read manifest: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Score one directory end-to-end
# ---------------------------------------------------------------------------


def _score_one_dir(
    *, infer_outputs: Path, args: argparse.Namespace, work_dir: Path
) -> Dict[str, Any]:
    """Patch template → run ais_bench → parse. Returns ``{"ok": ..., ...}``."""
    try:
        patched = patch_config(
            data_path=str(infer_outputs),
            vbench_cache_dir=args.vbench_cache_dir,
            full_json_dir=args.full_json_dir,
            out_path=work_dir / "eval_vbench_patched.py",
        )
    except TemplateError as exc:
        return _err(map_template_err(str(exc)), str(exc))

    try:
        invocation = _run_aisbench(
            patched_config=patched,
            work_dir=work_dir,
            max_num_workers=args.max_num_workers,
            timeout_sec=args.timeout_sec,
        )
        results_dir, summary_dir, _source = _resolve_output_dirs(work_dir)
        per_dim = _load_per_dim_scores(results_dir, args._score_dimensions)
        aggregates = _load_aggregates(summary_dir)
    except ScoreError as exc:
        return _err(exc.code, str(exc))

    if not per_dim:
        return _err("AISBENCH_NO_DIMENSIONS", f"No vbench_*.json under {results_dir}")

    return {
        "ok": True,
        "scores": {d: round(s, 6) for d, s in sorted(per_dim.items())},
        "score_dimensions": sorted(per_dim.keys()),
        "quality_score": round(aggregates["vbench_quality"], 6),
        "semantic_score": round(aggregates["vbench_semantic"], 6),
        "overall_score": round(aggregates["vbench_total"], 6),
        "commands": [{"name": "vbench_score", "command": " ".join(invocation["cmd"])}],
        "duration_sec": round(invocation["duration_sec"], 3),
    }


# ---------------------------------------------------------------------------
# Envelope + CLI
# ---------------------------------------------------------------------------


def _build_envelope(
    *, status: str, output: Optional[Dict[str, Any]] = None, error: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "protocol": "msagent.subagent_io",
        "subagent_type": "quant-tuning-score-dit",
        "status": status,
    }
    if output is not None:
        payload["output"] = output
    if error is not None:
        payload["error"] = error
    return payload


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Score DiT inference outputs via AISBench-VBench (single-process).",
    )
    p.add_argument("--infer-outputs", required=True, help="Directory of mp4s from quant-tuning-evaluate DiT workflow.")
    p.add_argument("--full-json-dir", required=True, help="VBench_kmeans_info*.json file.")
    p.add_argument("--vbench-cache-dir", required=True,
                   help="Pre-populated VBench small-model cache dir; this skill never downloads.")
    p.add_argument("--baseline-outputs", default=None, help="Optional FP baseline dir (from quant-tuning-evaluate DiT workflow with --ckpt_dir pointing to FP weights).")
    p.add_argument("--score-dimensions", default=None, help="Comma-separated subset of dims to report.")
    p.add_argument("--max-num-workers", type=int, default=1, help="Passed to ais_bench.")
    p.add_argument("--work-dir", default=None, help="Explicit workdir root for scoring outputs; defaults to {infer_outputs}/..")
    p.add_argument("--baseline-tolerance", type=float, default=DEFAULT_BASELINE_TOLERANCE,
                   help="is_satisfied ⇔ loss_vs_baseline ≥ -tolerance.")
    p.add_argument("--round", type=int, default=None)
    p.add_argument("--timeout-sec", type=int, default=7200)
    p.add_argument("--output-json", default=None, help="Also write JSON payload to disk.")
    return p


def _emit(payload: Dict[str, Any], output_json: Optional[str]) -> int:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if output_json:
        out = Path(output_json).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return 0 if payload.get("status") == "ok" else 2


def main() -> int:
    args = _build_parser().parse_args()
    start = time.perf_counter()

    err = _validate_args(args)
    if err:
        return _emit(_build_envelope(status="failed", error=err), args.output_json)

    args._score_dimensions = (
        [d.strip() for d in args.score_dimensions.split(",") if d.strip()]
        if args.score_dimensions else None
    )

    work_root = Path(args.work_dir or args.infer_outputs).expanduser().resolve().parent
    work_root.mkdir(parents=True, exist_ok=True)
    main_work = work_root / "vbench_inputs"

    main_payload = _score_one_dir(
        infer_outputs=Path(args.infer_outputs).expanduser().resolve(),
        args=args, work_dir=main_work,
    )
    if not main_payload.get("ok"):
        return _emit(_build_envelope(status="failed", error=main_payload["error"]), args.output_json)

    # Optional baseline re-score.
    baseline_extra: Dict[str, Any] = {}
    if args.baseline_outputs:
        baseline_dir = Path(args.baseline_outputs).expanduser()
        if not baseline_dir.is_dir():
            baseline_extra = {"baseline_error": {"code": "BASELINE_NOT_DIR",
                                                  "message": f"--baseline-outputs not a dir: {baseline_dir}"}}
        else:
            baseline_payload = _score_one_dir(
                infer_outputs=baseline_dir.resolve(),
                args=args, work_dir=work_root / "baseline_vbench_inputs",
            )
            if not baseline_payload.get("ok"):
                baseline_extra = {"baseline_error": baseline_payload.get(
                    "error", {"code": "BASELINE_FAILED", "message": "baseline failed"})}
            else:
                main_overall = main_payload["overall_score"]
                baseline_overall = baseline_payload["overall_score"]
                if baseline_overall is None or main_overall is None:
                    baseline_extra = {"baseline_error": {"code": "BASELINE_SCORE_MISSING",
                                                          "message": "baseline produced no overall_score"}}
                else:
                    loss = round(main_overall - baseline_overall, 6)
                    tol = float(args.baseline_tolerance)
                    baseline_extra = {
                        "baseline_overall_score": baseline_overall,
                        "baseline_scores": baseline_payload["scores"],
                        "loss_vs_baseline": loss,
                        "is_satisfied": bool(loss >= -tol),
                        "baseline_tolerance": tol,
                    }

    inference_params = _load_run_manifest(Path(args.infer_outputs).expanduser().resolve())

    output: Dict[str, Any] = {
        "ok": True,
        "round": args.round,
        "scorer": "vbench",
        "scores": main_payload["scores"],
        "score_dimensions": main_payload["score_dimensions"],
        "quality_score": main_payload["quality_score"],
        "semantic_score": main_payload["semantic_score"],
        "overall_score": main_payload["overall_score"],
        "duration_sec": round(time.perf_counter() - start, 3),
        "commands": main_payload["commands"],
        "data_path": str(Path(args.infer_outputs).expanduser().resolve()),
        "work_dir": str(main_work),
    }
    if inference_params is not None:
        output["inference_params"] = inference_params
    output.update(baseline_extra)

    return _emit(_build_envelope(status="ok", output=output), args.output_json)


if __name__ == "__main__":
    sys.exit(main())