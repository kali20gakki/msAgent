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

﻿#!/usr/bin/env python3
"""Recommend first-run msmodeling optix search ranges.

Input is a JSON context. The script deliberately returns `need_more_info`
instead of guessing when required fields are missing.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


GB = 1024**3
DEFAULT_BENCHMARK = "vllm_benchmark"
MINDIE_BENCHMARK = "ais_bench"      # MindIE only supports ais_bench
TARGETS = {"throughput", "ttft", "tpot", "balanced"}
CONFIG_SKILL_SCRIPT = "skills/modeling/msmodeling-optix-param-recommend/scripts/auto_config.py"
CONFIG_PATH_HINT = "optix/config.toml"
VLLM_COMMAND_ARG_BY_NAME = {
    "MAX_MODEL_LEN": "--max-model-len",
    "TENSOR_PARALLEL_SIZE": "--tensor-parallel-size",
    "PIPELINE_PARALLEL_SIZE": "--pipeline-parallel-size",
    "GPU_MEMORY_UTILIZATION": "--gpu_memory_utilization",
    "BLOCK_SIZE": "--block-size",
    "MAX_NUM_PARTIAL_PREFILLS": "--max-num-partial-prefills",
    "LONG_PREFILL_TOKEN_THRESHOLD": "--long-prefill-token-threshold",  # nosec B105
}
VLLM_INLINE_FLAG_NAMES = {
    "COMPILATION_CONFIG",
    "DISABLE_CHUNKED_MM_INPUT",
    "ENABLE_EXPERT_PARALLEL",
}
VLLM_BUILTIN_COMMAND_FIELDS = {"MAX_NUM_BATCHED_TOKENS", "MAX_NUM_SEQS"}

REQUIRED_FIELDS = (
    "engine",
    "hardware.single_card_mem_gb",
    "hardware.world_size",
    "hardware.num_per_nodes",
    "hardware.num_nodes",
    "workload.input_len_avg",
    "workload.input_len_max",
    "workload.output_len_avg",
    "workload.output_len_max",
    "target",
)

MODEL_FIELDS = (
    "model.hidden_size",
    "model.num_hidden_layers",
    "model.num_attention_heads",
    "model.num_key_value_heads",
    "model.torch_dtype",
    "model.max_position_embeddings",
)


def nested_get(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def nested_set(data: Dict[str, Any], path: str, value: Any) -> None:
    cur = data
    parts = path.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def is_truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def dtype_bytes(dtype: Any) -> float:
    text = str(dtype or "bfloat16").lower()
    if "int4" in text or "uint4" in text or "nf4" in text or text in ("fp4", "float4"):
        return 0.5
    if "int8" in text or "uint8" in text:
        return 1
    if "32" in text:
        return 4
    return 2


def divisors(value: int) -> List[int]:
    if value <= 0:
        return [1]
    result = []
    for item in range(1, value + 1):
        if value % item == 0:
            result.append(item)
    return result


def clamp_int(value: float, low: int, high: int) -> int:
    return max(low, min(high, int(round(value))))


def load_model_config(context: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    model = context.setdefault("model", {})
    config_path = model.get("config_path")
    notes = []
    if not config_path:
        return context, notes

    path = Path(config_path).expanduser()
    if not path.exists():
        notes.append(f"model.config_path does not exist: {path}")
        return context, notes

    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    # Some configs nest model parameters under text_config (e.g., Qwen3.5, Qwen3-VL, Kimi)
    text_config = config.get("text_config", {})
    if text_config:
        config = {**config, **text_config}

    mapping = {
        "hidden_size": "hidden_size",
        "intermediate_size": "intermediate_size",
        "num_hidden_layers": "num_hidden_layers",
        "num_attention_heads": "num_attention_heads",
        "num_key_value_heads": "num_key_value_heads",
        "torch_dtype": "torch_dtype",
        "max_position_embeddings": "max_position_embeddings",
        "vocab_size": "vocab_size",
        "model_type": "model_type",
        "architectures": "architectures",
        "num_experts": "num_experts",
        "num_local_experts": "num_local_experts",
        "n_routed_experts": "n_routed_experts",
        "n_shared_experts": "n_shared_experts",
        "head_dim": "head_dim",
        # Total parameter counts (present in some config.json files —
        # preferred over formula estimation when available).
        "num_parameters": "num_parameters",
        "num_total_parameters": "num_total_parameters",
        "total_parameters": "total_parameters",
    }
    for target, source in mapping.items():
        if target not in model and source in config:
            model[target] = config[source]

    if "num_key_value_heads" not in model and "num_attention_heads" in model:
        model["num_key_value_heads"] = model["num_attention_heads"]
    if "torch_dtype" not in model and "dtype" in config:
        model["torch_dtype"] = config["dtype"]
    if not model.get("is_moe"):
        model["is_moe"] = any(key in model for key in ("num_experts", "num_local_experts", "n_routed_experts"))
    return context, notes


def missing_required_fields(context: Dict[str, Any]) -> List[str]:
    missing = [field for field in REQUIRED_FIELDS if nested_get(context, field) in (None, "")]
    has_model_config = nested_get(context, "model.config_path") not in (None, "")
    if not has_model_config:
        missing.extend(field for field in MODEL_FIELDS if nested_get(context, field) in (None, ""))
    return sorted(set(missing))


def next_question_for(missing: List[str]) -> str:
    if not missing:
        return ""
    groups = [
        ("hardware.", "Please provide hardware info: single_card_mem_gb, world_size, num_per_nodes, num_nodes."),
        (
            "model.",
            "Please provide model config.json path, or hidden_size, layers, attention heads, KV heads, dtype, max context length.",
        ),
        ("workload.", "Please provide workload: avg/max input tokens and avg/max output tokens."),
        ("target", "Please provide optimization target: throughput, ttft, tpot or balanced."),
    ]
    for prefix, question in groups:
        if any(field.startswith(prefix) or field == prefix for field in missing):
            return question
    return f"Please supply missing fields: {', '.join(missing)}."


def estimate_model_weight_gb(model: Dict[str, Any]) -> Tuple[float, str]:
    # 1. User-provided explicit weight (most accurate)
    if model.get("model_weight_gb"):
        return to_float(model["model_weight_gb"]), "provided model_weight_gb"

    # 2. Total parameter count — prioritise config.json fields over rough formula
    total_params_from_config = to_int(
        model.get("num_parameters") or model.get("num_total_parameters") or model.get("total_parameters")
    )
    if total_params_from_config:
        return total_params_from_config * dtype_bytes(model.get("torch_dtype")) / GB, "from config.json total_parameters"

    # 3. User-provided approximate parameter count
    if model.get("num_parameters_billion"):
        params = to_float(model["num_parameters_billion"]) * 1_000_000_000
        return params * dtype_bytes(model.get("torch_dtype")) / GB, "estimated from num_parameters_billion"

    # 4. Formula-based rough estimate (fallback)
    hidden = to_int(model.get("hidden_size"))
    intermediate = to_int(model.get("intermediate_size"), hidden * 4)
    layers = to_int(model.get("num_hidden_layers"))
    heads = max(1, to_int(model.get("num_attention_heads")))
    kv_heads = max(1, to_int(model.get("num_key_value_heads"), heads))
    vocab = to_int(model.get("vocab_size"), 150000)
    bytes_per_param = dtype_bytes(model.get("torch_dtype"))
    kv_width = hidden * kv_heads / heads
    attn_params = hidden * hidden + 2 * hidden * kv_width + hidden * hidden
    mlp_params = 3 * hidden * intermediate
    total_params = vocab * hidden + layers * (attn_params + mlp_params) + vocab * hidden

    # MoE correction
    num_experts = to_int(model.get("num_local_experts") or model.get("num_experts") or model.get("n_routed_experts"))
    if num_experts > 1:
        n_shared = to_int(model.get("n_shared_experts"))
        if n_shared:
            # Shared Expert architecture (DeepSeek-V3 style):
            # 1 shared expert + (num_experts - n_shared) routed experts.
            # Shared expert weights are stored once; routed experts are
            # stored independently per expert.
            n_routed = max(1, num_experts - n_shared)
            total_params += layers * (n_shared - 1 + n_routed) * mlp_params
            source = f"MoE estimate ({n_shared} shared + {n_routed} routed experts)"
        else:
            # Standard MoE without shared experts: every expert is independent.
            total_params += layers * (num_experts - 1) * mlp_params
            source = f"rough MoE estimate ({num_experts} experts)"
    else:
        source = "rough decoder-only estimate"

    return total_params * bytes_per_param / GB, source


def choose_parallelism(context: Dict[str, Any], model_weight_gb: float) -> Dict[str, int]:
    hardware = context["hardware"]
    model = context["model"]
    world = max(1, to_int(hardware["world_size"], 1))
    per_node = max(1, to_int(hardware["num_per_nodes"], world))
    card_mem = to_float(hardware["single_card_mem_gb"])
    heads = max(1, to_int(model["num_attention_heads"]))
    layers = max(1, to_int(model["num_hidden_layers"]))

    tp_candidates = [item for item in divisors(world) if item <= per_node and heads % item == 0]
    if not tp_candidates:
        tp_candidates = [1]

    reserve = min(8.0, card_mem * 0.12)
    tp = tp_candidates[-1]
    for candidate in reversed(tp_candidates):
        if card_mem * 0.82 - model_weight_gb / candidate - reserve > card_mem * 0.1:
            tp = candidate
            break

    pp_candidates = [item for item in divisors(world // tp) if item <= max(1, world // tp) and layers % item == 0]
    pp = 1
    if to_int(hardware["num_nodes"], 1) > 1 and world // tp > per_node:
        pp = min(max(1, to_int(hardware["num_nodes"], 1)), pp_candidates[-1] if pp_candidates else 1)

    dp = max(1, world // max(1, tp * pp))
    return {"tp": tp, "pp": pp, "dp": dp}


def kv_capacity(context: Dict[str, Any], parallel: Dict[str, int], model_weight_gb: float) -> Dict[str, int]:
    hardware = context["hardware"]
    model = context["model"]
    workload = context["workload"]
    card_mem = to_float(hardware["single_card_mem_gb"])
    gpu_util = to_float(context.get("gpu_memory_utilization"), 0.88)
    reserve = min(8.0, card_mem * 0.12)
    kv_mem_gb = max(0.0, card_mem * gpu_util - model_weight_gb / parallel["tp"] - reserve)

    hidden = to_int(model["hidden_size"])
    heads = max(1, to_int(model["num_attention_heads"]))
    kv_heads = max(1, to_int(model["num_key_value_heads"]))
    layers = max(1, to_int(model["num_hidden_layers"]))
    block = to_int(context.get("cache_block_size"), 128)
    head_size = to_float(model.get("head_dim"), hidden / heads)
    per_token_bytes = layers * head_size * kv_heads * 2 * dtype_bytes(model.get("torch_dtype")) / parallel["tp"]
    if per_token_bytes <= 0:
        return {"safe_batch": 1, "avg_batch": 1, "kv_mem_gb": int(kv_mem_gb), "per_token_bytes": 0.0}

    total_blocks = (kv_mem_gb * GB) / (block * per_token_bytes)
    max_blocks = math.ceil(to_int(workload["input_len_max"]) / block) + math.ceil(
        to_int(workload["output_len_max"]) / block
    )
    avg_blocks = math.ceil(to_int(workload["input_len_avg"]) / block) + math.ceil(
        to_int(workload["output_len_avg"]) / block
    )
    safe_batch = max(1, int(total_blocks // max(1, max_blocks)))
    avg_batch = max(safe_batch, int(total_blocks // max(1, avg_blocks)))

    # Statistical-multiplexing batch: under uniform-random workload, sequences
    # are on average halfway through their lifecycle (prefill & decode).
    # This reflects real block allocator behaviour much better than the
    # worst-case-max-length assumption behind safe_batch.
    stat_input_blocks = math.ceil(to_int(workload["input_len_avg"]) * 0.5 / block)
    stat_output_blocks = math.ceil(to_int(workload["output_len_avg"]) * 0.5 / block)
    stat_blocks_per_seq = stat_input_blocks + stat_output_blocks
    stat_batch = max(1, int(total_blocks // max(1, stat_blocks_per_seq)))

    return {
        "safe_batch": safe_batch,
        "avg_batch": avg_batch,
        "stat_batch": stat_batch,
        "kv_mem_gb": int(kv_mem_gb),
        "per_token_bytes": per_token_bytes,
    }


def recommendation(
    name: str,
    section: str,
    dtype: str,
    value: Any,
    reason: str,
    min_value: Any = None,
    max_value: Any = None,
    config_position: str = "env",
    search: bool = True,
    dtype_param: Any = None,
    target_field: bool = True,
) -> Dict[str, Any]:
    item = {
        "section": section,
        "name": name,
        "dtype": dtype,
        "value": value,
        "search": search,
        "target_field": target_field,
        "config_position": config_position,
        "reason": reason,
    }
    if min_value is not None:
        item["min"] = min_value
    if max_value is not None:
        item["max"] = max_value
    if dtype_param is not None:
        item["dtype_param"] = dtype_param
    return item


def fixed_presence_flag(name: str, flag: str, reason: str) -> Dict[str, Any]:
    return recommendation(name, "vllm", "enum", flag, reason, search=False, dtype_param=[flag], target_field=False)


def default_presence_setting(name: str, reason: str) -> Dict[str, Any]:
    return recommendation(name, "vllm", "str", "vllm_default", reason, search=False, target_field=False)


def is_moe_model(model: Dict[str, Any]) -> bool:
    flag = model.get("is_moe")
    if isinstance(flag, str):
        return flag.strip().lower() in {"1", "true", "yes", "y"}
    return bool(flag or model.get("num_experts") or model.get("num_local_experts") or model.get("n_routed_experts"))


def _detect_platform(context: Dict[str, Any]) -> str:
    """Best-effort platform detection. Returns 'ascend', 'cuda', or 'unknown'."""
    model = context.get("model", {})
    hints = (
        str(model.get("config_path", "")).lower()
        + str(model.get("name_pattern", "")).lower()
        + str(context.get("hardware", {}).get("platform", "")).lower()
    )
    if "ascend" in hints or "npu" in hints:
        return "ascend"
    quant = str(model.get("quantization", "")).lower()
    if "ascend" in quant:
        return "ascend"
    return "unknown"


# ---------------------------------------------------------------------------
# Ascend platform ceiling constants (910B baseline — override via context)
# ---------------------------------------------------------------------------
# These are UPPER BOUNDS, not recommended values.  KV-cache estimation
# usually produces lower, safer values.  The ceilings only activate for
# very short sequences where pure arithmetic would suggest unrealistically
# high concurrency.
#
# Override via context for other NPU generations:
#
#   { "hardware": {
#       "max_num_seqs_ceiling": 768,
#       "max_num_batched_tokens_ceiling": 65536
#   } }
#
# =========  ==============================  =====
# Chip       Stream/event pool               Ceiling
# =========  ==============================  =====
# Ascend 910B  Limited; 32768 tokens safe    512 / 384 MoE
# Ascend 910B3 Larger; may tolerate higher   TBD
# =========  ==============================  =====
ASCEND_MAX_NUM_SEQS_DENSE = 512
ASCEND_MAX_NUM_SEQS_MOE = 384
ASCEND_MAX_NUM_BATCHED_TOKENS = 32768


def practical_seqs_ceiling(context: Dict[str, Any], parallel: Dict[str, int]) -> int:
    """Practical upper bound for MAX_NUM_SEQS beyond pure KV-cache math.

    KV-cache estimation can produce unrealistically high batch sizes for
    short sequences.  This ceiling accounts for platform ACL-graph limits,
    MoE all-to-all communication overhead, and NPU stream resources that
    KV-cache arithmetic alone cannot model.

    Set ``hardware.max_num_seqs_ceiling`` in the input context to override
    the default for your NPU generation.
    """
    model = context.get("model", {})
    platform = _detect_platform(context)
    moe = is_moe_model(model)

    if platform == "ascend":
        override = to_int(nested_get(context, "hardware.max_num_seqs_ceiling"))
        if override:
            return override
        return ASCEND_MAX_NUM_SEQS_MOE if moe else ASCEND_MAX_NUM_SEQS_DENSE
    return 512 if moe else 4096


def practical_batched_tokens_ceiling(context: Dict[str, Any]) -> int:
    """Practical upper bound for MAX_NUM_BATCHED_TOKENS.

    On Ascend NPUs the ``compile_ranges_endpoints`` derived from this value
    control how many graph streams are allocated during capture.  Values
    above 32768 frequently exhaust the driver stream/event resource pool
    (``Alloc sq cq fail``), forcing an --enforce-eager fallback.

    Set ``hardware.max_num_batched_tokens_ceiling`` in the input context to
    override the default for your NPU generation.
    """
    if _detect_platform(context) == "ascend":
        override = to_int(nested_get(context, "hardware.max_num_batched_tokens_ceiling"))
        if override:
            return override
        return ASCEND_MAX_NUM_BATCHED_TOKENS
    return 131072


# ---------------------------------------------------------------------------
# MAX_MODEL_LEN margin constants
# ---------------------------------------------------------------------------
# Baseline derived from EXP-001 (Qwen3.5-27B W8A8 GQA, Ascend 2×54GB, 8K ctx):
#   business_max = 3.5k(input) + 1.5k(output) = 5000
#   margin = 8192 (context) - 5000 (business) = 3192
#
# The baseline applies to the *sparsest* KV-cache regime (W8A8 + GQA/MLA).
# For denser KV caches, the margin must be tighter — see
# :func:`compute_model_len_margin`.
MODEL_LEN_MARGIN_SPARSE = 3192   # W8A8 + GQA/MLA  (EXP-001 baseline)
MODEL_LEN_MARGIN_MEDIUM = 2048   # W8A8 or GQA alone
MODEL_LEN_MARGIN_DENSE = 1024    # FP16 + MHA  (tightest headroom)
MODEL_LEN_ALIGN = 1024           # round-up alignment (CUDA-friendly)
FAST_MAXLEN = 8192               # fixed value when input+output < FAST_THRESHOLD
FAST_THRESHOLD = 5000            # below this total, use FAST_MAXLEN directly


def compute_model_len_margin(model: Dict[str, Any]) -> int:
    """Context margin tuned to per-token KV-cache density.

    The margin compensates for block-allocation fragmentation and scheduling
    overhead above the business ``input_max + output_max``.  Denser KV caches
    (FP16, MHA) consume more HBM per token, so the margin must be tighter to
    keep ``max_model_len`` within available memory.

    ============  ===========  ======
    KV density    Condition    Margin
    ============  ===========  ======
    Sparse        W8A8 + GQA   3192  (EXP-001 baseline)
    Medium        W8A8 or GQA  2048
    Dense         FP16 + MHA   1024
    ============  ===========  ======
    """
    heads = max(1, to_int(model.get("num_attention_heads", 32)))
    kv_heads = max(1, to_int(model.get("num_key_value_heads", heads)))
    is_gqa = kv_heads < heads

    bp = dtype_bytes(model.get("torch_dtype"))
    is_quantized = bp <= 1.0

    if is_gqa and is_quantized:
        return MODEL_LEN_MARGIN_SPARSE
    elif is_gqa or is_quantized:
        return MODEL_LEN_MARGIN_MEDIUM
    else:
        return MODEL_LEN_MARGIN_DENSE


def compute_max_model_len(model_max_context: int, input_max: int, output_max: int, margin: int | None = None) -> int:
    """Compute MAX_MODEL_LEN with density-aware margin and alignment.

    Short workloads (< FAST_THRESHOLD total tokens) use a fixed 8K context.
    Longer workloads add a headroom margin derived from KV-cache density
    and round up to a CUDA-friendly alignment boundary.

    If *margin* is ``None``, the baseline ``MODEL_LEN_MARGIN_SPARSE`` is
    used as a safe default; callers should prefer passing the result of
    :func:`compute_model_len_margin` for density-aware behaviour.
    """
    total = input_max + output_max
    if total < FAST_THRESHOLD:
        return min(model_max_context, FAST_MAXLEN)
    effective_margin = margin if margin is not None else MODEL_LEN_MARGIN_SPARSE
    with_margin = total + effective_margin
    aligned = ((with_margin + MODEL_LEN_ALIGN - 1) // MODEL_LEN_ALIGN) * MODEL_LEN_ALIGN
    return min(model_max_context, aligned)


# ---------------------------------------------------------------------------
# γ = β × (input_avg / max_model_len)²  — prefill concurrency ratio
# ---------------------------------------------------------------------------
# Calibrated against limited empirical data.  Override these via context
# field ``gamma_beta`` to tune for your own hardware/model combinations:
#
#   { "gamma_beta": { "throughput": 0.30, "balanced": 0.18, "ttft": 0.10 } }
#
# =========  ======  ==================================================
# Target     β       Source
# =========  ======  ==================================================
# throughput 0.22    Qwen3.5-27B W8A8 Ascend 2×54GB, ratio≈0.70, γ≈0.07
# balanced   0.15    EXP-001 (same hardware), ratio≈0.43, γ≈0.028
# ttft       0.08    Conservative lower bound for latency-sensitive runs
# =========  ======  ==================================================
GAMMA_BETA_DEFAULTS: Dict[str, float] = {
    "throughput": 0.22,
    "balanced": 0.15,
    "ttft": 0.08,
}


def estimate_gamma(
    target: str,
    input_avg: int,
    output_avg: int,
    max_model_len: int,
    beta_overrides: Optional[Dict[str, float]] = None,
) -> float:
    """Estimate the prefill concurrency ratio γ for continuous batching.

    In steady state most sequences are in decode (1 token each).  Only a
    fraction of ``MAX_NUM_SEQS`` slots are in prefill simultaneously.

    γ = β × (input_avg / max_model_len)²

    The square penalises high input-to-context ratios — long prefill
    relative to context means fewer sequences can prefill concurrently.

    *beta_overrides* allows per-target tuning of β without code changes.
    Pass a dict like ``{"throughput": 0.30, "balanced": 0.18}`` to
    replace the default coefficients.  Unknown targets fall back to the
    ``"balanced"`` default.

    **Limitations**: (1) sample size is small — these are engineering
    heuristics, not statistically rigorous calibrations; (2) β does not
    yet differentiate by model architecture (MHA/GQA/MLA) or quantization
    (FP16/W8A8/INT4); (3) γ is always bounded to [0.02, 0.50] and the
    downstream ``MAX_NUM_BATCHED_TOKENS`` gets a ±30 % search range, so
    moderate β errors are absorbed by the optimizer.
    """
    betas = {**GAMMA_BETA_DEFAULTS, **(beta_overrides or {})}
    beta = betas.get(target, betas.get("balanced", 0.15))

    input_ratio = input_avg / max(max_model_len, 1)
    gamma = beta * (input_ratio ** 2)
    return max(0.02, min(0.50, gamma))


def _hbm_tpot_ceiling(context: Dict[str, Any], per_token_bytes: float) -> int:
    """HBM-bandwidth constrained CONCURRENCY ceiling from TPOT SLO.

    In decode, every step must read ALL active KV-cache from HBM for
    every sequence.  There is no cross-sequence attention in decode,
    only per-sequence self-attention on the prefix context: **O(C)**.

    HBM read volume per decode step::

        C × total_context_tokens × per_token_kv_bytes

    Combined with per-step compute + schedule overhead, TPOT grows
    linearly with C.  We solve for the largest C within the SLO.
    """
    tpot_slo_ms = to_float(context.get("tpot_slo_ms"), 0.0)
    if tpot_slo_ms <= 0 or per_token_bytes <= 0:
        return 10 ** 9

    platform = _detect_platform(context)
    # Effective HBM bandwidth (GB/s) for decode access pattern.
    # Conservative — real achievable BW depends on access granularity
    # and NPU/GPU generation.
    if platform == "ascend":
        hbm_bw_gbps = 1000.0
    else:
        hbm_bw_gbps = 1500.0

    workload = context.get("workload", {})
    total_context = to_int(workload.get("input_len_avg")) + to_int(
        workload.get("output_len_avg")
    )
    if total_context <= 0:
        return 10 ** 9

    # Per-step: scheduler launch + compute floor + MoE all-to-all
    overhead_ms = 5.0
    available_ms = max(1.0, tpot_slo_ms - overhead_ms)

    # HBM read per sequence per decode step (GB)
    hbm_read_gb_per_seq = total_context * per_token_bytes / 1e9

    # TPOT ≈ overhead + C × (hbm_read_gb_per_seq × 1000 / hbm_bw_gbps)
    # C_max  = available_ms × hbm_bw_gbps / (hbm_read_gb_per_seq × 1000)
    ceiling = available_ms * hbm_bw_gbps / (hbm_read_gb_per_seq * 1000.0)
    return max(1, int(ceiling))


def benchmark_recommendations(
    capacity: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    max_seqs_hi: int = 4096,
) -> List[Dict[str, Any]]:
    """Recommend CONCURRENCY / REQUESTRATE from physical constraints.

    Three independent ceilings are computed, and the *smallest* wins:

    1. **HBM bandwidth** — decode step reads all KV-cache every step.
       O(C) in concurrency, bounded by TPOT SLO.
    2. **KV-cache capacity** — how many sequences physically fit in HBM.
       Derived from ``capacity["avg_batch"]``.
    3. **Burst safety** — benchmark Poisson arrival means instantaneous
       concurrency can exceed the target.  ``max_seqs_hi // 2`` ensures
       ``MAX_NUM_SEQS ≥ CONCURRENCY × 2``.
    """
    per_token_bytes = capacity.get("per_token_bytes", 0.0)

    # --- ceiling 1: HBM bandwidth vs TPOT SLO ---------------------------
    hbm_ceiling = 10 ** 9
    if context is not None:
        hbm_ceiling = _hbm_tpot_ceiling(context, per_token_bytes)

    # --- ceiling 2: KV-cache capacity -----------------------------------
    kv_ceiling = capacity.get("avg_batch", 4096)

    # --- ceiling 3: burst-safety ratio vs MAX_NUM_SEQS ------------------
    burst_ratio = 1.5
    burst_ceiling = max(8, int(max_seqs_hi / burst_ratio))

    # --- effective ceiling = min of all three ---------------------------
    effective_ceiling = min(hbm_ceiling, kv_ceiling, burst_ceiling)
    effective_ceiling = max(8, effective_ceiling)

    # Build a search range with meaningful width
    concurrency_lo = max(8, effective_ceiling // 4)
    concurrency_hi = effective_ceiling
    concurrency_default = max(concurrency_lo, effective_ceiling // 2)

    ceiling_source = "burst-safety" if burst_ceiling == effective_ceiling else (
        "HBM-bandwidth/TPOT" if hbm_ceiling == effective_ceiling else "KV-cache capacity"
    )
    concurrency_reason = (
        f"CONCURRENCY bounded by {ceiling_source} "
        f"(HBM={hbm_ceiling}, KV={kv_ceiling}, burst-ratio={burst_ratio}×)."
    )

    engine = str(context.get("engine", "")).lower() if context else ""
    section = "vllm" if engine == "vllm" else "ais_bench"
    return [
        recommendation(
            "CONCURRENCY",
            section,
            "int",
            concurrency_default,
            concurrency_reason,
            concurrency_lo,
            concurrency_hi,
            "env",
        ),
        recommendation(
            "REQUESTRATE",
            section,
            "float",
            max(1, concurrency_default // 2),
            "Request rate starts below concurrency; optimizer can raise it after measuring throughput.",
            1,
            concurrency_hi,
            "env",
        ),
    ]


def recommend_mindie(
    context: Dict[str, Any], parallel: Dict[str, int], capacity: Dict[str, int]
) -> List[Dict[str, Any]]:
    target = context["target"]
    model = context["model"]
    hardware = context["hardware"]
    workload = context["workload"]
    max_batch_hi = clamp_int(capacity["avg_batch"] * (1.3 if target == "throughput" else 1.0), 2, 4096)
    max_batch_lo = clamp_int(capacity["safe_batch"] * (0.5 if target == "ttft" else 0.7), 1, max_batch_hi)
    default_batch = clamp_int(capacity["safe_batch"], max_batch_lo, max_batch_hi)
    prefill_min, prefill_max, prefill_value = (0.1, 0.4, 0.25) if target == "ttft" else (0.3, 0.7, 0.5)
    max_prefill_tokens = max(to_int(workload["input_len_max"]), to_int(workload["input_len_avg"]) * default_batch)
    items = [
        recommendation(
            "max_batch_size",
            "mindie",
            "int",
            default_batch,
            "Bounded by estimated KV cache capacity under max input/output length.",
            max_batch_lo,
            max_batch_hi,
            "BackendConfig.ScheduleConfig.maxBatchSize",
        ),
        recommendation(
            "max_prefill_batch_size",
            "mindie",
            "ratio",
            prefill_value,
            "Ratio of maxBatchSize; lower favors TTFT, higher favors throughput.",
            prefill_min,
            prefill_max,
            "BackendConfig.ScheduleConfig.maxPrefillBatchSize",
            dtype_param="max_batch_size",
        ),
        recommendation(
            "max_prefill_token",
            "mindie",
            "int",
            max_prefill_tokens,
            "Total prefill tokens should cover the first-run workload shape.",
            to_int(workload["input_len_max"]),
            max(max_prefill_tokens * 2, to_int(workload["input_len_max"])),
            "BackendConfig.ScheduleConfig.maxPrefillTokens",
        ),
        recommendation(
            "max_queue_deloy_mircroseconds",
            "mindie",
            "range",
            10000 if target == "ttft" else 100000,
            "Queue delay trades batching opportunity against TTFT.",
            500,
            1000000,
            "BackendConfig.ScheduleConfig.maxQueueDelayMicroseconds",
            dtype_param=100,
        ),
        recommendation(
            "support_select_batch",
            "mindie",
            "bool",
            target != "ttft",
            "Disable for TTFT-first runs; enable for throughput or balanced runs.",
            0,
            1,
            "BackendConfig.ScheduleConfig.supportSelectBatch",
            search=target == "balanced",
        ),
        recommendation(
            "prefill_time_ms_per_req",
            "mindie",
            "range",
            0,
            "Keep searchable but conservative for first-run schedule exploration.",
            0,
            1000,
            "BackendConfig.ScheduleConfig.prefillTimeMsPerReq",
            dtype_param=10,
        ),
        recommendation(
            "decode_time_ms_per_req",
            "mindie",
            "range",
            0,
            "Keep searchable but conservative for first-run schedule exploration.",
            0,
            1000,
            "BackendConfig.ScheduleConfig.decodeTimeMsPerReq",
            dtype_param=10,
        ),
        recommendation(
            "max_preempt_count",
            "mindie",
            "ratio",
            0.1 if target == "throughput" else 0,
            "Allows limited preemption exploration; keep low for first-run stability.",
            0,
            0.3,
            "BackendConfig.ScheduleConfig.maxPreemptCount",
            dtype_param="max_batch_size",
        ),
        recommendation(
            "prefill_policy_type",
            "mindie",
            "enum",
            0,
            "First-run schedule policy candidate; keep enum values aligned with MindIE config support.",
            0,
            1,
            "BackendConfig.ScheduleConfig.prefillPolicyType",
            dtype_param=[0, 1, 3],
        ),
        recommendation(
            "decode_policy_type",
            "mindie",
            "enum",
            0,
            "First-run decode schedule policy candidate; keep enum values aligned with MindIE config support.",
            0,
            1,
            "BackendConfig.ScheduleConfig.decodePolicyType",
            dtype_param=[0, 1, 3],
        ),
        recommendation(
            "tp",
            "mindie",
            "enum",
            parallel["tp"],
            "Selected from attention-head-compatible divisors of world_size.",
            0,
            1,
            "BackendConfig.ModelDeployConfig.ModelConfig.0.tp",
            dtype_param=sorted(set([1, parallel["tp"]])),
        ),
        recommendation(
            "dp",
            "mindie",
            "int",
            parallel["dp"],
            "Derived to satisfy DP * TP * PP == world_size; keep fixed in first run.",
            parallel["dp"],
            parallel["dp"],
            "BackendConfig.ModelDeployConfig.ModelConfig.0.dp",
            search=False,
        ),
    ]
    if is_moe_model(model):
        world = max(1, to_int(hardware["world_size"], 1))
        expert_count = max(
            1,
            to_int(model.get("num_experts") or model.get("num_local_experts") or model.get("n_routed_experts"), world),
        )
        ep_candidates = [item for item in divisors(world) if item <= expert_count]
        moe_ep = ep_candidates[-1] if ep_candidates else 1
        items.extend(
            [
                recommendation(
                    "moe_ep",
                    "mindie",
                    "enum",
                    moe_ep,
                    "MoE model detected; explore expert parallelism with divisors of world_size.",
                    0,
                    1,
                    "BackendConfig.ModelDeployConfig.ModelConfig.0.moe_ep",
                    dtype_param=sorted(set([1, moe_ep])),
                ),
                recommendation(
                    "moe_tp",
                    "mindie",
                    "factories",
                    max(1, world // moe_ep),
                    "Derived from moe_ep so moe_ep * moe_tp tracks available parallel resources.",
                    0,
                    0,
                    "BackendConfig.ModelDeployConfig.ModelConfig.0.moe_tp",
                    dtype_param={"target_name": "moe_ep", "product": world, "dtype": "int"},
                ),
            ]
        )
    return items + benchmark_recommendations(capacity, context, max_batch_hi)


def recommend_vllm(context: Dict[str, Any], parallel: Dict[str, int], capacity: Dict[str, int]) -> List[Dict[str, Any]]:
    target = context["target"]
    workload = context["workload"]
    model = context["model"]
    max_model_len = compute_max_model_len(
        to_int(model["max_position_embeddings"]),
        to_int(workload["input_len_max"]),
        to_int(workload["output_len_max"]),
        compute_model_len_margin(model),
    )
    # Workload-aware ceiling: safe_batch is the worst-case floor (all
    # sequences at max length — will never happen under uniform-random
    # load).  stat_batch accounts for statistical multiplexing: sequences
    # are on average at 50 % of their lifecycle, which matches real block
    # allocator behaviour.  Use the larger of the two as the KV-capacity
    # bound so that the optimizer can explore the real headroom.
    workload_batch = max(capacity["safe_batch"], capacity["stat_batch"])
    seqs_ceiling = min(
        practical_seqs_ceiling(context, parallel),
        workload_batch,
    )
    kv_hi = capacity["avg_batch"] * (1.3 if target == "throughput" else 1.0)
    kv_lo = capacity["safe_batch"] * (0.5 if target == "ttft" else 0.7)
    max_num_seqs_hi = clamp_int(min(kv_hi, seqs_ceiling), 2, seqs_ceiling)

    # Lower bound: grounded in the SAME physical quantity that drives hi.
    # - When hi is KV-capacity-bound, lo = safe_batch × 0.7 (KV 70% load).
    # - When hi is practical-ceiling-bound (short sequences), lo = hi // 2
    #   to give the optimizer meaningful search width.
    kv_bound = seqs_ceiling == workload_batch
    if kv_bound:
        max_num_seqs_lo = clamp_int(kv_lo, 1, max_num_seqs_hi)
        max_num_seqs_value = clamp_int(
            (max_num_seqs_lo + max_num_seqs_hi) // 2, max_num_seqs_lo, max_num_seqs_hi
        )
    else:
        max_num_seqs_lo = clamp_int(max_num_seqs_hi // 2, 1, max_num_seqs_hi)
        max_num_seqs_value = clamp_int(max_num_seqs_hi // 2, max_num_seqs_lo, max_num_seqs_hi)

    # MAX_NUM_BATCHED_TOKENS: γ-based continuous-batching estimate.
    # In steady state most sequences are in decode (1 token each); only a
    # fraction of MAX_NUM_SEQS slots are in prefill simultaneously.
    # γ = β × (input_avg / max_model_len)² captures this prefill concurrency
    # ratio, bounded and platform-capped.
    batched_tokens_ceiling = practical_batched_tokens_ceiling(context)
    gamma_beta = context.get("gamma_beta") if isinstance(context.get("gamma_beta"), dict) else None
    gamma_val = estimate_gamma(
        target,
        to_int(workload["input_len_avg"]),
        to_int(workload["output_len_avg"]),
        max_model_len,
        gamma_beta,
    )
    batched_tokens_est = int(max_model_len * max_num_seqs_hi * gamma_val) + max_num_seqs_hi
    batched_tokens_value = min(batched_tokens_est, batched_tokens_ceiling)
    # Search range: ±30 % around the point estimate
    batched_tokens_lo = max(max_model_len, int(batched_tokens_value * 0.7))
    batched_tokens_hi = min(int(batched_tokens_value * 1.3), batched_tokens_ceiling)
    return [
        recommendation(
            "MAX_MODEL_LEN",
            "vllm",
            "int",
            max_model_len,
            "Includes context margin above business max tokens, aligned to CUDA-friendly boundary.",
            max_model_len,
            max_model_len,
            "env",
            search=False,
        ),
        recommendation(
            "MAX_NUM_SEQS",
            "vllm",
            "int",
            max_num_seqs_value,
            "Bounded by estimated KV cache capacity under the first-run workload.",
            max_num_seqs_lo,
            max_num_seqs_hi,
            "env",
        ),
        recommendation(
            "MAX_NUM_BATCHED_TOKENS",
            "vllm",
            "int",
            batched_tokens_value,
            f"γ={gamma_val:.3f}-based continuous-batching estimate; prefill concurrency tuned for {target}.",
            batched_tokens_lo,
            batched_tokens_hi,
            "env",
        ),
        recommendation(
            "TENSOR_PARALLEL_SIZE",
            "vllm",
            "enum",
            parallel["tp"],
            "Selected from attention-head-compatible divisors of world_size.",
            0,
            1,
            "env",
            dtype_param=sorted(set([1, parallel["tp"]])),
        ),
        recommendation(
            "PIPELINE_PARALLEL_SIZE",
            "vllm",
            "enum",
            parallel["pp"],
            "Keep at 1 for first run unless multi-node or very large model requires PP.",
            0,
            1,
            "env",
            dtype_param=sorted(set([1, parallel["pp"]])),
        ),
        recommendation(
            "DATA_PARALLEL_SIZE",
            "vllm",
            "int",
            parallel["dp"],
            "Derived to satisfy DP * TP * PP == world_size; wire into startup only if your vLLM mode supports it.",
            parallel["dp"],
            parallel["dp"],
            "env",
            search=False,
        ),
        recommendation(
            "GPU_MEMORY_UTILIZATION",
            "vllm",
            "float",
            0.9,
            "Conservative first-run utilization for avoiding startup OOM.",
            0.85,
            0.92,
            "env",
        ),
        recommendation(
            "BLOCK_SIZE",
            "vllm",
            "enum",
            16,
            "KV cache block-size candidate; verify available values with `vllm serve --help` on your environment.",
            0,
            1,
            "env",
            dtype_param=[16, 32, 64, 128],
        ),
        default_presence_setting(
            "ENABLE_PREFIX_CACHING", "Use vLLM's model-aware default; do not pass an explicit flag for first runs."
        ),
        default_presence_setting(
            "ENABLE_CHUNKED_PREFILL", "Use vLLM's model-aware default; do not pass an explicit flag for first runs."
        ),
        recommendation(
            "COMPILATION_CONFIG",
            "vllm",
            "enum",
            "",
            "Compile/cudagraph config. The empty string means no compilation config (safe default). "
            "The full --compilation-config flag is embedded in dtype_param values.",
            0,
            1,
            "env",
            dtype_param=["", "--compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}'"],
        ),
    ] + benchmark_recommendations(capacity, context, max_num_seqs_hi)

def load_discovery_help_text(context: Dict[str, Any]) -> Tuple[str, List[str]]:
    discovery = context.get("discovery") if isinstance(context.get("discovery"), dict) else {}
    if not is_truthy(discovery.get("enabled")):
        return "", []
    notes = []
    if discovery.get("vllm_help_text"):
        return str(discovery["vllm_help_text"]), notes
    help_path = discovery.get("vllm_help_text_path")
    if not help_path:
        notes.append("discovery.enabled is true but no vllm_help_text or vllm_help_text_path was provided.")
        return "", notes
    path = Path(str(help_path)).expanduser()
    if not path.exists():
        notes.append(f"discovery.vllm_help_text_path does not exist: {path}")
        return "", notes
    return path.read_text(encoding="utf-8", errors="replace"), notes


def discovered_recommendation(item: Dict[str, Any], flag: str) -> Dict[str, Any]:
    item["source"] = "vllm --help"
    item["optional"] = True
    item["placeholder_hint"] = f"Add `{flag}` with `${item['name']}` in vllm.command.others if you keep this field."
    return item


def discover_vllm_optional_recommendations(context: Dict[str, Any], help_text: str) -> List[Dict[str, Any]]:
    if not help_text:
        return []
    text = help_text.lower()
    workload = context["workload"]
    model = context["model"]
    target = context["target"]
    long_prefill = to_int(workload["input_len_max"]) >= 4096 or to_int(workload["input_len_avg"]) >= 2048
    items = []

    if long_prefill and "--max-num-partial-prefills" in text:
        value = 2 if target in {"throughput", "balanced"} else 1
        items.append(
            discovered_recommendation(
                recommendation(
                    "MAX_NUM_PARTIAL_PREFILLS",
                    "vllm",
                    "int",
                    value,
                    "Discovered from vLLM help; useful for long-prefill workloads with chunked prefill.",
                    1,
                    4,
                    "env",
                ),
                "--max-num-partial-prefills",
            )
        )
    if long_prefill and "--long-prefill-token-threshold" in text:
        input_avg = to_int(workload["input_len_avg"])
        input_max = to_int(workload["input_len_max"])
        value = clamp_int(max(512, input_avg), 1, input_max)
        items.append(
            discovered_recommendation(
                recommendation(
                    "LONG_PREFILL_TOKEN_THRESHOLD",
                    "vllm",
                    "int",
                    value,
                    "Discovered from vLLM help; separates long prompts for chunked-prefill scheduling.",
                    0,
                    input_max,
                    "env",
                ),
                "--long-prefill-token-threshold",
            )
        )
    if is_truthy(model.get("is_multimodal")) and "--disable-chunked-mm-input" in text:
        items.append(
            discovered_recommendation(
                fixed_presence_flag(
                    "DISABLE_CHUNKED_MM_INPUT",
                    "--disable-chunked-mm-input",
                    "Discovered from vLLM help; keep multimodal items from being partially chunked.",
                ),
                "--disable-chunked-mm-input",
            )
        )
    if is_moe_model(model) and "--enable-expert-parallel" in text:
        items.append(
            discovered_recommendation(
                fixed_presence_flag(
                    "ENABLE_EXPERT_PARALLEL",
                    "--enable-expert-parallel",
                    "Discovered from vLLM help for MoE models; enable expert parallel serving.",
                ),
                "--enable-expert-parallel",
            )
        )
    return items


def merge_recommendations(base: List[Dict[str, Any]], extra: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    names = {item["name"] for item in base}
    merged = list(base)
    for item in extra:
        if item["name"] not in names:
            merged.append(item)
            names.add(item["name"])
    return merged


def quote_arg(value: Any) -> str:
    return shlex.quote(str(value))


def cli_arg_for(item: Dict[str, Any]) -> Optional[str]:
    if not item.get("target_field", True):
        return None
    if item["section"] != "vllm":
        return None
    if item["name"] in VLLM_INLINE_FLAG_NAMES:
        return ""
    return VLLM_COMMAND_ARG_BY_NAME.get(item["name"])


def vllm_command_others(recs: List[Dict[str, Any]]) -> str:
    parts = []
    by_name = {item["name"]: item for item in recs if item["section"] == "vllm"}
    for name, cli_arg in VLLM_COMMAND_ARG_BY_NAME.items():
        if name in by_name and by_name[name].get("target_field", True):
            parts.extend([cli_arg, f"${name}"])
    for name in VLLM_INLINE_FLAG_NAMES:
        if name in by_name:
            item = by_name[name]
            if item.get("target_field", True):
                parts.append(f"${name}")
            else:
                parts.append(str(item["value"]))
    return " ".join(parts)


def normalized_target_field(item: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "section",
        "name",
        "config_position",
        "dtype",
        "min",
        "max",
        "value",
        "dtype_param",
        "search",
        "target_field",
        "reason",
        "source",
        "optional",
        "placeholder_hint",
    )
    result = {key: item[key] for key in keys if key in item}
    result["config_skill_cli_arg"] = cli_arg_for(item)
    return result


def append_option(args: List[str], option: str, value: Any) -> None:
    text = str(value)
    if text.startswith("-"):
        args.append(f"{option}={text}")
    else:
        args.extend([option, text])


def command_for_config_skill(item: Dict[str, Any]) -> Optional[str]:
    if not item.get("target_field", True):
        return None
    section = item["section"]
    if section not in {"mindie", "vllm", "ais_bench"}:
        return None
    if section == "vllm" and item["name"] in VLLM_BUILTIN_COMMAND_FIELDS:
        return None

    mode = "--add-search-param"
    args = [
        "python",
        CONFIG_SKILL_SCRIPT,
        mode,
        "--engine",
        section,
        "--param-name",
        item["name"],
        "--config-position",
        item.get("config_position", "env"),
        "--dtype",
        item["dtype"],
    ]
    append_option(args, "--value", item.get("value", ""))

    if "min" in item:
        args.extend(["--min", item["min"]])
    if "max" in item:
        args.extend(["--max", item["max"]])
    if item["dtype"] == "enum" and "dtype_param" in item:
        args.extend(["--enum-values", json.dumps(item["dtype_param"], ensure_ascii=False)])
    elif item["dtype"] in {"ratio", "range", "times"} and "dtype_param" in item:
        dtype_param = item["dtype_param"]
        if isinstance(dtype_param, (dict, list)):
            dtype_param = json.dumps(dtype_param, ensure_ascii=False)
        args.extend(["--dtype-param", dtype_param])
    elif item["dtype"] == "factories" and "dtype_param" in item:
        args.extend(["--factories-config", json.dumps(item["dtype_param"], ensure_ascii=False)])

    cli_arg = cli_arg_for(item)
    if cli_arg is not None:
        append_option(args, "--cli-arg", cli_arg)

    return " ".join(quote_arg(arg) for arg in args)


def build_config_skill_handoff(engine: str, recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    target_fields = [normalized_target_field(item) for item in recs if item.get("target_field", True)]
    commands = [command for item in recs if (command := command_for_config_skill(item))]
    unsupported = [item["name"] for item in recs if item["section"] not in {"mindie", "vllm", "ais_bench"}]
    notes = [
        "Review generated commands before writing config.toml.",
        "VLLM MAX_NUM_BATCHED_TOKENS and MAX_NUM_SEQS are already placeholders in VllmCommand.",
    ]
    if unsupported:
        notes.append(
            f"Current config skill CLI does not directly manage these sections: {', '.join(sorted(set(unsupported)))}."
        )
    if engine == "vllm":
        notes.append("Set vllm.command.others to include vllm_command_others placeholders before running optimizer.")
    return {
        "version": 1,
        "consumer_skill": "msmodeling-optix-param-recommend",
        "handoff_type": "target_fields_and_commands",
        "config_path_hint": CONFIG_PATH_HINT,
        "target_fields": target_fields,
        "vllm_command_others": vllm_command_others(recs) if engine == "vllm" else "",
        "apply_commands": commands,
        "notes": notes,
    }


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{key} = {toml_value(item)}" for key, item in value.items()) + "}"
    return str(value)


def target_field_toml(engine: str, item: Dict[str, Any]) -> str:
    lines = [
        f"[[{engine}.target_field]]",
        f"name = {toml_value(item['name'])}",
        f"config_position = {toml_value(item['config_position'])}",
        f"dtype = {toml_value(item['dtype'])}",
    ]
    if item.get("search", True):
        if "min" in item:
            lines.append(f"min = {toml_value(item['min'])}")
        if "max" in item:
            lines.append(f"max = {toml_value(item['max'])}")
    else:
        string_enum = item["dtype"] == "enum" and isinstance(item["value"], str)
        if string_enum:
            lines.append("min = 0")
            lines.append("max = 1")
        else:
            lines.append(f"min = {toml_value(item['value'])}")
            lines.append(f"max = {toml_value(item['value'])}")
    if "dtype_param" in item:
        lines.append(f"dtype_param = {toml_value(item['dtype_param'])}")
    lines.append(f"value = {toml_value(item['value'])}")
    return "\n".join(lines)


def build_toml(
    engine: str,
    benchmark: str,
    recs: List[Dict[str, Any]],
    world_size: int,
    workload: Optional[Dict[str, Any]] = None,
) -> str:
    if engine == "vllm":
        constraint_expr = "$DATA_PARALLEL_SIZE * $TENSOR_PARALLEL_SIZE * $PIPELINE_PARALLEL_SIZE == $NPU_COUNT"
    else:
        constraint_expr = "$dp * $tp == $NPU_COUNT"
    sections = [
        "# Generated by msmodeling-optix-param-recommend. Review before applying.",
        f"# Parallelism constraint: {constraint_expr}  (world_size = {world_size})",
        "# MindIE v1 assumes PP=1 unless your MindIE config explicitly supports PP.",
        "#",
        "# [constraint] section should already be present in your config.toml template.",
        "# Update it to match:",
        f"#   enable = true",
        f"#   npu_count = {world_size}",
        f"#   expressions = [{toml_value(constraint_expr)}]",
        "",
    ]
    engine_items = [item for item in recs if item["section"] == engine and item.get("target_field", True)]
    sections.extend(target_field_toml(engine, item) + "\n" for item in engine_items)
    if benchmark == "ais_bench":
        sections.append("[ais_bench.command]")
        sections.append("num_prompts = 3000")
        sections.append("")
        sections.append("# CONCURRENCY and REQUESTRATE for ais_bench are kept in JSON handoff.")
        sections.append("# They are not emitted as ais_bench target-field blocks because the current Settings loader")
        sections.append("# calls range_to_enum(self.ais_bench) instead of range_to_enum(self.ais_bench.target_field).")
    elif benchmark == "vllm_benchmark":
        wl = workload or {}
        input_len = wl.get("input_len_avg", 1024)
        output_len = wl.get("output_len_avg", 256)
        sections.append("[vllm_benchmark.command]")
        sections.append('dataset_name = "random"')
        sections.append("num_prompts = 3000")
        sections.append(f'others = "--random-input-len {input_len} --random-output-len {output_len} --temperature 0"')
        sections.append("")
        sections.append("# CONCURRENCY and REQUESTRATE are emitted as [[vllm.target_field]] above.")
        sections.append("# The framework auto-injects --max-concurrency $CONCURRENCY and --request-rate $REQUESTRATE")
        sections.append("# into vllm bench serve. Both variables are resolved from [[vllm.target_field]].")
    return "\n".join(sections)


def recommend(context: Dict[str, Any]) -> Dict[str, Any]:
    context = json.loads(json.dumps(context))
    context, load_notes = load_model_config(context)
    engine = str(context.get("engine", "")).lower()
    if engine:
        context["engine"] = engine
    if context.get("target"):
        context["target"] = str(context["target"]).lower()
    # Engine-aware benchmark default: MindIE only supports ais_bench.
    benchmark = str(context.get("benchmark_policy") or "")
    if not benchmark:
        benchmark = MINDIE_BENCHMARK if engine == "mindie" else DEFAULT_BENCHMARK
    context["benchmark_policy"] = benchmark

    missing = missing_required_fields(context)
    if engine not in {"mindie", "vllm"} and "engine" not in missing:
        missing.append("engine")
    if context.get("target") not in TARGETS and "target" not in missing:
        missing.append("target")
    if missing:
        return {
            "status": "need_more_info",
            "benchmark_policy": benchmark,
            "missing_fields": sorted(set(missing)),
            "next_question": next_question_for(sorted(set(missing))),
            "notes": load_notes,
        }

    model_weight_gb, weight_source = estimate_model_weight_gb(context["model"])
    parallel = choose_parallelism(context, model_weight_gb)
    capacity = kv_capacity(context, parallel, model_weight_gb)
    discovery_notes = []
    if engine == "mindie":
        recs = recommend_mindie(context, parallel, capacity)
        discovered_recs = []
        command = "msmodeling optix -e mindie -b ais_bench"
    else:
        help_text, discovery_notes = load_discovery_help_text(context)
        recs = recommend_vllm(context, parallel, capacity)
        discovered_recs = discover_vllm_optional_recommendations(context, help_text)
        recs = merge_recommendations(recs, discovered_recs)
        command = "msmodeling optix -e vllm -b vllm_benchmark"

    expression = f"DP * TP * PP == world_size ({parallel['dp']} * {parallel['tp']} * {parallel['pp']} == {context['hardware']['world_size']})"
    discovery_enabled = is_truthy(nested_get(context, "discovery.enabled", False))
    return {
        "status": "ok",
        "engine": engine,
        "benchmark_policy": benchmark,
        "assumptions": [
            "benchmark_policy defaults to "
            f"{MINDIE_BENCHMARK if engine == 'mindie' else DEFAULT_BENCHMARK}"
            if "benchmark_policy" not in context
            else f"benchmark_policy = {benchmark}",
            f"model weight is {model_weight_gb:.2f} GB ({weight_source})",
            f"estimated KV cache budget is {capacity['kv_mem_gb']} GB per card",
        ]
        + load_notes
        + discovery_notes,
        "parallelism": parallel,
        "constraints": [
            {
                "expression": expression,
                "recommendation": "Use DP * TP * PP == world_size for first-run full-card recommendations.",
            }
        ],
        "discovery": {
            "enabled": discovery_enabled,
            "added_parameters": [item["name"] for item in discovered_recs],
            "notes": discovery_notes,
        },
        "recommendations": recs,
        "config_skill_handoff": build_config_skill_handoff(engine, recs),
        "toml_snippet": build_toml(
            engine, benchmark, recs, to_int(context["hardware"]["world_size"]), context.get("workload")
        ),
        "next_command": command,
    }


def print_template() -> None:
    template = {
        "engine": "vllm",
        "hardware": {
            "single_card_mem_gb": 64,
            "world_size": 8,
            "num_per_nodes": 8,
            "num_nodes": 1,
        },
        "model": {"config_path": "/path/to/model/config.json"},
        "workload": {
            "input_len_avg": 1024,
            "input_len_max": 4096,
            "output_len_avg": 256,
            "output_len_max": 512,
        },
        "target": "balanced",
    }
    print(json.dumps(template, indent=2, ensure_ascii=False))


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Recommend first-run msmodeling optix parameter ranges.")
    parser.add_argument("--context", type=Path, help="Path to JSON context.")
    parser.add_argument("--print-template", action="store_true", help="Print an example context JSON.")
    args = parser.parse_args(argv)

    if args.print_template:
        print_template()
        return 0
    if not args.context:
        parser.error("--context is required unless --print-template is used")

    with args.context.open("r", encoding="utf-8") as handle:
        context = json.load(handle)
    print(json.dumps(recommend(context), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

