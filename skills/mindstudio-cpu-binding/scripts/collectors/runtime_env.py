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

"""Infer PyTorch / runtime settings from cmdline and /proc/<pid>/environ.

This collector intentionally uses a strict environment whitelist. Full process
environments can contain credentials; only operational CPU/NPU/runtime keys are
copied into the Snapshot.
"""

from __future__ import annotations

from typing import Any

ENV_WHITELIST = {
    "LOCAL_RANK",
    "RANK",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "KMP_AFFINITY",
    "KMP_BLOCKTIME",
    "ASCEND_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "TORCH_DISTRIBUTED_BACKEND",
    "BACKEND",
}


def collect_pytorch_env(
    cmdline: str,
    environ: str,
    torch_num_threads: int | None,
    torch_num_interop_threads: int | None,
    dataloader_workers: int | None,
    dataloader_pin_memory: bool | None,
    dataloader_prefetch_factor: int | None,
) -> dict[str, Any]:
    env = _parse_environ(environ)
    npu_detected = _is_npu_backend(cmdline, env)
    dataloader_source = (
        "cli"
        if _has_dataloader_cli_value(dataloader_workers, dataloader_pin_memory, dataloader_prefetch_factor)
        else None
    )

    return {
        "pytorch": {
            "detected": _is_pytorch_runtime(cmdline),
            "version": None,
            "npu_backend": {
                "detected": npu_detected,
                "name": "torch_npu" if npu_detected else None,
                "version": None,
            },
            "distributed": {
                "enabled": bool(env.get("RANK") or env.get("LOCAL_RANK") or env.get("WORLD_SIZE")),
                "world_size": _safe_int(env.get("WORLD_SIZE")),
                "rank": _safe_int(env.get("RANK")),
                "local_rank": _safe_int(env.get("LOCAL_RANK")),
                "backend": env.get("BACKEND") or env.get("TORCH_DISTRIBUTED_BACKEND"),
            },
            "threading": {
                "torch_num_threads": torch_num_threads,
                "torch_num_interop_threads": torch_num_interop_threads,
                "omp_num_threads": env.get("OMP_NUM_THREADS"),
                "mkl_num_threads": env.get("MKL_NUM_THREADS"),
                "openblas_num_threads": env.get("OPENBLAS_NUM_THREADS"),
                "goto_num_threads": env.get("GOTO_NUM_THREADS"),
                "kmp_affinity": env.get("KMP_AFFINITY"),
                "kmp_blocktime": env.get("KMP_BLOCKTIME"),
            },
            "dataloader": {
                "num_workers": dataloader_workers,
                "pin_memory": dataloader_pin_memory,
                "prefetch_factor": dataloader_prefetch_factor,
                "persistent_workers": None,
                "source": dataloader_source,
            },
            "env": {key: value for key, value in env.items() if key in ENV_WHITELIST},
        }
    }


def _parse_environ(environ: str) -> dict[str, str]:
    result: dict[str, str] = {}
    if not environ:
        return result
    for entry in environ.split("\x00"):
        if "=" not in entry:
            continue
        key, _, value = entry.partition("=")
        if key:
            result[key] = value
    return result


def _is_pytorch_runtime(cmdline: str) -> bool:
    lowered = cmdline.lower()
    if "python" not in lowered:
        return False
    return any(keyword in lowered for keyword in ("torch", "vllm", "sglang", "train.py", "infer"))


def _is_npu_backend(cmdline: str, env: dict[str, str]) -> bool:
    lowered = cmdline.lower()
    return "torch_npu" in lowered or "ASCEND_VISIBLE_DEVICES" in env


def _has_dataloader_cli_value(*values: Any) -> bool:
    return any(value is not None for value in values)


def _safe_int(  # pylint: disable=duplicate-code
    value: str | None,
) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
