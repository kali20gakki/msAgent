import json
import re
from pathlib import Path
from typing import Any

EPS = 1e-12
THRESHOLDS_OK = 1e-4
THRESHOLDS_WARN = 1e-2

STRUCTURAL_PATTERNS = {
    "linear_q_down_proj": "fused_qkv_a_proj",
    "linear_kv_down_proj": "fused_qkv_a_proj",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_module_keys(dump_data: dict[str, Any]) -> list[str]:
    data = dump_data.get("data", {})
    if not isinstance(data, dict):
        return []
    return [key for key in data if isinstance(key, str) and key.startswith("Module.")]


def get_module_items(dump_data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    data = dump_data.get("data", {})
    if not isinstance(data, dict):
        return []
    return [(k, v) for k, v in data.items() if isinstance(k, str) and k.startswith("Module.")]


def parse_layer_idx(key: str) -> int | None:
    match = re.search(r"\.layers\.(\d+)\.", key)
    return int(match.group(1)) if match else None


def infer_block(key: str) -> str:
    lowered = key.lower()
    if ".mlp." in lowered:
        return "mlp"
    if "self_attn" in lowered or "self_attention" in lowered or "attention" in lowered:
        return "attn"
    if "norm" in lowered:
        return "norm"
    if "embed" in lowered:
        return "embed"
    return "other"


def structural_false_positive(train_key: str, rollout_key: str) -> str:
    lowered_train = train_key.lower()
    lowered_rollout = rollout_key.lower()
    for key, value in STRUCTURAL_PATTERNS.items():
        if key in lowered_train and value in lowered_rollout:
            return f"structural fusion pattern: {key} -> {value}"
    return ""


def is_fused_qkv_pair(train_key: str, rollout_key: str) -> bool:
    return bool(structural_false_positive(train_key, rollout_key))
