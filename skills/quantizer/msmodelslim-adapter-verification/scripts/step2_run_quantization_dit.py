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
"""Step2 DiT 扩展：全回退量化（apiversion: multimodal_sd_modelslim_v1）。

LLM/VLM 用 step2_run_quantization.py；DiT 由本脚本单独兜住三件事：
  1. ``spec.dataset`` 必填且必须是字符串 —— 缺省时自动落一份自建校准集
  2. ``inference_config`` 各推理仓字段集不同（extra="forbid"）——
     ``--inference-config-json`` 注入；全回退阶段可留空
  3. 推理仓要注入 sys.path —— ``--inference-repo`` 设 ``WAN_INFERENCE_REPO``

预检（``_preflight``）把「dataset 缺失 / 写成字典 / enable_dump:true」等错误在
本地拦下，不让它飘到 msmodelslim 跑一半才抛离根因很远的报错。
三条硬约束详见 ``references/dit/README.md``。
"""

import argparse
import json
import os
import subprocess
import sys

import yaml


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REFERENCES_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "references")

# spec.dataset 占位符，见 references/dit/fallback_config.yaml
_CALIB_PLACEHOLDER = "_CALIB_DATASET_"

_DIT_API = "multimodal_sd_modelslim_v1"

# msmodelslim/lab_calib/ 下的现成 DiT prompt 集
_KNOWN_CALIB_SHORTNAMES = {
    "hunyuanvideo",
    "wan2_2_t2v",
    "wan2_2_i2v",
    "wan2_2_ti2v",
}


def _template_path() -> str:
    path = os.path.join(_REFERENCES_DIR, "dit", "fallback_config.yaml")
    if not os.path.isfile(path):
        raise SystemExit(
            f"[ERROR] 找不到 DiT 全回退模板: {path}\n"
            f"        参考 references/dit/ 目录，或用 --config-path 指定现成配置。"
        )
    return path


def _assert_output_isolated(model_path, output_path):
    """DiT skip 模式下消费真实权重目录 —— output_path 一旦落在原目录里/上就
    会覆盖原始权重（曾打坏 Wan2.2-T2V-A14B 几十 GB 权重，详见 SKILL.md）。"""
    src = os.path.realpath(model_path)
    dst = os.path.realpath(output_path)

    if src == dst:
        raise SystemExit(
            f"[ERROR] --output-path 与 --model-path 指向同一目录: {src}\n"
            f"        量化产物会覆盖原始权重。请换一个独立的 --output-path。"
        )
    common = os.path.commonpath([src, dst])
    if common == src:
        raise SystemExit(
            f"[ERROR] --output-path 在 --model-path 内部:\n"
            f"          model-path  = {src}\n"
            f"          output-path = {dst}\n"
            f"        量化产物会写进原始权重目录。请换一个目录外的 --output-path。"
        )
    if common == dst:
        raise SystemExit(
            f"[ERROR] --output-path 是 --model-path 的父目录:\n"
            f"          model-path  = {src}\n"
            f"          output-path = {dst}\n"
            f"        写入可能波及原始权重。请换一个独立的 --output-path。"
        )


def _materialize_calib_dataset(output_path: str) -> str:
    """落一份自建校准集到 <output_path>/calib_data/，返回目录绝对路径。

    用目录形态（而非文件）是因为它走 IndexedDirectoryDatasetLoader，行为最可预期；
    文件名必须是 index.jsonl，叫别的会掉到 legacy 兼容分支。
    """
    example = os.path.join(_REFERENCES_DIR, "dit", "index.example.jsonl")
    if not os.path.isfile(example):
        raise SystemExit(f"[ERROR] 缺少校准集样例: {example}")

    calib_dir = os.path.abspath(os.path.join(output_path, "calib_data"))
    index_path = os.path.join(calib_dir, "index.jsonl")
    os.makedirs(calib_dir, exist_ok=True)
    if not os.path.exists(index_path):
        with open(example, "r", encoding="utf-8") as src:
            content = src.read()
        with open(index_path, "w", encoding="utf-8") as dst:
            dst.write(content)
        print(f"[INFO] 已生成自建校准集: {index_path}")
    else:
        print(f"[INFO] 复用已有校准集: {index_path}")
    return calib_dir


def _looks_like_path(value: str) -> bool:
    return os.sep in value or value.startswith(".") or value.endswith((".json", ".jsonl"))


def _check_calib_dataset_resolvable(value: str) -> None:
    """校验 spec.dataset 能被 VLMDatasetLoader 的责任链解析。三种形态见 references/dit/README.md。"""
    if not _looks_like_path(value):
        if value not in _KNOWN_CALIB_SHORTNAMES:
            print(
                f"[WARN] spec.dataset='{value}' 不是已知短名 "
                f"({', '.join(sorted(_KNOWN_CALIB_SHORTNAMES))})；"
                f"将由 msmodelslim 在 lab_calib/ 下解析。"
            )
        return

    if not os.path.exists(value):
        raise SystemExit(
            f"[ERROR] spec.dataset 指向的路径不存在: {value}（可用形态见 references/dit/README.md）"
        )

    if os.path.isdir(value):
        has_index = any(
            os.path.exists(os.path.join(value, name))
            for name in ("index.json", "index.jsonl")
        )
        if not has_index:
            raise SystemExit(
                f"[ERROR] spec.dataset 是目录但缺少 index.json / index.jsonl: {value} "
                f"（样例见 references/dit/index.example.jsonl）"
            )
        return

    if os.path.basename(value) not in ("index.json", "index.jsonl"):
        print(
            f"[WARN] spec.dataset 文件名不是 index.json / index.jsonl: {value}；"
            f"建议改为 index.jsonl 并传所在目录。"
        )


def _load_inference_config_override(raw: str) -> dict:
    """--inference-config-json 支持 JSON 串或 JSON 文件路径。"""
    if os.path.isfile(raw):
        with open(raw, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise SystemExit(
                f"[ERROR] --inference-config-json 既不是可读文件、也不是合法 JSON: {e}"
            )
    if not isinstance(data, dict):
        raise SystemExit(
            f"[ERROR] --inference-config-json 必须是 JSON 对象，得到 {type(data).__name__}"
        )
    return data


def _write_fallback_yaml(path, calib_dataset, inference_config):
    """DiT 模板含 _CALIB_DATASET_ 占位符；可选注入 inference_config。"""
    with open(_template_path(), "r", encoding="utf-8") as f:
        text = f.read()

    if not calib_dataset:
        raise SystemExit(
            f"[ERROR] DiT 模板含 {_CALIB_PLACEHOLDER} 占位符但未提供校准集。\n"
            f"        请传 --calib-dataset <短名|目录绝对路径|index.jsonl 路径>。"
        )
    text = text.replace(_CALIB_PLACEHOLDER, calib_dataset)

    if inference_config:
        # 需要重新序列化以合并进 multimodal_sd_config.inference_config —— 注释会丢
        data = yaml.safe_load(text)
        sd_config = data.setdefault("spec", {}).setdefault("multimodal_sd_config", {})
        merged = dict(sd_config.get("inference_config") or {})
        merged.update(inference_config)
        sd_config["inference_config"] = merged
        text = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _preflight(config_path: str) -> None:
    """DiT 专属预检。硬约束见 references/dit/README.md。"""
    with open(config_path, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise SystemExit(f"[ERROR] 配置不是合法 YAML ({config_path}): {e}")

    if not isinstance(data, dict):
        raise SystemExit(f"[ERROR] 配置顶层必须是 mapping: {config_path}")

    if data.get("apiversion") != _DIT_API:
        raise SystemExit(
            f"[ERROR] apiversion 不匹配 DiT：期望 '{_DIT_API}'，实际 '{data.get('apiversion')}'。"
        )

    spec = data.get("spec")
    if not isinstance(spec, dict):
        raise SystemExit(f"[ERROR] 配置缺少 spec 或 spec 不是 mapping: {config_path}")

    if "dataset" not in spec:
        raise SystemExit(
            "[ERROR] spec.dataset 缺失。请传 --calib-dataset 或在配置里显式写 "
            "spec.dataset（详见 references/dit/README.md 硬约束 #1）。"
        )
    dataset = spec["dataset"]
    if not isinstance(dataset, str):
        raise SystemExit(
            f"[ERROR] spec.dataset 必须是字符串（实际是 {type(dataset).__name__}）。"
            f"详见 references/dit/README.md 硬约束 #1。"
        )
    if not dataset.strip():
        raise SystemExit("[ERROR] spec.dataset 为空字符串。")
    _check_calib_dataset_resolvable(dataset)

    sd_config = spec.get("multimodal_sd_config") or {}
    if sd_config.get("inference_config") is not None and "model_config" in sd_config:
        raise SystemExit(
            "[ERROR] multimodal_sd_config 同时含 inference_config 与 model_config，两者互斥"
            "（msmodelslim 会报 SchemaValidateError: inference_config and model_config "
            "are mutually exclusive）。新 YAML 只保留 inference_config，删掉 model_config"
            "（详见 msmodelslim-model-adapt/references/dit/pitfalls.md §I.1）。"
        )
    dump_config = sd_config.get("dump_config") or {}
    if dump_config.get("enable_dump") is False:
        print(
            "[OK] dump_config.enable_dump: false —— DiT data-free 默认"
            "（短路 prepare_calib_data，不消费校准数据）。"
        )
    elif dump_config.get("enable_dump") is True:
        print(
            "[WARN] dump_config.enable_dump: true —— 会触发无谓浮点推理 dump"
            "（DiT data-free 不消费校准数据，应显式 false；"
            "详见 references/dit/README.md 硬约束 #2）。"
        )
    if not sd_config.get("inference_config"):
        print(
            "[WARN] multimodal_sd_config.inference_config 为空。"
            "step2 全回退可接受，step4 前必须按目标推理仓补齐"
            "（详见 references/dit/README.md 硬约束 #3）。"
        )

    print(f"[OK] 配置预检通过: {config_path}")


def _inject_inference_repo(env, inference_repo: str) -> None:
    """设 WAN_INFERENCE_REPO 并前插 PYTHONPATH。Wan 系必传，否则 ImportError。"""
    repo = os.path.abspath(inference_repo)
    if not os.path.isdir(repo):
        raise SystemExit(f"[ERROR] --inference-repo 不是目录: {repo}")
    env["WAN_INFERENCE_REPO"] = repo
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo}{os.pathsep}{existing}" if existing else repo
    print(f"[INFO] 推理仓已注入: WAN_INFERENCE_REPO={repo}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model-type", required=True)
    parser.add_argument(
        "--device",
        default="cpu",
        help="DiT 必传 'npu'；物理卡号通过 ASCEND_RT_VISIBLE_DEVICES 指定。"
        "详见 references/verification_guide.md。",
    )
    parser.add_argument("--config-path", default="")
    parser.add_argument(
        "--calib-dataset",
        default="",
        help="spec.dataset 值：lab_calib 短名 / 含 index.jsonl 的目录绝对路径 / "
        "index.jsonl 文件路径。缺省时自动落自建集到 <output-path>/calib_data/。"
        "三种形态见 references/dit/README.md。",
    )
    parser.add_argument(
        "--inference-config-json",
        default="",
        help="JSON 串或 JSON 文件路径，合并进 multimodal_sd_config.inference_config。"
        "字段名见 references/dit/README.md 速查表。",
    )
    parser.add_argument(
        "--inference-repo",
        default="",
        help="推理仓根目录。设 WAN_INFERENCE_REPO 并前插 PYTHONPATH；"
        "Wan 系必传。",
    )
    args = parser.parse_args()

    if args.device == "npu" and not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        print(
            "[WARN] --device npu 但未设置 ASCEND_RT_VISIBLE_DEVICES，"
            "CANN 将默认使用 0 号物理卡（不报错、静默执行）。"
            "如需指定其它卡，请先 export ASCEND_RT_VISIBLE_DEVICES=<物理卡号> 再重跑"
            "（详见 references/verification_guide.md 的设备与卡号选择小节）。",
            file=sys.stderr,
        )

    os.makedirs(args.output_path, exist_ok=True)
    _assert_output_isolated(args.model_path, args.output_path)

    config_path = args.config_path or os.path.join(args.output_path, "fallback_config.yaml")
    if not os.path.exists(config_path):
        calib_dataset = args.calib_dataset or _materialize_calib_dataset(args.output_path)
        inference_config = (
            _load_inference_config_override(args.inference_config_json)
            if args.inference_config_json
            else {}
        )
        _write_fallback_yaml(config_path, calib_dataset, inference_config)
        print(f"[INFO] 已生成配置: {config_path}")
    else:
        print(f"[INFO] 复用已有配置: {config_path}")

    _preflight(config_path)

    env = os.environ.copy()
    if args.inference_repo:
        _inject_inference_repo(env, args.inference_repo)

    cmd = [
        sys.executable,
        "-m",
        "msmodelslim.cli",
        "quant",
        "--model_path",
        args.model_path,
        "--save_path",
        args.output_path,
        "--device",
        args.device,
        "--model_type",
        args.model_type,
        "--config_path",
        config_path,
        "--trust_remote_code",
        "True",
    ]
    rc = subprocess.run(cmd, env=env, check=False).returncode
    if rc != 0:
        print("[ERROR] step2失败")
        return rc
    print(f"[OK] step2完成: {args.output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())