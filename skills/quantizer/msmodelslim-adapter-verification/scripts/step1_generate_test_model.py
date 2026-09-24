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
"""步骤1：生成随机权重测试模型。LLM/VLM 默认基于 transformers AutoModel 构造；
DiT 无统一随机权重工厂，须走 --skip-random-model 零拷贝路径。
"""

import argparse
import json
import os
import shutil
import sys

import torch
from transformers import AutoConfig
import transformers


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _copy_non_weight_files(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for name in os.listdir(src_dir):
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.isdir(src):
            continue
        if name.endswith(".safetensors"):
            continue
        if name.endswith(".index.json"):
            continue
        shutil.copy2(src, dst)


def _validate_json_file(path, label):
    """Fail-loudly if path is missing or not valid JSON. Returns parsed dict."""
    if not os.path.isfile(path):
        raise SystemExit(f"[ERROR] {label} 缺少配置文件: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise SystemExit(f"[ERROR] {label} config.json 解析失败 ({path}): {e}")


def _run_skip_mode(args):
    """仅校验 model_path 可读（dir + config.json），不写盘。step2 直接消费 model_path；step3 用 --reference-weights 指回同一路径。"""
    if not os.path.isdir(args.model_path):
        raise SystemExit(
            f"[ERROR] --model-path 不是目录: {args.model_path}"
        )

    if args.model_family == "dit":
        expert_subdirs = _detect_expert_subdirs(args.model_path)
        if not expert_subdirs:
            cfg_path = os.path.join(args.model_path, "config.json")
            _validate_json_file(cfg_path, label="DiT 顶层")
            print(f"[INFO] DiT 单 expert 模型: {args.model_path}")
        else:
            for sub_name, sub_cfg in expert_subdirs:
                _validate_json_file(sub_cfg, label=f"DiT expert '{sub_name}'")
            print(
                f"[INFO] DiT MoE 模型，共 {len(expert_subdirs)} 个 expert "
                f"子目录: {args.model_path}"
            )
    else:
        cfg_path = os.path.join(args.model_path, "config.json")
        _validate_json_file(cfg_path, label=f"{args.model_family} 顶层")
        print(f"[INFO] {args.model_family} 模型: {args.model_path}")

    print(
        f"[OK] step1完成 ({args.model_family}, skip模式，未复制权重): "
        f"model_path 将被 step2 直接消费: {args.model_path}"
    )
    return 0


def _shrink_config(cfg, num_layers):
    """Shrink a transformers-style config to `num_layers` hidden layers.

    Handles text-in-text-out (`text_config.num_hidden_layers`) and the flat
    `num_hidden_layers` form used by some HF causal LMs.
    """
    out = dict(cfg)
    if "text_config" in out:
        text_cfg = dict(out["text_config"])
        text_cfg["num_hidden_layers"] = num_layers
        if isinstance(text_cfg.get("layer_types"), list):
            text_cfg["layer_types"] = text_cfg["layer_types"][:num_layers]
        out["text_config"] = text_cfg
    else:
        out["num_hidden_layers"] = num_layers
        if isinstance(out.get("layer_types"), list):
            out["layer_types"] = out["layer_types"][:num_layers]
    return out


def _detect_expert_subdirs(model_path):
    """返回 (子目录名, config.json 路径) 列表，仅匹配 `*_model` 子目录；空列表表示顶层布局。"""
    subdirs = []
    for name in sorted(os.listdir(model_path)):
        sub = os.path.join(model_path, name)
        if not os.path.isdir(sub):
            continue
        cfg = os.path.join(sub, "config.json")
        if os.path.exists(cfg) and name.endswith("_model"):
            subdirs.append((name, cfg))
    return subdirs


def _build_random_model_from_config(config):
    candidate_auto_model_names = [
        "AutoModelForCausalLM",
        "AutoModelForImageTextToText",
        "AutoModel",
    ]
    errors = []
    for cls_name in candidate_auto_model_names:
        auto_cls = getattr(transformers, cls_name, None)
        if auto_cls is None:
            continue
        try:
            model = auto_cls.from_config(
                config, trust_remote_code=True, torch_dtype=torch.float32
            )
            return model, cls_name
        except Exception as e:  # pragma: no cover - best-effort fallback chain
            errors.append(f"{cls_name}: {repr(e)}")

    raise RuntimeError(
        "无法根据配置构建模型。已尝试: "
        + ", ".join(candidate_auto_model_names)
        + "\n错误详情:\n"
        + "\n".join(errors)
    )


def _assert_output_isolated(model_path, output_path, what):
    """拒绝任何会写进原始权重目录的 --output-path（save_pretrained 会删旧分片，曾打坏 Wan2.2 高 GB 权重，详见 SKILL.md「权重安全」）。"""
    src = os.path.realpath(model_path)
    dst = os.path.realpath(output_path)

    if src == dst:
        raise SystemExit(
            f"[ERROR] --output-path 与 --model-path 指向同一目录: {src}\n"
            f"        {what}会覆盖原始权重（save_pretrained 会删掉旧分片）。\n"
            f"        请换一个独立的 --output-path；若想直接用真实权重，"
            f"加 --skip-random-model。"
        )

    if os.path.commonpath([src, dst]) == src:
        raise SystemExit(
            f"[ERROR] --output-path 在 --model-path 内部:\n"
            f"          model-path  = {src}\n"
            f"          output-path = {dst}\n"
            f"        {what}会往原始权重目录里写。请换一个目录外的 --output-path。"
        )

    if os.path.commonpath([src, dst]) == dst:
        raise SystemExit(
            f"[ERROR] --output-path 是 --model-path 的父目录:\n"
            f"          model-path  = {src}\n"
            f"          output-path = {dst}\n"
            f"        写入可能波及原始权重。请换一个独立的 --output-path。"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument(
        "--device",
        default="cpu",
        help="LLM/VLM 默认 cpu 即可。NPU 卡号通过 ASCEND_RT_VISIBLE_DEVICES 指定"
        "（详见 references/verification_guide.md）。",
    )
    parser.add_argument(
        "--model-family",
        default="llm",
        choices=["llm", "vlm", "dit"],
        help="llm/vlm 走 transformers AutoModel；dit 仅用于 --skip-random-model 路径。",
    )
    parser.add_argument(
        "--skip-random-model",
        action="store_true",
        help="跳过随机权重构造，仅校验 --model_path，不写盘。step2 直接消费 "
        "--model_path；step3 用 --reference-weights 指回同一路径。DiT 必传。",
    )
    args = parser.parse_args()

    # -------- Skip-random-model short-circuit (uses real weights) --------
    if args.skip_random_model:
        return _run_skip_mode(args)

    # 随机权重路径会写盘 —— 先确保写不到原始权重目录里
    _assert_output_isolated(args.model_path, args.output_path, "随机权重生成")

    # -------- LLM / VLM (transformers AutoModel) path --------
    src_cfg = os.path.join(args.model_path, "config.json")
    if not os.path.exists(src_cfg):
        print(f"[ERROR] 缺少配置文件: {src_cfg}")
        return 1

    _copy_non_weight_files(args.model_path, args.output_path)
    cfg = _read_json(src_cfg)
    _write_json(
        os.path.join(args.output_path, "config.json"),
        _shrink_config(cfg, args.num_layers),
    )

    config = AutoConfig.from_pretrained(args.output_path, trust_remote_code=True)
    model, used_cls_name = _build_random_model_from_config(config)
    print(f"[INFO] 使用模型类: {used_cls_name}")
    model = model.to(args.device).eval()
    model.save_pretrained(args.output_path)
    stale_index = os.path.join(args.output_path, "model.safetensors.index.json")
    if os.path.exists(stale_index) and os.path.exists(
        os.path.join(args.output_path, "model.safetensors")
    ):
        os.remove(stale_index)
    print(f"[OK] step1完成: {args.output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
