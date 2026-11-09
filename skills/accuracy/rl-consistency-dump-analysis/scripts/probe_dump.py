#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训推两侧 dump 的通用探查脚本(架构无关)。

对已采集的两侧 dump(training prefill vs inference prefill)快速摸清:
  1. 元数据(dump.json / construct.json / stack.json 关键字段)
  2. 两侧算子轨迹(construct.json 的 key,顺序即执行顺序)
  3. dump_tensor_data 文件清单(按算子聚合 input/output/weight)
  4. 两侧同名算子(可直接比对的候选边界)与采样 shape/dtype
输出终端摘要 + 可选 JSON。只读,不修改任何 dump 文件。

用法:
  python3 probe_dump.py \
      --train <train>/step0/rank0 \
      --infer <infer>/step0/rank0 \
      --sample 12 \
      --output probe_results.json

不绑定任何框架/模型:依赖的是 msprobe PrecisionDebugger(或等价)的通用
dump 布局 —— dump.json(元数据) + construct.json(算子轨迹) +
stack.json + dump_tensor_data/*.pt(文件名 = 算子路径 + forward 序号 +
input/output/parameters.weight)。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from typing import Any

import torch


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_pt_meta(path: str) -> tuple[list[int], str]:
    """读取 tensor 的 shape/dtype(注: torch.load 会完整反序列化整个文件, 大文件有 IO/内存开销,
    抽样建议调小 --sample; 若只需统计文件数, 用 collect_tensors 不用本函数)。"""
    t = torch.load(path, map_location="cpu", weights_only=True)
    return list(t.shape), str(t.dtype)


def parse_tensor_filename(name: str) -> tuple[str, str] | None:
    """解析 dump_tensor_data 文件名 → (算子路径, kind)。

    kind ∈ {input.<n>, output.<n>, parameters.weight}。算子的 forward 序号有两种形态:
      无序号:  Functional.dropout.0.forward.input.0.pt
                  → op="Functional.dropout.0.forward", kind="input.0"
      有序号:  ...VocabParallelEmbedding.forward.0.output.0.pt
                  → op="...VocabParallelEmbedding.forward.0", kind="output.0"
    规则:按 "." 切分后定位 "forward";紧随其后是纯数字则属于算子名(有序号),
    否则直接是 kind。
    """
    if not name.endswith(".pt"):
        return None
    stem = name[: -len(".pt")]
    parts = stem.split(".")
    try:
        fidx = parts.index("forward")
    except ValueError:
        return None
    if fidx + 1 < len(parts) and parts[fidx + 1].isdigit():
        op = ".".join(parts[: fidx + 2])
        kind = ".".join(parts[fidx + 2 :])
    else:
        op = ".".join(parts[: fidx + 1])
        kind = ".".join(parts[fidx + 1 :])
    if not kind:
        return None
    return op, kind


def collect_tensors(directory: str) -> dict[str, list[str]]:
    """扫描 dump_tensor_data,返回 {op: [kind, ...]}。"""
    per_op: dict[str, list[str]] = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(directory, "*.pt"))):
        name = os.path.basename(path)
        parsed = parse_tensor_filename(name)
        if parsed is None:
            per_op["__unparsed__"].append(name)
            continue
        op, kind = parsed
        per_op[op].append(kind)
    return per_op


def summarize_directory(directory: str, label: str) -> dict[str, Any]:
    out: dict[str, Any] = {"label": label, "directory": directory}
    meta_files = {
        "dump.json": None,
        "construct.json": None,
        "stack.json": None,
    }
    for name in meta_files:
        p = os.path.join(directory, name)
        if os.path.exists(p):
            try:
                meta_files[name] = load_json(p)
            except Exception as exc:  # noqa: BLE001
                meta_files[name] = {"__error__": str(exc)}
    dj = meta_files["dump.json"]
    if isinstance(dj, dict):
        out["task"] = dj.get("task")
        out["level"] = dj.get("level")
        out["framework"] = dj.get("framework")
        out["dump_data_dir"] = dj.get("dump_data_dir")
    cj = meta_files["construct.json"]
    if isinstance(cj, dict):
        out["construct_op_count"] = len(cj)
        out["construct_first_ops"] = list(cj.keys())[:15]
        out["construct_last_ops"] = list(cj.keys())[-5:]

    tensor_dir = os.path.join(directory, "dump_tensor_data")
    if not os.path.isdir(tensor_dir):
        out["tensor_error"] = "缺少 dump_tensor_data 子目录"
        return out
    per_op = collect_tensors(tensor_dir)
    kinds: Counter[str] = Counter()
    for op, kl in per_op.items():
        for k in kl:
            head = k.split(".")[0]  # input / output / parameters
            kinds[head] += 1
    out["tensor_file_count"] = sum(len(v) for v in per_op.values())
    out["tensor_op_count"] = len(per_op)
    out["tensor_kind_dist"] = dict(kinds)
    out["ops"] = sorted(per_op.keys())
    out["per_op_kinds"] = {op: sorted(kl) for op, kl in per_op.items()}
    return out


def sample_tensors(
    summary: dict[str, Any], n: int
) -> list[dict[str, Any]]:
    """抽样 shape/dtype。优先取两侧同名 op 的文件(探查可比性)。"""
    directory = summary["directory"]
    tensor_dir = os.path.join(directory, "dump_tensor_data")
    samples: list[dict[str, Any]] = []
    ops = summary.get("ops", [])
    for op in ops[: n * 2]:
        for kind in sorted(summary["per_op_kinds"].get(op, [])):
            if len(samples) >= n:
                break
            fname = f"{op}.{kind}.pt"
            fpath = os.path.join(tensor_dir, fname)
            if not os.path.exists(fpath):
                continue
            try:
                shape, dtype = load_pt_meta(fpath)
                samples.append(
                    {"op": op, "kind": kind, "shape": shape, "dtype": dtype}
                )
            except Exception as exc:  # noqa: BLE001
                samples.append({"op": op, "kind": kind, "error": str(exc)})
        if len(samples) >= n:
            break
    return samples


def op_name_prefix(op: str) -> str:
    """算子路径第一个 token,用于观察两侧命名规律。"""
    return op.split(".")[0]


def run(train_dir: str, infer_dir: str, sample: int, output: str) -> None:
    ts = summarize_directory(train_dir, "train")
    inf = summarize_directory(infer_dir, "infer")

    print("== 元数据 ==")
    for s in (ts, inf):
        print(f"  [{s['label']}] task={s.get('task')} level={s.get('level')} "
              f"framework={s.get('framework')}")
        print(f"       construct ops={s.get('construct_op_count')} "
              f"tensor files={s.get('tensor_file_count')} ops={s.get('tensor_op_count')} "
              f"kind={s.get('tensor_kind_dist')}")

    print("\n== 算子命名前缀规律 ==")
    for s in (ts, inf):
        c = Counter(op_name_prefix(op) for op in s.get("ops", []))
        print(f"  [{s['label']}] {dict(c.most_common(8))}")

    # 两侧同名算子(无需语义映射即可直接比对的候选边界)
    to = set(ts.get("ops", []))
    io = set(inf.get("ops", []))
    common = sorted(to & io)
    print(f"\n== 两侧同名算子 ==")
    print(f"  train={len(to)} infer={len(io)} common={len(common)}")
    for op in common[:30]:
        print(f"   {op}")
    if len(common) > 30:
        print(f"   ... 共 {len(common)} 个")

    # 融合/结构差异提示:仅一侧出现的算子中,名字含常见融合关键词的
    fused_keywords = ("fused", "merge", "moe", "grouped", "add_rms", "swiglu", "flash")
    train_only_fused = [
        op for op in (to - io)
        if any(k in op.lower() for k in fused_keywords)
    ][:15]
    infer_only_fused = [
        op for op in (io - to)
        if any(k in op.lower() for k in fused_keywords)
    ][:15]
    if train_only_fused or infer_only_fused:
        print("\n== 融合算子候选(仅一侧出现,按语义切片对齐的常见来源)==")
        if train_only_fused:
            print("  train-only:", " | ".join(train_only_fused))
        if infer_only_fused:
            print("  infer-only:", " | ".join(infer_only_fused))

    print("\n== 采样 tensor shape/dtype ==")
    ts_samples = sample_tensors(ts, sample)
    inf_samples = sample_tensors(inf, sample)
    # 并排展示同名 op(若有)
    tin = {f"{x['op']}::{x['kind']}": x for x in ts_samples}
    iin = {f"{x['op']}::{x['kind']}": x for x in inf_samples}
    shown = 0
    for key in sorted(set(tin) & set(iin)):
        a, b = tin[key], iin[key]
        sa = a.get("shape", ["?"]); sb = b.get("shape", ["?"])
        da = a.get("dtype", "?"); db = b.get("dtype", "?")
        mark = " 同shape" if sa == sb else "  ← shape 不同"
        print(f"  [{key}] train={sa}/{da} vs infer={sb}/{db}{mark}")
        shown += 1
        if shown >= sample:
            break
    if shown == 0:
        for s, samples in (("train", ts_samples), ("infer", inf_samples)):
            print(f"  [{s}] 前 {min(sample, len(samples))} 个:")
            for x in samples[: sample]:
                print(f"    {x.get('op')}::{x.get('kind')} "
                      f"{x.get('shape', x.get('error'))} / {x.get('dtype', '')}")

    result = {
        "train": ts,
        "infer": inf,
        "common_ops": common,
        "sample_train": ts_samples,
        "sample_infer": inf_samples,
    }
    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n结果写至 {output}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True, help="训练侧 step0/rank0 dump 目录")
    ap.add_argument("--infer", required=True, help="推理侧 step0/rank0 dump 目录")
    ap.add_argument("--sample", type=int, default=12, help="抽样 tensor 数")
    ap.add_argument("--output", help="结果 JSON 输出路径")
    args = ap.parse_args()
    run(args.train, args.infer, args.sample, args.output)


if __name__ == "__main__":
    main()
