#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""参数对齐校验(阶段 5)静态 config 比对脚本 —— 通用版, 不绑定任何具体模型。

读训练侧实际生效 config 与推理侧 hf_config, 逐项比对输出 param_alignment_report.json。
两种模式:
  1. 传 --mapping <字段映射 json>: 按映射逐项语义比对(两侧字段名可不同, 支持嵌套点分路径
     与结构断言 infer_expected)。映射写法见 param_alignment_mapping.example.json。
  2. 不传 --mapping: 自动同名键比对(零配置兜底)——枚举两侧顶层键并集, 同名键直接比,
     单侧缺失标 MISSING_SIDE, 嵌套 dict 值标 SKIP。适用快速筛查与建映射前的摸底。

纯 Python, 无 torch 依赖, 可在任意容器/宿主跑(权重/结构比对是动态层, 见 param-alignment.md §5.3)。

用法:
  python3 param_align_check.py \
    --train-cfg <训练侧实际生效 config.json(provider/bridge 转换结果)> \
    --infer-cfg <推理侧 config.json> \
    [--mapping <字段映射 json>] [--model <模型标识>] [--out <报告路径>]

注意:训练侧 config 必须是"实际转换结果", 不能只读原始 config.json——转换字段名写错时
原始文件有值但训练侧没接住, 只比原始文件会漏。
"""
import argparse
import json
import sys


def load_cfg(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get(cfg, field):
    """取字段值; 支持嵌套点分路径。字段不存在返回 (None, True)。field 为 None 表示语义上没有该字段。"""
    if field is None:
        return None, False
    cur = cfg
    for part in field.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None, True
    return cur, False


def compare(a, b):
    """比较两侧值: 标量/字符串/等长 list 逐元素。返回 (equal, detail)。
    不可比(嵌套 dict 等)返回 (None, 原因), 由调用方标 SKIP。
    bool 与数值/字符串类型不一致时显式判不等——True==1、1==1.0 在 Python 中为真,
    但两侧 config 分属不同框架时 bool/int 表述差异应暴露为 DIFF, 不能静默算 EXACT。"""
    if a is None or b is None:
        return a is b, f"{a} vs {b}"
    if isinstance(a, bool) or isinstance(b, bool):
        return (isinstance(a, bool) and isinstance(b, bool) and a == b), f"{a!r} vs {b!r}"
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b, f"{a} vs {b}"
    if isinstance(a, str) and isinstance(b, str):
        return a == b, f"{a} vs {b}"
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False, f"list len {len(a)} vs {len(b)}"
        diffs = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
        return not diffs, f"list len={len(a)}, first_diff={diffs[:3]}"
    if isinstance(a, dict) or isinstance(b, dict):
        return None, "嵌套 dict 值, 非标量不自动比(映射模式可指定点分路径深比)"
    return a == b, f"{a} vs {b}"


def compare_item(train_cfg, infer_cfg, semantic, tfield, ifield, infer_expected=None):
    """单项比对。返回 item dict。"""
    tv, tv_missing = get(train_cfg, tfield)
    iv, iv_missing = get(infer_cfg, ifield)
    if tfield is None and ifield is None:
        status, detail = "N/A", "两侧均无对应字段"
    elif infer_expected is not None:
        # 结构断言: 一侧无独立字段(该侧恒启用/恒为某值), 用期望值校验另一侧
        equal = (tv == infer_expected) and not tv_missing
        status = "EXACT" if equal else "DIFF"
        detail = f"train={tv} expected={infer_expected}"
    elif tfield is None or ifield is None:
        # 一侧在语义上没有对应字段: 标 MISSING_SIDE(不是值不一致), 查该侧代码是否硬编码同值
        side = "train" if tfield is None else "infer"
        status = "MISSING_SIDE"
        detail = f"{side} 侧无对应字段; 查该侧代码是否硬编码同值"
    elif tv_missing or iv_missing:
        status = "MISSING_SIDE"
        detail = f"train_missing={tv_missing} infer_missing={iv_missing}"
    else:
        equal, detail = compare(tv, iv)
        if equal is None:
            status = "SKIP"
        else:
            status = "EXACT" if equal else "DIFF"
    return {
        "semantic": semantic,
        "train": {"field": tfield, "value": tv, "missing": tv_missing},
        "infer": {"field": ifield, "value": iv, "missing": iv_missing},
        "status": status,
        "detail": detail,
    }


def build_mapping_items(train_cfg, infer_cfg, mapping):
    """按映射生成比对条目(带 note)。"""
    for it in mapping.get("items", []):
        item = compare_item(
            train_cfg, infer_cfg,
            it["semantic"], it.get("train"), it.get("infer"), it.get("infer_expected"),
        )
        item["note"] = it.get("note", "")
        yield item


def build_auto_items(train_cfg, infer_cfg):
    """无映射时自动同名键比对: 枚举两侧顶层键并集。"""
    keys = list(dict.fromkeys([*train_cfg.keys(), *infer_cfg.keys()]))
    items = []
    skipped = []
    for k in keys:
        item = compare_item(
            train_cfg, infer_cfg,
            k, k if k in train_cfg else None, k if k in infer_cfg else None, None,
        )
        if item["status"] == "SKIP":
            skipped.append(k)
        items.append(item)
    return items, skipped


def run(train_cfg, infer_cfg, mapping):
    if mapping is not None:
        items = list(build_mapping_items(train_cfg, infer_cfg, mapping))
        prefix = "P"
        # 映射模式也要收集 SKIP(嵌套 dict 值)/N/A(两侧均无字段)条目,否则这些未比项被静默吞掉,
        # 汇总会误报"全部 EXACT"(假成功)。
        skipped = [it["semantic"] for it in items if it["status"] in ("SKIP", "N/A")]
        mode = "mapping"
    else:
        items, skipped = build_auto_items(train_cfg, infer_cfg)
        prefix = "A"
        mode = "auto"
    for i, it in enumerate(items):
        it["id"] = f"{prefix}{i + 1}"
    mismatch = [it["id"] for it in items if it["status"] in ("DIFF", "MISSING_SIDE")]
    return {"mode": mode, "items": items, "skipped": skipped, "mismatch_points": mismatch}


def main():
    ap = argparse.ArgumentParser(description="参数对齐校验(阶段 5)静态 config 比对 —— 通用版")
    ap.add_argument("--train-cfg", required=True, help="训练侧实际生效 config.json(provider/bridge 转换结果)")
    ap.add_argument("--infer-cfg", required=True, help="推理侧 config.json")
    ap.add_argument("--mapping", default=None,
                    help="字段映射 json(写法见 param_alignment_mapping.example.json); 不传则自动同名键比对")
    ap.add_argument("--model", default="unknown", help="模型标识(写入报告, 映射文件有 model 则优先)")
    ap.add_argument("--out", default=None, help="输出 param_alignment_report.json 路径(默认 stdout)")
    args = ap.parse_args()

    train_cfg = load_cfg(args.train_cfg)
    infer_cfg = load_cfg(args.infer_cfg)
    mapping = load_cfg(args.mapping) if args.mapping else None

    res = run(train_cfg, infer_cfg, mapping)
    model = (mapping.get("model") or args.model) if mapping is not None else args.model

    mismatches = res["mismatch_points"]
    skipped = res["skipped"]
    if mismatches:
        summary = f"参数对齐: {len(mismatches)} 个不一致点 {mismatches}, 需对齐后采"
    elif skipped:
        # 有 SKIP/N-A 未比项时不能宣称"全部 EXACT",仅值一致但未全覆盖,属待确认而非可采。
        summary = (f"参数对齐: 无值不一致, 但 {len(skipped)} 个条目未比"
                   f"(嵌套 dict 值或两侧均无字段, 见 skipped_keys), 非全部 EXACT, 建议补映射确认后再采")
    else:
        summary = "参数对齐: 全部 EXACT, 可采"
    if skipped and mismatches:
        summary += f"; 另有 {len(skipped)} 个条目未比(见 skipped_keys, 可补映射深比)"

    report = {
        "model": model,
        "mode": res["mode"],
        "checked_at": "<现场填写>",
        "items": res["items"],
        "skipped_keys": res["skipped"],
        "mismatch_points": res["mismatch_points"],
        "weights": {"status": "N/A(动态层, 见 param-alignment.md §5.3)"},
        "structure": {"status": "N/A(动态层, 见 param-alignment.md §5.3)"},
        "summary": summary,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"written: {args.out}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    print(summary)
    sys.exit(0 if not res["mismatch_points"] else 1)


if __name__ == "__main__":
    main()
