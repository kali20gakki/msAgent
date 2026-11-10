#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dump_analyzer.py — msprobe PyTorch statistics dump 的确定性解析、对齐与差异分析。

职责（确定性工作，LLM 不手工重复）：
  1. 发现两侧 dump 目录中的 dump.json / stack.json / construct.json（递归，不依赖固定布局）
  2. 解析并标准化节点、槽位、统计量（含嵌套结构、标量参数、DTensor 附加字段）
  3. 节点对齐（精确名匹配）+ 匹配质量评估（族计数漂移、名称序列公共前缀、反向形状自洽）
  4. 统计差异计算（Max/Min/Mean/Norm 有符号差、相对误差、NRE、dtype/shape/标量值/槽位缺失等结构分歧、噪声地板）
  5. 关键证据整理：首分歧锚点（按 dump 顺序）、参数梯度差异与 gnorm 估计、差异 Top 表、未匹配节点分组、多 rank 汇总

不做的事：
  - 不做根因判定。输出是供 LLM 做因果分析的证据材料。
  - 统计差异排名只是线索，不等于根因排序。

用法：
  python3 dump_analyzer.py --target <target侧目录> --golden <golden侧目录> \
      [--out <工作目录>] [--rank <rank>] [--step <step>] [--top N] \
      [--json <file>] [--report <file>]

输入目录递归查找 dump.json；按 step/rank（或 proc）分组配对。多 rank 时逐 rank 分析并汇总。
所有输出写入 --out 指定的工作目录（默认 ./dump_analysis_out），不修改原始输入。
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

TENSOR_TYPE_MARKERS = ("tensor",)  # dtype 字段出现即按张量处理（torch.Tensor / DTensor / FakeTensor ...）

# 各 dtype 的 L2 统计量噪声地板（低于该量级视为数值噪声，NRE 分母伪影）
DTYPE_FLOOR = {
    "torch.float64": 1e-15, "torch.double": 1e-15,
    "torch.float32": 1.2e-7, "torch.float": 1.2e-7,
    "torch.float16": 1e-3, "torch.half": 1e-3,
    "torch.bfloat16": 1e-2,
}
DEFAULT_FLOOR = 1e-7
STAT_KEYS = ("Max", "Min", "Mean", "Norm")


# ---------------------------------------------------------------- 基础工具

def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def finite(v):
    return is_num(v) and math.isfinite(v)


def fmt(v):
    """数值格式化：保持精度，NaN/Inf 可读。"""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return str(v)
    if is_num(v):
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "Inf" if v > 0 else "-Inf"
        return repr(v)
    return str(v)


def shape_str(s):
    return "x".join(str(d) for d in s) if isinstance(s, list) else str(s)


def dtype_floor(dtype):
    return DTYPE_FLOOR.get(dtype, DEFAULT_FLOOR)


def parse_num(v):
    """宽容解析统计值（可能是字符串形式的 nan/inf/数字）。"""
    if is_num(v):
        return float(v)
    if isinstance(v, str):
        s = v.strip().rstrip("%")
        try:
            return float(s)
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------- 节点名解析

def parse_node_name(name):
    """
    解析 dump 节点名。返回 dict(kind, family, idx, dir)。
      API:   <NS>.<op>.<idx>.<forward|backward>       例 Tensor.permute.0.forward
      Module:<path>.<forward|backward>.<idx>          例 Module.time_embedding.0.Linear.forward.0
      ParamGrad: Module.<path>.parameters_grad.<idx>  例 Module.proj_out.Linear.parameters_grad.0
    不符合已知模式的归为 other，不猜测。
    """
    toks = name.split(".")
    if name.startswith("Module."):
        if len(toks) >= 3 and toks[-2] == "parameters_grad":
            return {"kind": "paramgrad", "family": ".".join(toks[:-1]), "idx": int(toks[-1]) if toks[-1].isdigit() else -1, "dir": "forward"}
        if len(toks) >= 3 and toks[-2] in ("forward", "backward") and toks[-1].isdigit():
            return {"kind": "module", "family": ".".join(toks[:-2]) + "." + toks[-2], "idx": int(toks[-1]), "dir": toks[-2]}
        return {"kind": "module", "family": name, "idx": -1, "dir": None}
    if len(toks) >= 3 and toks[-1] in ("forward", "backward") and toks[-2].isdigit():
        return {"kind": "api", "family": ".".join(toks[:-2]), "idx": int(toks[-2]), "dir": toks[-1]}
    return {"kind": "other", "family": name, "idx": -1, "dir": None}


# ---------------------------------------------------------------- 槽位展开

def expand_slot_value(v, path, role, out):
    """递归展开槽位元素。叶子：张量 dict / 标量 dict / None / 其他。"""
    if v is None:
        out.append({"role": role, "path": path, "leaf": "none"})
        return
    if isinstance(v, list):
        for i, e in enumerate(v):
            expand_slot_value(e, "%s.%d" % (path, i), role, out)
        return
    if isinstance(v, dict):
        vt = v.get("type")
        if isinstance(vt, str) and "tensor" in vt.lower():
            rec = {"role": role, "path": path, "leaf": "tensor",
                   "dtype": v.get("dtype"), "shape": v.get("shape")}
            for k in STAT_KEYS:
                rec[k] = parse_num(v.get(k)) if v.get(k) is not None else None
            rec["requires_grad"] = v.get("requires_grad")
            extras = {k: val for k, val in v.items()
                      if k not in ("type", "dtype", "shape", "Max", "Min", "Mean", "Norm", "requires_grad")}
            if extras:
                rec["extras"] = extras
            out.append(rec)
        elif "value" in v:
            out.append({"role": role, "path": path, "leaf": "scalar",
                        "stype": v.get("type"), "value": v.get("value")})
        else:
            # 未知 dict（可能是嵌套 kwargs 等）：按键继续展开
            for k, val in v.items():
                expand_slot_value(val, "%s.%s" % (path, k), role, out)
        return
    out.append({"role": role, "path": path, "leaf": "scalar", "stype": type(v).__name__, "value": v})


def expand_node_slots(node_data):
    """把节点数据展开为槽位列表。兼容 forward / backward / parameters_grad 布局。"""
    slots = []
    for role, val in node_data.items():
        if isinstance(val, dict):
            for k, e in val.items():
                expand_slot_value(e, "%s.%s" % (role, k), role, slots)
        elif isinstance(val, list):
            for i, e in enumerate(val):
                expand_slot_value(e, "%s.%d" % (role, i), role, slots)
        else:
            expand_slot_value(val, role, role, slots)
    return slots


# ---------------------------------------------------------------- 数据发现与解析

STEP_RE = re.compile(r"step(\d+)", re.IGNORECASE)
RANK_RE = re.compile(r"(?:rank|proc|device)(\d+)", re.IGNORECASE)


def discover_dump_root(root):
    """在目录下递归发现 dump.json，按 (step, rank) 分组。"""
    found = defaultdict(dict)
    for dirpath, _dirnames, filenames in os.walk(root):
        if "dump.json" in filenames:
            rel = os.path.relpath(dirpath, root)
            m_s = STEP_RE.search(rel)
            m_r = RANK_RE.search(rel)
            step = int(m_s.group(1)) if m_s else 0
            rank = int(m_r.group(1)) if m_r else 0
            entry = {"dir": dirpath,
                     "dump": os.path.join(dirpath, "dump.json"),
                     "stack": os.path.join(dirpath, "stack.json") if "stack.json" in filenames else None,
                     "construct": os.path.join(dirpath, "construct.json") if "construct.json" in filenames else None}
            if (step, rank) in found:
                print("[warn] 同 (step,rank) 发现多个 dump，保留第一个: %s" % (found[(step, rank)]["dir"]), file=sys.stderr)
            else:
                found[(step, rank)] = entry
    return dict(found)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_side_dump(path):
    """解析 dump.json → (meta, {name: node_record})，保持 dump 顺序。"""
    raw = load_json(path)
    meta = {k: raw.get(k) for k in ("task", "level", "framework", "dump_data_dir")}
    data = raw.get("data") or {}
    nodes = {}
    for i, (name, nd) in enumerate(data.items()):
        info = parse_node_name(name)
        nd = nd if isinstance(nd, dict) else {}
        nodes[name] = {
            "name": name, "order": i, "kind": info["kind"], "family": info["family"],
            "idx": info["idx"], "dir": info["dir"], "slots": expand_node_slots(nd),
        }
    return meta, nodes


def load_stack_map(path):
    """stack.json: {idx: [[names...], [frames...]]} → name → frames"""
    m = {}
    if not path:
        return m
    try:
        raw = load_json(path)
    except Exception as e:
        print("[warn] stack.json 解析失败 %s: %s" % (path, e), file=sys.stderr)
        return m
    for _, v in raw.items():
        try:
            names, frames = v[0], v[1]
        except (IndexError, TypeError):
            continue
        for n in names or []:
            m.setdefault(n, frames or [])
    return m


def load_construct_map(path):
    """construct.json: {node_name: label|null} → 仅保留非空映射"""
    m = {}
    if not path:
        return m
    try:
        raw = load_json(path)
    except Exception as e:
        print("[warn] construct.json 解析失败 %s: %s" % (path, e), file=sys.stderr)
        return m
    for k, v in raw.items():
        if v is not None:
            m[k] = v
    return m


def user_frames(frames, limit=3):
    """优先返回非 site-packages 的调用帧（文件:行 与 代码行）。
    保留帧内的代码行文本——更新类循环的语义消歧（EMA vs Adam exp_avg 等同构
    公式）依据正是调用文本中的操作数名（如 src_param / grad），栈证据必须带代码行。"""
    out = []
    for fr in frames or []:
        line = str(fr).strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("\n") if p.strip()]
        if "site-packages" in parts[0] or "conda" in parts[0]:
            continue
        out.append(parts[0])
        if len(out) >= limit:
            break
        if len(parts) > 1:  # 帧的代码行（调用文本）
            out.append(parts[1])
            if len(out) >= limit:
                break
    return out


# ---------------------------------------------------------------- 差异计算

def tensor_pair_diff(t, g):
    """张量槽位对差异。t/g 为槽位记录。"""
    d = {"path": t["path"]}
    d["dtype_t"], d["dtype_g"] = t.get("dtype"), g.get("dtype")
    d["shape_t"], d["shape_g"] = t.get("shape"), g.get("shape")
    d["dtype_diff"] = d["dtype_t"] != d["dtype_g"]
    d["shape_diff"] = d["shape_t"] != d["shape_g"]
    nt, ng = t.get("Norm"), g.get("Norm")
    d["stats"] = {}
    for k in STAT_KEYS:
        tv, gv = t.get(k), g.get(k)
        if tv is None and gv is None:
            continue
        item = {"t": tv, "g": gv}
        if finite(tv) and finite(gv):
            item["diff"] = tv - gv
            item["rel"] = abs(tv - gv) / abs(gv) if abs(gv) > 0 else (0.0 if abs(tv) == 0 else None)
        d["stats"][k] = item
    nre = None
    if finite(nt) and finite(ng):
        nre = abs(nt - ng) / abs(ng) if abs(ng) > 0 else (0.0 if nt == 0 else None)
    d["nre"] = nre
    # NaN / Inf 参与情况
    d["nonfinite"] = any((not finite(x)) and x is not None
                         for x in [t.get(k) for k in STAT_KEYS] + [g.get(k) for k in STAT_KEYS])
    # 噪声判定：两侧 Norm 都低于该 dtype 地板
    fl_t = dtype_floor(d["dtype_t"] or "")
    fl_g = dtype_floor(d["dtype_g"] or "")
    floor = max(fl_t, fl_g)
    d["floor"] = floor
    d["noise"] = (finite(nt) and finite(ng) and abs(nt) < floor and abs(ng) < floor)
    d["one_side_zero"] = (finite(nt) and finite(ng) and
                          ((abs(nt) < floor) != (abs(ng) < floor)))
    d["req_grad_diff"] = t.get("requires_grad") != g.get("requires_grad")
    # DTensor 附加结构字段差异（切分方式不同 → 统计量不可比提示）
    extras_t, extras_g = t.get("extras") or {}, g.get("extras") or {}
    ex_keys = set(extras_t) | set(extras_g)
    ex_diff = {}
    for k in ex_keys:
        if k in ("requires_grad",):
            continue
        if extras_t.get(k) != extras_g.get(k):
            ex_diff[k] = [extras_t.get(k), extras_g.get(k)]
    d["extras_diff"] = ex_diff
    return d


def classify_pair_diff(d):
    """差异分级：structural / scalar / significant / minor / none。"""
    if d.get("dtype_diff") or d.get("shape_diff") or d.get("slot_missing") or d.get("extras_diff") or d.get("req_grad_diff"):
        return "structural"
    if d.get("nonfinite"):
        return "significant"
    if d.get("one_side_zero"):
        return "significant"
    nre = d.get("nre")
    if nre is None:
        # 无 Norm 可比时，看其他统计量的绝对差是否显著
        for k in ("Max", "Min", "Mean"):
            it = d["stats"].get(k)
            if it and it.get("diff") is not None and it["diff"] != 0 and it.get("rel") not in (None,):
                return "significant" if (it.get("rel") or 0) >= 1e-3 else "minor"
        return "none"
    if d.get("noise"):
        return "noise" if nre == 0 or nre < 1e-2 else "minor"
    return "significant" if nre >= DIRTY_NRE else ("minor" if nre >= MINOR_NRE else "none")


DIRTY_NRE = 1e-3
MINOR_NRE = 1e-6


RANK_SUFFIX_RE = re.compile(r"_rank\d+$")


def norm_family(family):
    """归一化族名：剥离 _rank<N> 后缀（分片变体命名），用于跨命名变体匹配。"""
    return RANK_SUFFIX_RE.sub("", family)


# ---------------------------------------------------------------- 对齐与配对

def match_nodes(t_nodes, g_nodes):
    t_names, g_names = set(t_nodes), set(g_nodes)
    common = [n for n in t_nodes if n in g_names]  # 保持 target 侧（dump）顺序
    t_only = [n for n in t_nodes if n not in g_names]
    g_only = [n for n in g_nodes if n not in t_names]
    return common, t_only, g_only


def slot_map(node, tensor_only=False, non_output=False):
    """(role, path) → 槽位记录：两侧 path 展开规则一致，此键即可跨侧对齐槽位。
    tensor_only 只保留张量槽位；non_output 排除输出槽位（输入侧聚合检查用）。"""
    out = {}
    for s in node["slots"]:
        if tensor_only and s["leaf"] != "tensor":
            continue
        if non_output and s["role"] == "output":
            continue
        out[(s["role"], s["path"])] = s
    return out


def slot_shapes(node):
    """该节点全部张量槽位的 shape 集合（用于命名变体匹配的形状门槛）。"""
    return set(shape_str(s["shape"]) for s in node["slots"] if s["leaf"] == "tensor" and s.get("shape"))


def match_name_variants(t_only, g_only, t_nodes, g_nodes):
    """
    第二阶段匹配：族名归一化（剥离 _rank<N>）后按 (族, 调用号, 方向) 匹配，
    且要求两侧至少有一个相同 shape 的张量槽位（防分片/不可比配对）。
    每个键在两侧都必须唯一（1:1），否则放弃。返回 {t_name: g_name}。
    """
    def key_map(names, nodes):
        km = defaultdict(list)
        for n in names:
            nd = nodes[n]
            km[(norm_family(nd["family"]), nd["idx"], nd["dir"])].append(n)
        return km

    kt, kg = key_map(t_only, t_nodes), key_map(g_only, g_nodes)
    pairs = {}
    for k, tns in kt.items():
        gns = kg.get(k)
        if not gns or len(tns) != 1 or len(gns) != 1:
            continue
        t_name, g_name = tns[0], gns[0]
        if not (slot_shapes(t_nodes[t_name]) & slot_shapes(g_nodes[g_name])):
            continue
        pairs[t_name] = g_name
    return pairs


def name_seq_common_prefix(seq_t, seq_g):
    n = 0
    for a, b in zip(seq_t, seq_g):
        if a != b:
            break
        n += 1
    return n


def family_seq_stats(t_nodes, g_nodes):
    """执行序（dump 顺序）的归一化 family 序列对比。"""
    seq_t = [norm_family(nd["family"]) for nd in t_nodes.values()]
    seq_g = [norm_family(nd["family"]) for nd in g_nodes.values()]
    pre = name_seq_common_prefix(seq_t, seq_g)
    suf = name_seq_common_prefix(seq_t[::-1], seq_g[::-1])
    return {"len_t": len(seq_t), "len_g": len(seq_g),
            "common_prefix": pre, "common_suffix": suf,
            "ctx_t": seq_t[max(0, pre - 6):pre + 6], "ctx_g": seq_g[max(0, pre - 6):pre + 6],
            "diverge_at": pre if pre < min(len(seq_t), len(seq_g)) else None}


def family_count_table(t_nodes, g_nodes, matched_names):
    """族计数漂移表（按归一化族名聚合）。"""
    ft = Counter(norm_family(nd["family"]) for nd in t_nodes.values())
    fg = Counter(norm_family(nd["family"]) for nd in g_nodes.values())
    fams = set(ft) | set(fg)
    mcount = Counter(norm_family(t_nodes[n]["family"]) for n in matched_names)
    rows = []
    for f in fams:
        rows.append({"family": f, "n_t": ft.get(f, 0), "n_g": fg.get(f, 0),
                     "matched": mcount.get(f, 0)})
    rows.sort(key=lambda r: (-(abs(r["n_t"] - r["n_g"])), -(r["n_t"] + r["n_g"])))
    return rows


def shape_consistency(pairs, t_nodes, g_nodes):
    """
    匹配对的跨侧 output 形状一致率（按族+方向聚合）：
    同名/变体配对的形状一致率过低 → 疑似调用编号错位或切分差异。
    """
    fam_tot, fam_ok = Counter(), Counter()
    for t_name, g_name, _tier in pairs:
        nd_t, nd_g = t_nodes[t_name], g_nodes[g_name]
        if nd_t["dir"] != nd_g["dir"]:
            continue
        fam = "%s[%s]" % (norm_family(nd_t["family"]), nd_t["dir"] or "?")
        outs_t = [s.get("shape") for s in nd_t["slots"] if s["role"] == "output" and s["leaf"] == "tensor"]
        outs_g = [s.get("shape") for s in nd_g["slots"] if s["role"] == "output" and s["leaf"] == "tensor"]
        if not outs_t or not outs_g:
            continue
        fam_tot[fam] += 1
        if outs_t == outs_g:
            fam_ok[fam] += 1
    rows = []
    for fam, tot in fam_tot.items():
        ok = fam_ok.get(fam, 0)
        rows.append({"family": fam, "pairs": tot, "shape_match": ok,
                     "rate": round(ok / tot, 3) if tot else None,
                     "suspect": tot >= 3 and ok / tot < 0.5})
    rows.sort(key=lambda r: (r["suspect"], -(r["pairs"] - r["shape_match"])))
    return rows


def divergence_census(pairs, t_nodes, g_nodes):
    """
    全窗口分歧普查：精确配对节点中同形状 output 张量槽位按
    forward/backward 分列统计——槽位总数、Norm rel=0 数、minor(<1e-3) 数、
    significant(>=1e-3) 数、跨侧比值 >=1e3x 案例数、非有限值数、rel min/max。
    供报告"逐位一致/噪声级/无数值后果"类全窗口量词断言引用（普查义务，HC-B6）。
    """
    res = {}
    for direction in ("forward", "backward"):
        n = {"same_shape_slots": 0, "rel0": 0, "minor": 0, "significant": 0,
             "one_side_zero": 0, "ratio_ge_1e3": 0, "nonfinite": 0}
        rels = []
        top = []
        for t_name, g_name, _tier in pairs:
            nd_t = t_nodes[t_name]
            if nd_t["dir"] != direction:
                continue
            o_t = [s for s in nd_t["slots"] if s["role"] == "output" and s["leaf"] == "tensor"]
            o_g = [s for s in g_nodes[g_name]["slots"] if s["role"] == "output" and s["leaf"] == "tensor"]
            if not o_t or len(o_t) != len(o_g):
                continue
            for s_t, s_g in zip(o_t, o_g):
                if not s_t.get("shape") or s_t.get("shape") != s_g.get("shape"):
                    continue
                d = tensor_pair_diff(s_t, s_g)
                n["same_shape_slots"] += 1
                if d.get("nonfinite"):
                    n["nonfinite"] += 1
                if d.get("one_side_zero"):
                    n["one_side_zero"] += 1
                nt_, ng_ = s_t.get("Norm"), s_g.get("Norm")
                if finite(nt_) and finite(ng_):
                    big, small = max(abs(nt_), abs(ng_)), min(abs(nt_), abs(ng_))
                    if big > 0 and (small == 0 or big / small >= 1e3):
                        n["ratio_ge_1e3"] += 1
                nre = d.get("nre")
                if nre is None:
                    continue
                if nre == 0:
                    n["rel0"] += 1
                elif nre < DIRTY_NRE:
                    n["minor"] += 1
                else:
                    n["significant"] += 1
                rels.append(nre)
                rec = (nre, t_name, s_t.get("path"), nt_, ng_)
                if len(top) < 3:
                    top.append(rec)
                    top.sort(key=lambda x: -x[0])
                elif nre > top[-1][0]:
                    top[2] = rec
                    top.sort(key=lambda x: -x[0])
        n["rel_min"] = min(rels) if rels else None
        n["rel_max"] = max(rels) if rels else None
        n["top_examples"] = [{"node": t, "path": p, "nre": r, "norm_t": a, "norm_g": b}
                             for r, t, p, a, b in top]
        res[direction] = n
    return res


def neighbor_context(name, t_nodes, common_set, k=1):
    """该节点在 dump 序中相邻的已匹配节点（用于未匹配节点定位）。"""
    names = list(t_nodes)
    try:
        i = names.index(name)
    except ValueError:
        return {}
    prev_n = next_n = None
    for j in range(i - 1, -1, -1):
        if names[j] in common_set:
            prev_n = names[j]
            break
    for j in range(i + 1, len(names)):
        if names[j] in common_set:
            next_n = names[j]
            break
    return {"prev_matched": prev_n, "next_matched": next_n, "order": i}


# ---------------------------------------------------------------- 锚点：分侧守恒检验

def _reduce_factor(shape_in, shape_out):
    """
    归约结构识别：两形状除一个维度外全部相同，且该维 shape_in[d] = N*shape_out[d]（N≥2 整数）。
    输出侧该维被移除（keepdim=False 的沿轴归约）时尝试左/右对齐。返回 (N, dim) 或 None。
    """
    if not (isinstance(shape_in, list) and isinstance(shape_out, list)):
        return None
    if not all(isinstance(x, int) for x in list(shape_in) + list(shape_out)):
        return None
    a, b = list(shape_in), list(shape_out)
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        if len(diff) == 1:
            i = diff[0]
            if b[i] > 0 and a[i] % b[i] == 0 and a[i] // b[i] >= 2:
                return (a[i] // b[i], i)
        return None
    if len(a) == len(b) + 1:
        for pos in (0, len(a) - 1):        # 被移除的是第 0 维（左对齐）或最后一维（右对齐）
            aa = a[1:] if pos == 0 else a[:-1]
            if aa == b and a[pos] >= 2:
                return (a[pos], pos)
    return None


def conservation_check(pairs, t_nodes, g_nodes, rel_tol=0.02):
    """
    归约类反向节点的分侧守恒检验（mean 级，只需 Mean 统计量）：
      等分求和的数学恒等式 mean(out) = N·mean(in)；N_eff = mean(out)/mean(in)。
    逐侧计算 N_eff 并与形状导出的 N 比较；一侧守恒、一侧违反（或两侧 N_eff 明显不同）
    的节点输出为锚点——违反侧为缺陷侧第一指向（该侧归约实现/精度不足，或该侧捕获不忠实）。
    两侧同样违反（N_eff≈N_eff）多为 mean 类反向的语义性"违反"，不属于分歧锚点，不输出。
    """
    rows = []
    for t_name, g_name, tier in pairs:
        nd_t, nd_g = t_nodes[t_name], g_nodes[g_name]
        if nd_t["dir"] != "backward" or nd_g["dir"] != "backward":
            continue

        def first_tensor(slots, role):
            for s in slots:
                if s["leaf"] == "tensor" and s["role"] == role:
                    return s
            return None

        s_in_t, s_out_t = first_tensor(nd_t["slots"], "input"), first_tensor(nd_t["slots"], "output")
        s_in_g, s_out_g = first_tensor(nd_g["slots"], "input"), first_tensor(nd_g["slots"], "output")
        if not all([s_in_t, s_out_t, s_in_g, s_out_g]):
            continue
        rf = _reduce_factor(s_in_t.get("shape"), s_out_t.get("shape"))
        if not rf:
            continue
        if s_in_g.get("shape") != s_in_t.get("shape") or s_out_g.get("shape") != s_out_t.get("shape"):
            continue
        mi_t, mo_t = s_in_t.get("Mean"), s_out_t.get("Mean")
        mi_g, mo_g = s_in_g.get("Mean"), s_out_g.get("Mean")
        vals = (mi_t, mo_t, mi_g, mo_g)
        if not all(finite(x) for x in vals):
            continue
        if any(abs(x) < 1e-12 for x in vals):
            continue
        N = rf[0]
        row = {"node": t_name, "node_g": g_name, "match": tier, "order": nd_t["order"],
               "shape_in": s_in_t.get("shape"), "shape_out": s_out_t.get("shape"), "N": N,
               "mean_in_t": mi_t, "mean_out_t": mo_t, "N_eff_t": mo_t / mi_t,
               "mean_in_g": mi_g, "mean_out_g": mo_g, "N_eff_g": mo_g / mi_g}
        row["cons_t"] = abs(row["N_eff_t"] / N - 1.0) <= rel_tol
        row["cons_g"] = abs(row["N_eff_g"] / N - 1.0) <= rel_tol
        rows.append(row)
    # 锚点条件：分侧不一致（一侧守恒一侧违反），或两侧都不守恒但 N_eff 明显不同
    flagged = [r for r in rows
               if (r["cons_t"] != r["cons_g"]) or
               (not r["cons_t"] and not r["cons_g"] and
                abs(r["N_eff_t"] - r["N_eff_g"]) / max(abs(r["N_eff_t"]), abs(r["N_eff_g"]), 1e-30) > 0.05)]
    flagged.sort(key=lambda r: r["order"])
    return rows, flagged


# ---------------------------------------------------------------- 锚点：操作数身份跨侧对照

def _output_quad_index(nodes):
    """同侧输出四元组索引：(Max,Min,Mean,Norm) → [节点名,...]。仅收全有限值四元组。"""
    idx = defaultdict(list)
    for name, nd in nodes.items():
        for s in nd["slots"]:
            if s["leaf"] == "tensor" and s["role"] == "output":
                q = (s.get("Max"), s.get("Min"), s.get("Mean"), s.get("Norm"))
                if all(finite(x) for x in q):
                    idx[q].append(name)
    return idx


def operand_identity_check(pairs, t_nodes, g_nodes, idx_t, idx_g, drift_fams=None):
    """
    匹配节点对操作数槽位的同侧身份命中对照：
      正常情况下操作数 ≡ 其生产者输出（同一张量两次捕获），两侧都应有命中；
      一侧操作数与本侧**任何**节点输出逐位相同、另一侧无任何命中 → 命中存在性不对称，
      提示"单侧消费了非其生产链上的张量"（传错张量/缓冲区别名/参数派生操作数单侧异常）。
    命中 ≠ 定罪：捕获时序、缓冲别名也会造成不对称，须按 mechanism 3.6 消费点复核。
    按归一化族 + 槽位路径聚合，避免逐节点刷屏。
    置信度：全部来自精确匹配对且族计数无漂移 → high；含变体匹配对或族计数
    两侧不等（漂移/分片区，错位配对是主要假阳性来源）→ low。
    """
    def operand_leaves(nd):
        for s in nd["slots"]:
            if s["leaf"] == "tensor" and s["role"] != "output":
                yield s

    agg = defaultdict(lambda: {"count": 0, "examples": [], "n_variant": 0, "fam_drifted": False})
    for t_name, g_name, tier in pairs:
        nd_t, nd_g = t_nodes[t_name], g_nodes[g_name]
        map_t = {(s["role"], s["path"]): s for s in operand_leaves(nd_t)}
        map_g = {(s["role"], s["path"]): s for s in operand_leaves(nd_g)}
        for key in sorted(set(map_t) & set(map_g)):
            s_t, s_g = map_t[key], map_g[key]
            q_t = (s_t.get("Max"), s_t.get("Min"), s_t.get("Mean"), s_t.get("Norm"))
            q_g = (s_g.get("Max"), s_g.get("Min"), s_g.get("Mean"), s_g.get("Norm"))
            if not (all(finite(x) for x in q_t) and all(finite(x) for x in q_g)):
                continue
            # 近零/常量广播类四元组命中无信息量
            if (abs(q_t[3]) < dtype_floor(s_t.get("dtype") or "") or
                    abs(q_g[3]) < dtype_floor(s_g.get("dtype") or "")):
                continue
            hits_t = idx_t.get(q_t, [])
            hits_g = idx_g.get(q_g, [])
            if bool(hits_t) == bool(hits_g):
                continue
            fam = norm_family(nd_t["family"])
            direction = "target_only" if hits_t else "golden_only"
            a = agg[(fam, key[1], direction)]
            a["count"] += 1
            if tier != "exact":
                a["n_variant"] += 1
            if drift_fams and fam in drift_fams:
                a["fam_drifted"] = True
            if len(a["examples"]) < 3:
                q_out = None
                for s in nd_t["slots"]:
                    if s["leaf"] == "tensor" and s["role"] == "output":
                        qo = (s.get("Max"), s.get("Min"), s.get("Mean"), s.get("Norm"))
                        if all(finite(x) for x in qo):
                            q_out = [qo[0], qo[1], qo[2], qo[3]]
                            break
                a["examples"].append({
                    "node": t_name, "node_g": g_name, "order": nd_t["order"], "path": key[1],
                    "match": tier,
                    "hits_target": hits_t[:6], "hits_golden": hits_g[:6],
                    "quad_target": [q_t[0], q_t[1], q_t[2], q_t[3]],
                    "quad_golden": [q_g[0], q_g[1], q_g[2], q_g[3]],
                    "quad_output": q_out,
                    "shape_target": s_t.get("shape"), "shape_golden": s_g.get("shape")})
    out = []
    for (fam, path, direction), a in agg.items():
        conf = "high" if (a["n_variant"] == 0 and not a["fam_drifted"]) else "low"
        out.append({"family": fam, "path": path, "hits_side": direction, "count": a["count"],
                    "confidence": conf, "n_variant": a["n_variant"],
                    "examples": a["examples"]})
    out.sort(key=lambda g: (0 if g["confidence"] == "high" else 1, -g["count"]))
    return out


# ---------------------------------------------------------------- 锚点：梯度覆盖缺口

def _module_path_of(name, marker):
    """取名字中最后一个 `.<marker>.` 之前的部分（含 Module. 前缀的模块路径）。"""
    idx = name.rfind("." + marker + ".")
    return name[:idx] if idx > 0 else None


def coverage_gap_check(pairs, t_nodes, g_nodes, pg, max_rows=25):
    """
    覆盖缺口表：dNorm² 最大的参数梯度，其所属模块的反向节点是否在捕获窗口内。
    参数梯度 `Module.<path>.parameters_grad.<N>` 的模块路径为 `Module.<path>`；
    该路径的 Module backward 节点未捕获（任一侧缺失）或未配对 → 该参数的梯度分歧
    产生于未被捕获的反向计算段——"梯度差异来自窗口外"从推断升级为可核查的结构事实，
    机制层搜索必须回答：最后一个被捕获锚点与未捕获段之间，哪个覆盖内指纹能解释该
    模块族梯度的单侧不对称。
    """
    pair_of_t = {p[0]: p[1] for p in pairs}
    t_bwd = defaultdict(list)
    for n, nd in t_nodes.items():
        if nd.get("kind") == "module" and nd.get("dir") == "backward":
            mp = _module_path_of(n, "backward")
            if mp:
                t_bwd[mp].append(n)
    g_bwd = set()
    for n, nd in g_nodes.items():
        if nd.get("kind") == "module" and nd.get("dir") == "backward":
            mp = _module_path_of(n, "backward")
            if mp:
                g_bwd.add(mp)
    out = []
    seen_nodes = set()
    for rec in pg:
        if rec["node"] in seen_nodes:
            continue
        seen_nodes.add(rec["node"])
        mp = _module_path_of(rec["node"], "parameters_grad")
        if not mp:
            continue
        bwd_t = t_bwd.get(mp, [])
        matched = any(pair_of_t.get(b) is not None and
                      g_nodes.get(pair_of_t[b], {}).get("dir") == "backward" for b in bwd_t)
        out.append({"param_node": rec["node"], "param": rec["param"],
                    "norm_t": rec["norm_t"], "norm_g": rec["norm_g"],
                    "module": mp,
                    "bwd_captured_t": bool(bwd_t), "bwd_captured_g": mp in g_bwd,
                    "bwd_matched": matched, "bwd_examples_t": bwd_t[:2]})
        if len(out) >= max_rows:
            break
    return out


# ---------------------------------------------------------------- 锚点：聚合输入单侧悬殊（API-only 窗口外注入指纹）

_AGG_NORM_BASES = {"_foreach_norm", "foreach_norm", "vector_norm", "norm"}


def agg_single_sided_check(pairs, t_nodes, g_nodes, idx_t, idx_g, ratio_k=100.0, max_rows=12):
    """
    聚合归约节点（norm 类）输入槽位跨侧 Norm 单侧悬殊（一侧 ≥ ratio_k× 另一侧）。
    API-only dump 无 parameters_grad 节点时，clip_grad_norm_ 类聚合（_foreach_norm 等）
    的输入即 .grad 本体——该指纹 = 现象载体（聚合输入种群）单侧不对称的直接显形。
    window_out 判据（机械）：悬殊槽位四元组在同侧输出索引中**无生产者命中**的占多数
    → 这些输入的生产不在被捕获算子输出中（.grad 累积/注入路径未被捕获，或窗口外生产）
    → 触发"排除迁移禁令"（SKILL.md 硬性）：对被捕获路径的一切"已排除/非因果"结论，
    其有效域不覆盖该载体路径，自动降级为"对现象载体路径未排除"。
    命中 ≠ 定罪：单侧悬殊也可能是真实分歧的聚合显形；机制层候选必须回答
    "能否解释聚合输入的单侧不对称"。
    """
    bwd_max = {"target": 0.0, "golden": 0.0}
    n_bwd = 0
    for t_name, g_name, tier in pairs:
        nd_t, nd_g = t_nodes[t_name], g_nodes[g_name]
        if nd_t.get("dir") != "backward" or nd_g.get("dir") != "backward":
            continue
        n_bwd += 1
        for side, nd in (("target", nd_t), ("golden", nd_g)):
            for s in nd["slots"]:
                if s["leaf"] == "tensor" and s["role"] == "output":
                    nv = s.get("Norm")
                    if nv is not None and finite(nv):
                        bwd_max[side] = max(bwd_max[side], nv)
    flags = []
    for t_name, g_name, tier in pairs:
        nd_t, nd_g = t_nodes[t_name], g_nodes[g_name]
        base = (nd_t.get("family") or "").split(".")[-1]
        if base not in _AGG_NORM_BASES:
            continue
        map_t, map_g = (slot_map(nd_t, tensor_only=True, non_output=True),
                        slot_map(nd_g, tensor_only=True, non_output=True))
        rows = []
        for key in sorted(set(map_t) & set(map_g)):
            st, sg = map_t[key], map_g[key]
            nt, ng = st.get("Norm"), sg.get("Norm")
            if nt is None or ng is None or not (finite(nt) and finite(ng)):
                continue
            hi, lo = max(nt, ng), min(nt, ng)
            if hi <= 0 or lo * ratio_k > hi:
                continue
            side = "target" if nt > ng else "golden"
            quad = (st.get("Max"), st.get("Min"), st.get("Mean"), nt) if side == "target" \
                else (sg.get("Max"), sg.get("Min"), sg.get("Mean"), ng)
            hits = (idx_t if side == "target" else idx_g).get(quad)
            rows.append({"path": key[1], "norm_t": nt, "norm_g": ng,
                         "ratio": hi / max(lo, 1e-30), "side": side,
                         "no_producer": not hits,
                         "producers": (hits or [])[:3]})
        if len(rows) >= 2:
            rows.sort(key=lambda r: -r["ratio"])
            n_np = sum(1 for r in rows if r["no_producer"])
            flags.append({"node": t_name, "node_g": g_name, "order": nd_t["order"],
                          "n_flagged": len(rows), "side": rows[0]["side"],
                          "top_ratio": rows[0]["ratio"],
                          "n_no_producer": n_np,
                          "window_out": n_np * 2 >= len(rows),
                          "in_max_norm": max(r["norm_t"] if r["side"] == "target" else r["norm_g"]
                                             for r in rows),
                          "bwd_max_norm_same_side": bwd_max[rows[0]["side"]],
                          "rows": rows[:8]})
    flags.sort(key=lambda f: (not f["window_out"], -f["n_flagged"]))
    return flags[:max_rows]


def backward_created_check(pairs, t_nodes, g_nodes, max_rows=60):
    """
    反向内产生分歧锚点：匹配反向节点对，**输入槽位（grad_output 侧）四统计量
    逐位一致、输出槽位显著分歧** → 分歧在该反向算子内部产生，不是从上游继承。
    这是"继承 vs 产生"的机械判别（前向"首分歧之前干净"逻辑在反向的等价物）：
    - 该锚点节点 = 分歧引入点候选，机制层搜索应聚焦该反向的实现路径（含控制类
      操作数、dtype 分派），而不是继续上溯已一致的输入侧。
    - 输入逐位一致要求所有可比输入槽位（role=input 的张量叶子，四元组全有限）
      的 (Max,Min,Mean,Norm) 完全相等；任何结构分歧（dtype/shape）或近地板四元组
      → 该节点不可判，跳过。
    命中 ≠ 定罪：捕获失真仍可伪造"输入一致"，须与守恒检验、参数梯度交叉验证。
    """
    out = []
    for t_name, g_name, tier in pairs:
        nd_t, nd_g = t_nodes[t_name], g_nodes[g_name]
        if nd_t["dir"] != "backward" or nd_g["dir"] != "backward":
            continue
        map_t, map_g = slot_map(nd_t, tensor_only=True), slot_map(nd_g, tensor_only=True)
        ins = sorted(k for k in set(map_t) & set(map_g) if k[0] == "input")
        if not ins:
            continue
        ok = True
        for k in ins:
            s_t, s_g = map_t[k], map_g[k]
            if s_t.get("dtype") != s_g.get("dtype") or s_t.get("shape") != s_g.get("shape"):
                ok = False
                break
            q_t = (s_t.get("Max"), s_t.get("Min"), s_t.get("Mean"), s_t.get("Norm"))
            q_g = (s_g.get("Max"), s_g.get("Min"), s_g.get("Mean"), s_g.get("Norm"))
            if not all(finite(x) for x in q_t) or not all(finite(x) for x in q_g):
                ok = False
                break
            if any(abs(x) < dtype_floor(s_t.get("dtype") or "") for x in (q_t[3], q_g[3])):
                ok = False
                break
            if q_t != q_g:
                ok = False
                break
        if not ok:
            continue
        # 输入逐位一致：检查输出侧是否有显著数值分歧
        outs = sorted(k for k in set(map_t) & set(map_g) if k[0] == "output")
        worst = None
        for k in outs:
            s_t, s_g = map_t[k], map_g[k]
            if s_t.get("dtype") != s_g.get("dtype") or s_t.get("shape") != s_g.get("shape"):
                continue
            nt, ng = s_t.get("Norm"), s_g.get("Norm")
            if not (finite(nt) and finite(ng)):
                continue
            floor = dtype_floor(s_t.get("dtype") or "")
            if abs(nt) < floor and abs(ng) < floor:
                continue
            nre = abs(nt - ng) / abs(ng) if abs(ng) > 0 else None
            if nre is not None and nre >= DIRTY_NRE and (worst is None or nre > worst[1]):
                worst = (k[1], nre, nt, ng)
        if worst is None:
            continue
        out.append({"node": t_name, "node_g": g_name, "match": tier, "order": nd_t["order"],
                    "family": norm_family(nd_t["family"]),
                    "diverge_path": worst[0], "nre": worst[1],
                    "norm_t": worst[2], "norm_g": worst[3],
                    "n_input_slots": len(ins)})
    out.sort(key=lambda r: r["order"])
    return out[:max_rows]


# ---------------------------------------------------------------- 单 (step, rank) 分析

def analyze_pair(t_root, g_root, args, pair_key):
    meta_t, t_nodes = parse_side_dump(t_root["dump"])
    meta_g, g_nodes = parse_side_dump(g_root["dump"])
    stack_t = load_stack_map(t_root["stack"])
    stack_g = load_stack_map(g_root["stack"])
    cons_t = load_construct_map(t_root["construct"])
    cons_g = load_construct_map(g_root["construct"])

    res = {"pair": list(pair_key),
           "meta_t": meta_t, "meta_g": meta_g,
           "n_nodes_t": len(t_nodes), "n_nodes_g": len(g_nodes)}

    common, t_only, g_only = match_nodes(t_nodes, g_nodes)
    pairs = [(n, n, "exact") for n in common]

    # 第二阶段：族名归一化（剥离 _rank<N> 变体）+ 形状门槛 + 双侧唯一键，1:1 增配
    variant_pairs = match_name_variants(t_only, g_only, t_nodes, g_nodes)
    for t_name, g_name in variant_pairs.items():
        pairs.append((t_name, g_name, "name_variant"))
    pairs.sort(key=lambda p: t_nodes[p[0]]["order"])
    matched_t = [p[0] for p in pairs]
    rem_t_only = [n for n in t_only if n not in variant_pairs]
    rem_g_only = [n for n in g_only if n not in set(variant_pairs.values())]
    common_set = set(matched_t)

    res["alignment"] = {
        "matched": len(pairs), "matched_exact": len(common),
        "matched_name_variant": len(variant_pairs),
        "t_only": len(rem_t_only), "g_only": len(rem_g_only),
        "by_kind": {}, "family_seq": family_seq_stats(t_nodes, g_nodes),
    }
    kt = Counter(t_nodes[n]["kind"] for n in t_nodes)
    kg = Counter(g_nodes[n]["kind"] for n in g_nodes)
    km = Counter(t_nodes[p[0]]["kind"] for p in pairs)
    res["alignment"]["by_kind"] = {k: {"t": kt.get(k, 0), "g": kg.get(k, 0), "matched": km.get(k, 0)}
                                   for k in ("api", "module", "paramgrad", "other") if kt.get(k, 0) or kg.get(k, 0)}
    fam_rows = family_count_table(t_nodes, g_nodes, matched_t)
    res["alignment"]["family_drift_top"] = [r for r in fam_rows if r["n_t"] != r["n_g"]][:25]
    res["alignment"]["family_drift_n_families"] = sum(1 for r in fam_rows if r["n_t"] != r["n_g"])
    res["shape_consistency_suspect"] = [r for r in shape_consistency(pairs, t_nodes, g_nodes) if r["suspect"]][:15]
    res["divergence_census"] = divergence_census(pairs, t_nodes, g_nodes)
    # ---- 槽位对齐 + 差异 ----
    dirty = []          # significant / structural
    minor = []          # minor（供参考，不计入首分歧）
    scalar_diffs = []
    dtype_cluster = Counter()
    dtype_examples = defaultdict(list)
    shape_cluster = Counter()
    shape_examples = defaultdict(list)
    slot_missing = []
    first_fwd = first_bwd = first_struct = None
    n_compared = 0
    sig_count = 0
    noise_count = 0

    for t_name, g_name, tier in pairs:
        nd_t, nd_g = t_nodes[t_name], g_nodes[g_name]
        map_t, map_g = slot_map(nd_t), slot_map(nd_g)
        keys = sorted(set(map_t) | set(map_g))
        for key in keys:
            s_t, s_g = map_t.get(key), map_g.get(key)
            if s_t is None or s_g is None:
                slot_missing.append({"node": t_name, "path": key[1], "side": "target" if s_g is None else "golden"})
                continue
            lt, lg = s_t["leaf"], s_g["leaf"]
            if lt == "tensor" and lg == "tensor":
                n_compared += 1
                d = tensor_pair_diff(s_t, s_g)
                cls = classify_pair_diff(d)
                if cls == "noise":
                    noise_count += 1
                    continue
                if d["dtype_diff"]:
                    dt = (str(d["dtype_t"]), str(d["dtype_g"]))
                    dtype_cluster[dt] += 1
                    if len(dtype_examples[dt]) < 3:
                        dtype_examples[dt].append({"node": t_name, "path": d["path"]})
                if d["shape_diff"]:
                    fam = nd_t["family"]
                    shape_cluster[fam] += 1
                    if len(shape_examples[fam]) < 3:
                        shape_examples[fam].append({"node": t_name, "path": d["path"],
                                                    "shape_t": d["shape_t"], "shape_g": d["shape_g"]})
                if cls in ("structural", "significant"):
                    sig_count += 1
                    entry = {"node": t_name, "node_g": g_name if g_name != t_name else None,
                             "match": tier,
                             "kind": nd_t["kind"], "dir": nd_t["dir"], "order": nd_t["order"],
                             "family": nd_t["family"], "cls": cls, **d}
                    if cons_t.get(t_name) or cons_g.get(g_name):
                        entry["construct_t"] = cons_t.get(t_name)
                        entry["construct_g"] = cons_g.get(g_name)
                    dirty.append(entry)
                    if cls == "structural" and first_struct is None:
                        first_struct = entry
                    if first_fwd is None and nd_t["dir"] == "forward":
                        first_fwd = entry
                    if first_bwd is None and nd_t["dir"] == "backward":
                        first_bwd = entry
                elif cls == "minor":
                    minor.append({"node": t_name, "path": d["path"], "nre": d.get("nre"),
                                  "dtype_t": d["dtype_t"], "dtype_g": d["dtype_g"]})
            elif lt == "scalar" and lg == "scalar":
                if s_t.get("value") != s_g.get("value") or s_t.get("stype") != s_g.get("stype"):
                    scalar_diffs.append({"node": t_name, "kind": nd_t["kind"], "dir": nd_t["dir"],
                                         "order": nd_t["order"], "path": s_t["path"],
                                         "t": [s_t.get("stype"), s_t.get("value")],
                                         "g": [s_g.get("stype"), s_g.get("value")]})
            elif lt != lg:
                slot_missing.append({"node": t_name, "path": key[1],
                                     "side": "type_mismatch", "t_leaf": lt, "g_leaf": lg})

    dirty.sort(key=lambda e: e["order"])
    sig_ordered = [e for e in dirty if e["cls"] == "significant"]
    res["diff_overview"] = {"slots_compared_tensor": n_compared, "significant": sig_count,
                            "minor": len(minor), "noise": noise_count,
                            "scalar_diffs": len(scalar_diffs), "slot_missing": len(slot_missing)}
    res["first_divergence"] = {"forward": first_fwd, "backward": first_bwd, "structural": first_struct}
    res["dirty_order_sample"] = sig_ordered[:80]
    res["scalar_diffs"] = scalar_diffs[:200]
    res["scalar_diffs_total"] = len(scalar_diffs)
    res["slot_missing"] = slot_missing[:100]
    res["slot_missing_total"] = len(slot_missing)
    res["dtype_clusters"] = [{"dtypes": list(k), "count": v, "examples": dtype_examples[k]}
                             for k, v in dtype_cluster.most_common(20)]
    res["shape_clusters"] = [{"family": k, "count": v, "examples": shape_examples[k]}
                             for k, v in shape_cluster.most_common(20)]

    # ---- 参数梯度与 gnorm 估计 ----
    pg = []
    sq_t = sq_g = 0.0
    pg_t_cnt = pg_g_cnt = 0
    for t_name, g_name, _tier in pairs:
        nd_t, nd_g = t_nodes[t_name], g_nodes[g_name]
        if nd_t["kind"] != "paramgrad":
            continue
        map_t, map_g = slot_map(nd_t), slot_map(nd_g)
        for key in sorted(set(map_t) | set(map_g)):
            s_t, s_g = map_t.get(key), map_g.get(key)
            if s_t is None or s_g is None or s_t["leaf"] != "tensor" or s_g["leaf"] != "tensor":
                continue
            nt, ng = s_t.get("Norm"), s_g.get("Norm")
            rec = {"node": t_name, "param": key[1], "dtype_t": s_t.get("dtype"), "dtype_g": s_g.get("dtype"),
                   "shape_t": s_t.get("shape"), "shape_g": s_g.get("shape"),
                   "norm_t": nt, "norm_g": ng, "nre": (abs(nt - ng) / abs(ng) if finite(nt) and finite(ng) and abs(ng) > 0 else None)}
            pg.append(rec)
            if finite(nt):
                sq_t += nt * nt
                pg_t_cnt += 1
            if finite(ng):
                sq_g += ng * ng
                pg_g_cnt += 1
    pg.sort(key=lambda r: -abs((r["norm_t"] or 0) ** 2 - (r["norm_g"] or 0) ** 2))
    res["param_grad"] = {"count": len(pg), "gnorm_est_t": math.sqrt(sq_t) if pg_t_cnt > 0 else None,
                         "gnorm_est_g": math.sqrt(sq_g) if pg_g_cnt > 0 else None,
                         "n_t": pg_t_cnt, "n_g": pg_g_cnt,
                         "top": pg[:40], "worst_rel": sorted([p for p in pg if p["nre"] is not None],
                                                             key=lambda r: -r["nre"])[:15]}
    res["param_grad_full"] = pg if len(pg) <= 400 else None

    # ---- 锚点：分侧守恒检验（归约类反向，mean 级） ----
    cons_all, cons_flagged = conservation_check(pairs, t_nodes, g_nodes)
    res["conservation"] = {
        "checked": len(cons_all), "n_flagged": len(cons_flagged),
        "flagged": cons_flagged[:80],
        "n_hard": sum(1 for r in cons_flagged if r["cons_t"] != r["cons_g"]),
    }

    # ---- 锚点：操作数身份跨侧对照（带置信度标注） ----
    drift_fams = {r["family"] for r in fam_rows if r["n_t"] != r["n_g"]}
    idx_t = _output_quad_index(t_nodes)
    idx_g = _output_quad_index(g_nodes)
    oi_groups = operand_identity_check(pairs, t_nodes, g_nodes, idx_t, idx_g, drift_fams)
    res["operand_identity"] = {"n_groups": len(oi_groups),
                               "n_high": sum(1 for g in oi_groups if g["confidence"] == "high"),
                               "groups": oi_groups[:40]}

    # ---- 锚点：梯度覆盖缺口（参数梯度大分歧 × 所属模块反向未捕获/未配对） ----
    res["coverage_gap"] = coverage_gap_check(pairs, t_nodes, g_nodes, pg)

    # ---- 锚点：聚合输入单侧悬殊（API-only dump 的窗口外注入指纹） ----
    res["agg_single_sided"] = agg_single_sided_check(pairs, t_nodes, g_nodes, idx_t, idx_g)

    # ---- 锚点：反向内产生分歧（输入逐位一致、输出显著分歧 → 分歧在该反向内部产生） ----
    res["backward_created"] = backward_created_check(pairs, t_nodes, g_nodes)

    # ---- Top 差异表（仅形状/dtype 可比的对，结构分歧另行呈现） ----
    comparable = [e for e in dirty
                  if e["cls"] == "significant" and not e.get("dtype_diff") and not e.get("shape_diff")]
    res["top_nre"] = sorted(comparable, key=lambda e: (e.get("nre") if e.get("nre") is not None else -1.0),
                            reverse=True)[:args.top]
    res["top_absnorm"] = sorted(comparable, key=lambda e: -abs(((e["stats"].get("Norm") or {}).get("t") or 0)
                                                              - ((e["stats"].get("Norm") or {}).get("g") or 0)))[:args.top]

    # ---- 未匹配节点分组 ----
    def group_unmatched(names, nodes, side):
        groups = defaultdict(list)
        for n in names:
            groups[nodes[n]["family"]].append(n)
        out = []
        for fam, lst in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            out.append({"family": fam, "side": side, "count": len(lst),
                        "examples": lst[:3], "order_range": [nodes[lst[0]]["order"], nodes[lst[-1]]["order"]],
                        "ctx": neighbor_context(lst[0], nodes, common_set)})
        return out
    res["unmatched"] = (group_unmatched(rem_t_only, t_nodes, "target")[:30] +
                        group_unmatched(rem_g_only, g_nodes, "golden")[:30])
    res["unmatched_total"] = {"target": len(rem_t_only), "golden": len(rem_g_only)}
    res["name_variant_pairs"] = [{"target": t, "golden": g} for t, g in list(variant_pairs.items())[:200]]

    # ---- 关键锚点的堆栈信息 ----
    anchors = []
    for k in ("forward", "backward", "structural"):
        e = res["first_divergence"].get(k)
        if e:
            anchors.append(e["node"])
    for e in res["top_nre"][:10]:
        anchors.append(e["node"])
    # 速览锚点（高置信身份组/聚合悬殊/反向产生分歧/守恒违反）同样纳入栈渲染——
    # 更新类循环的语义消歧（EMA vs Adam exp_avg 同构公式）依赖调用文本，属审计现场材料。
    for grp in ((res.get("operand_identity") or {}).get("groups") or []):
        if grp.get("confidence") == "high":
            for ex in (grp.get("examples") or [])[:2]:
                anchors.append(ex["node"])
    for a in (res.get("agg_single_sided") or []):
        anchors.append(a["node"])
    for r in (res.get("backward_created") or []):
        anchors.append(r["node"])
    for r in ((res.get("conservation") or {}).get("flagged") or []):
        anchors.append(r["node"])
    stack_info = {}
    for n in set(anchors):
        if stack_t.get(n) or stack_g.get(n):
            stack_info[n] = {"t": user_frames(stack_t.get(n, []), limit=4), "g": user_frames(stack_g.get(n, []), limit=4)}
    res["anchor_stacks"] = stack_info

    # minor 汇总（不逐条展开）
    res["minor_sample"] = minor[:20]
    return res, t_nodes, g_nodes


# ---------------------------------------------------------------- 汇总输出

def _bc_row(r, indent=""):
    """反向内产生分歧锚点行：速览节与正文节共用同一行格式（防两处漂移）。"""
    return "%s- `%s`（order %s, %s）输入槽位 ×%d 逐位一致 | 输出 %s: NRE=%s Norm t=%s g=%s" % (
        indent, r["node"], r["order"], r["match"], r["n_input_slots"],
        r["diverge_path"], fmt(r["nre"]), fmt(r["norm_t"]), fmt(r["norm_g"]))


def md_anchor_brief(res, max_rows=5):
    """决定性锚点速览：摘要顶部的高价值锚点清单，供第一轮阅读强制逐条核对。
    命中 ≠ 定罪；但把根因归于覆盖外机制前必须逐条复核本节并留痕（SKILL.md 硬性）。"""
    L = []
    A = L.append
    A("### ⚡ 决定性锚点速览（先逐条核查本节，再进入深入探索）\n")
    n_items = 0
    hard = [r for r in ((res.get("conservation") or {}).get("flagged") or [])
            if r.get("cons_t") != r.get("cons_g")]
    if hard:
        A("- **分侧守恒（违反侧=缺陷侧第一指向）**:")
        for r in hard[:max_rows]:
            A("  - `%s` N=%s: target N_eff=%.6g（%s）| golden N_eff=%.6g（%s）" % (
                r["node"], r["N"], r["N_eff_t"], "守恒" if r["cons_t"] else "违反",
                r["N_eff_g"], "守恒" if r["cons_g"] else "违反"))
        n_items += 1
    hi = [g for g in ((res.get("operand_identity") or {}).get("groups") or [])
          if g.get("confidence") == "high"]
    if hi:
        A("- **高置信操作数身份不对称（单侧消费了非其生产链上的张量；须按 mechanism 3.6 消费点复核）**:")
        for grp in hi[:max_rows]:
            ex = grp["examples"][0]
            qo = ex.get("quad_output")
            st = (res.get("anchor_stacks") or {}).get(ex["node"]) or {}
            st_txt = ""
            if st.get("t") or st.get("g"):
                st_txt = " | 栈: target=%s golden=%s（语义角色判定须引用栈帧文本——同构更新公式 EMA/Adam 的标量指纹无判别力）" % (
                    " <- ".join(st["t"][:4]) or "无", " <- ".join(st["g"][:4]) or "无")
            A("  - `%s` %s [命中仅在 %s] ×%d | 例 `%s`(order %s): t命中=%s g命中=%s | t四元组=%s g四元组=%s%s%s" % (
                grp["family"], grp["path"], grp["hits_side"], grp["count"],
                ex["node"], ex["order"],
                ex["hits_target"] or "无", ex["hits_golden"] or "无",
                [fmt(x) for x in ex["quad_target"]], [fmt(x) for x in ex["quad_golden"]],
                " | 消费节点输出四元组=%s（复核'捕获陈旧'叙事前必须先对其做算术闭合）" % [fmt(x) for x in qo]
                if qo else "", st_txt))
        n_items += 1
    gaps = [g for g in (res.get("coverage_gap") or [])
            if not (g["bwd_captured_t"] and g["bwd_captured_g"] and g["bwd_matched"])]
    aggs = [a for a in (res.get("agg_single_sided") or []) if a.get("window_out")]
    if gaps:
        A("- **梯度覆盖缺口（参数梯度大分歧，所属模块反向不在窗口/未配对——分歧产生于未捕获反向段）**:")
        for g in gaps[:max_rows]:
            A("  - `%s`（%s）: Norm t=%s g=%s | 模块反向 target=%s golden=%s 配对=%s" % (
                g["param_node"], g["param"], fmt(g["norm_t"]), fmt(g["norm_g"]),
                "有" if g["bwd_captured_t"] else "无", "有" if g["bwd_captured_g"] else "无",
                "是" if g["bwd_matched"] else "否"))
        n_items += 1
    if aggs:
        A("- **聚合输入单侧悬殊（norm 类聚合的输入=现象载体单侧悬殊；无窗口内生产者命中 → 注入/累积路径未被捕获——API-only dump 的覆盖缺口等价物）**:")
        for a in aggs[:max_rows]:
            A("  - `%s`（order %s）: %d 个输入槽位 %s 侧 ≥%.3g×，其中 %d 个无窗口内生产者命中（例 %s: t=%s g=%s）| 悬殊输入 max Norm=%s vs 窗口内反向输出 max Norm=%s" % (
                a["node"], a["order"], a["n_flagged"], a["side"], a["top_ratio"],
                a["n_no_producer"],
                a["rows"][0]["path"], fmt(a["rows"][0]["norm_t"]), fmt(a["rows"][0]["norm_g"]),
                fmt(a["in_max_norm"]), fmt(a["bwd_max_norm_same_side"])))
        n_items += 1
    if gaps or aggs:
        A("  - ⚠ **覆盖缺口/窗口外注入指纹非空 ⇒ 排除迁移禁令生效（SKILL.md 硬性）**：现象载体"
          "（悬殊聚合输入/参数梯度）的单侧不对称产生于未被捕获的生产路径——此前对**被捕获路径**"
          "上一切结构簇（dtype/控制类操作数/形状簇）的“已排除/非因果”结论，其有效域不覆盖该"
          "载体路径，自动降级为“已核查（有效域=被验证路径）；对现象载体路径**未排除**”：不得"
          "再标“非因果”，该簇必须作为覆盖外候选的**机制层候选**保留在排序中，并明确回答"
          "“该簇能否解释聚合输入/参数梯度的单侧不对称”。")
    bc = res.get("backward_created") or []
    if bc:
        A("- **反向内产生分歧（输入逐位一致、输出显著分歧——分歧在该反向算子内部产生，非继承；机制层搜索聚焦该反向的实现路径）**:")
        for r in bc[:max_rows]:
            A(_bc_row(r, indent="  "))
        n_items += 1
    fd = res.get("first_divergence") or {}
    parts = []
    for label, key in (("前向", "forward"), ("反向", "backward"), ("结构", "structural")):
        e = fd.get(key)
        if e:
            extra = ""
            if key == "structural" and e.get("dtype_diff"):
                extra = " [分歧操作数 dtype %s vs %s → 最早显形归属候选=该 dtype 分歧簇（其消费点前执行序已证实一致时，不得归因于覆盖外'更早已分歧'类假设）]" % (
                    e.get("dtype_t"), e.get("dtype_g"))
            elif key == "structural" and e.get("shape_diff"):
                extra = " [分歧操作数 shape %s vs %s → 最早显形归属候选=该 shape 分歧簇]" % (
                    shape_str(e.get("shape_t")), shape_str(e.get("shape_g")))
            parts.append("%s `%s`（order %d, %s）%s" % (label, e["node"], e["order"], e["cls"], extra))
    if parts:
        A("- **首分歧**: " + " | ".join(parts))
        n_items += 1
    if res.get("dtype_clusters"):
        A("- **Top dtype 簇**: " + "; ".join(
            "%s vs %s ×%d" % (c["dtypes"][0], c["dtypes"][1], c["count"])
            for c in res["dtype_clusters"][:3]))
        n_items += 1
    if res.get("shape_clusters"):
        A("- **Top shape 簇**: " + "; ".join(
            "%s ×%d" % (c["family"], c["count"]) for c in res["shape_clusters"][:3]))
        n_items += 1
    if not n_items:
        A("- （本 rank 无显著锚点）")
    return "\n".join(L)


def md_pair_section(res, args):
    L = []
    A = L.append
    step, rank = res["pair"]
    A("## Pair step%s rank%s\n" % (step, rank))
    L.append(md_anchor_brief(res))
    A("- target: %d nodes | golden: %d nodes | matched: %d（精确 %d + 命名变体 %d）| target-only: %d | golden-only: %d" % (
        res["n_nodes_t"], res["n_nodes_g"], res["alignment"]["matched"],
        res["alignment"]["matched_exact"], res["alignment"]["matched_name_variant"],
        res["unmatched_total"]["target"], res["unmatched_total"]["golden"]))
    bk = res["alignment"]["by_kind"]
    if bk:
        A("- 按类型: %s" % "; ".join("%s(t=%d,g=%d,m=%d)" % (k, v["t"], v["g"], v["matched"]) for k, v in bk.items()))
    fs = res["alignment"]["family_seq"]
    A("- family 序列: t=%d, g=%d, 公共前缀=%d%s" % (fs["len_t"], fs["len_g"], fs["common_prefix"],
      (", 于 %s 处分叉" % fs["diverge_at"]) if fs["diverge_at"] is not None else ""))
    if fs["diverge_at"] is not None:
        A("  - target 上下文: %s" % " | ".join(fs["ctx_t"][:12]))
        A("  - golden 上下文: %s" % " | ".join(fs["ctx_g"][:12]))
    if res["alignment"]["family_drift_top"]:
        A("- 族计数漂移 Top（族: target vs golden, matched）:")
        for r in res["alignment"]["family_drift_top"][:15]:
            A("  - %s: %d vs %d (matched %d)" % (r["family"], r["n_t"], r["n_g"], r["matched"]))
    if res["shape_consistency_suspect"]:
        A("- 形状一致率可疑的族（疑似编号错位/切分差异）:")
        for r in res["shape_consistency_suspect"][:8]:
            A("  - %s: %d/%d 形状一致" % (r["family"], r["shape_match"], r["pairs"]))
    dc = res.get("divergence_census") or {}
    if dc:
        A("")
        A("### 全窗口分歧普查（\"逐位一致/无数值后果\"类断言须引用本表）")
        A("")
        A("| 方向 | 同形状 output 槽位 | rel=0 | minor(<1e-3) | significant(≥1e-3) | 比值≥1e3× | 单侧地板外 | 非有限 | rel min | rel max |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for direction in ("forward", "backward"):
            n = dc.get(direction) or {}
            A("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                direction, n.get("same_shape_slots"), n.get("rel0"), n.get("minor"),
                n.get("significant"), n.get("ratio_ge_1e3"), n.get("one_side_zero"),
                n.get("nonfinite"),
                fmt(n["rel_min"]) if n.get("rel_min") is not None else "-",
                fmt(n["rel_max"]) if n.get("rel_max") is not None else "-"))
        for direction in ("forward", "backward"):
            for e in (dc.get(direction) or {}).get("top_examples", [])[:3]:
                A("- %s top: `%s` %s nre=%s (t Norm=%s / g Norm=%s)" % (
                    direction, e["node"], e["path"], fmt(e["nre"]), fmt(e["norm_t"]), fmt(e["norm_g"])))
    do = res["diff_overview"]
    A("- 差异概览: 张量槽位比较 %d | 显著 %d | minor %d | 噪声 %d | 标量参数差 %d | 槽位缺失 %d" % (
        do["slots_compared_tensor"], do["significant"], do["minor"], do["noise"],
        do["scalar_diffs"], do["slot_missing"]))

    fd = res["first_divergence"]
    for label, key in (("前向首分歧", "forward"), ("反向首分歧", "backward"), ("结构首分歧", "structural")):
        e = fd.get(key)
        if not e:
            continue
        A("\n### %s: `%s` (order=%d)\n" % (label, e["node"], e["order"]))
        if e.get("construct_t") or e.get("construct_g"):
            A("- construct: target=%s | golden=%s" % (e.get("construct_t"), e.get("construct_g")))
        A("- dtype: %s vs %s | shape: %s vs %s" % (e.get("dtype_t"), e.get("dtype_g"),
                                                   shape_str(e.get("shape_t")), shape_str(e.get("shape_g"))))
        for k, it in (e.get("stats") or {}).items():
            A("- %s: t=%s g=%s diff=%s rel=%s" % (k, fmt(it.get("t")), fmt(it.get("g")), fmt(it.get("diff")), fmt(it.get("rel"))))
        A("- NRE=%s | cls=%s | extras_diff=%s" % (fmt(e.get("nre")), e["cls"], e.get("extras_diff") or "-"))
        st = res["anchor_stacks"].get(e["node"])
        if st:
            if st.get("t"):
                A("- target 栈: %s" % " <- ".join(st["t"][:2]))
            if st.get("g"):
                A("- golden 栈: %s" % " <- ".join(st["g"][:2]))

    if res["scalar_diffs"]:
        A("\n### 标量参数差异（前 %d / 共 %d）\n" % (len(res["scalar_diffs"]), res["scalar_diffs_total"]))
        for s in res["scalar_diffs"][:40]:
            A("- `%s` %s: target=%s golden=%s" % (s["node"], s["path"], s["t"], s["g"]))

    if res["dtype_clusters"]:
        A("\n### dtype 分歧簇\n")
        for c in res["dtype_clusters"][:10]:
            A("- %s vs %s: %d 处 (例: %s)" % (c["dtypes"][0], c["dtypes"][1], c["count"],
                                             ", ".join(e["node"] + ":" + e["path"] for e in c["examples"][:2])))
    if res["shape_clusters"]:
        A("\n### shape 分歧簇\n")
        for c in res["shape_clusters"][:10]:
            ex = c["examples"][0] if c["examples"] else {}
            A("- %s: %d 处 (例: %s %s t=%s g=%s)" % (c["family"], c["count"],
                                                    ex.get("node", ""), ex.get("path", ""),
                                                    shape_str(ex.get("shape_t")), shape_str(ex.get("shape_g"))))

    pg = res["param_grad"]
    if pg["count"]:
        A("\n### 参数梯度（count=%d）\n" % pg["count"])
        A("- gnorm 估计(仅捕获参数): target=%s golden=%s" % (fmt(pg["gnorm_est_t"]), fmt(pg["gnorm_est_g"])))
        for p in pg["top"][:20]:
            A("- `%s` %s: Norm t=%s g=%s (nre=%s)" % (p["node"], p["param"], fmt(p["norm_t"]), fmt(p["norm_g"]), fmt(p["nre"])))

    cons = res["conservation"]
    if cons["checked"]:
        A("\n### 分侧守恒检验（归约类反向，mean 级：N_eff = mean(out)/mean(in)，等分求和守恒 ⇔ N_eff≈N）\n")
        A("- 检验 %d 个归约类反向节点；分侧不一致（一侧守恒/一侧违反）**%d 个**" % (cons["checked"], cons["n_hard"]))
        if cons["n_hard"]:
            A("- **一侧守恒、一侧违反的节点：违反侧为缺陷侧第一指向（该侧归约实现/精度不足，或该侧捕获不忠实）**")
        for r in cons["flagged"][:30]:
            A("- `%s` N=%s: target %s(N_eff=%.6g) | golden %s(N_eff=%.6g)" % (
                r["node"], r["N"],
                "守恒" if r["cons_t"] else "违反", r["N_eff_t"],
                "守恒" if r["cons_g"] else "违反", r["N_eff_g"]))
        if cons["n_flagged"] > 30:
            A("- （其余 %d 个见 JSON conservation.flagged）" % (cons["n_flagged"] - 30))

    bc = res.get("backward_created") or []
    if bc:
        A("\n### 反向内产生分歧（输入逐位一致、输出显著分歧）\n")
        A("- 输入（grad_output 侧）全部可比槽位四统计量逐位相等、输出显著分歧 → 分歧在该反向算子内部产生（非从上游继承）。机制层搜索聚焦该反向实现路径（控制类操作数、dtype 分派）；命中 ≠ 定罪，须与守恒检验、参数梯度交叉验证。")
        A("- 注意：反向还消费**不可见的前向保存张量**——若该反向对应前向节点的输出两侧一致，分歧在该反向实现本身；若前向输出已有分歧，则可能继承自保存张量，用前向对应节点输出对比判别。")
        for r in bc[:25]:
            A(_bc_row(r))
        if len(bc) > 25:
            A("- （其余 %d 个见 JSON backward_created）" % (len(bc) - 25))

    oi = res["operand_identity"]
    if oi["n_groups"]:
        A("\n### 操作数身份跨侧对照（一侧操作数与本侧某节点输出四元组逐位相同、另一侧无任何命中）\n")
        A("- 正常情况操作数 ≡ 生产者输出（两侧都应有命中）；命中不对称提示单侧消费了非其生产链上的张量。命中 ≠ 定罪，须按 mechanism 3.6 消费点复核。")
        A("- 置信度：high=全部来自精确匹配对且族计数无漂移；low=含变体匹配对或位于计数漂移族（错位配对为主要假阳性来源）。high 组 %d / 共 %d。" % (
            oi.get("n_high", 0), oi["n_groups"]))
        for grp in oi["groups"][:15]:
            ex = grp["examples"][0]
            qo = ex.get("quad_output")
            A("- [%s] %s `%s` [命中仅在 %s] ×%d | 例 `%s`(order %s): t命中=%s g命中=%s | t四元组=%s g四元组=%s%s" % (
                grp.get("confidence", "?"), grp["family"], grp["path"], grp["hits_side"], grp["count"],
                ex["node"], ex["order"],
                ex["hits_target"] or "无", ex["hits_golden"] or "无",
                [fmt(x) for x in ex["quad_target"]], [fmt(x) for x in ex["quad_golden"]],
                " | 消费节点输出四元组=%s" % [fmt(x) for x in qo] if qo else ""))

    if res["top_nre"]:
        A("\n### NRE Top（显著差异，前 %d）\n" % len(res["top_nre"]))
        for e in res["top_nre"][:args.top]:
            A("- `%s` %s [%s] nre=%s Norm t=%s g=%s" % (e["node"], e["path"], e["dir"], fmt(e.get("nre")),
                                                        fmt((e["stats"].get("Norm") or {}).get("t")),
                                                        fmt((e["stats"].get("Norm") or {}).get("g"))))
    if res["unmatched"]:
        A("\n### 未匹配节点分组（前 %d 组）\n" % len(res["unmatched"]))
        for u in res["unmatched"][:25]:
            A("- [%s] %s × %d (order %s; 前邻 `%s` / 后邻 `%s`)" % (
                u["side"], u["family"], u["count"], u["order_range"],
                (u["ctx"] or {}).get("prev_matched"), (u["ctx"] or {}).get("next_matched")))
    A("")
    return "\n".join(L)


def main():
    global DIRTY_NRE, MINOR_NRE
    ap = argparse.ArgumentParser(description="msprobe statistics dump 对齐与差异分析")
    ap.add_argument("--target", required=True, help="target 侧 dump 目录")
    ap.add_argument("--golden", required=True, help="golden 侧 dump 目录")
    ap.add_argument("--out", default="dump_analysis_out", help="输出工作目录")
    ap.add_argument("--rank", type=int, default=None, help="仅分析指定 rank")
    ap.add_argument("--step", type=int, default=None, help="仅分析指定 step")
    ap.add_argument("--top", type=int, default=30, help="Top 表大小")
    ap.add_argument("--dirty-nre", type=float, default=1e-3, help="显著 NRE 阈值")
    ap.add_argument("--minor-nre", type=float, default=1e-6, help="minor NRE 阈值")
    ap.add_argument("--json", default=None, help="JSON 输出文件（默认 <out>/analysis.json）")
    ap.add_argument("--report", default=None, help="Markdown 报告（默认 <out>/analysis_summary.md）")
    args = ap.parse_args()
    DIRTY_NRE, MINOR_NRE = args.dirty_nre, args.minor_nre

    t_groups = discover_dump_root(args.target)
    g_groups = discover_dump_root(args.golden)
    if not t_groups or not g_groups:
        print("[error] 未在输入目录中发现 dump.json: target=%d golden=%d" % (len(t_groups), len(g_groups)), file=sys.stderr)
        sys.exit(2)
    steps_t = {s for s, _ in t_groups}
    steps_g = {s for s, _ in g_groups}
    common_steps = sorted(steps_t & steps_g)
    if args.step is not None:
        common_steps = [s for s in common_steps if s == args.step]
    os.makedirs(args.out, exist_ok=True)

    full = {"target_root": os.path.abspath(args.target), "golden_root": os.path.abspath(args.golden),
            "steps": {}, "warnings": []}
    md = ["# dump 对齐与差异分析摘要\n",
          "- target: `%s`" % args.target, "- golden: `%s`" % args.golden, ""]
    if steps_t != steps_g:
        full["warnings"].append("两侧 step 集合不一致: t=%s g=%s" % (sorted(steps_t), sorted(steps_g)))

    for step in common_steps:
        ranks_t = sorted({r for (s, r) in t_groups if s == step})
        ranks_g = sorted({r for (s, r) in g_groups if s == step})
        rank_pairs = []  # (label, target_rank_id, golden_rank_id)
        if args.rank is not None:
            if args.rank in ranks_t and args.rank in ranks_g:
                rank_pairs = [(args.rank, args.rank, args.rank)]
            else:
                full["warnings"].append("step %d 未找到指定 rank %s: t=%s g=%s" % (step, args.rank, ranks_t, ranks_g))
        elif set(ranks_t) & set(ranks_g):
            # 同标识 rank 配对
            inter = sorted(set(ranks_t) & set(ranks_g))
            rank_pairs = [(r, r, r) for r in inter]
            if set(ranks_t) != set(ranks_g):
                full["warnings"].append("step %d 两侧 rank 集合不一致，仅配对同标识 rank: t=%s g=%s" % (step, ranks_t, ranks_g))
        elif len(ranks_t) == len(ranks_g):
            # 标识不同（如各自 proc<PID>）但数量一致：按序号配对，明确告警
            rank_pairs = [(rt, rt, rg) for rt, rg in zip(ranks_t, ranks_g)]
            full["warnings"].append("step %d 两侧 rank 标识不同且无交集（t=%s g=%s），已按排序序号一一配对；"
                                    "若标识并非 rank 语义（如 proc<PID>）该配对通常成立，请在报告中说明" % (step, ranks_t, ranks_g))
        else:
            full["warnings"].append("step %d 两侧 rank 数不一致且无同标识 rank，无法可靠配对: t=%s g=%s" % (step, ranks_t, ranks_g))
        if not rank_pairs:
            continue

        step_res = {}
        for r, r_t, r_g in rank_pairs:
            res, _, _ = analyze_pair(t_groups[(step, r_t)], g_groups[(step, r_g)], args, (step, r))
            step_res[str(r)] = res
        full["steps"][str(step)] = step_res

        md.append("\n# Step %d（ranks: %s）\n" % (step, [p[0] for p in rank_pairs]))
        md.append(md_pair_section_per_rank(step_res, args))

    if full["warnings"]:
        md.append("\n## 警告\n")
        for w in full["warnings"]:
            md.append("- %s" % w)

    json_path = args.json or os.path.join(args.out, "analysis.json")
    report_path = args.report or os.path.join(args.out, "analysis_summary.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full, f, ensure_ascii=False, default=str)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("JSON: %s" % json_path)
    print("报告: %s" % report_path)


def md_pair_section_per_rank(step_res, args):
    """多 rank：逐 rank 摘要 + 跨 rank 共识。"""
    if len(step_res) == 1:
        return md_pair_section(next(iter(step_res.values())), args)
    L = []
    A = L.append
    consensus = defaultdict(list)
    cons_cons = defaultdict(set)
    oi_cons = defaultdict(set)
    gap_cons = defaultdict(set)
    bc_cons = defaultdict(set)
    cens = {}
    for r, res in step_res.items():
        do = res["diff_overview"]
        fd = res["first_divergence"]
        A("## rank %s\n" % r)
        A("- nodes t=%d g=%d matched=%d | 显著差异槽位 %d | 标量差 %d | 首分歧(前向) `%s` (反向) `%s`" % (
            res["n_nodes_t"], res["n_nodes_g"], res["alignment"]["matched"], do["significant"],
            do["scalar_diffs"],
            (fd.get("forward") or {}).get("node"), (fd.get("backward") or {}).get("node")))
        pg = res["param_grad"]
        if pg["count"]:
            A("- gnorm 估计: target=%s golden=%s" % (fmt(pg["gnorm_est_t"]), fmt(pg["gnorm_est_g"])))
        cons = res.get("conservation") or {}
        if cons.get("checked"):
            A("- 分侧守恒检验: 检验 %d，分侧不一致 %d（其中一侧守恒/一侧违反 %d；详见 JSON）" % (
                cons["checked"], cons["n_flagged"], cons.get("n_hard", 0)))
        oi = res.get("operand_identity") or {}
        if oi.get("n_groups"):
            A("- 操作数身份命中不对称组: %d（高置信 %d；详见 JSON operand_identity.groups）" % (
                oi["n_groups"], oi.get("n_high", 0)))
        for e in (res["top_nre"] or [])[:10]:
            if e.get("family"):
                consensus[e["family"]].append(r)
        for row in (cons.get("flagged") or []):
            if row.get("cons_t") != row.get("cons_g"):
                cons_cons[row["node"]].add(r)
        for grp in (oi.get("groups") or []):
            if grp.get("confidence") == "high":
                oi_cons[(grp["family"], grp["path"], grp["hits_side"])].add(r)
        for g in (res.get("coverage_gap") or []):
            if not (g["bwd_captured_t"] and g["bwd_captured_g"] and g["bwd_matched"]):
                gap_cons[(g["param_node"], g["param"])].add(r)
        for row in (res.get("backward_created") or []):
            bc_cons[row["node"]].add(r)
        A("")
        A(md_anchor_brief(res, max_rows=3))
        A("")
        for _d in ("forward", "backward"):
            _c = (res.get("divergence_census") or {}).get(_d)
            if _c:
                _acc = cens.setdefault(_d, defaultdict(int))
                for _k in ("same_shape_slots", "rel0", "minor", "significant",
                           "one_side_zero", "ratio_ge_1e3", "nonfinite"):
                    _acc[_k] += _c.get(_k) or 0
    if cens:
        A("### 全窗口分歧普查（\"逐位一致/无数值后果\"类断言须引用本表；全部 rank 求和）\n")
        A("| 方向 | 同形状 output 槽位 | rel=0 | minor(<1e-3) | significant(≥1e-3) | 比值≥1e3× | 单侧地板外 | 非有限 |")
        A("|---|---|---|---|---|---|---|---|")
        for _d in ("forward", "backward"):
            _a = cens.get(_d) or {}
            A("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                _d, _a.get("same_shape_slots", 0), _a.get("rel0", 0), _a.get("minor", 0),
                _a.get("significant", 0), _a.get("ratio_ge_1e3", 0), _a.get("one_side_zero", 0),
                _a.get("nonfinite", 0)))
        A("")
    if cons_cons or oi_cons or gap_cons or bc_cons:
        A("### 锚点跨 rank 共识\n")
        for node, ranks in sorted(cons_cons.items(), key=lambda kv: -len(kv[1]))[:10]:
            A("- 分侧守恒不一致 `%s`: rank %s" % (node, sorted(ranks, key=str)))
        for node, ranks in sorted(bc_cons.items(), key=lambda kv: -len(kv[1]))[:10]:
            A("- 反向内产生分歧 `%s`: rank %s" % (node, sorted(ranks, key=str)))
        for (fam, path, side), ranks in sorted(oi_cons.items(), key=lambda kv: -len(kv[1]))[:10]:
            A("- 高置信操作数身份 %s `%s` [命中仅在 %s]: rank %s" % (fam, path, side, sorted(ranks, key=str)))
        for (pn, pm), ranks in sorted(gap_cons.items(), key=lambda kv: -len(kv[1]))[:10]:
            A("- 梯度覆盖缺口 `%s`（%s）: rank %s" % (pn, pm, sorted(ranks, key=str)))
        A("")
    A("### 跨 rank 共识（出现在多 rank Top 差异中的族）\n")
    for fam, ranks in sorted(consensus.items(), key=lambda kv: -len(kv[1]))[:15]:
        A("- %s: rank %s" % (fam, sorted(set(ranks))))
    A("\n---\n各 rank 明细见 JSON（analysis.json）。")
    return "\n".join(L)


if __name__ == "__main__":
    main()
