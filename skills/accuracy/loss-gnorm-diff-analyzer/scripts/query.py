#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
query.py — 原始 dump.json / stack.json / construct.json 抽查查询工具集。

目标：消灭「手写一次性查询脚本+调试」与「结构试探」两类固定成本。所有引值
输出自带出处（侧别标签、文件、节点名、order、槽位路径、全精度数值），可直接
抄入 evidence.json 的 evidence_log——查询输出即回查（HC-B3 by construction），
取数后不再单独做"引值回查"式重验。本工具只报告事实，不做对齐/因果判断
（判断纪律见 hard-constraints.md）。

用法（python3 -I 运行；<dump> 为 dump.json 路径）：各子命令的详细用法、
适用场景与已知限制以 references/dump-analysis-mechanism.md §3.6 为权威文档，
本处只留速览；`--help` 可用。退出码：0=正常；1=无匹配（多为节点名笔误）；
2=文件/参数错误。

  node      <dump> <节点模式...> [--slot 槽位] [--limit N] [--label 侧标签]
            按名取节点全部槽位全精度统计量（模式=精确/正则/子串，并集；
            槽位路径形如 input_args.0 / output.0 / parameters.bias）
  order     <dump> [--range A:B] [--grep 模式] [--limit N]
            按执行序列出节点名（dump.data 键序=执行序）
  count     <dump> [--grep 模式]   节点计数
  quad      <dump> --from-node <节点名> [--from-slot 槽位] [--from-file <dump>]
            [--values Max Min Mean Norm] [--all] [--tolerance 相对容差]
            [--limit N]
            四元组身份反查（机制 §3.6）：源槽位 (Max,Min,Mean,Norm) 在本
            dump output 槽位（--all=全部槽位）匹配，列出命中节点+order+槽位。
            **省略 --from-slot=批量模式**：源节点全部张量槽位逐槽一行汇总。
            默认逐位相等；--tolerance REL 分量相对容差（双 NaN 亦命中）；
            容差 0 时含 NaN 四元组恒无命中（IEEE 语义，如实反馈）。
            命中≠语义生产者，须按执行序/栈帧再核对。
  diff      <dumpT> <dumpG> <节点模式...> [--limit N]
            同名节点双侧逐槽位对照：两侧统计量全精度 + NRE
            （=|Norm(T)-Norm(G)|/|Norm(G)|）+ ΔMean + 逐位一致标记（≡）；
            单侧独有节点/槽位照列（order 双侧并排，调用号漂移直接可见）。
  first     <dumpT> <dumpG> [节点模式...] [--threshold 1e-3]
            [--direction forward|backward|both]
            首分歧扫描：同名匹配节点按执行序逐槽位，报告①每方向首个
            NRE≥阈值的同形状同 dtype 有限槽位（含该节点全槽位双侧对照+
            前置核查行——更早匹配节点数与最大 NRE，证明「无更早」）；
            ②结构首分歧（首个 dtype/shape 不一致——可能在漂移族同名列，
            须按 align 纪律甄别后再作证据）；③首个非有限槽位（advisory，
            不参与数值首分歧判定）。模式缺省=全部同名节点。扫描口径=全部
            槽位（比"仅输出槽"严），引用断言时按实际口径如实表述。
  gain      <dumpT> <dumpG> <节点模式...> [--direction forward|backward|both]
            [--threshold 1e-3] [--amp 1e3] [--limit 60]
            链增益剖面：模式匹配同名节点按执行序列出 r_in（输入侧最大
            NRE）/ r_out（输出侧最大 NRE）与增益 r_out/r_in，标记
            ★分歧制造（输入一致而输出分歧——本地制造指纹）与
            ★本地制造候选（输入输出均分歧）；尾部 r_out 分位、首个
            r_out≥--amp 节点、分歧制造/继承节点计数与继承增益中位。
            r_in<阈值按「输入一致」近似（严格逐位一致仍用 diff）。
  align     <dumpT> <dumpG> [--family 族名]... [--grep 模式] [--context K]
            [--limit N]
            跨侧族对齐（防对齐返工）：族=剥去调用号的节点名。无 --family：
            族计数漂移总表（Δ≠0 按 |Δ| 降序）；--family：序号配对
            （T 第 k ↔ G 第 k）对齐表+±K 邻域族名对照（邻域不一致=错位点）。
            分片/插步/捕获差导致调用号错位时先用本命令定位漂移点。
  survey    <dumpT> <dumpG> {backward|forward|only} [--top N] [--grep 模式]
            [--limit N]
            批量普查：backward/forward=同名节点对逐公共 output 槽位 Norm
            对照——桶计数（逐位一致/minor<1e-3/≥1e-3/非有限/单侧空或0/
            无公共槽位）+ top-N 按 NRE 降序（order 双侧、Norm 双侧）；
            only=单侧独有节点按族汇总（计数+该侧最大 output Norm）。
  ranks     <dirT> <dirG> <节点模式...> [--slot 槽位] [--top N] [--quads]
            多 rank 复核（HC-B8 工具化）：dirT/dirG 为含 rank*/dump.json
            的目录（按名序一一配对；目录直接含 dump.json=单 rank）。逐 rank
            逐槽位对照：汇总行（节点对数/分桶/maxNRE/最大悬殊比）+
            ⚠rank 集中显形 / ⚠rank 量级集中（≥1e3×）提示 + top-N 显著明细。
            --quads：逐 rank 打印模式匹配节点全量四元组明细（承重族逐 rank
            数值分列呈报的取数工具）。**patterns 必须传族名/节点名定向复核**
            ——逐 rank Top-N 表有截断，读 Top 表不能替代定向复核。
  params    <dumpT> <dumpG> [--top N] [--grep 模式] [--threshold 1e-3]
            参数槽全库普查：同名节点 parameters 全部参数槽 + 顶层张量叶子
            节点（param_grad 形态）跨侧对照，分桶：逐位一致/聚合差
            （Max-Min 恒等+Mean-Norm 分歧——§3.9 程序二 H-content/
            H-artifact 形态识别输入）/统计量全差/shape 差/单侧缺；输出
            NRE≥阈值计数与方向统计（T>G / T<G 槽数——随机归约伪影方向
            随机、系统性内容差方向一致）+聚合差 NRE 分位+top-N 全精度明细。
  stack     <stack.json> <节点名或模式> [--limit N]
            节点栈帧（文件:行号）。默认前 20 帧。
  construct <construct.json> <节点模式...> [--include-null] [--label 模式]
            construct 标签查询：默认 节点→标签（仅非空）；--label 反查
            标签→节点列表。
  census    <rank目录父路径> (--backward | --patterns <模式...>) [--slot 槽位]
            [--top N] [--label 标签]
            窗口级量词闭合（HC-C10② 工具化，枚举闭包口径）：--backward=
            枚举全部 rank 全部 .backward 节点 output 槽位；--patterns=全
            rank×匹配节点×全槽位（或 --slot 指定）。逐 rank 汇总+全局 Top；
            单 rank 的界外推为窗口级结论是事实错误——本命令是其机械防线。
"""

import argparse
import glob
import json
import os
import re
import sys

MAX_LINE = 600
STAT_KEYS = ("Max", "Min", "Mean", "Norm")


def die(msg, code=2):
    print("[query.py] 错误: %s" % msg, file=sys.stderr)
    sys.exit(code)


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except OSError as e:
        die("无法读取 %s: %s" % (path, e))
    except json.JSONDecodeError as e:
        die("JSON 解析失败 %s: %s" % (path, e))


def dump_data(path):
    """加载 dump.json，打印 meta 头行，返回 data 字典（保持键序）。"""
    raw = load_json(path)
    if not isinstance(raw, dict) or not isinstance(raw.get("data"), dict):
        die("%s 不是 statistics dump（缺 data 字典）" % path)
    data = raw["data"]
    print("# dump=%s task=%s level=%s framework=%s nodes=%d"
          % (path, raw.get("task"), raw.get("level"), raw.get("framework"),
             len(data)))
    return data


def is_tensor_leaf(v):
    """张量统计叶子：dict 且含 type 与任一统计键（Max/Min/Mean/Norm）。
    torch.Tensor 及其子类（DTensor 等）的统计槽均算；标量/slice/ProcessGroup
    等无统计键，不匹配。判定只看统计键是否存在，统计量值可为 null。"""
    return (isinstance(v, dict) and "type" in v
            and any(k in v for k in STAT_KEYS))


def iter_leaves(obj, path=()):
    """递归产出 (槽位路径, 叶子值)。路径成分以 . 连接，列表成分为下标。"""
    if isinstance(obj, dict) and "type" in obj:
        yield ".".join(path), obj
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from iter_leaves(v, path + (str(k),))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from iter_leaves(v, path + (str(i),))
    else:
        yield ".".join(path), obj


def resolve_slot(node, slot):
    """按点分路径取子结构；返回 (是否找到, 值)。列表成分用下标。"""
    cur = node
    for comp in slot.split("."):
        if isinstance(cur, dict):
            if comp in cur:
                cur = cur[comp]
            else:
                return False, None
        elif isinstance(cur, list):
            try:
                cur = cur[int(comp)]
            except (ValueError, IndexError):
                return False, None
        else:
            return False, None
    return True, cur


def fmt_val(x):
    if isinstance(x, float):
        return repr(x)
    return json.dumps(x, ensure_ascii=False)


def render_leaf(path, leaf, indent="   "):
    if is_tensor_leaf(leaf):
        parts = [str(leaf.get("type")), "dtype=%s" % leaf.get("dtype"),
                 "shape=%s" % json.dumps(leaf.get("shape"))]
        for k in STAT_KEYS:
            parts.append("%s=%s" % (k, fmt_val(leaf.get(k))))
        if "requires_grad" in leaf:
            parts.append("requires_grad=%s"
                         % json.dumps(leaf["requires_grad"]))
        for k, v in leaf.items():
            if k not in ("type", "dtype", "shape", "Max", "Min", "Mean",
                         "Norm", "requires_grad"):
                parts.append("%s=%s" % (k, json.dumps(v, ensure_ascii=False)))
        line = "%s : %s" % (path, " ".join(parts))
    elif isinstance(leaf, dict) and "value" in leaf:
        line = "%s : %s value=%s" % (path, leaf.get("type"),
                                     json.dumps(leaf["value"],
                                                ensure_ascii=False))
    elif isinstance(leaf, dict):
        line = "%s : %s" % (path, json.dumps(leaf, ensure_ascii=False))
    else:
        line = "%s : %s" % (path, fmt_val(leaf))
    if len(line) > MAX_LINE:
        line = line[:MAX_LINE] + " …(截断)"
    return indent + line


def match_names(order_map, patterns):
    """精确名优先，其次正则 search，编译失败退化为子串。返回按 order 排序名。"""
    hits = set()
    for p in patterns:
        if p in order_map:
            hits.add(p)
            continue
        rx = _rx_or_sub(p)
        for name in order_map:
            if (rx is not None and rx.search(name)) or \
               (rx is None and p in name):
                hits.add(name)
    return sorted(hits, key=lambda n: order_map[n])


def order_index(data):
    return {n: i for i, n in enumerate(data)}


def parse_quad_value(s):
    if s == "null":
        return None
    if s == "True":
        return True
    if s == "False":
        return False
    try:
        return float(s)
    except ValueError:
        die("四元组分量 %r 不是数值/null/True/False" % s)


# ---------------------------------------------------------------- 子命令

def cmd_node(args):
    data = dump_data(args.dump)
    order_map = order_index(data)
    matches = match_names(order_map, args.patterns)
    if not matches:
        die("无匹配节点（精确/正则/子串均未命中）: %s" % " ".join(args.patterns), 1)
    if len(matches) > args.limit:
        print("# 命中 %d 个节点，仅显示前 %d（--limit 调整；可用更精确模式）"
              % (len(matches), args.limit))
    tag = "[%s] " % args.label if args.label else ""
    for name in matches[:args.limit]:
        node = data[name]
        if not isinstance(node, dict):
            print("== %s%s  order=%d  [非字典节点数据]" % (tag, name,
                                                          order_map[name]))
            continue
        print("== %s%s  order=%d" % (tag, name, order_map[name]))
        if args.slot:
            ok, val = resolve_slot(node, args.slot)
            if ok:
                if isinstance(val, dict) and "type" not in val:
                    for pth, lf in iter_leaves(val, (args.slot,)):
                        print(render_leaf(pth, lf))
                else:
                    print(render_leaf(args.slot, val))
            else:
                print("   [槽位 %s 不存在；该节点可用槽位：]" % args.slot)
                for i, (pth, _lf) in enumerate(iter_leaves(node)):
                    print("     %s" % pth)
                    if i >= 39:
                        print("     …(仅列前 40)")
                        break
        else:
            for pth, lf in iter_leaves(node):
                print(render_leaf(pth, lf))


def cmd_order(args):
    data = dump_data(args.dump)
    names = list(data)
    if args.range:
        a, _, b = args.range.partition(":")
        try:
            i = int(a) if a else None
            j = int(b) if b else None
        except ValueError:
            die("--range 须为 A:B 形式（整数，可单侧省略）")
        sel = names[i:j]
        for k, n in enumerate(sel, i if i is not None else 0):
            print("%7d  %s" % (k, n))
        return
    if args.grep:
        rx = _rx_or_sub(args.grep)
        hits = [n for n in names
                if (rx is not None and rx.search(n)) or
                (rx is None and args.grep in n)]
        print("# grep 命中 %d / %d" % (len(hits), len(names)))
        for n in hits[:args.limit]:
            print("%7d  %s" % (names.index(n), n))
        if len(hits) > args.limit:
            print("# …仅显示前 %d（--limit 调整）" % args.limit)
        return
    print("# 未给 --range/--grep：总数 %d；前 15：" % len(names))
    for i, n in enumerate(names[:15]):
        print("%7d  %s" % (i, n))
    print("# …后 5：")
    for i in range(max(0, len(names) - 5), len(names)):
        print("%7d  %s" % (i, names[i]))
    print("# 提示：--range A:B 看执行序片段；--grep 按名过滤。")


def cmd_count(args):
    data = dump_data(args.dump)
    if args.grep:
        rx = _rx_or_sub(args.grep)
        n = sum(1 for name in data
                if (rx is not None and rx.search(name)) or
                (rx is None and args.grep in name))
        print("count(grep=%s) = %d / %d" % (args.grep, n, len(data)))
    else:
        print("count = %d" % len(data))


def _quad_of(leaf):
    return tuple(leaf.get(k) for k in STAT_KEYS)


def _quad_matches(qa, qb, tol):
    """逐位（tol=0）或相对容差匹配；容差模式下双 NaN 视为匹配。"""
    if tol == 0:
        # 元素级 ==：NaN 与自身不等；元组 == 会走 identity 短路把同一 NaN 对象判等
        return all(a == b for a, b in zip(qa, qb))
    for a, b in zip(qa, qb):
        if a == b:
            continue
        a_nan = isinstance(a, float) and a != a
        b_nan = isinstance(b, float) and b != b
        if a_nan and b_nan:
            continue
        if a_nan != b_nan:
            return False
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if abs(a - b) > tol * max(abs(a), abs(b), 1e-300):
                return False
        else:
            return False
    return True


def _quad_hits(quad, by_key, index, tol):
    if tol == 0:
        return sorted(by_key.get(quad, []))
    return sorted(e[1:] for e in index if _quad_matches(quad, e[0], tol))


def cmd_quad(args):
    data = dump_data(args.dump)
    order_map = order_index(data)
    tol = args.tolerance or 0.0
    # 目标索引：output 槽位（--all 时全部槽位）
    by_key, index = {}, []
    for name, node in data.items():
        if not isinstance(node, dict):
            continue
        for pth, lf in iter_leaves(node):
            if not args.all and not (pth == "output" or
                                     pth.startswith("output.")):
                continue
            if not is_tensor_leaf(lf):
                continue
            key = _quad_of(lf)
            if tol == 0:
                by_key.setdefault(key, []).append((order_map[name], name, pth))
            else:
                index.append((key, order_map[name], name, pth))
    if args.values:
        sources = [(tuple(parse_quad_value(s) for s in args.values),
                    "命令行 --values", "-")]
        batch = False
    else:
        if not args.from_node:
            die("须给 --from-node（省略 --from-slot=批量模式）或 --values")
        if args.from_file:
            src_data = dump_data(args.from_file)
        else:
            src_data = data
        node = src_data.get(args.from_node)
        if not isinstance(node, dict):
            die("源节点不存在: %s" % args.from_node)
        if args.from_slot:
            ok, leaf = resolve_slot(node, args.from_slot)
            if not ok or not is_tensor_leaf(leaf):
                die("源槽位 %s/%s 不是张量统计叶子" % (args.from_node,
                                                       args.from_slot))
            sources = [(_quad_of(leaf),
                        "%s / %s" % (args.from_file or args.dump,
                                     args.from_node),
                        args.from_slot)]
            batch = False
        else:
            sources = [(_quad_of(lf), "", pth)
                       for pth, lf in iter_leaves(node)
                       if is_tensor_leaf(lf)]
            if not sources:
                die("源节点 %s 无张量统计槽位" % args.from_node)
            print("# 批量反查 %d 槽位（源=%s / %s；匹配=%s）"
                  % (len(sources), args.from_file or args.dump,
                     args.from_node,
                     ("容差 %s" % tol) if tol else "逐位"))
            batch = True
    total_hits = 0
    hit_slots = 0
    for quad, src_desc, slot_desc in sources:
        hits = _quad_hits(quad, by_key, index, tol)
        total_hits += len(hits)
        if hits:
            hit_slots += 1
        if batch:
            if not hits:
                print("   [%s] 四元组=(%s) → 无命中"
                      % (slot_desc, ",".join(fmt_val(q) for q in quad)))
                continue
            inline = "; ".join("%d %s [%s]" % h for h in hits[:args.limit])
            more = "" if len(hits) <= args.limit else \
                " …(共 %d 命中)" % len(hits)
            print("   [%s] 四元组=(%s) → 命中 %d：%s%s"
                  % (slot_desc, ",".join(fmt_val(q) for q in quad),
                     len(hits), inline, more))
            continue
        print("# 反查四元组 (Max,Min,Mean,Norm)=(%s)  来源=%s  槽位=%s"
              % (",".join(fmt_val(q) for q in quad), src_desc, slot_desc))
        if any(isinstance(q, float) and q != q for q in quad):
            print("# 注意：四元组含 NaN——容差 0 下 IEEE NaN!=NaN，逐位反查恒无"
                  "命中（如实反馈；可用 --tolerance）")
        if not hits:
            print("无逐位命中")
            sys.exit(1)
        print("# 命中 %d 处（%s）：" % (len(hits),
              "output 槽位" if not args.all else "全部槽位"))
        for order, name, pth in hits[:args.limit]:
            print("%7d  %s  [%s]" % (order, name, pth))
        if len(hits) > args.limit:
            print("# …仅显示前 %d（--limit 调整）" % args.limit)
        print("# 提示（机制 §3.6）：命中≠语义生产者——须按执行序/栈帧核对消费点"
              "与语义归属；单侧命中/对侧无命中是不对称证据。")
        return
    if batch:
        print("# 批量完成：有命中槽位 %d / %d，命中总数 %d"
              % (hit_slots, len(sources), total_hits))
        print("# 提示（机制 §3.6）：命中≠语义生产者；单侧命中/对侧无命中是"
              "不对称证据。")


def _pair_lines(path, tl, gl):
    """diff 用：打印一对槽位（T/G 前缀）+ 一致性/NRE/ΔMean 摘要行。"""
    out = []
    if tl is None and gl is None:
        return out
    for tag, lf in (("T", tl), ("G", gl)):
        if lf is None:
            out.append("   %s [缺该槽位]" % tag)
        else:
            body = render_leaf(path, lf, indent="").strip()
            out.append("   %s %s" % (tag, body))
    if is_tensor_leaf(tl) and is_tensor_leaf(gl):
        ident = all(tl.get(k) == gl.get(k) for k in STAT_KEYS)
        tn, gn = tl.get("Norm"), gl.get("Norm")
        if isinstance(tn, (int, float)) and isinstance(gn, (int, float)) \
                and gn not in (0, None):
            nre = abs(tn - gn) / abs(gn)
            nre_s = repr(nre)
        else:
            nre_s = "n/a(Norm空或0)"
        tm, gm = tl.get("Mean"), gl.get("Mean")
        if isinstance(tm, (int, float)) and isinstance(gm, (int, float)):
            dm = repr(tm - gm)
        else:
            dm = "n/a"
        mark = "≡逐位一致" if ident else "≠"
        out.append("      %s  NRE=%s  ΔMean=%s" % (mark, nre_s, dm))
    return out


def cmd_diff(args):
    t_data = dump_data(args.dump_t)
    g_data = dump_data(args.dump_g)
    t_order = order_index(t_data)
    g_order = order_index(g_data)
    t_names = set(match_names(t_order, args.patterns))
    g_names = set(match_names(g_order, args.patterns))
    if not t_names and not g_names:
        die("两侧均无匹配节点: %s" % " ".join(args.patterns), 1)
    paired = sorted(t_names & g_names, key=lambda n: t_order[n])
    t_only = sorted(t_names - g_names, key=lambda n: t_order[n])
    g_only = sorted(g_names - t_names, key=lambda n: g_order[n])
    if len(paired) > args.limit:
        print("# 同名配对 %d 个，仅详析前 %d（--limit 调整）"
              % (len(paired), args.limit))
    for name in paired[:args.limit]:
        tn, gn = t_data[name], g_data[name]
        print("== %s  orderT=%d orderG=%s" % (name, t_order[name],
                                              g_order.get(name)))
        t_slots = dict(iter_leaves(tn)) if isinstance(tn, dict) else {}
        g_slots = dict(iter_leaves(gn)) if isinstance(gn, dict) else {}
        for pth in sorted(set(t_slots) | set(g_slots)):
            for ln in _pair_lines(pth, t_slots.get(pth), g_slots.get(pth)):
                print(ln)
    if t_only:
        print("# 仅 T 侧存在的匹配节点（%d）：" % len(t_only))
        for n in t_only[:args.limit]:
            print("   orderT=%d  %s" % (t_order[n], n))
    if g_only:
        print("# 仅 G 侧存在的匹配节点（%d）：" % len(g_only))
        for n in g_only[:args.limit]:
            print("   orderG=%d  %s" % (g_order[n], n))


def cmd_stack(args):
    raw = load_json(args.stack)
    if not isinstance(raw, dict):
        die("%s 不是 stack.json（顶层应为字典）" % args.stack)
    frames_map = {}
    for _k, v in raw.items():
        try:
            names, frames = v[0], v[1]
        except (IndexError, TypeError):
            continue
        for n in names or []:
            frames_map.setdefault(n, frames or [])
    pat = args.pattern
    matches = []
    if pat in frames_map:
        matches = [pat]
    else:
        rx = _rx_or_sub(pat)
        for n in frames_map:
            if (rx is not None and rx.search(n)) or (rx is None and pat in n):
                matches.append(n)
    if not matches:
        die("stack.json 无匹配节点: %s" % pat, 1)
    print("# stack=%s 命中 %d 节点，显示前 %d" % (args.stack, len(matches),
                                                 min(len(matches), 10)))
    for name in matches[:10]:
        frames = frames_map[name]
        print("== %s  (%d 帧)" % (name, len(frames)))
        for fr in frames[:args.limit]:
            print("   %s" % fr)
        if len(frames) > args.limit:
            print("   …(共 %d 帧，--limit 调整)" % len(frames))


def cmd_construct(args):
    raw = load_json(args.construct)
    if not isinstance(raw, dict):
        die("%s 不是 construct.json（顶层应为字典）" % args.construct)
    if not raw:
        print("construct.json 为空字典（本案无 construct 标签，如实）")
        sys.exit(1)
    if args.label:
        rx = _rx_or_sub(args.label)
        hits = []
        for node, lab in raw.items():
            if lab is None:
                continue
            if (rx is not None and rx.search(str(lab))) or \
               (rx is None and args.label in str(lab)):
                hits.append((node, lab))
        if not hits:
            die("无匹配标签: %s" % args.label, 1)
        print("# construct=%s 标签反查『%s』命中 %d：" % (args.construct,
                                                          args.label, len(hits)))
        for node, lab in hits[:args.limit]:
            print("   %s -> %s" % (node, lab))
        return
    order_map = {n: i for i, n in enumerate(raw)}
    matches = match_names(order_map, args.patterns)
    shown = 0
    for name in matches:
        lab = raw[name]
        if lab is None and not args.include_null:
            continue
        print("   %s -> %s" % (name, lab))
        shown += 1
        if shown >= args.limit:
            break
    if shown == 0:
        print("无匹配的非空标签（--include-null 查看空标签）")
        sys.exit(1)


FAM_API = re.compile(r"^(.*)\.(\d+)\.(forward|backward)$")
FAM_MODULE = re.compile(r"^(Module\..*\.(?:forward|backward))\.\d+$")


def family_of(name):
    """族名=剥去调用号：API `NS.op.N.forward`→`NS.op.forward`；
    Module `Module.p.forward.N`→`Module.p.forward`；其余原样。"""
    m = FAM_API.match(name)
    if m:
        return "%s.%s" % (m.group(1), m.group(3))
    m = FAM_MODULE.match(name)
    if m:
        return m.group(1)
    return name


def _rx_or_sub(pattern):
    try:
        return re.compile(pattern)
    except re.error:
        return None


def _grep_keep(grep):
    """--grep 过滤谓词：正则 search 优先，编译失败退化为子串；空=全保留。"""
    if not grep:
        return lambda _name: True
    rx = _rx_or_sub(grep)
    if rx is not None:
        return lambda name: rx.search(name) is not None
    return lambda name: grep in name


def _neighbor_fams(names, idx, k):
    lo = max(0, idx - k)
    hi = min(len(names), idx + k + 1)
    return [family_of(names[i]) for i in range(lo, hi)]


def cmd_align(args):
    t_data = dump_data(args.dump_t)
    g_data = dump_data(args.dump_g)
    t_order = order_index(t_data)
    g_order = order_index(g_data)
    fam_t, fam_g = {}, {}
    for n in t_data:
        fam_t.setdefault(family_of(n), []).append(n)
    for n in g_data:
        fam_g.setdefault(family_of(n), []).append(n)
    exact = set(t_data) & set(g_data)
    print("# 对齐总览：T 节点 %d / G 节点 %d / 精确同名 %d / 族数 T %d vs G %d"
          % (len(t_data), len(g_data), len(exact), len(fam_t), len(fam_g)))
    if not args.family:
        rx = _rx_or_sub(args.grep) if args.grep else None
        rows = []
        for f in set(fam_t) | set(fam_g):
            if rx is not None and not rx.search(f):
                continue
            if rx is None and args.grep and args.grep not in f:
                continue
            ct, cg = len(fam_t.get(f, [])), len(fam_g.get(f, []))
            if ct != cg:
                rows.append((abs(ct - cg), ct + cg, f, ct, cg))
        rows.sort(key=lambda r: (-r[0], -r[1], r[2]))
        print("# 计数漂移族 %d 个（|Δ| 降序；--family <族名> 深入对齐）"
              % len(rows))
        for _, _, f, ct, cg in rows[:args.limit]:
            print("   Δ=%+d  T=%-6d G=%-6d %s" % (ct - cg, ct, cg, f))
        if len(rows) > args.limit:
            print("# …仅显示前 %d（--limit 调整）" % args.limit)
        return
    names_t, names_g = list(t_data), list(g_data)
    for fam in args.family:
        t_inst = sorted(fam_t.get(fam, []), key=lambda n: t_order[n])
        g_inst = sorted(fam_g.get(fam, []), key=lambda n: g_order[n])
        print("== 族 %s：T %d 实例 / G %d 实例（序号配对：T 第 k ↔ G 第 k）"
              % (fam, len(t_inst), len(g_inst)))
        if not t_inst and not g_inst:
            print("   [两侧均无该族实例（核对族名拼写）]")
            continue
        kmax = max(len(t_inst), len(g_inst))
        detail_left = 3
        for k in range(kmax):
            tn = t_inst[k] if k < len(t_inst) else None
            gn = g_inst[k] if k < len(g_inst) else None
            flag = ""
            t_ne = g_ne = None
            if args.context and tn and gn:
                t_ne = _neighbor_fams(names_t, t_order[tn], args.context)
                g_ne = _neighbor_fams(names_g, g_order[gn], args.context)
                if t_ne != g_ne:
                    flag = "  ⚠邻域不一致(错位嫌疑)"
            if (tn is None) != (gn is None):
                flag = "  [单侧余量]"
            print("   k=%-5d T %-52s orderT=%-7s | G %-52s orderG=%-7s%s"
                  % (k, tn or "-", t_order.get(tn, "-") if tn else "-",
                     gn or "-", g_order.get(gn, "-") if gn else "-", flag))
            if flag.startswith("  ⚠") and detail_left > 0:
                detail_left -= 1
                print("      T 邻域: %s" % " | ".join(t_ne))
                print("      G 邻域: %s" % " | ".join(g_ne))
            if k + 1 >= args.limit:
                print("   …(共 %d 行，--limit 调整)" % kmax)
                break


def _output_slots(node):
    out = []
    if isinstance(node, dict):
        for pth, lf in iter_leaves(node):
            if pth == "output" or pth.startswith("output."):
                if is_tensor_leaf(lf):
                    out.append((pth, lf))
    return out


def _norm_num(v):
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, float) and v != v:
        return None
    if isinstance(v, (int, float)):
        return v
    return None


def _finite_leaf(lf):
    for k in STAT_KEYS:
        v = lf.get(k)
        if isinstance(v, float) and v != v:
            return False
    return True


def cmd_survey(args):
    t_data = dump_data(args.dump_t)
    g_data = dump_data(args.dump_g)
    t_order = order_index(t_data)
    g_order = order_index(g_data)
    keep = _grep_keep(args.grep)

    if args.mode == "only":
        fams = {}
        for side, data in (("T", t_data), ("G", g_data)):
            for n in data:
                if n in (t_data if side == "G" else g_data):
                    continue
                if not keep(n):
                    continue
                f = family_of(n)
                d = fams.setdefault(f, {"T": 0, "G": 0, "nT": 0.0, "nG": 0.0})
                d[side] += 1
                for _pth, lf in _output_slots(data[n]):
                    v = _norm_num(lf.get("Norm"))
                    if v is not None and abs(v) > d["n%s" % side]:
                        d["n%s" % side] = abs(v)
        rows = sorted(fams.items(),
                      key=lambda kv: -(kv[1]["T"] + kv[1]["G"]))
        t_only = sum(d["T"] for d in fams.values())
        g_only = sum(d["G"] for d in fams.values())
        print("# 单侧独有节点：T %d / G %d，涉及族 %d（按总数降序）"
              % (t_only, g_only, len(fams)))
        for f, d in rows[:args.limit]:
            print("   T=%-5d (maxNorm=%-12.4g)  G=%-5d (maxNorm=%-12.4g)  %s"
                  % (d["T"], d["nT"], d["G"], d["nG"], f))
        if len(rows) > args.limit:
            print("# …仅显示前 %d（--limit 调整）" % args.limit)
        return

    phase = args.mode

    def is_phase(name):
        if phase == "backward":
            return name.endswith(".backward") or ".backward." in name
        return name.endswith(".forward") or ".forward." in name

    buckets = {"rel0": 0, "minor": 0, "significant": 0, "nonfinite": 0,
               "one_side_zero": 0, "no_common_slot": 0}
    rows = []
    pairs = 0
    for name in t_data:
        if name not in g_data or not is_phase(name) or not keep(name):
            continue
        pairs += 1
        t_slots = dict(_output_slots(t_data[name]))
        g_slots = dict(_output_slots(g_data[name]))
        common = sorted(set(t_slots) & set(g_slots))
        if not common:
            buckets["no_common_slot"] += 1
            continue
        for p in common:
            tl, gl = t_slots[p], g_slots[p]
            nt, ng = _norm_num(tl.get("Norm")), _norm_num(gl.get("Norm"))
            if not (_finite_leaf(tl) and _finite_leaf(gl)):
                buckets["nonfinite"] += 1
                rows.append((2.0, name, p, nt, ng, "非有限"))
                continue
            if nt is None or ng is None or (nt == 0 and ng == 0):
                if (nt in (None, 0)) != (ng in (None, 0)):
                    buckets["one_side_zero"] += 1
                    rows.append((1.5, name, p, nt, ng, "单侧空或0"))
                else:
                    buckets["rel0"] += 1
                continue
            if all(tl.get(k) == gl.get(k) for k in STAT_KEYS):
                buckets["rel0"] += 1
                continue
            nre = abs(nt - ng) / abs(ng) if ng else float("inf")
            if nre < 1e-3:
                buckets["minor"] += 1
                continue
            buckets["significant"] += 1
            rows.append((nre, name, p, nt, ng, "≥1e-3"))
    print("# %s 同名节点对 %d；公共 output 槽位桶计数：逐位一致=%d "
          "minor(<1e-3)=%d significant(≥1e-3)=%d 非有限=%d 单侧空或0=%d "
          "无公共槽位节点=%d"
          % (phase, pairs, buckets["rel0"], buckets["minor"],
             buckets["significant"], buckets["nonfinite"],
             buckets["one_side_zero"], buckets["no_common_slot"]))
    rows.sort(key=lambda r: -r[0])
    for nre, name, p, nt, ng, tag in rows[:args.top]:
        print("   NRE=%-11.5g orderT=%-7d orderG=%-7d NormT=%-16r "
              "NormG=%-16r %s [%s]"
              % (nre, t_order[name], g_order.get(name, -1), nt, ng, name, p))
    if len(rows) > args.top:
        print("# …top %d / 显著+异常槽位共 %d（--top 调整）"
              % (args.top, len(rows)))


# ----------------------------------------------------------------

def _discover_rank_dumps(base):
    """发现多 rank dump：base/rank*/dump.json（按名排序，rank 目录序=rank
    序）；base 自身含 dump.json 时按单 rank 处理；base 下无直接 rank 目录时
    向下搜一层 step*/*rank*/dump.json（step 层级布局自动发现）。
    返回 [(rank 标签, 路径)]。"""
    cands = []
    if os.path.isdir(base):
        for d in sorted(glob.glob(os.path.join(base, "rank*"))):
            f = os.path.join(d, "dump.json")
            if os.path.isfile(f):
                cands.append((os.path.basename(d), f))
        if not cands:
            f = os.path.join(base, "dump.json")
            if os.path.isfile(f):
                cands.append(("单rank", f))
        if not cands:
            # base 为 case/dump 根（step 目录在其下一层）时自动下探
            deeper = []
            for sd in sorted(glob.glob(os.path.join(base, "*"))):
                if not os.path.isdir(sd):
                    continue
                sub = []
                for d in sorted(glob.glob(os.path.join(sd, "rank*"))):
                    f = os.path.join(d, "dump.json")
                    if os.path.isfile(f):
                        sub.append((os.path.basename(d), f))
                if sub:
                    deeper.append((os.path.basename(sd), sub))
            if len(deeper) == 1:
                step_name, sub = deeper[0]
                print("# ranks: %s 下无直接 rank*/dump.json，自动下探到 %s/"
                      "（%d rank）" % (base, step_name, len(sub)))
                cands = sub
            elif len(deeper) > 1:
                die("目录 %s 下有 %d 个含 rank*/dump.json 的子目录（%s）——"
                    "请传单一 step 层目录（多 step 须按 step 分组分别复核）"
                    % (base, len(deeper), "/".join(s for s, _ in deeper[:4])))
    if not cands:
        die("目录 %s 下未发现 rank*/dump.json、dump.json 或 */rank*/dump.json"
            "（多 rank 复核需两侧同为 rank 目录布局）" % base)
    return cands


def _leaf_quad_ident(tl, gl):
    return all(tl.get(k) == gl.get(k) for k in STAT_KEYS)


def cmd_census(args):
    """census：多 rank 单侧窗口级全量普查——一次调用枚举全部 rank，
    输出窗口级量词的枚举闭包口径（HC-C10② by construction）。
    模式一 --backward：枚举全部 rank 全部 .backward 节点的 output 槽位，
    报告逐 rank 最大发射 Norm 与全局 Top——替代"窗内反向发射量级界"的
    逐 rank 手工复核（单 rank 界外推为窗口级结论是事实错误——本命令是其机械防线）。
    模式二 --patterns：按模式匹配节点（默认 output.0 槽位或 --slot 指定），
    逐 rank 列最大 Norm 槽位——替代"全 N 槽 × 全 rank"的逐 rank 交互取数。"""
    ranks = _discover_rank_dumps(args.base)
    label = args.label or args.base
    print("# census: base=%s（%d rank）%s" % (label, len(ranks),
          "模式=backward 全枚举" if args.backward else
          "模式=patterns %s slot=%s" % (list(args.patterns), args.slot)))
    rows = []   # (norm, rank, name, order, slot)
    per_rank = []
    for rl, path in ranks:
        data = load_json(path).get("data", {})
        order = {n: i for i, n in enumerate(data)}
        if args.backward:
            names = [n for n in data if ".backward" in n]
        else:
            if not args.patterns:
                die("census：--backward 与 --patterns 二选一")
            names = sorted(match_names(order, list(args.patterns)),
                           key=lambda n: order[n])
        r_max, r_cnt = 0.0, 0
        for name in names:
            node = data.get(name)
            if not isinstance(node, dict):
                continue
            if args.slot:
                ok, v = resolve_slot(node, args.slot)
                slots = {args.slot: v} if ok else {}
            else:
                slots = dict(iter_leaves(node))
            for pth, lf in slots.items():
                if not is_tensor_leaf(lf):
                    continue
                nv = _norm_num(lf.get("Norm"))
                if nv is None or not _finite_leaf(lf):
                    continue
                r_cnt += 1
                rows.append((nv, rl, name, order[name], pth))
                r_max = max(r_max, nv)
        per_rank.append((rl, r_cnt, r_max))
    print("# 逐 rank 汇总（枚举闭包口径：全 rank×全匹配节点×全张量槽位）：")
    for rl, cnt, mx in per_rank:
        print("   %-10s 槽位数=%-7d maxNorm=%-12.6g" % (rl, cnt, mx))
    rows.sort(key=lambda r: -r[0])
    n = max(1, args.top)
    print("# 全局 Top-%d（跨 rank，含 order 与槽位路径；全精度即查即回查）：" % n)
    for nv, rl, name, od, pth in rows[:n]:
        print("   Norm=%-14.8g rank=%-10s %s (order=%d) %s"
              % (nv, rl, name, od, pth))
    if not rows:
        print("# （无张量槽位命中——检查模式/槽位口径）")


def cmd_ranks(args):
    t_ranks = _discover_rank_dumps(args.dir_t)
    g_ranks = _discover_rank_dumps(args.dir_g)
    print("# ranks: T=%s（%d rank）/ G=%s（%d rank）；路径=<dir>/<rank>/dump.json"
          % (args.dir_t, len(t_ranks), args.dir_g, len(g_ranks)))
    if len(t_ranks) != len(g_ranks):
        print("# ⚠两侧 rank 数不等：按名序 zip 配对，多出侧忽略（如实呈报）")
    summary = []
    sig_rows = []   # (nre, ratio, rank, name, slot, nt, ng)
    quad_rows = []  # --quads: (rank, name, slot, tl, gl)
    for (rl_t, pt), (rl_g, pg) in zip(t_ranks, g_ranks):
        t_data = load_json(pt).get("data", {})
        g_data = load_json(pg).get("data", {})
        t_order = {n: i for i, n in enumerate(t_data)}
        t_names = set(match_names(t_order, args.patterns))
        g_names = set(match_names({n: i for i, n in enumerate(g_data)},
                                  args.patterns))
        paired = t_names & g_names
        b = {"rel0": 0, "minor": 0, "sig": 0, "nonfinite": 0, "zero": 0}
        max_nre, max_ratio = 0.0, 1.0
        for name in sorted(paired, key=lambda n: t_order[n]):
            tn, gn = t_data[name], g_data[name]
            if not (isinstance(tn, dict) and isinstance(gn, dict)):
                continue
            if args.slot:
                ok_t, v_t = resolve_slot(tn, args.slot)
                ok_g, v_g = resolve_slot(gn, args.slot)
                t_slots = {args.slot: v_t} if ok_t else {}
                g_slots = {args.slot: v_g} if ok_g else {}
            else:
                t_slots = dict(iter_leaves(tn))
                g_slots = dict(iter_leaves(gn))
            for pth in sorted(set(t_slots) | set(g_slots)):
                tl, gl = t_slots.get(pth), g_slots.get(pth)
                if args.quads:
                    quad_rows.append((rl_t, name, pth, tl, gl))
                if not (is_tensor_leaf(tl) and is_tensor_leaf(gl)):
                    b["zero"] += 1
                    continue
                if _leaf_quad_ident(tl, gl):
                    b["rel0"] += 1
                    continue
                if not (_finite_leaf(tl) and _finite_leaf(gl)):
                    b["nonfinite"] += 1
                    continue
                nt, ng = _norm_num(tl.get("Norm")), _norm_num(gl.get("Norm"))
                if nt is None or ng is None or (nt == 0 and ng == 0):
                    b["zero"] += 1
                    continue
                nre = abs(nt - ng) / abs(ng) if ng else float("inf")
                ratio = abs(nt) / abs(ng) if ng else float("inf")
                r_disp = max(ratio, 1.0 / ratio) if ratio not in (0,
                                                                  float("inf")) \
                    else float("inf")
                if nre < 1e-3:
                    b["minor"] += 1
                    continue
                b["sig"] += 1
                sig_rows.append((nre, r_disp, rl_t, name, pth, nt, ng))
                max_nre = max(max_nre, nre)
                max_ratio = max(max_ratio, r_disp)
        summary.append((rl_t, len(paired), b, max_nre, max_ratio))
    print("# 逐 rank 汇总（槽位对照分桶；NRE=|NormT-NormG|/|NormG|，"
          "悬殊比=max(NormT/NormG, 倒数)）：")
    for rl, npairs, b, mn, mr in summary:
        print("   %-8s 节点对=%-6d 逐位一致=%-7d minor=%-6d 显著=%-6d "
              "非有限=%-4d 单侧空0=%-5d maxNRE=%-10.4g max悬殊=%-10.4g"
              % (rl, npairs, b["rel0"], b["minor"], b["sig"], b["nonfinite"],
                 b["zero"], mn, mr))
    sig_ranks = [rl for rl, _n, b, _m, _r in summary if b["sig"] > 0]
    calm_ranks = [rl for rl, _n, b, _m, _r in summary if b["sig"] == 0]
    if sig_ranks and calm_ranks:
        print("# ⚠rank 集中显形：显著分歧仅在 rank [%s] 出现，rank [%s] 逐 rank "
              "一致——单 rank 一致≠窗口级一致（HC-B8：一致断言须注明 rank 有效域）"
              % (",".join(sig_ranks), ",".join(calm_ranks)))
    if sig_ranks and not calm_ranks:
        nres = [s[3] for s in summary if s[3] > 0]
        if nres and max(nres) / min(nres) >= 1e3:
            top_rl = max(summary, key=lambda s: s[3])[0]
            print("# ⚠rank 量级集中：全部 rank 有显著分歧但跨 rank maxNRE 悬殊"
                  "（%g → %g，≥1e3×，最极端 rank=%s）——悬殊 rank 须分列数值"
                  "呈报，不得以单一 rank 代表全分布" % (min(nres), max(nres),
                                                         top_rl))
        else:
            print("# 全部 rank 均有显著分歧（无集中性）")
    if not sig_ranks:
        print("# 全部 rank 逐槽位一致（在匹配模式与槽位范围内）")
    if sig_rows:
        sig_rows.sort(key=lambda r: -r[0])
        print("# 显著明细 top %d（NRE 降序）：" % args.top)
        for nre, rd, rl, name, pth, nt, ng in sig_rows[:args.top]:
            print("   NRE=%-10.4g 悬殊=%-10.4g %-8s %s [%s] NormT=%r NormG=%r"
                  % (nre, rd, rl, name, pth, nt, ng))
        if len(sig_rows) > args.top:
            print("# …top %d / 显著槽位共 %d（--top 调整）" % (args.top,
                                                              len(sig_rows)))
    if args.quads and quad_rows:
        print("# 逐 rank 四元组明细（模式定向全量，非仅显著；每 rank 前 %d 行，"
              "NRE=—表示不可比：dtype/shape 差、非有限或无统计）：" % args.top)
        shown = {}
        for rl, name, pth, tl, gl in quad_rows:
            shown[rl] = shown.get(rl, 0)
            if shown[rl] >= args.top:
                continue
            shown[rl] += 1
            kind, nre = _slot_cmp(tl, gl)
            if kind == "nre":
                disp = "NRE=%.4g" % nre
            elif kind == "dtype":
                disp = "（dtype差）"
            elif kind == "shape":
                disp = "（shape差）"
            elif kind == "nonfinite":
                disp = "（非有限）"
            else:
                disp = "（无统计）"
            tq = ",".join(fmt_val(q) for q in _quad_of(tl)) \
                if is_tensor_leaf(tl) else "缺"
            gq = ",".join(fmt_val(q) for q in _quad_of(gl)) \
                if is_tensor_leaf(gl) else "缺"
            print("   %-8s %s [%s]" % (rl, name, pth))
            print("      T=(%s)" % tq)
            print("      G=(%s)  %s" % (gq, disp))
    if not sig_rows and not (args.quads and quad_rows):
        sys.exit(1)


def cmd_params(args):
    t_data = dump_data(args.dump_t)
    g_data = dump_data(args.dump_g)
    keep = _grep_keep(args.grep)

    counts = {"identical": 0, "agg_diff": 0, "stat_diff": 0,
              "shape_diff": 0, "missing": 0, "pairs": 0, "dir_t": 0,
              "dir_l": 0}
    agg_nres = []
    top_rows = []

    def compare(name, slot_desc, ts, gs):
        if not (is_tensor_leaf(ts) and is_tensor_leaf(gs)):
            counts["missing"] += 1
            return
        counts["pairs"] += 1
        if ts.get("shape") != gs.get("shape"):
            counts["shape_diff"] += 1
            top_rows.append((float("inf"), name, slot_desc, ts, gs, "shape差"))
            return
        if _leaf_quad_ident(ts, gs):
            counts["identical"] += 1
            return
        tq, gq = _quad_of(ts), _quad_of(gs)
        nt, ng = _norm_num(ts.get("Norm")), _norm_num(gs.get("Norm"))
        nre = None
        if nt is not None and ng is not None and ng != 0:
            nre = abs(nt - ng) / abs(ng)
        if tq[0] == gq[0] and tq[1] == gq[1]:
            counts["agg_diff"] += 1
            if nre is not None:
                agg_nres.append(nre)
            tag = "聚合差(Max-Min恒等+Mean-Norm分歧)"
        else:
            counts["stat_diff"] += 1
            tag = "统计量全差"
        if nre is not None and nre >= args.threshold:
            if nt > ng:
                counts["dir_t"] += 1
            elif nt < ng:
                counts["dir_l"] += 1
            top_rows.append((nre, name, slot_desc, ts, gs, tag))

    for name in t_data:
        if name not in g_data or not keep(name):
            continue
        tn, gn = t_data[name], g_data[name]
        if not (isinstance(tn, dict) and isinstance(gn, dict)):
            continue
        tp = tn.get("parameters")
        gp = gn.get("parameters")
        if isinstance(tp, dict) and isinstance(gp, dict):
            for pk in sorted(set(tp) | set(gp)):
                compare(name, "parameters.%s" % pk, tp.get(pk), gp.get(pk))
        elif is_tensor_leaf(tn) and is_tensor_leaf(gn):
            compare(name, "（param_grad 节点）", tn, gn)
    universe = ("grep=%s" % args.grep) if args.grep else "全部同名节点"
    print("# params 普查（universe=%s；同名节点 parameters 槽 + param_grad "
          "顶层节点）：对照槽位 %d；分桶：逐位一致=%d 聚合差=%d 统计量全差=%d "
          "shape差=%d 单侧缺=%d"
          % (universe, counts["pairs"], counts["identical"],
             counts["agg_diff"], counts["stat_diff"], counts["shape_diff"],
             counts["missing"]))
    n_sig = len(top_rows)
    print("# NRE≥%g 的参数槽 %d 个；方向统计（按 Norm）：T>G %d 槽 / T<G %d 槽"
          "——方向单一性是 §3.9 程序二 H-content/H-artifact 判别输入（随机归约"
          "伪影方向随机，系统性内容差方向一致）"
          % (args.threshold, n_sig, counts["dir_t"], counts["dir_l"]))
    if agg_nres:
        s = sorted(agg_nres)
        n = len(s)
        print("# 聚合差 NRE 分布（n=%d）：min=%.3e p25=%.3e med=%.3e p75=%.3e "
              "max=%.3e；NRE≥1e-3：%d" % (n, s[0], s[n // 4], s[n // 2],
                                          s[3 * n // 4], s[-1],
                                          sum(1 for x in s if x >= 1e-3)))
    top_rows.sort(key=lambda r: -r[0])
    print("# 明细 top %d（NRE 降序；shape 差置顶）：" % args.top)
    for nre, name, slot_desc, ts, gs, tag in top_rows[:args.top]:
        nre_s = "%.4g" % nre if nre != float("inf") else "n/a(shape)"
        print("   NRE=%s  %s  %s  [%s]" % (nre_s, name, slot_desc, tag))
        print("      T=(%s)" % ",".join(fmt_val(q) for q in _quad_of(ts)))
        print("      G=(%s)" % ",".join(fmt_val(q) for q in _quad_of(gs)))
    if len(top_rows) > args.top:
        print("# …top %d / 记录共 %d（--top 调整）" % (args.top,
                                                       len(top_rows)))


# ----------------------------------------------------------------

def _slot_cmp(tl, gl):
    """单槽位跨侧可比性判定（first/gain 共用）。返回 (kind, nre)：
    kind="nre"=同 dtype 同形状有限统计（nre=|NormT-NormG|/|NormG|）；
    "dtype"/"shape"=结构分歧；"nonfinite"=含 NaN/Inf（可能是捕获伪影，
    不参与数值首分歧判定）；"na"=无 Norm 统计等不可比。"""
    if not (is_tensor_leaf(tl) and is_tensor_leaf(gl)):
        return "na", None
    if tl.get("dtype") != gl.get("dtype"):
        return "dtype", None
    if tl.get("shape") != gl.get("shape"):
        return "shape", None
    if not (_finite_leaf(tl) and _finite_leaf(gl)):
        return "nonfinite", None
    nt, ng = _norm_num(tl.get("Norm")), _norm_num(gl.get("Norm"))
    if nt is None or ng is None or ng == 0:
        return "na", None
    return "nre", abs(nt - ng) / abs(ng)


def _is_backward_name(name):
    return name.endswith(".backward")


def _fmt_g(v):
    return "%.4g" % v if isinstance(v, (int, float)) else "—"


def cmd_first(args):
    t_data = dump_data(args.dump_t)
    g_data = dump_data(args.dump_g)
    t_order = order_index(t_data)
    g_order = order_index(g_data)
    if args.patterns:
        t_names = set(match_names(t_order, args.patterns))
        names = sorted(t_names & set(g_data), key=lambda n: t_order[n])
    else:
        names = [n for n in t_data if n in g_data]
    print("# first：同名匹配节点 %d 对（universe=%s）；阈值=%g；方向=%s"
          % (len(names), " ".join(args.patterns) if args.patterns
             else "全部同名", args.threshold, args.direction))
    found = {}          # d -> (name, pth, nre, pre_cnt, pre_max)
    prefix_max = {"forward": 0.0, "backward": 0.0}
    scanned = {"forward": 0, "backward": 0}
    struct_first = None
    nonfinite_first = None
    for name in names:
        d = "backward" if _is_backward_name(name) else "forward"
        if args.direction != "both" and d != args.direction:
            continue
        scanned[d] += 1
        pre_max = prefix_max[d]
        pre_cnt = scanned[d] - 1
        tn, gn = t_data[name], g_data[name]
        t_slots = dict(iter_leaves(tn)) if isinstance(tn, dict) else {}
        g_slots = dict(iter_leaves(gn)) if isinstance(gn, dict) else {}
        node_max = None
        hit = None
        for pth in sorted(set(t_slots) | set(g_slots)):
            kind, nre = _slot_cmp(t_slots.get(pth), g_slots.get(pth))
            if kind == "nonfinite" and nonfinite_first is None:
                nonfinite_first = (name, pth)
            if kind in ("shape", "dtype") and struct_first is None:
                struct_first = (name, pth, kind)
            if kind != "nre":
                continue
            node_max = nre if node_max is None else max(node_max, nre)
            if d not in found and nre >= args.threshold and hit is None:
                hit = (name, pth, nre, pre_cnt, pre_max)
        if hit is not None:
            found[d] = hit
        elif node_max is not None:
            prefix_max[d] = max(prefix_max[d], node_max)
    for d in ("forward", "backward"):
        if args.direction not in ("both", d):
            continue
        if d in found:
            name, pth, nre, pre_cnt, pre_max = found[d]
            print("== [%s] 首个显著数值分歧（NRE≥%g，同形状同 dtype 有限槽位）："
                  % (d, args.threshold))
            print("   %s  orderT=%d orderG=%s  槽位=%s  NRE=%r"
                  % (name, t_order[name], g_order.get(name), pth, nre))
            tn, gn = t_data[name], g_data[name]
            t_slots = dict(iter_leaves(tn)) if isinstance(tn, dict) else {}
            g_slots = dict(iter_leaves(gn)) if isinstance(gn, dict) else {}
            for p2 in sorted(set(t_slots) | set(g_slots)):
                for ln in _pair_lines(p2, t_slots.get(p2), g_slots.get(p2)):
                    print(ln)
            print("   # 前置核查：该方向更早匹配节点 %d 个，其槽位最大 NRE=%r"
                  "（< 阈值）→ 无更早显著分歧" % (pre_cnt, pre_max))
        else:
            print("# [%s] 无 NRE≥%g 的可比槽位（在匹配范围内）"
                  % (d, args.threshold))
    if struct_first:
        name, pth, kind = struct_first
        print("# 结构首分歧（首个 %s 不一致槽位）：%s [%s]  orderT=%d"
              "——可能是漂移族同名列，作证据前须按 align 纪律甄别"
              % (kind, name, pth, t_order[name]))
    if nonfinite_first:
        name, pth = nonfinite_first
        print("# 首个非有限槽位（advisory：可能是捕获伪影，不参与数值首分歧"
              "判定）：%s [%s]  orderT=%d" % (name, pth, t_order[name]))
    if not found and not struct_first and not nonfinite_first:
        sys.exit(1)


def cmd_gain(args):
    t_data = dump_data(args.dump_t)
    g_data = dump_data(args.dump_g)
    t_order = order_index(t_data)
    names = [n for n in match_names(t_order, args.patterns) if n in g_data]
    if not names:
        die("两侧均无匹配节点: %s" % " ".join(args.patterns), 1)
    print("# gain：链增益剖面（同名匹配 %d 节点，按 order 排序）；阈值=%g；"
          "方向=%s" % (len(names), args.threshold, args.direction))
    for d in ("forward", "backward"):
        if args.direction != "both" and d != args.direction:
            continue
        rows = []
        for name in names:
            if _is_backward_name(name) != (d == "backward"):
                continue
            tn, gn = t_data[name], g_data[name]
            t_slots = dict(iter_leaves(tn)) if isinstance(tn, dict) else {}
            g_slots = dict(iter_leaves(gn)) if isinstance(gn, dict) else {}
            r_in, r_out, out_slot = None, None, None
            for pth in sorted(set(t_slots) | set(g_slots)):
                kind, nre = _slot_cmp(t_slots.get(pth), g_slots.get(pth))
                if kind != "nre":
                    continue
                if pth == "output" or pth.startswith("output."):
                    if r_out is None or nre > r_out:
                        r_out, out_slot = nre, pth
                elif pth.startswith("input"):
                    if r_in is None or nre > r_in:
                        r_in = nre
            if r_in is None and r_out is None:
                continue
            rows.append((t_order[name], name, r_in, r_out, out_slot))
        if not rows:
            print("# [%s] 无可剖节点（无可比数值槽位）" % d)
            continue
        print("# [%s] %d 节点（r_in=输入侧最大 NRE，r_out=输出侧最大 NRE）"
              "——仅列前 %d 行，统计覆盖全部：" % (d, len(rows), args.limit))
        for o, name, r_in, r_out, oslot in rows[:args.limit]:
            if (r_in is None or r_in < args.threshold) and \
                    r_out is not None and r_out >= args.threshold:
                mark = "  ★分歧制造（输入一致而输出分歧）"
            elif r_in is not None and r_in >= args.threshold and \
                    r_out is not None and r_out >= args.threshold:
                mark = "  ★本地制造候选（输入输出均分歧，增益=%s）" \
                       % _fmt_g(r_out / r_in)
            elif r_in is not None and r_in > 0 and r_out is not None:
                mark = "  增益=%s" % _fmt_g(r_out / r_in)
            else:
                mark = ""
            print("   orderT=%-7d %-56s r_in=%-10s r_out=%-10s [%s]%s"
                  % (o, name[:56], _fmt_g(r_in) if r_in is not None else "—",
                     _fmt_g(r_out) if r_out is not None else "—",
                     oslot or "-", mark))
        outs = sorted(r[3] for r in rows if r[3] is not None)
        if outs:
            n = len(outs)

            def q(f):
                return outs[min(int(f * (n - 1)), n - 1)]
            print("#   r_out 分布（n=%d）：p10=%s p25=%s p50=%s p75=%s p90=%s "
                  "max=%s" % (n, _fmt_g(q(0.10)), _fmt_g(q(0.25)),
                              _fmt_g(q(0.50)), _fmt_g(q(0.75)),
                              _fmt_g(q(0.90)), _fmt_g(outs[-1])))
        amp = [r for r in rows if r[3] is not None and r[3] >= args.amp]
        if amp:
            o, name, r_in, r_out, _s = amp[0]
            print("#   首个 r_out≥%g：%s（orderT=%d，r_in=%s，本地增益=%s）"
                  % (args.amp, name, o, _fmt_g(r_in) if r_in is not None
                     else "—", _fmt_g(r_out / r_in) if r_in else "∞"))
        created = [r for r in rows if (r[2] is None or r[2] < args.threshold)
                   and r[3] is not None and r[3] >= args.threshold]
        print("#   分歧制造节点（r_in<%g 且 r_out≥%g）：%d 个%s"
              % (args.threshold, args.threshold, len(created),
                 "：" + ", ".join(r[1] for r in created[:8])
                 if created else ""))
        gains = [r[3] / r[2] for r in rows
                 if r[2] is not None and r[2] > 0 and r[3] is not None]
        if gains:
            sg = sorted(gains)
            print("#   继承节点 %d 个（r_in≥阈值），增益 r_out/r_in：p50=%s "
                  "max=%s（>1=放大，<1=衰减）"
                  % (len(gains), _fmt_g(sg[len(sg) // 2]), _fmt_g(sg[-1])))


# ---------------------------------------------------------------- 入口

def build_parser():
    p = argparse.ArgumentParser(prog="query.py",
                                description="dump 抽查查询工具集（用法见文件头）")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("node")
    s.add_argument("dump")
    s.add_argument("patterns", nargs="+")
    s.add_argument("--slot")
    s.add_argument("--limit", type=int, default=10)
    s.add_argument("--label", default=None)
    s.set_defaults(func=cmd_node)

    s = sub.add_parser("order")
    s.add_argument("dump")
    s.add_argument("--range", dest="range")
    s.add_argument("--grep")
    s.add_argument("--limit", type=int, default=200)
    s.set_defaults(func=cmd_order)

    s = sub.add_parser("count")
    s.add_argument("dump")
    s.add_argument("--grep")
    s.set_defaults(func=cmd_count)

    s = sub.add_parser("quad")
    s.add_argument("dump")
    s.add_argument("--from-node", dest="from_node")
    s.add_argument("--from-slot", dest="from_slot")
    s.add_argument("--from-file", dest="from_file")
    s.add_argument("--values", nargs=4)
    s.add_argument("--all", action="store_true")
    s.add_argument("--tolerance", type=float, default=None)
    s.add_argument("--limit", type=int, default=50)
    s.set_defaults(func=cmd_quad)

    s = sub.add_parser("diff")
    s.add_argument("dump_t")
    s.add_argument("dump_g")
    s.add_argument("patterns", nargs="+")
    s.add_argument("--limit", type=int, default=10)
    s.set_defaults(func=cmd_diff)

    s = sub.add_parser("first")
    s.add_argument("dump_t")
    s.add_argument("dump_g")
    s.add_argument("patterns", nargs="*")
    s.add_argument("--threshold", type=float, default=1e-3)
    s.add_argument("--direction", choices=["forward", "backward", "both"],
                   default="both")
    s.set_defaults(func=cmd_first)

    s = sub.add_parser("gain")
    s.add_argument("dump_t")
    s.add_argument("dump_g")
    s.add_argument("patterns", nargs="+")
    s.add_argument("--direction", choices=["forward", "backward", "both"],
                   default="both")
    s.add_argument("--threshold", type=float, default=1e-3)
    s.add_argument("--amp", type=float, default=1e3)
    s.add_argument("--limit", type=int, default=60)
    s.set_defaults(func=cmd_gain)

    s = sub.add_parser("align")
    s.add_argument("dump_t")
    s.add_argument("dump_g")
    s.add_argument("--family", action="append", default=[])
    s.add_argument("--grep")
    s.add_argument("--context", type=int, default=1)
    s.add_argument("--limit", type=int, default=40)
    s.set_defaults(func=cmd_align)

    s = sub.add_parser("survey")
    s.add_argument("dump_t")
    s.add_argument("dump_g")
    s.add_argument("mode", choices=["backward", "forward", "only"])
    s.add_argument("--top", type=int, default=30)
    s.add_argument("--grep")
    s.add_argument("--limit", type=int, default=50)
    s.set_defaults(func=cmd_survey)

    s = sub.add_parser("ranks")
    s.add_argument("dir_t")
    s.add_argument("dir_g")
    s.add_argument("patterns", nargs="+")
    s.add_argument("--slot")
    s.add_argument("--top", type=int, default=30)
    s.add_argument("--quads", action="store_true")
    s.set_defaults(func=cmd_ranks)

    s = sub.add_parser("params")
    s.add_argument("dump_t")
    s.add_argument("dump_g")
    s.add_argument("--top", type=int, default=15)
    s.add_argument("--grep")
    s.add_argument("--threshold", type=float, default=1e-3)
    s.set_defaults(func=cmd_params)

    s = sub.add_parser("stack")
    s.add_argument("stack")
    s.add_argument("pattern")
    s.add_argument("--limit", type=int, default=20)
    s.set_defaults(func=cmd_stack)

    s = sub.add_parser("census")
    s.add_argument("base", help="rank 目录父路径（单侧；自动发现 rank*/dump.json）")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--patterns", nargs="+",
                   help="模式匹配节点（子串；多个模式取并集）")
    g.add_argument("--backward", action="store_true",
                   help="枚举全部 .backward 节点的 output 槽位")
    s.add_argument("--slot", help="只普查该槽位路径（如 output.0 / input_args.1）")
    s.add_argument("--top", type=int, default=8, help="全局 Top-N 明细行数")
    s.add_argument("--label", help="输出标签（默认=base 路径）")
    s.set_defaults(func=cmd_census)

    s = sub.add_parser("construct")
    s.add_argument("construct")
    s.add_argument("patterns", nargs="+")
    s.add_argument("--include-null", dest="include_null", action="store_true")
    s.add_argument("--label")
    s.add_argument("--limit", type=int, default=200)
    s.set_defaults(func=cmd_construct)

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
