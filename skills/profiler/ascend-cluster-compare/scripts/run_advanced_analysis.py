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
"""
run_advanced_analysis.py
msprof-analyze 进阶分析编排脚本。

职责:
  1. 检测集群数据格式（DB/TEXT）与 msprof-analyze 工具可用性;
  2. 判断哪些进阶 feature 可运行, 必要时自动补跑前置 cluster_time_summary;
  3. 执行进阶 feature, 已有输出时直接复用（--force 可强制重跑）;
  4. 解析输出（CSV / SQLite 表）为结构化数据, 输出 advanced_analysis.json,
     供 generate_cluster_report.py --advanced 注入 HTML 报告「进阶分析」章节。

进阶分析能力对照（源自 msprof-analyze advanced_features 文档）:
  - cluster_time_compare_summary : 双集群 rank/step 级时间拆解对比, 需两侧 DB 均含
    ClusterTimeSummary 表（缺失时本脚本自动补跑）; 纯 TEXT 模式数据不支持。
  - free_analysis                : 单集群空闲原因分析, 支持 --export_type text 输出 CSV
    （cluster_analysis_output/FreeAnalysis/free_analysis.csv）。
  - communication_bottleneck     : 单集群通信瓶颈定位（慢/快 rank + 原因）, 支持
    --export_type text（cluster_analysis_output/CommunicationBottleneckAnalysis/
    communication_bottleneck.csv）。
  其余 feature（freq_analysis / ep_load_balance / slow_rank / slow_link / hccl_sum /
  communication_matrix_sum / mstx_sum / cann_api_sum / export_summary 等）可按相同
  模式扩展: 补一个 run_xxx + parse_xxx 即可。

用法:
  python run_advanced_analysis.py --cluster-a <dirA> [--cluster-b <dirB>] --output <advanced.json>
  python run_advanced_analysis.py --cluster-a <dirA> --cluster-b <dirB> --output adv.json --dry-run

注意:
  单集群 feature（free_analysis / communication_bottleneck）会对 A、B 两个集群分别执行,
  msprof 输出目录按集群隔离（<work-dir>/cluster_A、<work-dir>/cluster_B）, 结果记录带
  cluster 标注; 若提取器（cluster_data_extractor.py）已在各集群数据目录生成过进阶分析
  输出, 本脚本会自动复用（status=cached）, 不会重复执行。
"""
import argparse
import csv
import json
import os
import shutil
import sqlite3
import subprocess
from datetime import datetime

FEAT_TIME_CMP = "cluster_time_compare_summary"
FEAT_FREE = "free_analysis"
FEAT_COMM_BN = "communication_bottleneck"

TIME_SUMMARY_TABLE = "ClusterTimeSummary"
COMPARE_TABLE = "ClusterTimeCompareSummary"

CSV_FREE = "free_analysis.csv"
CSV_COMM_BN = "communication_bottleneck.csv"

DEFAULT_TIMEOUT = 7200  # 秒, 集群级分析可能较慢


def log(msg):
    print(msg, flush=True)


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _pick(row, *names):
    """大小写不敏感、忽略空格/下划线差异地从 DictReader 行中取列值。"""
    norm = {}
    for k, v in row.items():
        if k is None:
            continue
        norm[k.strip().lower().replace(" ", "").replace("_", "")] = (v or "").strip()
    for n in names:
        v = norm.get(n.strip().lower().replace(" ", "").replace("_", ""))
        if v:
            return v
    return ""


# ==================== 环境与数据探测 ====================

def find_msprof(bin_name):
    return shutil.which(bin_name)


def get_tool_version(msprof):
    try:
        p = subprocess.run([msprof, "--version"], capture_output=True, text=True, timeout=60)
        out = (p.stdout or "").strip() or (p.stderr or "").strip()
        return out.splitlines()[0][:120] if out else ""
    except Exception:
        return ""


def detect_db(data_dir):
    for cand in (
        os.path.join(data_dir, "cluster_analysis_output", "cluster_analysis.db"),
        os.path.join(data_dir, "cluster_analysis.db"),
        os.path.join(data_dir, "cluster.db"),
    ):
        if os.path.isfile(cand):
            return cand
    return None


def detect_text_dir(data_dir):
    for sub in (os.path.join(data_dir, "cluster_analysis_output"), data_dir):
        if sub and os.path.isfile(os.path.join(sub, "cluster_step_trace_time.csv")):
            return sub
    return None


def table_exists(db_path, table):
    try:
        con = sqlite3.connect(db_path)
        try:
            row = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            return row is not None
        finally:
            con.close()
    except Exception:
        return False


def detect_cluster(data_dir):
    info = {
        "data_dir": data_dir,
        "db_path": None,
        "db_kind": None,
        "text_dir": None,
        "format": "unknown",
        "has_time_summary_table": False,
    }
    if not data_dir or not os.path.isdir(data_dir):
        info["format"] = "not_found"
        return info
    db = detect_db(data_dir)
    if db:
        info["db_path"] = db
        info["db_kind"] = "db_new" if os.path.basename(db) == "cluster_analysis.db" else "db_old"
        info["format"] = "db"
        info["has_time_summary_table"] = table_exists(db, TIME_SUMMARY_TABLE)
    txt = detect_text_dir(data_dir)
    if txt:
        info["text_dir"] = txt
        info["format"] = info["format"] if info["db_path"] else "text"
    return info


def run_tool(msprof, args_list, timeout=DEFAULT_TIMEOUT):
    cmd = [msprof] + args_list
    log(f"  [exec] {' '.join(cmd)}")
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or ""), (p.stderr or "")
    except FileNotFoundError:
        return 127, "", f"命令不存在: {msprof}"
    except subprocess.TimeoutExpired:
        return 124, "", "执行超时"
    except Exception as e:
        return 1, "", str(e)


def find_output_file(roots, filename, max_depth=4):
    """在候选根目录下限深递归搜索文件, 返回第一个命中路径。"""
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if dirpath[len(root):].count(os.sep) >= max_depth:
                dirnames[:] = []
                continue
            if filename in filenames:
                return os.path.join(dirpath, filename)
    return None


def find_db_with_table(roots, table, max_depth=4):
    """搜索包含指定表的 cluster_analysis.db。"""
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if dirpath[len(root):].count(os.sep) >= max_depth:
                dirnames[:] = []
                continue
            if "cluster_analysis.db" in filenames:
                p = os.path.join(dirpath, "cluster_analysis.db")
                if table_exists(p, table):
                    return p
    return None


# ==================== 输出解析 ====================

def parse_free_csv(csv_path, top_n=10):
    """free_analysis text 输出解析: 列含 Rank ID/Start Time(us)/End Time(us)/
    Duration(us)/Pytorch Idle Time(us)/Cann Idle Time(us)/Reason。按 Reason 聚合。"""
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row:
                rows.append(row)
    agg = {}
    total_us = 0.0
    for r in rows:
        reason = _pick(r, "Reason") or "未标注原因"
        dur_us = _to_float(_pick(r, "Duration(us)", "Duration"))
        rank = _pick(r, "Rank ID", "Rank")
        total_us += dur_us
        a = agg.setdefault(reason, {"count": 0, "duration_us": 0.0, "ranks": set()})
        a["count"] += 1
        a["duration_us"] += dur_us
        if rank:
            a["ranks"].add(rank)
    reasons = []
    for reason, a in agg.items():
        reasons.append({
            "reason": reason[:200],
            "count": a["count"],
            "total_ms": round(a["duration_us"] / 1000.0, 2),
            "pct": round(a["duration_us"] / total_us * 100, 1) if total_us > 0 else 0,
            "rank_count": len(a["ranks"]),
            "sample_ranks": sorted(a["ranks"])[:8],
        })
    reasons.sort(key=lambda x: x["total_ms"], reverse=True)
    return {
        "rows_total": len(rows),
        "total_ms": round(total_us / 1000.0, 2),
        "reasons": reasons[:top_n],
    }


def parse_comm_bn_csv(csv_path, top_n=20):
    """communication_bottleneck text 输出解析: 列含 Communication Op/
    Slow Rank ID/Fast Rank ID/Reason 等（列名以实际文件为准）。"""
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = [
            {k.strip(): (v or "").strip() for k, v in r.items() if k}
            for r in csv.DictReader(f) if r
        ]
    items = []
    for r in rows:
        op = _pick(r, "Communication Op", "CommunicationOp", "Op Name", "Op")
        slow = _pick(r, "Slow Rank ID", "SlowRankID", "Slow Rank")
        fast = _pick(r, "Fast Rank ID", "FastRankID", "Fast Rank")
        reason = _pick(r, "Reason")
        dur_us = _to_float(_pick(r, "Duration(us)", "Elapsed(us)", "Time(us)", "Transit Time(us)"))
        items.append({
            "op": op[:120],
            "slow_rank": slow,
            "fast_rank": fast,
            "reason": reason[:300],
            "duration_ms": round(dur_us / 1000.0, 2),
        })
    items.sort(key=lambda x: x["duration_ms"], reverse=True)
    return {
        "rows_total": len(rows),
        "columns": list(rows[0].keys()) if rows else [],
        "items": items[:top_n],
    }


def parse_compare_summary(db_path, top_n=10):
    """ClusterTimeCompareSummary 表解析: 字段 rank/step/{metrics}/{metrics}Base/
    {metrics}Diff, 时间单位 μs → 统一转 ms。"""
    con = sqlite3.connect(db_path)
    try:
        cur = con.execute(f"SELECT * FROM {COMPARE_TABLE}")
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        con.close()
    if not rows:
        return {"rows_total": 0}
    diff_cols = [c for c in cols if c.endswith("Diff")]
    metrics = [c[:-4] for c in diff_cols]
    key_cols = [c for c in cols if not c.endswith("Base") and not c.endswith("Diff")]

    def _nums(r, col):
        v = r.get(col)
        return v if isinstance(v, (int, float)) else None

    overall = {}
    for m in metrics:
        b = [v for r in rows if (v := _nums(r, m + "Base")) is not None]
        c = [v for r in rows if (v := _nums(r, m)) is not None]
        d = [v for r in rows if (v := _nums(r, m + "Diff")) is not None]
        overall[m] = {
            "base_avg_ms": round(sum(b) / len(b) / 1000.0, 2) if b else None,
            "cur_avg_ms": round(sum(c) / len(c) / 1000.0, 2) if c else None,
            "diff_avg_ms": round(sum(d) / len(d) / 1000.0, 2) if d else None,
        }

    rank_col = next(
        (c for c in key_cols if c.strip().lower() in ("rank", "rank_id", "rankid", "index")),
        None,
    )
    top_ranks = []
    if rank_col and metrics:
        stage_metric = next((m for m in metrics if "stage" in m.lower()), metrics[0])
        by_rank = {}
        for r in rows:
            by_rank.setdefault(r.get(rank_col), []).append(r)
        for rk, rs in by_rank.items():
            item = {"rank": rk, "step_count": len(rs)}
            for m in metrics:
                vals = [v for x in rs if (v := _nums(x, m + "Diff")) is not None]
                item[m + "_diff_ms"] = round(sum(vals) / len(vals) / 1000.0, 2) if vals else None
            top_ranks.append(item)
        top_ranks.sort(key=lambda x: (x.get(stage_metric + "_diff_ms") or 0), reverse=True)
        top_ranks = top_ranks[:top_n]
    return {
        "rows_total": len(rows),
        "key_cols": key_cols,
        "metrics": metrics,
        "overall": overall,
        "rank_col": rank_col,
        "top_ranks": top_ranks,
    }


# ==================== feature 编排 ====================

def feature_availability(name, tool_available, ca, cb):
    """返回 (available, reason)。"""
    if not tool_available:
        return False, "msprof-analyze 工具不可用（未安装或不在 PATH）"
    if name == FEAT_TIME_CMP:
        if not (ca and cb):
            return False, "cluster_time_compare_summary 需要提供两个集群目录（缺少 --cluster-b）"
        for tag, c in (("集群A", ca), ("集群B", cb)):
            if not c["db_path"]:
                return False, f"{tag} 无 DB 格式数据（当前格式: {c['format']}），该 feature 仅支持 DB 输入"
        return True, ""
    target = cb or ca
    if not target or target["format"] in ("unknown", "not_found"):
        return False, "目标集群目录下未识别到 cluster_analysis_output / profiling 数据"
    return True, ""


def ensure_time_summary(msprof, c, tag):
    """确保集群 DB 含 ClusterTimeSummary 表; 缺失时自动补跑。返回 (ok, msg)。"""
    if c["has_time_summary_table"]:
        return True, ""
    log(f"  [prep] {tag} 缺少 {TIME_SUMMARY_TABLE} 表, 自动补跑 cluster_time_summary ...")
    rc, out, err = run_tool(msprof, ["-m", "cluster_time_summary", "-d", c["data_dir"]])
    if rc != 0:
        tail = (err or out or "").strip().splitlines()[-5:]
        return False, f"{tag} 补跑 cluster_time_summary 失败(rc={rc}): " + " | ".join(tail)[-400:]
    c["has_time_summary_table"] = True
    return True, ""


def run_feature_time_cmp(msprof, ca, cb, work_dir, top_num, force):
    feat = {
        "name": FEAT_TIME_CMP,
        "label": "集群时间拆解对比",
        "desc": "双集群 rank/step 级时间指标对比（基准/当前/差值），需两侧 DB 含 ClusterTimeSummary 表",
    }
    ok, reason = feature_availability(FEAT_TIME_CMP, bool(msprof), ca, cb)
    feat["available"], feat["reason"] = ok, reason
    if not ok:
        feat["status"] = "skipped" if not (ca and cb) else "not_available"
        return feat

    ok1, msg1 = ensure_time_summary(msprof, ca, "集群A(基准)")
    ok2, msg2 = ensure_time_summary(msprof, cb, "集群B(对比)")
    if not (ok1 and ok2):
        feat["status"] = "failed"
        feat["reason"] = msg1 or msg2
        return feat

    existing = find_db_with_table([work_dir, cb["data_dir"], ca["data_dir"]], COMPARE_TABLE)
    if existing and not force:
        feat["status"] = "cached"
        feat["data"] = parse_compare_summary(existing, top_num)
        feat["data"]["source_db"] = existing
        return feat

    cmd = ["-m", FEAT_TIME_CMP, "-d", cb["data_dir"], "--bp", ca["data_dir"], "-o", work_dir]
    feat["command"] = " ".join(cmd)
    rc, out, err = run_tool(msprof, cmd)
    db_hit = find_db_with_table([work_dir, cb["data_dir"]], COMPARE_TABLE) if rc == 0 else None
    if rc == 0 and db_hit:
        feat["status"] = "success"
        feat["data"] = parse_compare_summary(db_hit, top_num)
        feat["data"]["source_db"] = db_hit
    else:
        tail = (err or out or "").strip().splitlines()[-5:]
        feat["status"] = "failed"
        feat["reason"] = (f"执行失败(rc={rc}): " + " | ".join(tail))[-500:] if tail else "执行完成但未找到结果 DB"
    return feat


def run_feature_on_target(msprof, target, work_dir, top_num, force,
                          feat_name, feat_label, feat_desc, csv_name,
                          cluster_tag=None):
    """单集群 feature（free_analysis / communication_bottleneck）通用编排。

    cluster_tag: 集群标识（"A"/"B"）。提供时 msprof 输出目录改为 <work_dir>/cluster_<tag>,
    避免多集群共享同一 work_dir 时输出互相覆盖; 缓存搜索同时覆盖该子目录与集群数据目录。
    """
    feat = {"name": feat_name, "label": feat_label, "desc": feat_desc}
    if cluster_tag:
        feat["cluster"] = cluster_tag
    ok, reason = feature_availability(feat_name, bool(msprof), target, None)
    feat["available"], feat["reason"] = ok, reason
    if not ok:
        feat["status"] = "not_available"
        return feat

    out_dir = os.path.join(work_dir, f"cluster_{cluster_tag}") if cluster_tag else work_dir
    os.makedirs(out_dir, exist_ok=True)
    roots = [out_dir, work_dir, target["data_dir"], target.get("text_dir")]
    existing = find_output_file(roots, csv_name)
    if existing and not force:
        feat["status"] = "cached"
        feat["data"] = (parse_free_csv if feat_name == FEAT_FREE else parse_comm_bn_csv)(existing, top_num)
        feat["data"]["source_csv"] = existing
        return feat

    cmd = ["-m", feat_name, "-d", target["data_dir"],
           "--export_type", "text", "--top_num", str(top_num), "-o", out_dir]
    feat["command"] = " ".join(cmd)
    rc, out, err = run_tool(msprof, cmd)
    csv_hit = find_output_file([out_dir, work_dir, target["data_dir"]], csv_name) if rc == 0 else None
    if rc == 0 and csv_hit:
        feat["status"] = "success"
        feat["data"] = (parse_free_csv if feat_name == FEAT_FREE else parse_comm_bn_csv)(csv_hit, top_num)
        feat["data"]["source_csv"] = csv_hit
    else:
        tail = (err or out or "").strip().splitlines()[-5:]
        feat["status"] = "failed"
        feat["reason"] = (f"执行失败(rc={rc}): " + " | ".join(tail))[-500:] if tail else "执行完成但未找到输出 CSV"
    return feat


# ==================== 主流程 ====================

def main():
    ap = argparse.ArgumentParser(description="msprof-analyze 进阶分析编排（供集群比对报告注入）")
    ap.add_argument("--cluster-a", required=True, help="集群A目录（基准; 单集群模式下同时作为分析目标）")
    ap.add_argument("--cluster-b", default=None, help="集群B目录（对比; 提供后单集群 feature 优先分析 B）")
    ap.add_argument("--output", required=True, help="输出 advanced_analysis.json 路径")
    ap.add_argument("--msprof", default="msprof-analyze", help="msprof-analyze 可执行文件名/路径")
    ap.add_argument("--top-num", type=int, default=10, help="各 feature 结果 Top N")
    ap.add_argument("--work-dir", default=None, help="msprof 输出目录（默认: <output 同目录>/advanced_work）")
    ap.add_argument("--dry-run", action="store_true", help="仅检测可用性，不执行任何分析")
    ap.add_argument("--force", action="store_true", help="忽略已有输出，强制重新执行")
    args = ap.parse_args()

    out_path = os.path.abspath(args.output)
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    work_dir = os.path.abspath(args.work_dir or os.path.join(out_dir, "advanced_work"))
    if not args.dry_run:
        os.makedirs(work_dir, exist_ok=True)

    msprof = find_msprof(args.msprof)
    tool = {
        "bin": args.msprof,
        "available": msprof is not None,
        "path": msprof or "",
        "version": "",
    }
    if tool["available"]:
        tool["version"] = get_tool_version(msprof)

    ca = detect_cluster(os.path.abspath(args.cluster_a))
    cb = detect_cluster(os.path.abspath(args.cluster_b)) if args.cluster_b else None
    log(f"[detect] 集群A: format={ca['format']}, db={ca['db_path']}, "
        f"time_summary={ca['has_time_summary_table']}")
    if cb:
        log(f"[detect] 集群B: format={cb['format']}, db={cb['db_path']}, "
            f"time_summary={cb['has_time_summary_table']}")

    result = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dry_run": bool(args.dry_run),
        "tool": tool,
        "clusters": {"A": ca, "B": cb},
        "features": [],
    }

    if not tool["available"]:
        log("[warn] msprof-analyze 不可用, 所有 feature 标记为 not_available")

    # 1) 双集群时间拆解对比（DB only）
    result["features"].append(run_feature_time_cmp(msprof, ca, cb, work_dir, args.top_num, args.force))

    # 2/3) 单集群 feature: free_analysis / communication_bottleneck（对 A、B 分别执行）
    for tag, cluster in (("A", ca), ("B", cb)):
        if cluster is None:
            continue
        tag_label = "基准" if tag == "A" else "对比"
        for feat_name, feat_label, feat_desc, csv_name in (
            (FEAT_FREE, "空闲时间原因分析",
             "定位单集群空闲（Free）时间的成因分类与耗时排名, 辅助 Host 下发瓶颈定位",
             CSV_FREE),
            (FEAT_COMM_BN, "通信瓶颈定位",
             "定位单集群慢卡/快卡与通信瓶颈原因, 辅助链路劣化排查",
             CSV_COMM_BN),
        ):
            feat = run_feature_on_target(
                msprof, cluster, work_dir, args.top_num, args.force,
                feat_name, feat_label, feat_desc, csv_name, cluster_tag=tag)
            result["features"].append(feat)
            log(f"[feat] 集群{tag}({tag_label}) {feat_name}: {feat['status']}"
                + (f"（{feat['reason']}）" if feat.get("reason") and feat["status"] != "success" else ""))

    if args.dry_run:
        for f in result["features"]:
            f["status"] = "not_available" if not f.get("available") else "pending"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log(f"[done] 进阶分析结果已写入: {out_path}")
    for f in result["features"]:
        log(f"  - {f['name']}: {f['status']}" + (f"（{f['reason']}）" if f.get("reason") and f["status"] != "success" else ""))


if __name__ == "__main__":
    main()
