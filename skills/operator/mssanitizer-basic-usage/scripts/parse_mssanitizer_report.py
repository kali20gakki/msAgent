#!/usr/bin/env python3
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
"""parse_mssanitizer_report.py — 解析 msSanitizer 检测报告，输出合并的 .md 最终报告

用法:
    python parse_mssanitizer_report.py <报告文件1.txt> [<报告文件2.txt> ...]

示例:
    python parse_mssanitizer_report.py mssanitizer_origin_memcheck_20260617_120000.txt
    python parse_mssanitizer_report.py mssanitizer_origin_*.txt

输入文件命名规则: mssanitizer_origin_<tool>_<timestamp>.txt
输出文件命名规则: mssanitizer_analysis_<timestamp>.md（多文件合并为一份报告）

报告头部标记（由 run_mssanitizer.py 写入，缺失时相应内容标注"未记录"）:
    [MSSANITIZER_CMD]        算子运行命令
    [MSSANITIZER_REAL_CMD]   实际执行的 mssanitizer 检测命令
    [MSSANITIZER_TOOL]       检测工具名
    [MSSANITIZER_PARAMS]     检测核心参数
    [MSSANITIZER_ENV]        环境信息（key=value; key=value 形式）
    [MSSANITIZER_LOGDIR]     运行明细日志目录（mindstudio_sanitizer_log）

汇总符号规则: ERROR 数量为 0 写 `✅ 0`，大于 0 写 `❌ N`；
              WARNING 数量为 0 写 `✅ 0`，大于 0 写 `⚠️ N`。

"错误原因分析"与"修改方式与回归结果"章节由本脚本生成待填模板，
需按 SKILL.md 步骤六（误报排查/根因分析/修复回归）人工补全。
"""

import logging
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Tuple

TOOL_ORDER = ["memcheck", "racecheck", "initcheck", "synccheck"]

ENV_LABELS = [
    ("mssanitizer", "mssanitizer 版本"),
    ("cann", "CANN 版本"),
    ("npu", "芯片型号"),
    ("host", "主机"),
    ("cwd", "执行目录"),
    ("date", "检测时间"),
]

ALARM_PATTERNS = {
    "非法读取 (illegal read)": r"illegal read",
    "非法写入 (illegal write)": r"illegal write",
    "多核踩踏 (out of bounds)": r"out of bounds",
    "非对齐访问 (misaligned)": r"misaligned",
    "内存泄漏 (LeakCheck/leak)": r"LeakCheck|Direct leak",
    "非法释放 (illegal free)": r"illegal free",
    "未使用内存 (Unused memory)": r"Unused memory",
    "未初始化读取 (uninitialized)": r"uninitialized",
    "数据竞争 (hazard)": r"hazard detected",
    "同步异常 (Unpaired/Redundant/Sync error)": r"Unpaired|Redundant|Sync error",
    "寄存器告警 (Register)": r"Register.*not reset",
}


def now() -> datetime:
    """返回带本地时区信息的当前时间"""
    return datetime.now(timezone.utc).astimezone()


def now_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """返回格式化后的当前时间字符串"""
    return now().strftime(fmt)


def fmt_err(n: int) -> str:
    """ERROR 数量: 0 写 ✅，大于 0 写 ❌"""
    return f"✅ {n}" if n == 0 else f"❌ {n}"


def fmt_warn(n: int) -> str:
    """WARNING 数量: 0 写 ✅，大于 0 写 ⚠️"""
    return f"✅ {n}" if n == 0 else f"⚠️ {n}"


@dataclass
class ReportData:
    """单个原始报告的解析结果"""
    tool: str
    header: dict
    error_lines: List[Tuple[int, str]]
    warn_lines: List[Tuple[int, str]]
    leak_summaries: List[str]
    type_counts: Counter


def _strip_alarm_prefix(line: str) -> str:
    """去掉告警行前缀（====== 或 [mssanitizer]）"""
    return re.sub(r"^(====== |\[mssanitizer\] )", "", line).rstrip()


def _extract_tool_from_line(line: str, tool_name: str) -> str:
    """从行内容提取工具名（无文件名规则时的兜底）"""
    if tool_name:
        return tool_name
    for t in TOOL_ORDER:
        if f"Start {t}" in line:
            return t
    return tool_name


def parse_single_report(input_path: str) -> ReportData:
    """解析单个原始报告，返回 ReportData"""
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    error_lines: List[Tuple[int, str]] = []
    warn_lines: List[Tuple[int, str]] = []
    leak_summaries: List[str] = []
    tool_name = ""
    header: dict = {}

    # 从文件名提取工具名
    m = re.search(r"mssanitizer_origin_(memcheck|racecheck|initcheck|synccheck)_",
                  os.path.basename(input_path))
    if m:
        tool_name = m.group(1)

    for i, line in enumerate(lines):
        # 报告头部元信息标记（真实命令/核心参数/环境等），不计入告警
        hm = re.match(r"^\[(MSSANITIZER_[A-Z_]+)\]\s?(.*)$", line)
        if hm:
            header[hm.group(1)] = hm.group(2).strip()
            continue
        # 从内容中提取工具名（无文件名规则时的兜底）
        tool_name = _extract_tool_from_line(line, tool_name)
        if line.startswith("====== ERROR:") or line.startswith("[mssanitizer] Error:"):
            error_lines.append((i + 1, _strip_alarm_prefix(line)))
        elif line.startswith("====== WARNING:") or line.startswith("[mssanitizer] Warning:"):
            warn_lines.append((i + 1, _strip_alarm_prefix(line)))
        elif "SUMMARY:" in line:
            leak_summaries.append(line.strip())

    # 从头部标记补充工具名
    if not tool_name and "MSSANITIZER_TOOL" in header:
        tool_name = header["MSSANITIZER_TOOL"]

    # 按告警类型分类统计
    full_text = "".join(lines)
    type_counts = Counter()
    for label, pat in ALARM_PATTERNS.items():
        type_counts[label] = len(re.findall(pat, full_text))

    return ReportData(
        tool=tool_name,
        header=header,
        error_lines=error_lines,
        warn_lines=warn_lines,
        leak_summaries=leak_summaries,
        type_counts=type_counts,
    )


def parse_env_line(env_line: str) -> dict:
    """解析 'key=value; key=value' 形式的环境信息行"""
    env = {}
    for item in env_line.split("; "):
        if "=" in item:
            k, v = item.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def shorten(msg: str, limit: int = 80) -> str:
    """截断告警信息用于表格展示"""
    msg = msg.replace("|", "\\|")
    return msg if len(msg) <= limit else msg[:limit] + "..."


@dataclass
class ReportContext:
    """合并报告所需的中间数据"""
    tool_data: dict
    all_error_lines: List[Tuple[str, int, str]]
    all_warn_lines: List[Tuple[str, int, str]]
    merged_type_counts: Counter
    global_header: dict
    real_cmds: List[Tuple[str, str]]
    params_list: List[Tuple[str, str]]
    log_dir: str
    ordered_tools: List[str]
    user_cmd: str


def _read_user_cmd_from_file(path: str) -> str:
    """从报告文件头部读取算子运行命令，无则返回空串"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return ""
    for line in lines:
        if line.startswith("[MSSANITIZER_CMD] "):
            return line[len("[MSSANITIZER_CMD] "):].strip()
    return ""


def _resolve_user_cmd(global_header: dict, input_paths: List[str]) -> str:
    """算子运行命令：优先取头部标记，其次从报告文件首行读取"""
    user_cmd = global_header.get("MSSANITIZER_CMD", "")
    if user_cmd:
        return user_cmd
    for path in input_paths:
        user_cmd = _read_user_cmd_from_file(path)
        if user_cmd:
            return user_cmd
    return ""


def _merge_report(ctx: ReportContext, path: str) -> ReportContext:
    """解析单个报告并合并到上下文"""
    try:
        data = parse_single_report(path)
    except FileNotFoundError:
        logging.warning("警告: 文件不存在，跳过 — %s", path)
        return ctx
    tool = data.tool or os.path.basename(path)
    ctx.tool_data[tool] = {
        "errors": data.error_lines,
        "warns": data.warn_lines,
        "leaks": data.leak_summaries,
        "counts": data.type_counts,
        "file": os.path.abspath(path),
    }
    for ln, msg in data.error_lines:
        ctx.all_error_lines.append((tool, ln, msg))
    for ln, msg in data.warn_lines:
        ctx.all_warn_lines.append((tool, ln, msg))
    ctx.merged_type_counts += data.type_counts

    if "MSSANITIZER_ENV" in data.header and "MSSANITIZER_ENV" not in ctx.global_header:
        ctx.global_header = data.header
    if "MSSANITIZER_REAL_CMD" in data.header:
        ctx.real_cmds.append((tool, data.header["MSSANITIZER_REAL_CMD"]))
    if "MSSANITIZER_PARAMS" in data.header:
        ctx.params_list.append((tool, data.header["MSSANITIZER_PARAMS"]))
    if "MSSANITIZER_LOGDIR" in data.header and not ctx.log_dir:
        ctx.log_dir = data.header["MSSANITIZER_LOGDIR"]
    return ctx


def collect_reports(input_paths: List[str]) -> ReportContext:
    """解析并聚合所有原始报告，返回上下文"""
    ctx = ReportContext(
        tool_data={},
        all_error_lines=[],
        all_warn_lines=[],
        merged_type_counts=Counter(),
        global_header={},
        real_cmds=[],
        params_list=[],
        log_dir="",
        ordered_tools=[],
        user_cmd="",
    )
    for path in input_paths:
        ctx = _merge_report(ctx, path)

    ctx.user_cmd = _resolve_user_cmd(ctx.global_header, input_paths)

    # 检测命令兜底：头部无记录时按工具名拼出
    if not ctx.real_cmds:
        for t in ctx.tool_data:
            ctx.real_cmds.append((t, f"mssanitizer --tool={t} -- {ctx.user_cmd}".strip()))

    ctx.ordered_tools = [t for t in TOOL_ORDER if t in ctx.tool_data] + \
                        [t for t in ctx.tool_data if t not in TOOL_ORDER]
    return ctx


class MarkdownWriter:
    """收集 Markdown 行并写出到文件"""

    def __init__(self) -> None:
        self.lines: List[str] = []

    def w(self, s: str = "") -> None:
        self.lines.append(s)

    def write(self, output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))


def _write_env_section(w, ctx: ReportContext) -> None:
    w("## 1. 环境信息")
    w()
    env = parse_env_line(ctx.global_header.get("MSSANITIZER_ENV", ""))
    if env:
        w("| 项目 | 值 |")
        w("|------|-----|")
        for key, label in ENV_LABELS:
            if key in env:
                w(f"| {label} | {env[key]} |")
        w()
    else:
        w("> 原始报告头部未记录环境信息（手动运行或旧版报告），请人工补充："
          "mssanitizer 版本、CANN 版本、芯片型号、算子仓库及执行目录、检测时间。")
        w()


def _write_cmd_section(w, ctx: ReportContext) -> None:
    w("## 2. 真实命令与核心参数")
    w()
    if ctx.user_cmd:
        w(f"- 算子运行命令: `{ctx.user_cmd}`")
    w("- 检测命令（逐工具，实际执行）:")
    for tool, cmd in ctx.real_cmds:
        w(f"  - {tool}: `{cmd}`")
    w("- 检测核心参数（逐工具）:")
    for tool, params in ctx.params_list:
        w(f"  - {tool}: `{params}`")
    if not ctx.params_list:
        w("  - 未记录（默认仅 `--tool=<工具名>`，如有附加参数请人工补充）")
    w()


def _write_summary_section(w, ctx: ReportContext) -> None:
    total_errors = sum(len(d["errors"]) for d in ctx.tool_data.values())
    total_warns = sum(len(d["warns"]) for d in ctx.tool_data.values())
    w("## 3. 检测结果汇总")
    w()
    w("| 检测工具 | ERROR | WARNING | 源文件 |")
    w("|----------|-------|---------|--------|")
    for tool_name in ctx.ordered_tools:
        d = ctx.tool_data[tool_name]
        w(f"| {tool_name} | {fmt_err(len(d['errors']))} | {fmt_warn(len(d['warns']))} | `{d['file']}` |")
    w(f"| **合计** | **{fmt_err(total_errors)}** | **{fmt_warn(total_warns)}** | |")
    w()
    w("> 符号规则: ERROR 为 0 写 ✅ 0、大于 0 写 ❌ N；WARNING 为 0 写 ✅ 0、大于 0 写 ⚠️ N。")
    w()


def _write_locations_section(w, ctx: ReportContext) -> None:
    w("## 4. 报告与日志位置")
    w()
    if ctx.log_dir:
        w(f"- 运行明细日志目录: `{ctx.log_dir}`（内含 `mssanitizer_<时间戳>_<pid>.log`，"
          "记录逐条内存/搬运记录与告警详情）")
    else:
        w("- 运行明细日志目录: `<算子仓库根>/mindstudio_sanitizer_log/`（mssanitizer 在执行目录自动生成）")
    w("- 原始检测报告:")
    for tool_name in ctx.ordered_tools:
        w(f"  - {tool_name}: `{ctx.tool_data[tool_name]['file']}`")
    w("- result.log: `<算子仓库根>/result.log`（手动运行并重定向 `> result.log 2>&1` 时生成，"
      "内含结论摘要与 `[mssanitizer] logging to file:` 溯源行）")
    w()


def _write_tool_error_block(w, ctx: ReportContext, tool_name: str) -> None:
    tool_errors = [(ln, msg) for t, ln, msg in ctx.all_error_lines if t == tool_name]
    if not tool_errors:
        return
    w(f"### {tool_name}")
    w()
    for ln, msg in tool_errors:
        w(f"- **[行{ln}]** {msg}")
    w()


def _write_error_section(w, ctx: ReportContext) -> None:
    w("## 5. ERROR 级别告警列表")
    w()
    if ctx.all_error_lines:
        for tool_name in ctx.ordered_tools:
            _write_tool_error_block(w, ctx, tool_name)
    else:
        w("*(无)*")
        w()


def _write_tool_warning_block(w, ctx: ReportContext, tool_name: str) -> None:
    tool_warns = [(ln, msg) for t, ln, msg in ctx.all_warn_lines if t == tool_name]
    if not tool_warns:
        return
    w(f"### {tool_name}")
    w()
    for ln, msg in tool_warns:
        w(f"- **[行{ln}]** {msg}")
    w()


def _write_warning_section(w, ctx: ReportContext) -> None:
    w("## 6. WARNING 级别告警列表")
    w()
    if ctx.all_warn_lines:
        for tool_name in ctx.ordered_tools:
            _write_tool_warning_block(w, ctx, tool_name)
    else:
        w("*(无)*")
        w()


def _write_leak_section(w, ctx: ReportContext) -> None:
    all_leaks = []
    for tool_name in ctx.ordered_tools:
        all_leaks.extend(ctx.tool_data[tool_name]["leaks"])
    if all_leaks:
        w("## 7. 汇总信息")
        w()
        for s in all_leaks:
            w(f"- `{s}`")
        w()


def _write_type_section(w, ctx: ReportContext) -> None:
    w("## 8. 按告警类型分布（全工具汇总）")
    w()
    w("| 告警类型 | 数量 |")
    w("|----------|------|")
    for label, count in ctx.merged_type_counts.most_common():
        w(f"| {label} | {count} |")
    w()
    w("> 详细告警字段说明请参见 [references/alarms.md](references/alarms.md)")
    w()


def _write_analysis_section(w, ctx: ReportContext) -> None:
    w("## 9. 错误原因分析（待补全）")
    w()
    w("> 逐条核对告警调用栈指向的源码位置与明细日志，先给出误报排查结论"
      "（确认真实 / 判定误报+依据），再对确认真实的错误写明根因。")
    w()
    if ctx.all_error_lines:
        w("| 编号 | 告警（工具/类型/位置） | 误报排查结论 | 根因 |")
        w("|------|------------------------|--------------|------|")
        for idx, (tool, ln, msg) in enumerate(ctx.all_error_lines, 1):
            w(f"| E{idx} | {tool} {shorten(msg)} @ 行{ln} | <确认真实 / 判定误报（依据：...）> | <根因> |")
        w()
    else:
        w("本次检测无 ERROR，无需填写。")
        w()


def _write_fix_section(w, ctx: ReportContext) -> None:
    w("## 10. 修改方式与回归结果（待补全）")
    w()
    w("> 逐条写明修复动作（修改的文件、位置、改动内容），以及修复后重新编译、"
      "回归检测的结果（对应工具 ERROR 是否清零 ✅）。")
    w()
    if ctx.all_error_lines:
        w("| 编号 | 修改内容（文件/位置/改动） | 回归检测结果 |")
        w("|------|----------------------------|--------------|")
        for idx in range(1, len(ctx.all_error_lines) + 1):
            w(f"| E{idx} | <待填写> | <待填写> |")
        w()
    else:
        w("本次检测无 ERROR，无需修改。")
        w()


def generate_merged_report(input_paths: List[str], output_path: str) -> None:
    """解析多个原始报告，合并输出一份 .md 最终报告（待补全模板）"""
    ctx = collect_reports(input_paths)
    writer = MarkdownWriter()
    w = writer.w

    w("# msSanitizer 检测分析报告")
    w()
    w(f"> 生成时间: {now_str()}")
    w("> 本报告由解析脚本自动生成；\"错误原因分析\"与\"修改方式与回归结果\"章节为待填模板，"
      "请按 SKILL.md 步骤六（误报排查/根因分析/修复回归）与步骤七（最终报告规范）补全。")
    w()

    _write_env_section(w, ctx)
    _write_cmd_section(w, ctx)
    _write_summary_section(w, ctx)
    _write_locations_section(w, ctx)
    _write_error_section(w, ctx)
    _write_warning_section(w, ctx)
    _write_leak_section(w, ctx)
    _write_type_section(w, ctx)
    _write_analysis_section(w, ctx)
    _write_fix_section(w, ctx)

    writer.write(output_path)
    logging.info("合并分析报告已生成: %s", output_path)


def _filter_existing(input_paths: List[str]) -> List[str]:
    """过滤出存在的输入文件，并提示跳过不存在的文件"""
    valid_paths = []
    for p in input_paths:
        if os.path.exists(p):
            valid_paths.append(os.path.abspath(p))
        else:
            logging.warning("警告: 文件不存在，跳过 — %s", p)
    return valid_paths


def _extract_timestamp(path: str) -> str:
    """从文件名提取时间戳，缺失时使用当前时间"""
    m = re.search(r"_(\d{8}_\d{6})", os.path.basename(path))
    if m:
        return m.group(1)
    return now_str("%Y%m%d_%H%M%S")


def main() -> None:
    """主流程"""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    if len(sys.argv) < 2:
        logging.error("用法: python parse_mssanitizer_report.py <报告文件1.txt> [<报告文件2.txt> ...]")
        logging.error("示例: python parse_mssanitizer_report.py mssanitizer_origin_*.txt")
        sys.exit(1)

    input_paths = sys.argv[1:]
    # 验证输入文件存在
    valid_paths = _filter_existing(input_paths)

    if not valid_paths:
        logging.error("错误: 没有有效的输入文件")
        sys.exit(1)

    # 从第一个文件名提取时间戳
    timestamp = _extract_timestamp(valid_paths[0])

    output_dir = os.path.dirname(valid_paths[0])
    output_path = os.path.join(output_dir, f"mssanitizer_analysis_{timestamp}.md")

    generate_merged_report(valid_paths, output_path)


if __name__ == "__main__":
    main()
