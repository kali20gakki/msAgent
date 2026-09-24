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
"""run_mssanitizer.py — 批量运行 msSanitizer 四种检测工具

用法:
    python run_mssanitizer.py [--output-dir <目录>] [--extra-args "<mssanitizer参数>"] -- <用户自定义命令>

示例:
    python run_mssanitizer.py -- ./execute_add_op
    python run_mssanitizer.py --output-dir ./add_example -- bash build.sh --run_example AddCustom
    python run_mssanitizer.py --extra-args "--leak-check=yes" -- ./execute_add_op
    python run_mssanitizer.py -- bash run.sh
    python run_mssanitizer.py -- python test_add.py

脚本会依次执行 memcheck → racecheck → initcheck → synccheck，
每个检测的输出实时打印到终端并保存到带时间戳的 .txt 文件。
报告头部会记录真实命令、核心参数与环境信息，供解析脚本生成最终报告。

注意:
    - 请在算子仓库根目录下执行本脚本：mssanitizer 会将运行明细日志写入
      当前工作目录的 mindstudio_sanitizer_log/，result.log 等产物同样落在
      当前工作目录。
    - 通过 --output-dir 指定原始报告输出目录（建议为算子目录），默认为当前工作目录。
"""

import logging
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple


TOOLS = [
    {"name": "memcheck", "desc": "内存检测"},
    {"name": "racecheck", "desc": "竞争检测"},
    {"name": "initcheck", "desc": "未初始化检测"},
    {"name": "synccheck", "desc": "同步检测"},
]


@dataclass
class RunContext:
    """检测运行共享上下文"""
    output_dir: str
    user_cmd: List[str]
    user_cmd_str: str
    extra_args_strip: str
    timestamp: str
    env_line: str
    log_dir: str


def now() -> datetime:
    """返回带本地时区信息的当前时间"""
    return datetime.now(timezone.utc).astimezone()


def now_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """返回格式化后的当前时间字符串"""
    return now().strftime(fmt)


def _run_capture(argv: List[str], timeout: int = 30) -> str:
    """执行命令并返回标准输出，失败返回空字符串"""
    try:
        result = subprocess.run(
            argv, shell=False, capture_output=True, text=True,
            errors="replace", timeout=timeout,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _read_first_line(path: str) -> str:
    """读取文件首个非空行，失败返回空字符串"""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return ""
    for line in lines:
        if line.strip():
            return line.strip()
    return ""


def collect_env_info() -> dict:
    """尽力采集环境信息，采集失败的项目记为 unknown"""
    env = {}

    ms_ver = _run_capture(["mssanitizer", "-v"])
    if "command not found" in ms_ver or not ms_ver:
        ms_ver = "unknown"
    env["mssanitizer"] = ms_ver.replace("\n", " / ")

    # CANN 版本：依次尝试常见 version.info 位置
    cann_ver = ""
    candidates = [
        os.path.join(os.environ.get("ASCEND_HOME_PATH", ""), "version.info"),
        "/usr/local/Ascend/cann/version.info",
        "/usr/local/Ascend/version.info",
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            cann_ver = _read_first_line(p)
            if cann_ver:
                break
    env["cann"] = cann_ver or "unknown"

    # 芯片型号：优先 acl 接口，失败记为 unknown
    soc = _run_capture(
        ["python3", "-c", "import acl; print(acl.get_soc_name())"]
    )
    env["npu"] = soc.splitlines()[0] if soc else "unknown"

    env["host"] = _run_capture(["hostname"]) or "unknown"
    env["cwd"] = os.getcwd()
    env["date"] = now_str()
    return env


def parse_args(argv: List[str]) -> Optional[Tuple[str, str, List[str]]]:
    """解析 --output-dir / --extra-args 与 -- 之后的用户命令，不合法返回 None"""
    output_dir = None
    extra_args = ""
    try:
        sep_idx = argv.index("--")
    except ValueError:
        logging.error("错误: 未找到 '--' 分隔符")
        logging.error("用法: python run_mssanitizer.py [--output-dir <目录>] "
                      "[--extra-args \"<mssanitizer参数>\"] -- <用户自定义命令>")
        logging.error("示例: python run_mssanitizer.py --output-dir ./add_example -- ./execute_add_op")
        return None

    pre_args = argv[:sep_idx]
    user_cmd = argv[sep_idx + 1:]

    # 解析 --output-dir / --extra-args
    i = 0
    while i < len(pre_args):
        if pre_args[i] == "--output-dir" and i + 1 < len(pre_args):
            output_dir = pre_args[i + 1]
            i += 2
        elif pre_args[i] == "--extra-args" and i + 1 < len(pre_args):
            extra_args = pre_args[i + 1]
            i += 2
        else:
            logging.warning("警告: 未知参数 — %s", pre_args[i])
            i += 1

    if not user_cmd:
        logging.error("错误: '--' 后未提供用户命令")
        logging.error("示例: python run_mssanitizer.py -- ./execute_add_op")
        return None

    return output_dir, extra_args, user_cmd


def build_real_cmd_str(tool_name: str, extra_args_strip: str, user_cmd_str: str) -> str:
    """构造报告头部记录的真实检测命令字符串"""
    cmd = f"mssanitizer --tool={tool_name}"
    if extra_args_strip:
        cmd += f" {extra_args_strip}"
    cmd += f" -- {user_cmd_str}"
    return cmd


def build_run_cmd(tool_name: str, extra_args_strip: str, user_cmd: List[str]) -> List[str]:
    """构造最终 mssanitizer 运行命令（shell=False 列表形式）"""
    cmd = ["mssanitizer", f"--tool={tool_name}"]
    if extra_args_strip:
        cmd.extend(shlex.split(extra_args_strip))
    cmd.append("--")
    cmd.extend(user_cmd)
    return cmd


def write_report_header(report_file: str, tool_name: str, real_cmd_str: str,
                        ctx: RunContext) -> None:
    """将运行命令、核心参数与环境信息写入报告文件头部"""
    params = f"--tool={tool_name}"
    if ctx.extra_args_strip:
        params += f" {ctx.extra_args_strip}"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"[MSSANITIZER_CMD] {ctx.user_cmd_str}\n")
        f.write(f"[MSSANITIZER_REAL_CMD] {real_cmd_str}\n")
        f.write(f"[MSSANITIZER_TOOL] {tool_name}\n")
        f.write(f"[MSSANITIZER_PARAMS] {params}\n")
        f.write(f"[MSSANITIZER_ENV] {ctx.env_line}\n")
        f.write(f"[MSSANITIZER_LOGDIR] {ctx.log_dir}\n")


def _stream_output(cmd: List[str], report_file: str) -> int:
    """运行命令，实时将 stdout/stderr 输出到终端并追加到报告文件，返回退出码"""
    proc = subprocess.Popen(
        cmd, shell=False, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, errors="replace",
    )
    with open(report_file, "a", encoding="utf-8") as f:
        for line in proc.stdout:
            logging.info("%s", line.rstrip("\n"))
            f.write(line)
    proc.wait()
    return proc.returncode


def count_alarms(report_file: str) -> Tuple[int, int]:
    """统计报告文件中的 ERROR / WARNING 数量"""
    error_count = 0
    warn_count = 0
    if not os.path.exists(report_file):
        return error_count, warn_count
    with open(report_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("====== ERROR:") or line.startswith("[mssanitizer] Error:"):
                error_count += 1
            elif line.startswith("====== WARNING:") or line.startswith("[mssanitizer] Warning:"):
                warn_count += 1
    return error_count, warn_count


def run_single_tool(index: int, tool: dict, ctx: RunContext) -> Tuple[str, int, int]:
    """运行单个检测工具，返回 (报告文件路径, ERROR 数, WARNING 数)"""
    tool_name = tool["name"]
    report_file = os.path.join(ctx.output_dir, f"mssanitizer_origin_{tool_name}_{ctx.timestamp}.txt")
    real_cmd_str = build_real_cmd_str(tool_name, ctx.extra_args_strip, ctx.user_cmd_str)

    logging.info("[%d/4] %s (%s)", index, tool["desc"], tool_name)
    logging.info("      报告文件: %s", report_file)
    logging.info("      真实命令: %s", real_cmd_str)
    logging.info("-" * 40)

    write_report_header(report_file, tool_name, real_cmd_str, ctx)

    run_cmd = build_run_cmd(tool_name, ctx.extra_args_strip, ctx.user_cmd)
    try:
        exit_code = _stream_output(run_cmd, report_file)
    except FileNotFoundError:
        logging.error("错误: 未找到 mssanitizer 命令，请确认工具已安装")
        logging.error("      跳过 %s...", tool["desc"])
        return report_file, 0, 0

    error_count, warn_count = count_alarms(report_file)
    logging.info("      退出码: %s", exit_code)
    logging.info("      ERROR: %d  WARNING: %d", error_count, warn_count)
    return report_file, error_count, warn_count


def print_summary(total_errors: int, total_warnings: int,
                  report_files: List[str], log_dir: str) -> None:
    """打印检测完成汇总信息"""
    logging.info("=" * 60)
    logging.info("  检测完成")
    logging.info("  结束时间: %s", now_str())
    logging.info("  总计 ERROR: %d  WARNING: %d", total_errors, total_warnings)
    logging.info("")
    logging.info("  生成的报告文件:")
    for report_file in report_files:
        if os.path.exists(report_file):
            logging.info("    %s  (%d bytes)", report_file, os.path.getsize(report_file))
        else:
            logging.info("    %s  (未生成)", report_file)
    logging.info("")
    logging.info("  运行明细日志目录: %s", log_dir)
    logging.info("  （result.log 仅在手动运行并重定向时生成，同样位于执行目录）")
    logging.info("=" * 60)


def main() -> None:
    """主流程"""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    parsed = parse_args(sys.argv[1:])
    if parsed is None:
        sys.exit(1)
    output_dir, extra_args, user_cmd = parsed

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.getcwd()

    user_cmd_str = " ".join(user_cmd)
    timestamp = now_str("%Y%m%d_%H%M%S")
    env_info = collect_env_info()
    env_line = "; ".join(f"{k}={v}" for k, v in env_info.items())
    log_dir = os.path.join(os.getcwd(), "mindstudio_sanitizer_log")
    extra_args_strip = extra_args.strip()

    logging.info("=" * 60)
    logging.info("  msSanitizer 全量检测")
    logging.info("  用户命令: %s", user_cmd_str)
    if extra_args_strip:
        logging.info("  附加参数: %s", extra_args_strip)
    logging.info("  执行目录: %s", env_info["cwd"])
    logging.info("  开始时间: %s", env_info["date"])
    logging.info("  明细日志目录: %s", log_dir)
    logging.info("=" * 60)
    logging.info("")

    ctx = RunContext(
        output_dir=output_dir,
        user_cmd=user_cmd,
        user_cmd_str=user_cmd_str,
        extra_args_strip=extra_args_strip,
        timestamp=timestamp,
        env_line=env_line,
        log_dir=log_dir,
    )

    total_errors = 0
    total_warnings = 0
    report_files = []
    for i, tool in enumerate(TOOLS, 1):
        report_file, error_count, warn_count = run_single_tool(i, tool, ctx)
        report_files.append(report_file)
        total_errors += error_count
        total_warnings += warn_count

    print_summary(total_errors, total_warnings, report_files, log_dir)


if __name__ == "__main__":
    main()
