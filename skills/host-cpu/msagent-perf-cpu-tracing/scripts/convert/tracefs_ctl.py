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

"""tracefs 直采控制（仅 Linux；八步实现清单见设计 §3.1.3）。

不使用 trace-cmd：全部通过 debugfs / sysfs 文本接口完成。
"""

import glob
import os
import re
import shutil
import threading
import time
import uuid

EVENT_PART_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


class TracefsError(RuntimeError):
    pass


def _write(path, value):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(str(value))


def _read(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def _read_nonblocking(fd):
    """非阻塞读；EAGAIN/空数据返回空串（poll 阈值兜底用）。"""
    try:
        return os.read(fd, 1 << 16)
    except OSError:
        return b""


class TracefsInstance(object):
    def __init__(self, base, name, buffer_kb=2048):
        self.base = base
        self.name = name
        self.buffer_kb = int(buffer_kb)
        self.path = os.path.join(base, "instances", name)
        self.enabled = []

    def ensure(self):
        instances = os.path.join(self.base, "instances")
        if not os.path.isdir(instances):
            raise TracefsError("tracefs instances 目录不可用：%s" % instances)
        if os.path.isdir(self.path):
            # 不删别人的 instance：同名即报错（名字已带时间戳与随机后缀，正常不会冲突）
            raise TracefsError("instance 已存在，拒绝覆盖（可能被其它采集占用）：%s" % self.path)
        os.makedirs(self.path, exist_ok=True)
        # 时钟：global（ktime_get ≈ CLOCK_MONOTONIC，跨 CPU 可比）
        _write(os.path.join(self.path, "trace_clock"), "global")
        # buffer：按核设置，overwrite=0 保留丢事件计数
        try:
            _write(os.path.join(self.path, "buffer_size_kb"), self.buffer_kb)
        except OSError:
            pass
        try:
            _write(os.path.join(self.path, "options/overwrite"), "0")
        except OSError:
            pass
        # poll 唤醒阈值（实测 A3 主机被前置调成 50：大 buffer 下需 50% 才唤醒，
        # reader 读空后 poll 长期不触发造成断流）。降到 1% 尽早唤醒；旧内核
        # 无此文件时静默忽略（capture 内另有非 poll 直读兜底）。
        try:
            _write(os.path.join(self.path, "buffer_percent"), "1")
        except OSError:
            pass
        _write(os.path.join(self.path, "tracing_on"), "0")
        return self

    def enable_events(self, events):
        resolved = [self._resolve_event(event) for event in events]
        for event, path in resolved:
            try:
                _write(path, "1")
                self.enabled.append((event, path))
            except OSError as exc:
                raise TracefsError("启用事件失败 %s: %s" % (event, exc))

    def _resolve_event(self, event):
        """把事件名解析为 instance 内的 enable 路径。

        裸名（如 sched_switch）必须经 available_events 解析出唯一 group；
        直接写 events/enable 会把 instance 内全部事件打开（实测泄漏）。
        """
        if "/" in event:
            group, name = event.split("/", 1)
            group = self._event_part(group)
            name = self._event_part(name)
            return event, os.path.join(self.path, "events", group, name, "enable")
        mapping = {}
        for line in _read(os.path.join(self.base, "available_events")).splitlines():
            item = line.strip()
            if not item or ":" not in item:
                continue
            group, name = item.split(":", 1)
            if name == event:
                mapping.setdefault(event, []).append(group)
        if not mapping.get(event):
            raise TracefsError("事件不存在：%s（available_events 无匹配）" % event)
        if len(mapping[event]) > 1:
            raise TracefsError("事件名歧义 %s: %s（请给 group/name 全名）"
                               % (event, ", ".join(mapping[event])))
        group = mapping[event][0]
        return "%s:%s" % (group, event), os.path.join(self.path, "events", group, event, "enable")

    @staticmethod
    def _event_part(part):
        """事件名片段仅允许字母/数字/下划线，拒绝 . 与 .. 及绝对路径，避免拼出 tracefs 之外的路径。"""
        if not part or not EVENT_PART_PATTERN.match(part):
            raise TracefsError("非法事件名片段 %r：只允许字母、数字、下划线（如 sched/sched_switch）" % part)
        return part

    def trim_cpus(self, cpus):
        """尽力关闭目标核集之外的 per-CPU buffer。

        实测部分内核（6.6 aarch64）instance 内不暴露可写的 per_cpu/cpuN/enable，
        此时静默跳过（返回 0），数据量由转换期 --cpus 过滤兜底。
        """
        wanted = set(int(item) for item in cpus)
        trimmed = 0
        pattern = os.path.join(self.path, "per_cpu", "cpu*", "enable")
        for path in glob.glob(pattern):
            cpu_no = path.split(os.sep)[-2][3:]
            try:
                if int(cpu_no) not in wanted:
                    _write(path, "0")
                    trimmed += 1
            except (OSError, ValueError):
                continue
        return trimmed

    def set_filters(self, cpus=None, comms=None):
        """内核侧事件过滤：common_cpu 限核（全部事件）+ comm 限任务（仅 sched 事件）。

        实测 A3（6.6 aarch64）：instance 内 per_cpu/cpuN/enable 不可写（trim_cpus
        无效），于是全核产流（640 核约 40MB/s），而 trace_pipe 读接口上限只有
        ~1.2MB/s（单次 read 最多 ~4KB、约 300 次/秒），读线程追不上，5s 窗口
        只读到最前面 0.14s。改用事件 filter 的 common_cpu 字段在采集侧限核后，
        8/640 核降到 0.39MB/s，5s 窗口实测跨度 4.992s（完整）。

        filter 是覆盖写，cpu 与 comm 条件必须合并成一条表达式写一次。
        """
        cpu_expr = self._cpu_expr(cpus) if cpus else ""
        comm_expr = " || ".join("comm == \"%s\"" % item for item in comms) if comms else ""
        applied = {}
        for event, path in self.enabled:
            conditions = []
            if cpu_expr:
                conditions.append(cpu_expr)
            if comm_expr and event in ("sched:sched_switch", "sched:sched_wakeup"):
                conditions.append("(%s)" % comm_expr)
            if not conditions:
                continue
            filter_path = path[: -len("enable")] + "filter"
            try:
                _write(filter_path, " && ".join(conditions))
                applied[event] = _read(filter_path).strip()
            except OSError:
                applied[event] = ""
        return applied

    @staticmethod
    def _cpu_expr(cpus):
        """把核集压成连续的 common_cpu 区间表达式（始终加括号，便于 && 组合）。"""
        ordered = sorted(set(int(item) for item in cpus))
        if not ordered:
            return ""
        spans, start, prev = [], ordered[0], ordered[0]
        for cpu in ordered[1:]:
            if cpu == prev + 1:
                prev = cpu
                continue
            spans.append((start, prev))
            start = prev = cpu
        spans.append((start, prev))
        parts = []
        for a, b in spans:
            parts.append("common_cpu == %d" % a if a == b
                         else "common_cpu >= %d && common_cpu <= %d" % (a, b))
        return "(%s)" % " || ".join(parts)

    def start(self):
        _write(os.path.join(self.path, "tracing_on"), "1")

    def stop(self):
        try:
            _write(os.path.join(self.path, "tracing_on"), "0")
        except OSError:
            pass

    def capture(self, out_path, duration_s, stop_event=None):
        """专用阻塞读线程流式读 trace_pipe 落盘，到时间即停。

        实测 A3（6.6 aarch64）四条内核约束：
          1. trace_pipe 的 poll 单次唤醒只读一小块，会把消费限到 ~1MB/s；
          2. 「select 门控 + 抽干」虽能追平突发产流，但产流稀疏（如 comm 过滤后）
             时抽干一旦结束，poll 在剩余窗口内不再重唤醒（buffer_percent=1 也不行），
             窗口只剩第一段突发（实测 10s 只采到 0.47s）；
          3. buffer 为空时 os.read 即使 O_NONBLOCK 也会阻塞；
          4. 读接口硬上限 ~1.2MB/s（单次 read 最多 ~4KB、约 300 次/秒），
             640 核全核产流 ~40MB/s，必须靠 set_filters 在采集侧限核才追得上。
        因此改为专用阻塞读线程：空缓冲时阻塞等数据（不依赖 poll 重唤醒），
        deadline 由主线程控制 tracing_on=0；读线程 daemon 化，停止后被遗弃
        （采集为一次性 CLI 进程，退出即回收；fd 泄漏上界 = 段数 <= 6）。
        """
        pipe = os.path.join(self.path, "trace_pipe")
        written = 0
        done = threading.Event()
        failure = []
        try:
            fd = os.open(pipe, os.O_RDONLY)
        except OSError as exc:
            raise TracefsError("打开 trace_pipe 失败: %s" % exc)

        def _reader():
            nonlocal written
            try:
                with open(out_path, "w", encoding="utf-8") as dst:
                    while not done.is_set():
                        try:
                            chunk = os.read(fd, 1 << 16)
                        except OSError as exc:
                            if not done.is_set():
                                failure.append("读取 trace_pipe 失败: %s" % exc)
                            break
                        if not chunk:
                            if done.is_set():
                                break
                            time.sleep(0.02)
                            continue
                        try:
                            dst.write(chunk.decode("utf-8", "replace"))
                        except OSError as exc:
                            failure.append("读取线程写盘失败: %s" % exc)
                            break
                        written += len(chunk)
            finally:
                done.set()

        reader = threading.Thread(target=_reader, daemon=True)
        reader.start()
        self.start()
        deadline = time.time() + float(duration_s)
        try:
            while time.time() < deadline and not (stop_event and stop_event.is_set()):
                time.sleep(0.1)
        finally:
            self.stop()
            done.set()
            reader.join(timeout=2.0)
        if not reader.is_alive():
            # 读线程可能仍阻塞在 read 上（空缓冲 + 已停采）；此时不能 close fd
            # （同进程多段采集时 fd 号复用会让旧线程读到新 pipe），留给进程退出回收。
            os.close(fd)
        if failure:
            # 写盘/读管道异常不能被当成「采到 0 字节」的成功结果
            raise TracefsError(failure[0])
        return written

    def overrun(self):
        total = 0
        pattern = os.path.join(self.path, "per_cpu", "cpu*", "stats")
        for path in glob.glob(pattern):
            for line in _read(path).splitlines():
                if line.lower().startswith("overrun"):
                    try:
                        total += int(line.split(":", 1)[1].strip())
                    except (IndexError, ValueError):
                        pass
        return total

    def cleanup(self):
        self.stop()
        for _event, path in self.enabled:
            try:
                _write(path, "0")
            except OSError:
                pass
        shutil.rmtree(self.path, ignore_errors=True)


def capture_window(base, cpus, events, duration_s, out_path, comms=None, buffer_kb=2048,
                   stop_event=None):
    """在指定核集上采一个窗口；按核裁剪失败时靠事件 filter 的 common_cpu 限核。"""
    # 名字带毫秒时间戳 + 随机后缀：避免毫秒取模回绕（周期约 1000s）撞名后误删他人实例
    name = "hostcpu_%d_%s" % (int(time.time() * 1000), uuid.uuid4().hex[:8])
    inst = TracefsInstance(base, name, buffer_kb=buffer_kb)
    try:
        # ensure 也可能中途失败（建目录后写 clock/buffer 报错），必须纳入 finally 清理
        inst.ensure()
        inst.trim_cpus(cpus)
        inst.enable_events(events)
        filters = inst.set_filters(cpus=cpus, comms=comms)
        bytes_written = inst.capture(out_path, duration_s, stop_event=stop_event)
        return {"bytes": bytes_written, "overrun": inst.overrun(),
                "instance": name, "filters": filters}
    finally:
        inst.cleanup()


def filter_cpus(source, target, cpus):
    """按核集把全核 trace 文本筛成目标核（现场采集若未按 per_cpu 关闭）。"""
    wanted = set(cpus)
    kept = 0
    line_re = re.compile(r"\[(\d+)\]")
    with open(source, "r", encoding="utf-8", errors="replace") as src, \
            open(target, "w", encoding="utf-8") as dst:
        for line in src:
            match = line_re.search(line)
            if match and int(match.group(1)) in wanted:
                dst.write(line)
                kept += 1
            elif line.startswith("#"):
                dst.write(line)
    return kept


def streaming_worker(fn, *args, **kwargs):
    thread = threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True)
    thread.start()
    return thread
