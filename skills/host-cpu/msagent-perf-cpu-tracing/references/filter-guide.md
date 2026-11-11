# 过滤、buffer 与配对陷阱

## 主闸门：核集

- `--cpus` 是数据量主闸门：128 核全开约 2~5 万事件/秒 → 8 核约 2 千事件/秒；
- 采集单位固定为单卡核集（≤32 核），不提供「一次全卡」的隐式行为。

## comm filter：两条互斥规则

- **需要识别抢占方**（「谁抢了我的线程」）→ **只能按 cpu 过滤**，comm/pid 留到转换期；
  否则 filter 会连带丢掉「他进程抢占目标线程」的事件，抢占方永远查不到。
- **只看目标线程自身调度** → 可以叠加 comm filter，事件量再降一档。

## sched_switch 配对陷阱

event filter 按 pid 过滤只会留下「prev 命中」或「next 命中」的一半事件，配对即断。
结论：**采集期不做 pid 过滤**，pid/tid 精确过滤统一在转换期（`pipe_convert`）执行。

## buffer 与丢事件

- `buffer_size_kb` 默认 2048/CPU，上限 8192/CPU（32 核 × 8MB = 256MB）；
- `options/overwrite=0`：不覆盖旧事件，溢出可被计数；
- 采集前后读 `instances/<id>/per_cpu/cpu*/stats` 的 `overrun`，与文本中 `LOST <n> EVENTS` 一并计入 `lost_events`；
- 覆盖率 = 1 − 丢事件/(事件+丢事件)；<90% 时结论强制 `UNOBSERVED`，禁止输出「无打扰/无漂移」。
