---
name: msagent-perf-cpu-config-audit
description: >-
  昇腾训练/推理主机的环境配置体检：对 NUMA 亲和、大页/THP、governor 频率、cgroup
  限流、IRQ smp_affinity、环境变量等 8 个配置域做单点快照审计，输出分级 Finding
  与可回滚修改建议，不改动运行状态。触发：检查环境、配置体检、新机器开箱、换容器、
  大页没配、governor 不对、cgroup 限流、中断亲和配置、NUMA 配置检查、绑核没生效。
  不用于：训练变慢/step 抖动/利用率低的根因归因（→ perf-cpu-perf-triage）；
  产出绑核方案（→ perf-cpu-bind-tune）；运行期实际落核观测（→ perf-cpu-tracing）。
metadata:
  layer: analysis
---

# msagent-perf-cpu-config-audit — 主机配置审计

对 Host CPU 侧的配置参数与运行时环境变量做只读预检（8 域检查项），输出审计报告、
Finding[]/Action[]、配置基线与 suspect_items。全程只读，不执行任何修改动作。

## 使用方式

```bash
# 最简：全自动（自动发现框架/NPU/业务进程），快检
python3 scripts/audit.py

# 全量深检（配置侧 + 现状侧）
python3 scripts/audit.py --full

# 可选：显式指定框架与期望卡集（校验"申请 vs 实际"是否一致）
python3 scripts/audit.py --framework vllm --npu-cards 0-7

# 事中模式：指定目标进程
python3 scripts/audit.py --pid 12345

# 聚焦域 + 基线对比
python3 scripts/audit.py --focus numa,hugepage --baseline <上轮目录>/hostcpu-baseline.json
```

## 参数

| 参数 | 必选 | 说明 | 缺省 |
| ---- | ---- | ---- | ---- |
| --framework | 否 | 辅助进程发现（缩小候选范围） | 自动发现 |
| --npu-cards | 否 | 期望卡集（H1 卡数校验，按封装口径；如 0-7） | 按实际发现，不校验期望 |
| --pid | 否 | 目标进程（事中模式，启用现状侧检查） | 自动取首个发现目标 |
| --quick / --full | 否 | 快检 / 深检 | quick |
| --focus | 否 | 聚焦域（numa,bind,hugepage,freq,irq,cgroup,env,npu） | 全量 |
| --baseline | 否 | 上一轮 hostcpu-baseline.json（对比输入） | 无 |
| --expected | 否 | 用户覆盖期望值（JSON；用 --dump-thresholds 导模板后改数值） | 无 |
| --dump-thresholds | 否 | 导出当前生效阈值模板（含逐键说明）后退出 | 不导出 |
| --output | 否 | 输出目录 | `<cwd>/hostcpu_audit_<host>_<时间戳>/` |

## 工作空间与输出

- skill 安装位置由部署决定，skill 自身不感知、不引用安装路径
- 运行产物默认落在当前工作目录：`<cwd>/hostcpu_audit_<host>_<时间戳>/`（skill 目录不写运行产物）
- `--output` 显式指定输出目录；`HOSTCPU_WORKSPACE` 环境变量可覆盖产物根目录（编排场景使用）
- 远端执行：skill 拷贝位置与 cwd 由远程执行插件控制，skill 只按 cwd/参数落盘

## 输出产物

| 产物 | 用途 |
| ---- | ---- |
| static-snapshot.json | 配置事实快照（归档与复检对比，非跨 skill 契约） |
| findings.json | Finding[]/Action[]（供 perf-triage 复用） |
| hostcpu-baseline.json | 配置原始值存档（复检对比 + 回滚依据） |
| hostcpu_config_report_<host>_<ts>.html | HTML 分节报告（主交付物；`--stamp` 可显式指定时间戳） |
| suspect_items.json | 建议后续观测项（供 os-observe 承接） |

## 职责边界

- 只做单点快照 + 配置项判定，不做差分/时序采样（持续确认 → os-observe）；
- 不做 PMU 定量（→ perf-collect）、不做运行中异常归因（→ perf-triage）；
- 绑核诊断输出 Finding + suggested_cpuset，方案归 bind-tune；
- NUMA 亲和偏差判定归 bind-tune（本 skill 的 A5/A6/A7 仅输出拓扑 FACT）；
- NPU 形态事实交接：双 die 机型输出同封装 die 分组 `npu_die_groups`（绑核不可跨 node 拆分），
  并输出 fabric 与亲和可得性 `npu_affinity_available`；UB/HCCS_SW 等距机型
  「无 NPU 亲和」属合法形态（A5 不告警、不建议标定），R001 仅给均衡切片示例并降置信。

## 判定口径

- 阈值三级覆盖：`references/check-items.md` 默认表 → 机型 profile → 用户 `--expected`；
- 机型 profile 按 `match.cpu/npu` 与本机身份（/proc/cpuinfo、DMI、npu-smi 型号）前缀匹配，
  多个命中时最长匹配优先（精确机型优先于家族通用），未命中走默认表；
- 判定态 OK/WARN/CRIT；无期望值时 FACT（仅报事实）；
- 工具缺失只降级（等价 /proc、/sys 读取）并标注 `view_limited`，不跳过。

## 阈值自定义（--expected）

用户不需要预先知道阈值键名——先导出模板，改数值后传回：

```bash
# 1. 导出当前生效阈值模板（默认 + 机型 profile 合并结果，含逐键说明）
python3 scripts/audit.py --dump-thresholds my_thresholds.json

# 2. 编辑模板中的数值（下划线开头的说明键会被忽略），然后审计时传入
python3 scripts/audit.py --expected my_thresholds.json
```

- 模板内 `expected_affinity` 可填期望绑核核集（如 `0-79`，留空不判）；
- 机型 profile（`references/profiles/`）同样只写差异项，`npu_numa_override`（NPU→NUMA
  标定映射，topo 无 Affinity 列的机型使用）与 `snc_expected` 为机型级扩展键。
