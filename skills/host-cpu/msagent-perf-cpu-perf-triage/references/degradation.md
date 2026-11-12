# 降级矩阵

每个依赖点缺失时的表现、降级行为与标注。降级不中断主流程，但**必须在报告与
`manifest.json` 中显式标注**，不允许静默留空。

## 配置维（本期可用）

| 依赖点 | 缺位表现 | 降级行为 | 标注 |
| ---- | ---- | ---- | ---- |
| config-audit skill | 找不到 `audit.py` | 配置维缺失，本期无运行时归因 → 结论「无法归因」 | `config_triage.status = UNAVAILABLE` |
| config-audit 执行失败 | 子进程非 0 退出 | 同上；保留 stderr 末 300 字符供排查 | `status = FAILED` |
| findings 过期（> 30min） | 时效校验失败 | 自动重跑快检；新旧差异标注 | `degradation` 记录 age |
| 显式 `--skip-audit` | 用户要求跳过 | 不审计，仅用现有 findings（可为空） | `status = SKIPPED` |

## 运行时维（后续 tracing 接入后启用）

| 依赖点 | 缺位表现 | 降级行为 | 标注 |
| ---- | ---- | ---- | ---- |
| 运行时数据缺失（本期默认） | 会话内无 `01-trace/phase_verdict_1.json` | 打扰 / 等待两维为 `null`，结论降级为「仅完成配置判型」 | `attribution.disturb_pct / wait_pct = null` |
| 运行时产物缺 `accounting` | 旧版本产物无该字段 | 两维按不可观测处理（`null`） | `degradation` 提示版本不匹配 |
| 多卡仅部分成功 | 部分卡 UNOBSERVED | 主结论取成功的卡；未成功卡列明细并标原因 | `basis = card<N>` |

## 其他维

| 依赖点 | 缺位表现 | 降级行为 | 标注 |
| ---- | ---- | ---- | ---- |
| perf-collect 缺位（第一期） | 阶段2 空 | 效率维 `not_observable`，其时间计入残余 | 报告显式标注「第二期」 |
| step 周期缺失 | 无 step 对齐 | 按观测窗分析，不声称与 step 对齐 | `未与 step 对齐` |
| os-observe 缺位（第一期） | 无 `anomaly_window` | 用户口述窗 / 运行时自定全窗 | `window = derived` |

## 降级与结论的关系

- 降级**降低置信度、不改变结论结构**：`AttributionResult` 字段不变，缺项为 `null`；
- 配置维缺失 + 运行时不可观测 → 结论只能是「无法归因」，需补齐依赖后重跑；
- 任一维降级都必须出现在 `index.html` 的「降级与口径说明」区块，供用户判断可信度。
