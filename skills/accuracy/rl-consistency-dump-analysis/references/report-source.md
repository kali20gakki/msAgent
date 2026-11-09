# report_source.json 字段说明(分析报告的事实源)

`scripts/generate_alignment_report.py` 从单一事实源 `report_source.json` 渲染 MD/HTML 两种格式。模板:`references/templates/alignment_report_source.zh-CN.json`。**先初始化再填,别手写裸 dict**:

```bash
python3 scripts/generate_alignment_report.py --init-source report_source.json
# 编辑 report_source.json 填事实 → 生成:
python3 scripts/generate_alignment_report.py --input report_source.json --output-dir reports/
```

生成器有 schema 校验,字段取值不合法会报 `ERROR: ...`(如 `long_horizon_required=false` 时 required_gates 不能含 `long_horizon_e2e`)。下面按字段说明**本 skill 的填法**。

## 顶层

- `schema_version: 1`、`language: "zh-CN"`、`title`、`generated_at`(ISO 日期)。
- `acceptance_contract`:本 skill 是纯分析,通常 `status="evidence_missing"`(存在已知差异)或 `not_run`;`objective` 写本次比对目标;`requested_model_scope` 写模型与规模;`required_gates` 从 9 个合法 gate 中选(本次通常去掉 `long_horizon_e2e`,因为非长程诊断);`long_horizon_required` 与 gate 是否含 long_horizon_e2e 必须一致;`stop_condition` 写「首差异有确定性根因并完成 ULP 定性」。

## scope

- `phase: "prefill"`、`model`、`coverage`(边界数与 token 数,如「44 个语义边界 × 单样本 seq=190」)。
- `training_backend` / `inference_backend` / `hardware_backend`(如「<训练框架> + <推理引擎> + <硬件>(TP/PP/EP)」)。
- `evidence_provider`:如「dump 离线比对(boundary_results.json + ULP 量化)+ 训练日志 prefill logprob diff 实测」。
- `topology.tp` **必须为 1**(schema 硬约束)。

## results

- `native_prefill_status` / `conditional_prefill_status` ∈ {`exact`, `nonzero`, `not_run`, `evidence_missing`}。纯分析无 replay 路径 → `conditional_prefill_status="not_run"`,`unresolved_replay_ids=[]`。
- `e2e_metric_name` / `e2e_metric_status` / `e2e_metric_value` / `e2e_metric_evidence`:
  - 有 prefill logprob diff 实测 → `e2e_metric_name` 填对应指标名(如 `prefill_logprob_diff_mean`),status=`nonzero`,value=mean,evidence 写实测行(n_tokens/max/std)。
  - 没实测 → status=`evidence_missing`,**value=null**(nonzero/zero 必须带数值,evidence_missing/not_run 必须 null)。

## verification_axes(9 轴)

| 轴 | status 合法值 | 本 skill 典型填法 |
|---|---|---|
| `tensor_prefill` | exact/nonzero/not_run/evidence_missing | **必须与 `native_prefill_status` 一致**;evidence 写边界汇总+首差异 |
| `weight_sync` | exact/mismatch/not_run/evidence_missing/not_applicable | 权重全部 bitwise → `exact` |
| 其余 7 轴 | pass/fail/not_run/evidence_missing/(部分含 not_applicable) | 未做的标 `not_run` + 原因 |

`production_patch_hygiene` 若确认采集代码已还原可标 `pass`。

## validation_matrix / metric_progression

- 纯单次分析通常 `validation_matrix.entries=[]`、`metric_progression=null`。schema 允许空。
- 若有多步 E2E 演进数据(不同 stage 的指标值)才填 metric_progression(注意 `results.e2e_metric_value` 必须等于 headline stage 值)。

## differences[]

每条一个已确认差异(通常 1 条,即首差异):

- `diff_id` 唯一(如 `D1-<首差异边界名>-<根因短描述>`)。
- `status`:整体 nonzero 时首差异条标 `excluded_known_gap`(已确认未解决差异);待验证用 `pending`;**`resolved` 仅在做过修复/replay 且 verification_gate 全真时用**,纯分析不伪装 resolved。
- `observation`:"native"(或 "conditional" 仅条件路径)。
- `root_cause_class` ∈ {Op path, Input precision, Config, Missing step, Framework bug, Test setup mismatch, Weight sync, State transition, Control plane}。首差异 GEMM 舍入 → `Op path`。
- `root_cause_confidence` ∈ {CONFIRMED, SUPPORTED, CANDIDATE}。ULP 指纹 + 前序 EXACT + 权重 bitwise → CONFIRMED。
- `discovery_max_abs_diff`(首差异 max_abs)、`summary`、`proof`(证据:门禁数值、ULP 倍数分布、排除依据)、`repair_result`(定性结论)、`rejected_attempts[]`(排除的假设清单)。

## comparison / provenance / artifacts / limitations

- `comparison`:prompt_token_length / train_input_token_length / infer_prefill_token_length(通常相等,TP=1 无 padding);`train_prompt_slice`(prompt-only 裁剪描述);`logit_causal_shift`(logprob 逐 token 对齐口径,如「覆盖 N 个 (position,next_token) 对」)。
- `provenance`:版本、模型摘要、样本摘要、dump 路径(自由 key)。
- `artifacts`:`raw_dump_included=true`(报告可引用 dump 目录)、`production_patch`(若有)、补丁卫生/还原审计;`raw_tensor_included=false`、`prompt_included=false`(不泄露原则)。
- `limitations[]`:如实记未覆盖边界(MoE 内部)、E2E 数值缺失、诊断模型范围。

## 常见校验报错与修法

| 报错 | 修法 |
|---|---|
| `long_horizon_e2e required gate requires long_horizon_required=true` | 移除该 gate 或改 long_horizon_required |
| `tensor_prefill must match results.native_prefill_status` | 两处保持一致 |
| `nonzero/zero E2E status requires numeric metric value` | 补 value,或改 status 为 evidence_missing 且 value=null |
| `f-string ... backslash` | 生成器需 Python 3.11+(f-string 表达式已改单引号写法,不再需要 3.12);用 3.11+ 解释器或容器,产物拷回 |
