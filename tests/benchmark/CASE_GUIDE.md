# Benchmark Case 编写指南

一个 benchmark case 应该回答三个问题：

- Agent 看到什么输入。
- Agent 应该完成什么任务。
- Judge 如何判断答案是否合格。

每个 case 使用一个 YAML 文件，放在 `benchmarks/` 下；输入数据放在 `data/` 下。

## 最小模板

```yaml
id: my_case
input_data_path: ../data/my_case
prompt: >
  请根据 input_data 中的数据回答问题，并给出关键证据。
must_include:
  - 必须覆盖的事实 A
  - 必须覆盖的事实 B
scoring_prompt: >
  按 0-5 分评分。高分答案必须结论准确、证据来自输入数据、推理清晰。
  如果遗漏关键事实、证据不足或出现编造，应明显扣分。
```

建议先从这个模板开始，再按需要增加 `must_include_regex`、`must_tool_use`、
`skill_path` 和工具效率字段。

## 字段说明

| 字段 | 必填 | 说明 |
| --- | :---: | --- |
| `id` | 否 | case 标识；不写时使用 YAML 文件名 |
| `input_data_path` | 是 | 输入数据目录或文件；相对 YAML 文件所在目录解析 |
| `skill_path` | 否 | 可选 skill 目录；会复制到隔离工作区的 `skill/` |
| `prompt` | 是 | 给被测 Agent 的任务说明 |
| `must_include` | 是 | 答案必须语义覆盖的要点 |
| `must_include_regex` | 否 | 对最终 `answer` 文本做正则硬校验 |
| `must_tool_use` | 否 | trace 中必须出现的工具调用 |
| `scoring_prompt` | 是 | judge 的 0-5 分评分标准 |
| `tool_budget` | 否 | 免费工具调用额度，默认 20 |
| `tool_penalty_per_call` | 否 | 超预算后每次工具调用的效率惩罚系数，默认 0.02 |
| `tool_penalty_floor` | 否 | 效率系数下限，默认 0.5 |

## 输入数据怎么放

推荐结构：

```text
data/
  my_case/
    README.md
    metrics.csv
    logs.txt
    events.jsonl
```

写 case 时让 `input_data_path` 指向目录：

```yaml
input_data_path: ../data/my_case
```

runner 会把这个目录复制到临时隔离工作区，并暴露为 `input_data/`。Agent 理论上只应该读取
`input_data/` 和可选的 `skill/`，不应该读取 benchmark 源码、case YAML 或历史 run 输出。

输入数据建议：

- 放 compact summary，例如 csv、txt、markdown、html 摘要。
- 大型 profiler 数据可以放，但最好同时提供可快速定位的 summary 文件。
- 避免让 case 必须依赖父目录、绝对路径或本机私有路径。
- 不要把金标准答案写进 `input_data/`，否则评测会变成抄答案。

## prompt 怎么写

`prompt` 是被测 Agent 唯一能看到的任务描述。它应该说明任务目标和输出重点，但不要泄露
`must_include` 里的标准答案。

较好的写法：

```yaml
prompt: >
  请根据 input_data 中的 Ascend profiling 数据诊断是否存在快慢卡现象。
  需要说明慢卡 Rank、瓶颈类型、关键证据和可执行优化建议。
```

不好的写法：

```yaml
prompt: >
  请证明 rank3 是慢卡，并解释它的 Free Time 异常。
```

第二种把答案直接告诉了 Agent，不适合作为评测。

## must_include 怎么写

`must_include` 是答案必须语义覆盖的要点，由 judge 判断。它适合放“必须说对”的事实或结论。

示例：

```yaml
must_include:
  - rank3
  - Host 下发慢或调度瓶颈
  - Free Time 异常
  - 不是通信慢链路
```

建议：

- 每一项尽量短，表达一个独立要点。
- 不要把多个独立要求塞进同一项，否则 judge 难以稳定判断。
- 可以接受同义改写时，用语义描述；需要严格字符串命中时，用 `must_include_regex`。
- `must_include` 是硬门槛，任一项未覆盖，最终 `score = 0`。

## must_include_regex 怎么写

`must_include_regex` 会对最终答案 JSON 的 `answer` 字段做 Python `re.search`。
默认 flags 为 `re.IGNORECASE | re.MULTILINE`，不启用 `DOTALL`。

示例：

```yaml
must_include_regex:
  - 'rank\s*3|Rank\s*3|rank3'
```

适合使用 regex 的场景：

- 必须出现某个 ID、编号、agent 名称或 rank。
- 要兜底检查答案文本里是否真的写出了关键实体。
- 不希望完全依赖 LLM judge 的语义判断。

注意：regex 也是硬门槛，任一条不匹配，最终 `score = 0`。

## must_tool_use 怎么写

`must_tool_use` 用来要求 Agent 在 trace 中真的调用过某类工具。匹配规则是大小写不敏感的
子串匹配。

示例：

```yaml
must_tool_use:
  - read_file
  - msprof-mcp
```

如果 trace 中出现 `msprof-mcp_analyze_kernel_details`，就能命中 `msprof-mcp`。

适合使用 `must_tool_use` 的场景：

- 要求 Agent 必须读取输入文件，而不是凭常识回答。
- 要求 Agent 必须调用某个专业 MCP 或脚本。
- 评测目标不仅是答案正确，还包括方法论合规。

注意：`must_tool_use` 是硬门槛。答案即使正确，只要要求的工具没被调用，最终
`score = 0`。

## skill_path 怎么写

如果要评测某个 skill 是否能指导 Agent 完成任务，可以配置 `skill_path`：

```yaml
skill_path: ../../../skills/profiling/profiling-analysis
```

runner 会把 skill 目录复制到隔离工作区的 `skill/`，并在提示词中要求 Agent 先阅读
`skill/SKILL.md`。

建议：

- `skill_path` 指向 skill 根目录，而不是某个单独文件。
- skill 中引用的脚本和 reference 文件应放在 skill 目录内，保证复制后仍可用。
- 如果 skill 依赖外部命令，case 的 `scoring_prompt` 可以要求 Agent 说明命令不可用时的替代证据。

## scoring_prompt 怎么写

`scoring_prompt` 是给 judge 的 rubric，范围 0-5 分。它不提供给被测 Agent。

推荐结构：

```yaml
scoring_prompt: >
  按 0-5 分评分。高分答案必须正确指出慢卡 rank，解释瓶颈类型，
  引用 input_data 中的关键证据，并给出可执行建议。
  如果结论错误、证据不来自输入数据、遗漏关键解释或泛泛而谈，应明显扣分。
```

建议：

- 明确高分答案必须满足什么。
- 明确哪些错误要扣分。
- 不要在 `scoring_prompt` 里写只有人类知道但无法从输入数据验证的要求。
- 不要让 judge 依据隐藏推理打分，只依据 trace 和最终答案。

## 工具效率字段

工具效率目前只用于报告，不影响最终 `score`。

```yaml
tool_budget: 30
tool_penalty_per_call: 0.02
tool_penalty_floor: 0.5
```

计算方式：

```text
over_budget = max(0, tool_calls - tool_budget)
efficiency_factor = clamp(1 - over_budget * tool_penalty_per_call, tool_penalty_floor, 1.0)
```

什么时候调大 `tool_budget`：

- 输入数据很多，需要 Agent 做多步探索。
- profiling、日志排障、数据库查询等任务天然需要更多工具调用。
- `must_tool_use` 要求调用多个工具。

## 示例：阅读理解 case

```yaml
id: agent_list
input_data_path: ../data/msagent_agents
prompt: >
  根据 input_data 中的资料回答：msAgent 内置了哪些 agent？请列出全部名称。
must_include:
  - Hermes
  - Zephyr
  - Icarus
  - Minos
  - Accuracy
must_tool_use:
  - read_file
scoring_prompt: >
  按 0-5 分评分。重点看是否完整列出全部 5 个内置 agent 名称，遗漏或编造名称应扣分。
```

这个 case 主要考阅读完整性。`must_tool_use: read_file` 要求 Agent 确实读取资料。

## 示例：profiling skill case

```yaml
id: cluster_fast_slow_rank_profiling_skill
input_data_path: ../data/kv_cache_type_page_seqlen_4096_bs_1_profile_count_0
skill_path: ../../../skills/profiling/profiling-analysis
prompt: >
  请按照提供的 profiling-analysis skill 对 input_data 中的 Ascend 集群 profiling
  数据做性能瓶颈与快慢卡诊断。需要明确回答：是否存在快慢卡现象；真正的慢卡 Rank ID
  和候选快卡 Rank ID；瓶颈类型；关键证据；以及可执行优化建议。
must_include:
  - rank3
  - Host 下发慢或调度瓶颈
  - Free Time 异常
  - 不是通信慢链路
must_include_regex:
  - 'rank\s*3|Rank\s*3|rank3'
tool_budget: 30
scoring_prompt: >
  按 0-5 分评分。高分答案必须正确指出慢卡是 rank3，解释 Host 下发慢/调度瓶颈，
  引用 input_data 中的 Free/Compute/Communication/API 或 step_trace_time 证据，
  排除通信慢链路，并给出具体可执行建议。结论错误、证据不来自输入数据或泛泛而谈
  应明显扣分。
```

这个 case 主要考 Agent 是否能按 skill 方法分析真实 profiling 数据。`tool_budget` 调高到
30，避免复杂任务在效率报告中显得过度苛刻。

## 新增 case 的建议流程

1. 在 `data/<case_id>/` 准备输入数据。
2. 在 `benchmarks/<case_id>.yaml` 写 case。
3. 先用本地 smoke 或一个便宜模型跑单 case。
4. 检查 `runs/<run_id>/traces/<case_id>.trace.json`，确认 Agent 没有读到不该读的文件。
5. 检查 `judge/<case_id>.judge.json`，确认 `must_include` 判断符合预期。
6. 检查 `report.md` 的 `Missing Items`、`Missing Tools` 和效率列。
7. 根据结果调整 prompt、rubric、regex 或 tool budget。

## 提交前检查清单

- case YAML 可以被 `PyYAML` 正常解析。
- `input_data_path` 和 `skill_path` 都是可复制的真实路径。
- `prompt` 没有泄露标准答案。
- `must_include` 覆盖了最关键的结果和证据要求。
- 必须严格出现的实体写进了 `must_include_regex`。
- 必须遵守的方法或工具写进了 `must_tool_use`。
- `scoring_prompt` 能区分高质量答案和泛泛回答。
- 至少跑过一次单 case，并检查了 `report.md`、`scores.json` 和 trace。
