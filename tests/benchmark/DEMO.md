# Benchmark 演示：四个 Case YAML 例子

本文用四个 case 演示 benchmark harness 的写法与评分：三个"读小资料答题"的简单 case，加一个"读真实大数据找慢卡"的实战 case。

---

## 一分钟看懂评测流程

```text
case.yaml ──▶ 复制 input_data 到隔离工作区 ──▶ 被测 Agent 只看 prompt + input_data/
                                                      │
                                                      ▼
                                        Agent 输出统一 JSON 答案
                                                      │
                                                      ▼
        Judge 对照 must_include + scoring_prompt 打分（must_include_regex / must_tool_use 确定性校验）
                                                      │
                                                      ▼
              落盘：trace / judge / metrics / scores.json / report.md
```

评分规则（最终 `score`，范围 0–1）：

```text
任一 must_include / must_include_regex 未覆盖  ->  score = 0
任一 must_tool_use 要求的工具未被调用          ->  score = 0
否则                                          ->  score = rubric_score / 5
```

两类**硬门槛**：

- `must_include` / `must_include_regex`：答案"说"对了什么（结果对不对）。
- `must_tool_use`：Agent"做"了什么（过程/方法论对不对）——要求的工具没调用，哪怕答案对也判 0。

工具使用效率（`efficiency_factor`）单独报告，**不影响** `score`。

---

## Case 字段速查

| 字段 | 必填 | 作用 | 谁能看到 |
| --- | :---: | --- | --- |
| `id` | 否 | case 标识，缺省用文件名 | — |
| `input_data_path` | 是 | 输入数据目录/文件（相对 YAML 解析） | Agent（复制进 `input_data/`） |
| `prompt` | 是 | 给 Agent 的任务 | Agent |
| `must_include` | 是 | 答案必须语义覆盖的要点（硬门槛） | 仅 Judge |
| `must_include_regex` | 否 | 对最终 `answer` 文本做确定性正则校验（硬门槛） | 仅 runner |
| `must_tool_use` | 否 | trace 中必须出现的工具调用（硬门槛，确定性校验） | 仅 runner |
| `scoring_prompt` | 是 | Judge 0–5 分评分标准 | 仅 Judge |
| `tool_budget` | 否 | 免费工具调用额度（默认 20，仅效率报告） | runner |

> `must_include` / `scoring_prompt` 对被测 Agent **不可见**，避免泄题。
> `must_tool_use` 的匹配是大小写不敏感的子串匹配：写 `msprof-mcp` 可命中
> `msprof-mcp_analyze_kernel_details`。

---

## 例子 1：`agent_default` — 单点事实问答

**场景**：问 msAgent 的默认 agent 是谁、职责是什么。考"结论 + 描述"。

```yaml
id: agent_default
input_data_path: ../data/msagent_agents
prompt: >
  根据 input_data 中的资料回答：msAgent 的默认 agent 是哪个？它的主要职责是什么？
must_include:
  - Hermes
  - Ascend NPU profiling 分析
must_include_regex:
  - "Hermes"
must_tool_use:
  - read_file
scoring_prompt: >
  按 0-5 分评分。重点看是否正确指出默认 agent 是 Hermes，以及是否正确描述其职责
  （Ascend NPU profiling 分析）。结论错误或张冠李戴应给低分。
```

**看点**：

- `must_include` 有两项——答出名字还不够，还要答对职责。
- `must_include_regex: Hermes` 是兜底硬校验：答案文本里没出现 `Hermes` 直接判 0 分。
- `must_tool_use: read_file` 要求 Agent 真的读了文件——靠瞎猜（不读文件）即使答对也判 0。

**实跑结果（msAgent + deepseek-chat）**

> Agent 答案：
>
> ```text
> msAgent 的默认 agent 是 Hermes。它的主要职责是作为 Ascend NPU profiling 分析
> agent，采用 msprof-mcp-first 工作流，负责 Ascend NPU 性能分析。
> ```

| score | judge | must_include | must_tool_use | tools / budget | eff | token | 耗时 |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| **1.0000** | 5.0/5 | 2/2 pass | read_file ✓ | 2 / 20 | 1.00 | 73,055 | ~17s |

Judge strengths：正确指出 Hermes 为默认 agent、职责描述准确、引用了 `agents.md` 证据。

---

## 例子 2：`agent_list` — 列表完整性

**场景**：要求列出全部内置 agent。考"覆盖是否完整"。

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

**看点**：

- 5 个 `must_include` 是"全有才过"的硬门槛——漏一个 `score` 就是 0。
- `scoring_prompt` 额外惩罚"编造名称"，区分"列全"和"列全且不瞎编"。
- `must_tool_use: read_file` 确保结论来自真实读取，而非凭空生成。

**实跑结果（msAgent + deepseek-chat）**

> Agent 答案：
>
> ```text
> msAgent 内置了 5 个 agent：Hermes、Zephyr、Icarus、Minos、Accuracy。
> ```

| score | judge | must_include | must_tool_use | tools / budget | eff | token | 耗时 |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| **1.0000** | 5.0/5 | 5/5 pass | read_file ✓ | 2 / 20 | 1.00 | 73,235 | ~18s |

Judge strengths：完整列出全部 5 个、无遗漏无编造、附带证据引用。

---

## 例子 3：`agent_pick` — 决策/匹配

**场景**：给一个需求，让 Agent 选对应的 agent。考"按描述做选择"。

```yaml
id: agent_pick
input_data_path: ../data/msagent_agents
prompt: >
  根据 input_data 中的资料回答：如果我要做 msModelSlim 的模型分析与适配，应该使用
  哪个 agent？
must_include:
  - Zephyr
must_include_regex:
  - "Zephyr"
must_tool_use:
  - read_file
scoring_prompt: >
  按 0-5 分评分。重点看是否正确选出 Zephyr，并说明它负责 msModelSlim 模型分析与适配。
  选错 agent 应给 0 分。
```

**看点**：

- 单一正确答案 `Zephyr`，选错即 0，最适合演示"硬门槛"。
- `must_tool_use: read_file` 同样要求决策建立在读取资料之上。

**实跑结果（msAgent + deepseek-chat）**

> Agent 答案：`Zephyr`

| score | judge | must_include | must_tool_use | tools / budget | eff | token | 耗时 |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| **1.0000** | 5.0/5 | 1/1 pass | read_file ✓ | 2 / 20 | 1.00 | 73,070 | ~16s |

Judge strengths：正确选出 Zephyr、引用 `agents.md` 证据、并说明了为何排除其他 agent。

---

> 以上三个 case 共用同一份小资料 `data/msagent_agents/agents.md`（5 个 agent 的一句话描述），所以无需准备真实大数据即可跑通、判分。

---

## 例子 4：`real_case1` — 真实数据实战（含 `must_tool_use` 门槛触发）

**场景**：丢给 Agent 一整个 Ascend profiler 数据包，让它自己分析并识别慢卡。这里额外
要求它必须走 msprof-mcp 方法论。

```yaml
id: real_agent_smoke
input_data_path: ../data/kv_cache_type_page_seqlen_4096_bs_1_profile_count_0
prompt: >
  识别数据中的慢卡
must_include:
  - rank3
must_tool_use:
  - read_file
  - msprof-mcp
scoring_prompt: >
  按 0-5 分评分。重点看结论是否准确、证据是否来自输入数据、建议是否可执行。
```

**看点**：

- `input_data_path` 指向真实 profiler 数据包（多 rank 目录），Agent 要自己挑文件、跑分析。
- `prompt` 极简（"识别数据中的慢卡"）——考验 Agent 自主探索能力，而非照抄资料。
- `must_include: rank3` 是金标准答案；`scoring_prompt` 进一步要求"证据来自输入数据、建议可执行"。
- `must_tool_use: msprof-mcp` 强制方法论：必须用 msprof-mcp 工具分析，而不是只靠
  读 csv + 自己算。
- 与前三个对比：前三个是"阅读理解"，这个是"端到端分析"。

**实跑结果（msAgent + deepseek-chat）—— 答案对，却因门槛判 0**

> Agent 答案：
>
> ```text
> Rank 3 is the slow card. Its Stage time (386,230 µs) is the highest among all
> ranks, driven by 69,916 µs of Free/idle time — ~19× the rank average ...
> ```

| score | judge | must_include | must_tool_use | tools / budget | eff | token | 耗时 |
| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| **0.0000** | 4.5/5 | 1/1 pass | read_file ✓ / **msprof-mcp ✗** | 36 / 20 | **0.68** | 1,004,346 | ~135s |

这是本演示**最有看点**的一条：

- 答案其实是对的（rank3），judge 也给到 4.5/5，`must_include` 通过。
- **但 Agent 这一轮没有调用 `msprof-mcp` 工具**（它改用 `execute` + `read_file` 手动算），
  `must_tool_use` 门槛未通过 → 最终 `score` 被直接打成 **0**。
- 这正是 `must_tool_use` 的价值：当你要考核的是"是否按规定方法论解题"而不只是"答案对不对"
  时，它能拦住"结果蒙对、过程不合规"的情况。
- 同一轮还顺带演示了效率：本轮用了 **36** 次工具调用，超预算（20）16 次，
  `efficiency_factor` 降到 **0.68**（注意：效率只报告，真正把分打成 0 的是 `must_tool_use`）。

> 提示：Agent 是否调用 msprof-mcp 在不同 run 间会波动——这种波动恰恰说明了为什么需要
> `must_tool_use` 来强约束方法论。若想看它通过，可放宽为只要求 `read_file`。

---

## 如何运行

本地 smoke（不调真实模型，验证管线）：

```bash
cd tests/benchmark
PYTHONPATH=src python3 -m run_benchmark \
  --config benchmarks/agent_list.yaml \
  --out runs/smoke --agent heuristic --judge heuristic
```

真实 msAgent（源码版 CLI + 模型）：

```bash
export MSAGENT_CLI="/path/to/python /path/to/msagent/run.py"   # 源码版 CLI
PYTHONPATH=src python3 -m run_benchmark \
  --config benchmarks/agent_list.yaml \
  --out runs/real --agent msagent-cli --judge msagent-cli --msagent-agent Hermes
```

跑整个目录（一次跑全部 case）把 `--config` 换成 `benchmarks` 即可。

---

## 四个 case 真实运行汇总（msAgent + deepseek-chat，agent = Hermes）

全部四个 case 都用真实 msAgent 跑 agent 与 judge，结果如下：

| Case | 类型 | Final | Judge | Must Include | Must Tool Use | Tools | Budget | Eff | Token | 耗时 |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `agent_default` | 单点事实 | 1.0000 | 5.00 | pass (2/2) | ✓ read_file | 2 | 20 | 1.00 | 73,055 | ~17s |
| `agent_list` | 列表完整性 | 1.0000 | 5.00 | pass (5/5) | ✓ read_file | 2 | 20 | 1.00 | 73,235 | ~18s |
| `agent_pick` | 决策匹配 | 1.0000 | 5.00 | pass (1/1) | ✓ read_file | 2 | 20 | 1.00 | 73,070 | ~16s |
| `real_case1` | 真实数据实战 | **0.0000** | 4.50 | pass (1/1) | ✗ msprof-mcp | 36 | 20 | 0.68 | 1,004,346 | ~135s |

**几个值得指出的点**：

- 三个阅读理解 case：`must_include` + `must_tool_use` 双门槛都过，judge 5/5，`score = 1.0`。
- `real_case1` 是反面教材：**答案对（rank3）、judge 4.5/5，却因没调用 `msprof-mcp`，
  `must_tool_use` 不通过，`score = 0`**——说明硬门槛能否决"结果对、过程不合规"的运行。
- 两个确定性维度各司其职：`must_tool_use` 直接决定通过/否（score 0），
  `efficiency_factor`（real_case1 = 0.68）只做效率报告、不改 score。
- 工具/token 随任务复杂度上升：阅读理解 2 次 / ~7.3w token；实战 36 次 / ~100w token。

> 复现命令见上一节；产物分别在 `runs/demo_<case_id>/`（含 trace / judge / metrics /
> report.md / runtime 原始输出）。各 run 的工具使用在不同 run 间会有波动。
