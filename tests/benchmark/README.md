# Benchmark 使用指南

`tests/benchmark` 是一个用于评估 Agent 的 benchmark harness。它会读取 case YAML，
把输入数据复制到隔离工作区，运行被测 Agent，保存 trace，再用 LLM as a judge
根据标准答案要点和评分 rubric 打分。

适合用来回答这类问题：

- 某个 Agent 能不能在固定输入数据上给出稳定、正确的结论。
- Agent 是否真的调用了要求的方法或工具，而不是只给出看起来正确的答案。
- 不同 Agent、不同模型、不同 skill 在同一批 case 上的效果、耗时、token 和工具调用差异。

## 文档入口

- [QUICKSTART.md](QUICKSTART.md)：最快跑通 benchmark 的命令。
- [CASE_GUIDE.md](CASE_GUIDE.md)：如何新增 benchmark case。
- [DEMO.md](DEMO.md)：几个 case 的完整示例和结果解读。

## 目录结构

```text
tests/benchmark/
  benchmarks/                 # case YAML，一个 YAML 对应一个 case
  data/                       # case 输入数据
  runs/                       # 运行产物，默认不提交具体 run
  src/
    run_benchmark.py          # CLI 入口
    schema.py                 # case 字段校验
    metrics.py                # token、耗时、工具调用统计
    judge.py                  # 本地 heuristic judge，用于 smoke test
    mock_agent.py             # 本地 heuristic agent，用于 smoke test
    msagent_cli.py            # msAgent adapter
    third_party/
      codex_cli.py            # Codex CLI adapter
      claude_cli.py           # Claude CLI adapter
```

## 快速运行

从仓库根目录进入 benchmark 目录：

```bash
cd tests/benchmark
```

安装为可执行命令：

```bash
python3 -m pip install -e .
```

跑一个本地 smoke case，不调用真实模型：

```bash
benchmark-builder \
  --config benchmarks/mock_agent_smoke.yaml \
  --out runs/smoke \
  --agent heuristic \
  --judge heuristic
```

不安装也可以直接运行：

```bash
PYTHONPATH=src python3 -m run_benchmark \
  --config benchmarks/mock_agent_smoke.yaml \
  --out runs/smoke \
  --agent heuristic \
  --judge heuristic
```

跑完整目录时，把 `--config` 指向 `benchmarks`：

```bash
PYTHONPATH=src python3 -m run_benchmark \
  --config benchmarks \
  --out runs/all-smoke \
  --agent heuristic \
  --judge heuristic
```

## 使用 msAgent

如果系统 PATH 里已经有 `msagent` 命令：

```bash
PYTHONPATH=src python3 -m run_benchmark \
  --config benchmarks/agent_list.yaml \
  --out runs/msagent-agent-list \
  --agent msagent-cli \
  --judge msagent-cli \
  --msagent-agent Hermes
```

从源码或虚拟环境运行 msAgent 时，用 `MSAGENT_CLI` 指定完整命令：

```bash
MSAGENT_CLI="uv --project /path/to/msagent run msagent" \
PYTHONPATH=src python3 -m run_benchmark \
  --config benchmarks/agent_list.yaml \
  --out runs/msagent-agent-list \
  --agent msagent-cli \
  --judge msagent-cli \
  --msagent-agent Hermes
```

常用 msAgent 相关选项：

- `--msagent-agent Hermes`：选择 msAgent 内置 agent；不传时使用 `MSAGENT_AGENT`，再不传默认 `Hermes`。
- `--model deepseek-chat`：给被测 Agent 指定模型。
- `--judge-model deepseek-chat`：给 judge 单独指定模型；不传时沿用 `--model`。
- `MSAGENT_APPROVAL_MODE=aggressive`：msAgent approval mode，默认就是 `aggressive`。

msAgent adapter 会为每个 case 创建临时隔离工作区，并复制当前仓库的 `.msagent`
配置。运行 trace 会写到 `runs/<run_id>/runtime/{agent,judge}/`。

## 使用 Codex CLI 或 Claude CLI

Codex CLI：

```bash
PYTHONPATH=src python3 -m run_benchmark \
  --config benchmarks/mock_agent_smoke.yaml \
  --out runs/codex-cli \
  --agent codex-cli \
  --judge codex-cli
```

如果命令不在 PATH 里，可以设置：

```bash
export CODEX_CLI=/path/to/codex
```

Claude CLI：

```bash
PYTHONPATH=src python3 -m run_benchmark \
  --config benchmarks/mock_agent_smoke.yaml \
  --out runs/claude-cli \
  --agent claude-cli \
  --judge claude-cli
```

如果命令不在 PATH 里，可以设置：

```bash
export CLAUDE_CLI=/path/to/claude
```

让 Codex CLI 或 Claude CLI 走 DeepSeek 时，设置共享开关，然后照常传
`--model deepseek-chat`。Codex CLI 需要先启动本地 Responses proxy
（默认 `http://127.0.0.1:8787/v1`），因为当前 Codex 自定义 provider 只支持
Responses API，而 DeepSeek 对外是 Chat Completions API：

```bash
export BENCHMARK_DEEPSEEK=1
# 如果要跑 Claude CLI，也需要：
# export DEEPSEEK_API_KEY=<your-deepseek-api-key>

PYTHONPATH=src python3 -m run_benchmark \
  --config benchmarks/real_case1.yaml \
  --out runs/real-case1-codex-deepseek \
  --agent codex-cli \
  --judge codex-cli \
  --model deepseek-chat \
  --judge-model deepseek-chat \
  --timeout-seconds 1800

PYTHONPATH=src python3 -m run_benchmark \
  --config benchmarks/real_case1.yaml \
  --out runs/real-case1-claude-deepseek \
  --agent claude-cli \
  --judge claude-cli \
  --model deepseek-chat \
  --judge-model deepseek-chat \
  --timeout-seconds 1800
```

也可以只打开单个 adapter：`CODEX_DEEPSEEK=1` 或 `CLAUDE_DEEPSEEK=1`。
Codex 默认注入本地 proxy 的 Responses provider；Claude 默认注入
`https://api.deepseek.com/anthropic` 并把 `DEEPSEEK_API_KEY` 映射为 `ANTHROPIC_API_KEY`。
如果你的 proxy 不在默认地址，可以设置 `CODEX_DEEPSEEK_BASE_URL`。

## 常用参数

```text
--config            必填。单个 case YAML，或包含多个 YAML 的目录。
--out               必填。运行产物输出目录。
--agent             被测 Agent：codex-cli / claude-cli / msagent-cli / heuristic。
--judge             Judge：codex-cli / claude-cli / msagent-cli / heuristic。
--model             被测 Agent 模型。
--judge-model       Judge 模型；不传时默认使用 --model。
--timeout-seconds   单个 agent/judge 子进程超时时间，默认 900 秒。
--msagent-agent     msAgent persona，默认 Hermes。
```

`heuristic` 只用于本地 smoke test，能验证读取 case、复制数据、生成报告等管线是否正常。
真实评测请使用 `msagent-cli`、`codex-cli` 或 `claude-cli`。

## Case 是什么

每个 case YAML 描述一次评测任务。最小示例：

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

字段含义简表：

| 字段 | 必填 | 说明 |
| --- | :---: | --- |
| `id` | 否 | case 标识；不写时使用 YAML 文件名 |
| `input_data_path` | 是 | 输入数据目录或文件，相对 YAML 文件解析 |
| `skill_path` | 否 | 可选 skill 目录；会复制为隔离工作区里的 `skill/` |
| `prompt` | 是 | 给被测 Agent 的任务说明 |
| `must_include` | 是 | 最终答案必须语义覆盖的要点 |
| `must_include_regex` | 否 | 对最终 `answer` 文本做正则硬校验 |
| `must_tool_use` | 否 | trace 中必须出现的工具调用 |
| `scoring_prompt` | 是 | judge 的 0-5 分评分标准 |
| `tool_budget` | 否 | 免费工具调用额度，默认 20，只影响效率报告 |

更完整的 case 编写建议见 [CASE_GUIDE.md](CASE_GUIDE.md)。

## Agent 输出格式

被测 Agent 必须返回一个 JSON object：

```json
{
  "answer": "最终答案文本",
  "evidence": ["支持答案的证据 1", "支持答案的证据 2"],
  "reasoning_summary": "简短、可审计的推理摘要",
  "confidence": 0.8
}
```

注意：

- `prompt` 和复制后的 `input_data/` 会提供给被测 Agent。
- `must_include`、`must_include_regex` 和 `scoring_prompt` 不会提供给被测 Agent。
- 如果配置了 `skill_path`，被测 Agent 会看到复制后的 `skill/` 目录，并被要求先读 `skill/SKILL.md`。
- Agent 不应该读取 benchmark 源码、case YAML、历史 run 输出或父目录。

## 评分规则

judge 负责两件事：

1. 判断 `must_include` 每一项是否被最终答案语义覆盖。
2. 根据 `scoring_prompt` 给出 `rubric_score`，范围 0 到 5。

runner 还会做两个确定性硬校验：

- `must_include_regex`：用 Python `re.search` 检查最终 `answer` 文本。
- `must_tool_use`：在 trace 的工具调用里做大小写不敏感子串匹配。

最终分数：

```text
任一 must_include 或 must_include_regex 未覆盖  ->  score = 0
任一 must_tool_use 未命中                       ->  score = 0
否则                                           ->  score = rubric_score / 5
```

工具使用效率只写入报告，不折进最终分数：

```text
over_budget = max(0, tool_calls - tool_budget)
efficiency_factor = clamp(1 - over_budget * tool_penalty_per_call, tool_penalty_floor, 1.0)
```

默认：

- `tool_budget = 20`
- `tool_penalty_per_call = 0.02`
- `tool_penalty_floor = 0.5`

## 输出结果怎么看

每次运行会写入：

```text
runs/<run_id>/
  traces/<case_id>.trace.json
  judge/<case_id>.judge.json
  metrics/<case_id>.metrics.json
  scores.json
  report.md
  runtime/
    agent/
    judge/
```

常看这几个文件：

- `report.md`：人看的汇总表，适合快速判断哪几个 case 没过。
- `scores.json`：机器可读的完整评分结果。
- `traces/<case_id>.trace.json`：Agent 过程 trace，用来排查工具调用和最终答案。
- `judge/<case_id>.judge.json`：judge 对每个 `must_include` 的判断和理由。
- `metrics/<case_id>.metrics.json`：token、耗时、工具调用聚合。
- `runtime/`：底层 CLI 的 stdout、stderr、jsonl 或 msAgent trace。

快速查看总分：

```bash
jq '.average_score, .scores[] | {case_id, score, judge_score, must_include_pass, must_tool_use_pass}' \
  runs/smoke/scores.json
```

查看某个 case 缺了什么：

```bash
jq '.scores[] | select(.score == 0) | {case_id, must_include_results, must_tool_use_results, weaknesses}' \
  runs/smoke/scores.json
```

## 常见问题

### 跑真实 Agent 前先跑什么

先跑 `heuristic + heuristic` 的 smoke case：

```bash
PYTHONPATH=src python3 -m run_benchmark \
  --config benchmarks/mock_agent_smoke.yaml \
  --out runs/smoke \
  --agent heuristic \
  --judge heuristic
```

如果这个失败，优先修 case YAML、依赖安装或输出路径问题；如果这个通过，再切真实
CLI agent。

### 为什么答案对了还是 0 分

通常是两个原因：

- `must_include_regex` 没匹配最终 `answer` 文本。
- `must_tool_use` 要求的工具没有在 trace 中出现。

这两个都是硬门槛，任何一项失败都会把最终 `score` 置为 0。可以打开
`runs/<run_id>/report.md` 看 `Missing Items` 和 `Missing Tools`。

### 为什么 token 或工具调用很高

benchmark 会把输入数据复制到隔离工作区，但不会阻止 Agent 探索大文件。写 case 时应在
`prompt` 和可选 `skill_path` 中提醒 Agent 优先读 compact summary，避免直接打开
`trace_view.json`、大型数据库、原始 profiler 目录等大文件。

### 能不能只用 msAgent 跑 agent，用别的模型 judge

可以。`--agent` 和 `--judge` 是独立参数，例如：

```bash
PYTHONPATH=src python3 -m run_benchmark \
  --config benchmarks/agent_list.yaml \
  --out runs/msagent-codex-judge \
  --agent msagent-cli \
  --judge codex-cli \
  --msagent-agent Hermes
```

### 输出目录可以复用吗

可以，但新运行会覆盖同名文件。建议每次使用新的 `runs/<run_id>`，例如带上日期、
模型名或 case 名。
