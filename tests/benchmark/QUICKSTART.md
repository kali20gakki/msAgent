# Benchmark Quickstart

这份文档只放最常用命令，适合第一次跑 benchmark 或调试环境时使用。

## 1. 进入目录

```bash
cd tests/benchmark
```

## 2. 安装依赖

推荐安装为本地可执行命令：

```bash
python3 -m pip install -e .
```

不想安装也可以后面统一加 `PYTHONPATH=src python3 -m run_benchmark`。

## 3. 跑本地 smoke

本地 smoke 不调用真实模型，用来验证 benchmark 管线是否正常：

```bash
benchmark-builder \
  --config benchmarks/mock_agent_smoke.yaml \
  --out runs/smoke \
  --agent heuristic \
  --judge heuristic
```

等价的不安装版本：

```bash
PYTHONPATH=src python3 -m run_benchmark \
  --config benchmarks/mock_agent_smoke.yaml \
  --out runs/smoke \
  --agent heuristic \
  --judge heuristic
```

成功后会看到类似输出：

```text
Ran 1 cases from mock_agent_smoke; average_score=1.0000
```

查看报告：

```bash
cat runs/smoke/report.md
```

## 4. 跑单个 msAgent case

如果 `msagent` 命令已经在 PATH 中：

```bash
PYTHONPATH=src python3 -m run_benchmark \
  --config benchmarks/agent_list.yaml \
  --out runs/msagent-agent-list \
  --agent msagent-cli \
  --judge msagent-cli \
  --msagent-agent Hermes
```

如果要指定源码版或虚拟环境里的 msAgent：

```bash
MSAGENT_CLI="uv --project /path/to/msagent run msagent" \
PYTHONPATH=src python3 -m run_benchmark \
  --config benchmarks/agent_list.yaml \
  --out runs/msagent-agent-list \
  --agent msagent-cli \
  --judge msagent-cli \
  --msagent-agent Hermes
```

指定模型：

```bash
PYTHONPATH=src python3 -m run_benchmark \
  --config benchmarks/agent_list.yaml \
  --out runs/msagent-agent-list-deepseek \
  --agent msagent-cli \
  --judge msagent-cli \
  --msagent-agent Hermes \
  --model deepseek-chat
```

## 5. 跑 Codex/Claude + DeepSeek

Codex CLI 和 Claude CLI adapter 都支持用 `BENCHMARK_DEEPSEEK=1` 临时切到 DeepSeek。
Codex CLI 默认连接本地 Responses proxy `http://127.0.0.1:8787/v1`，因为
Codex 自定义 provider 需要 Responses API；Claude CLI 直接连接 DeepSeek 的
Anthropic-compatible endpoint。这个开关只影响 benchmark 子进程：

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

## 6. 跑 profiling skill case

profiling 相关 case 会把 `skill_path` 指向仓库内的 skill 目录，Agent 会在隔离工作区中看到
复制后的 `skill/`：

```bash
PYTHONPATH=src python3 -m run_benchmark \
  --config benchmarks/cluster_fast_slow_rank_profiling_skill.yaml \
  --out runs/profiling-skill \
  --agent msagent-cli \
  --judge msagent-cli \
  --msagent-agent Hermes \
  --timeout-seconds 1800
```

真实 profiling 数据可能较大，建议适当调高 `--timeout-seconds`。

## 7. 跑整个目录

```bash
PYTHONPATH=src python3 -m run_benchmark \
  --config benchmarks \
  --out runs/all-msagent \
  --agent msagent-cli \
  --judge msagent-cli \
  --msagent-agent Hermes \
  --timeout-seconds 1800
```

## 8. 看结果

人看的汇总：

```bash
cat runs/msagent-agent-list/report.md
```

机器可读的完整结果：

```bash
jq '.' runs/msagent-agent-list/scores.json
```

只看每个 case 的分数：

```bash
jq '.scores[] | {case_id, score, judge_score, must_include_pass, must_tool_use_pass}' \
  runs/msagent-agent-list/scores.json
```

看工具调用是否命中：

```bash
jq '.scores[] | {case_id, must_tool_use_results}' runs/msagent-agent-list/scores.json
```

看 token、耗时和工具数量：

```bash
jq '{token_usage, duration_ms, tool_calls}' runs/msagent-agent-list/scores.json
```

## 9. 常见失败排查

`msAgent CLI was not found`

```bash
export MSAGENT_CLI="uv --project /path/to/msagent run msagent"
```

`Codex CLI was not found`

```bash
export CODEX_CLI=/path/to/codex
```

`Claude CLI was not found`

```bash
export CLAUDE_CLI=/path/to/claude
```

case 最终 `score = 0`

```bash
cat runs/<run_id>/report.md
jq '.scores[] | {case_id, must_include_results, must_tool_use_results, weaknesses}' runs/<run_id>/scores.json
```

重点看：

- `Missing Items`：答案缺了 `must_include` 或 `must_include_regex`。
- `Missing Tools`：trace 中没有命中 `must_tool_use`。
- `runtime/*/*.stderr.txt`：底层 CLI 报错。
