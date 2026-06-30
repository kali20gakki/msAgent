from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from third_party.claude_cli import ClaudeCliAgent, ClaudeCliJudge
from third_party.codex_cli import CodexCliAgent, CodexCliJudge
from judge import MockLLMJudge, normalized_judge_score
from metrics import build_case_metrics
from msagent_cli import MsagentCliAgent, MsagentCliJudge
from mock_agent import MockBenchmarkAgent
from schema import BenchmarkCase, load_suite


def run(
    config: Path,
    out_dir: Path,
    agent_kind: str = "codex-cli",
    judge_kind: str = "codex-cli",
    model: str | None = None,
    judge_model: str | None = None,
    timeout_seconds: int = 900,
    msagent_agent: str | None = None,
) -> dict[str, Any]:
    suite = load_suite(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = out_dir / "runtime"
    agent = _build_agent(
        agent_kind,
        workspace=Path.cwd(),
        artifact_dir=runtime_dir / "agent",
        model=model,
        timeout_seconds=timeout_seconds,
        msagent_agent=msagent_agent,
    )
    judge = _build_judge(
        judge_kind,
        workspace=Path.cwd(),
        artifact_dir=runtime_dir / "judge",
        model=judge_model or model,
        timeout_seconds=timeout_seconds,
        msagent_agent=msagent_agent,
    )

    traces_dir = out_dir / "traces"
    metrics_dir = out_dir / "metrics"
    traces_dir.mkdir(parents=True, exist_ok=True)
    judge_dir = out_dir / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    scores = []
    failures = []
    metrics_items = []
    for case in suite.cases:
        try:
            input_path = case.resolve_input_path()
            if not input_path.exists():
                raise FileNotFoundError(f"Missing input data for {case.id}: {input_path}")

            trace = agent.run(case, input_path)
            trace_path = traces_dir / f"{case.id}.trace.json"
            trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")

            judge_result = _normalize_judge_result(case, judge.judge(case, trace), trace)
            judge_path = judge_dir / f"{case.id}.judge.json"
            judge_path.write_text(
                json.dumps(judge_result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            case_metrics = build_case_metrics(case.id, trace, judge_result)
            metrics_path = metrics_dir / f"{case.id}.metrics.json"
            metrics_path.write_text(
                json.dumps(case_metrics, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            metrics_items.append(case_metrics)

            tool_use_results = _tool_use_results(case, case_metrics)
            must_tool_use_pass = all(item["used"] for item in tool_use_results)
            efficiency = _efficiency_report(case, case_metrics)
            final_score = _final_score(judge_result, must_tool_use_pass)
            score = {
                "case_id": case.id,
                "status": "completed",
                "must_include_pass": judge_result["must_include_pass"],
                "must_include_results": judge_result["must_include_results"],
                "must_tool_use_pass": must_tool_use_pass,
                "must_tool_use_results": tool_use_results,
                "judge_score": judge_result["rubric_score"],
                "score": final_score,
                "efficiency": efficiency,
                "strengths": judge_result.get("strengths", []),
                "weaknesses": judge_result.get("weaknesses", []),
                "trace_path": str(trace_path),
                "metrics_path": str(metrics_path),
                "judge_path": str(judge_path),
            }
            scores.append(score)
        except Exception as exc:
            score = _failed_case_score(case, exc, out_dir / "failures")
            scores.append(score)
            failures.append(
                {
                    "case_id": case.id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "failure_path": score["failure_path"],
                }
            )

    judge_scores = [float(item["judge_score"]) for item in scores]
    report = {
        "suite": suite.name,
        "case_count": len(suite.cases),
        "completed_count": len(scores) - len(failures),
        "failed_count": len(failures),
        "failed_cases": failures,
        "judge_enabled": True,
        "average_score": round(mean(item["score"] for item in scores), 4),
        "average_judge_score": round(mean(judge_scores), 4),
        "must_include_pass_rate": round(
            mean(1.0 if item["must_include_pass"] else 0.0 for item in scores),
            4,
        ),
        "must_tool_use_pass_rate": round(
            mean(1.0 if item["must_tool_use_pass"] else 0.0 for item in scores),
            4,
        ),
        "average_efficiency_factor": round(
            mean(item["efficiency"]["efficiency_factor"] for item in scores),
            4,
        ),
        "token_usage": _sum_token_usage(metrics_items),
        "duration_ms": _sum_durations(metrics_items),
        "tool_calls": _sum_tool_calls(metrics_items),
        "scores": scores,
    }

    (out_dir / "scores.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(_render_markdown_report(report), encoding="utf-8")
    return report


def _build_agent(
    agent_kind: str,
    *,
    workspace: Path,
    artifact_dir: Path,
    model: str | None,
    timeout_seconds: int,
    msagent_agent: str | None,
) -> Any:
    if agent_kind == "codex-cli":
        return CodexCliAgent(
            workspace=workspace,
            artifact_dir=artifact_dir,
            model=model,
            timeout_seconds=timeout_seconds,
        )
    if agent_kind == "claude-cli":
        return ClaudeCliAgent(
            workspace=workspace,
            artifact_dir=artifact_dir,
            model=model,
            timeout_seconds=timeout_seconds,
        )
    if agent_kind == "msagent-cli":
        return MsagentCliAgent(
            workspace=workspace,
            artifact_dir=artifact_dir,
            model=model,
            timeout_seconds=timeout_seconds,
            msagent_agent=msagent_agent,
        )
    if agent_kind == "heuristic":
        return MockBenchmarkAgent()
    raise ValueError(f"Unknown agent kind: {agent_kind}")


def _build_judge(
    judge_kind: str,
    *,
    workspace: Path,
    artifact_dir: Path,
    model: str | None,
    timeout_seconds: int,
    msagent_agent: str | None,
) -> Any:
    if judge_kind == "codex-cli":
        return CodexCliJudge(
            workspace=workspace,
            artifact_dir=artifact_dir,
            model=model,
            timeout_seconds=timeout_seconds,
        )
    if judge_kind == "claude-cli":
        return ClaudeCliJudge(
            workspace=workspace,
            artifact_dir=artifact_dir,
            model=model,
            timeout_seconds=timeout_seconds,
        )
    if judge_kind == "msagent-cli":
        return MsagentCliJudge(
            workspace=workspace,
            artifact_dir=artifact_dir,
            model=model,
            timeout_seconds=timeout_seconds,
            msagent_agent=msagent_agent,
        )
    if judge_kind == "heuristic":
        return MockLLMJudge()
    raise ValueError(f"Unknown judge kind: {judge_kind}")


def _normalize_judge_result(
    case: BenchmarkCase,
    judge_result: dict[str, Any],
    trace: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(judge_result)
    normalized["rubric_score"] = _clamp_score(normalized.get("rubric_score", 0.0))
    normalized["must_include_results"] = _normalize_must_include_results(case, normalized)
    normalized["must_include_results"].extend(_regex_must_include_results(case, trace))
    normalized["must_include_pass"] = all(item["covered"] for item in normalized["must_include_results"])
    return normalized


def _normalize_must_include_results(
    case: BenchmarkCase,
    judge_result: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_results = judge_result.get("must_include_results", [])
    by_item: dict[str, dict[str, Any]] = {}
    if isinstance(raw_results, list):
        for item in raw_results:
            if isinstance(item, dict):
                key = str(item.get("item", ""))
                if key and key not in by_item:
                    by_item[key] = item
    elif isinstance(raw_results, dict):
        for key, covered in raw_results.items():
            key_text = str(key)
            if key_text and key_text not in by_item:
                by_item[key_text] = {
                    "item": key_text,
                    "covered": bool(covered),
                    "reason": "Covered by judge output." if covered else "Not covered by judge output.",
                }

    normalized = []
    for required_item in case.must_include:
        raw = by_item.get(required_item)
        if raw is None:
            normalized.append(
                {
                    "item": required_item,
                    "covered": False,
                    "reason": "Judge did not return a result for this required item.",
                }
            )
            continue
        normalized.append(
            {
                "item": required_item,
                "covered": bool(raw.get("covered")),
                "reason": str(raw.get("reason", "")).strip(),
            }
        )
    return normalized


def _regex_must_include_results(
    case: BenchmarkCase,
    trace: dict[str, Any],
) -> list[dict[str, Any]]:
    if not case.must_include_regex:
        return []

    answer_text = _final_answer_text(trace)
    results = []
    for pattern in case.must_include_regex:
        covered = re.search(pattern, answer_text, re.IGNORECASE | re.MULTILINE) is not None
        results.append(
            {
                "item": pattern,
                "covered": covered,
                "reason": (
                    "Regex pattern matched the final answer text."
                    if covered
                    else "Regex pattern did not match the final answer text."
                ),
            }
        )
    return results


def _final_answer_text(trace: dict[str, Any]) -> str:
    for event in reversed(trace.get("events", [])):
        if event.get("type") != "final_answer":
            continue
        answer = event.get("answer")
        if not isinstance(answer, dict):
            return ""
        raw_answer = answer.get("answer", "")
        if isinstance(raw_answer, str):
            return raw_answer
        return json.dumps(raw_answer, ensure_ascii=False)
    return ""


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, min(5.0, score)), 2)


def _final_score(judge_result: dict[str, Any], must_tool_use_pass: bool = True) -> float:
    if not judge_result.get("must_include_pass"):
        return 0.0
    if not must_tool_use_pass:
        return 0.0
    return round(normalized_judge_score(judge_result), 4)


def _tool_use_results(case: BenchmarkCase, case_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Deterministically check whether required tools were actually called.

    Matching is case-insensitive and forgiving: a required entry is considered
    used if it equals or is a substring of any tool name in the trace (so
    ``msprof`` matches ``msprof-mcp_analyze_kernel_details``). This is a hard
    gate, like must_include: any required tool that was never called sets
    score=0.
    """
    by_tool = case_metrics.get("tool_calls", {}).get("agent", {}).get("by_tool", {})
    results = []
    for required in case.must_tool_use:
        needle = required.casefold()
        matched = {tool: int(count) for tool, count in by_tool.items() if needle in str(tool).casefold()}
        results.append(
            {
                "tool": required,
                "used": bool(matched),
                "count": sum(matched.values()),
                "matched_tools": sorted(matched),
            }
        )
    return results


def _efficiency_report(case: BenchmarkCase, case_metrics: dict[str, Any]) -> dict[str, Any]:
    """Report tool-use efficiency for human review.

    This is informational only: the efficiency factor is NOT folded into the
    final score. It surfaces how far a run exceeded its tool budget and what a
    multiplicative penalty would look like if it were applied later.
    """
    tool_calls = int(case_metrics.get("tool_calls", {}).get("agent", {}).get("count", 0))
    over_budget = max(0, tool_calls - case.tool_budget)
    raw_factor = 1.0 - over_budget * case.tool_penalty_per_call
    efficiency_factor = round(max(case.tool_penalty_floor, min(1.0, raw_factor)), 4)
    return {
        "tool_calls": tool_calls,
        "tool_budget": case.tool_budget,
        "over_budget": over_budget,
        "tool_penalty_per_call": case.tool_penalty_per_call,
        "tool_penalty_floor": case.tool_penalty_floor,
        "efficiency_factor": efficiency_factor,
        "applied_to_score": False,
    }


def _failed_case_score(case: BenchmarkCase, error: Exception, failures_dir: Path) -> dict[str, Any]:
    failures_dir.mkdir(parents=True, exist_ok=True)
    error_info = {
        "type": type(error).__name__,
        "message": str(error),
    }
    failure_path = failures_dir / f"{case.id}.failure.json"
    failure_payload = {
        "case_id": case.id,
        "status": "failed",
        "error": error_info,
    }
    failure_path.write_text(
        json.dumps(failure_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "case_id": case.id,
        "status": "failed",
        "must_include_pass": False,
        "must_include_results": [
            {
                "item": item,
                "covered": False,
                "reason": "Case failed before judging.",
            }
            for item in case.must_include
        ],
        "must_tool_use_pass": False,
        "must_tool_use_results": [
            {
                "tool": tool,
                "used": False,
                "count": 0,
                "matched_tools": [],
            }
            for tool in case.must_tool_use
        ],
        "judge_score": 0.0,
        "score": 0.0,
        "efficiency": _empty_efficiency_report(case),
        "strengths": [],
        "weaknesses": [error_info["message"]],
        "error": error_info,
        "failure_path": str(failure_path),
    }


def _empty_efficiency_report(case: BenchmarkCase) -> dict[str, Any]:
    return {
        "tool_calls": 0,
        "tool_budget": case.tool_budget,
        "over_budget": 0,
        "tool_penalty_per_call": case.tool_penalty_per_call,
        "tool_penalty_floor": case.tool_penalty_floor,
        "efficiency_factor": 1.0,
        "applied_to_score": False,
    }


def _render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# Benchmark Report: {report['suite']}",
        "",
        f"- Cases: {report['case_count']}",
        f"- Failed cases: {report.get('failed_count', 0)}",
        "- Judge: enabled",
        f"- Average score: {report['average_score']:.4f}",
        _format_judge_score(report.get("average_judge_score")),
        f"- Must-include pass rate: {report['must_include_pass_rate']:.4f}",
        f"- Must-tool-use pass rate: {report['must_tool_use_pass_rate']:.4f}",
        f"- Average efficiency factor (informational): {report['average_efficiency_factor']:.4f}",
        f"- Total tokens: {report['token_usage']['total_tokens']}",
        f"- Total duration: {report['duration_ms']['total']} ms",
        f"- Agent tool calls: {report['tool_calls']['agent']['count']}",
        "",
        "| Case | Final | Judge | Must Include | Must Tool Use | Tools | Budget | Over | Eff | Missing Items | Missing Tools |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for item in report["scores"]:
        missing_items = [result["item"] for result in item.get("must_include_results", []) if not result.get("covered")]
        missing_tools = [result["tool"] for result in item.get("must_tool_use_results", []) if not result.get("used")]
        efficiency = item.get("efficiency", {})
        lines.append(
            "| {case_id} | {score:.4f} | {judge_score:.2f} | {must_include} | {must_tool_use} | {tools} | {budget} | {over} | {eff:.2f} | {missing} | {missing_tools} |".format(
                case_id=item["case_id"],
                score=item["score"],
                judge_score=float(item["judge_score"]),
                must_include="pass" if item["must_include_pass"] else "fail",
                must_tool_use="pass" if item["must_tool_use_pass"] else "fail",
                tools=efficiency.get("tool_calls", 0),
                budget=efficiency.get("tool_budget", 0),
                over=efficiency.get("over_budget", 0),
                eff=float(efficiency.get("efficiency_factor", 1.0)),
                missing=", ".join(missing_items) or "[]",
                missing_tools=", ".join(missing_tools) or "[]",
            )
        )
    lines.append("")
    return "\n".join(lines)


def _format_judge_score(value: Any) -> str:
    if value is None:
        return "- Average judge score: n/a"
    return f"- Average judge score: {float(value):.2f}/5"


def _sum_token_usage(metrics_items: list[dict[str, Any]]) -> dict[str, int]:
    total: dict[str, Any] = {
        "available": True,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "unavailable_cases": [],
    }
    for item in metrics_items:
        usage = item.get("token_usage", {}).get("total", {})
        if not usage.get("available", True):
            total["available"] = False
            total["unavailable_cases"].append(item.get("case_id"))
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            total[key] += int(usage.get(key, 0))
    return total


def _sum_durations(metrics_items: list[dict[str, Any]]) -> dict[str, int]:
    total = {"agent": 0, "judge": 0, "total": 0}
    for item in metrics_items:
        duration = item.get("duration_ms", {})
        for key in total:
            total[key] += int(duration.get(key, 0))
    return total


def _sum_tool_calls(metrics_items: list[dict[str, Any]]) -> dict[str, Any]:
    total: dict[str, Any] = {
        "agent": {
            "count": 0,
            "by_tool": {},
        },
    }
    by_tool = total["agent"]["by_tool"]
    for item in metrics_items:
        agent_calls = item.get("tool_calls", {}).get("agent", {})
        total["agent"]["count"] += int(agent_calls.get("count", 0))
        for tool, count in agent_calls.get("by_tool", {}).items():
            by_tool[tool] = int(by_tool.get(tool, 0)) + int(count)
    total["agent"]["by_tool"] = dict(sorted(by_tool.items()))
    return total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a judge-scored benchmark suite.")
    parser.add_argument("--config", type=Path, required=True, help="Path to benchmark YAML.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for traces and scores.")
    parser.add_argument(
        "--agent",
        choices=["codex-cli", "claude-cli", "msagent-cli", "heuristic"],
        default="codex-cli",
        help=(
            "Agent adapter to run. codex-cli, claude-cli, and msagent-cli are real CLI agents; heuristic is local-only."
        ),
    )
    parser.add_argument(
        "--judge",
        choices=["codex-cli", "claude-cli", "msagent-cli", "heuristic"],
        default="codex-cli",
        help=(
            "Judge adapter to run. codex-cli, claude-cli, and msagent-cli are real LLM judges; heuristic is local-only."
        ),
    )
    parser.add_argument("--model", help="Model for the selected real CLI agent.")
    parser.add_argument("--judge-model", help="Model for the selected real CLI judge. Defaults to --model.")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument(
        "--msagent-agent",
        help="Built-in msAgent persona to use for msagent-cli runs. Defaults to MSAGENT_AGENT or Hermes.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run(
        args.config,
        args.out,
        agent_kind=args.agent,
        judge_kind=args.judge,
        model=args.model,
        judge_model=args.judge_model,
        timeout_seconds=args.timeout_seconds,
        msagent_agent=args.msagent_agent,
    )
    print(f"Ran {report['case_count']} cases from {report['suite']}; average_score={report['average_score']:.4f}")


if __name__ == "__main__":
    main()
