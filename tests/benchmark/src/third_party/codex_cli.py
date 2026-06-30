from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from schema import BenchmarkCase
from trace import TraceBuilder  # pylint: disable=no-name-in-module


AGENT_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer": {
            "type": "string",
            "description": "The final answer to the benchmark prompt.",
        },
        "evidence": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Short evidence snippets or file-derived facts supporting the answer.",
        },
        "reasoning_summary": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["answer", "evidence", "reasoning_summary", "confidence"],
}


JUDGE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "must_include_pass": {"type": "boolean"},
        "must_include_results": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "item": {"type": "string"},
                    "covered": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["item", "covered", "reason"],
            },
        },
        "rubric_score": {"type": "number", "minimum": 0, "maximum": 5},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "weaknesses": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "must_include_pass",
        "must_include_results",
        "rubric_score",
        "strengths",
        "weaknesses",
    ],
}


class CodexCliUnavailableError(RuntimeError):
    pass


class CodexCliAgent:
    agent_info = {
        "name": "codex-cli-agent",
        "runtime": "codex exec",
        "token_usage_mode": "codex-jsonl",
    }

    def __init__(
        self,
        *,
        workspace: Path,
        artifact_dir: Path,
        model: str | None = None,
        timeout_seconds: int = 900,
    ) -> None:
        self.workspace = workspace.resolve()
        self.artifact_dir = artifact_dir.resolve()
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.codex_path = resolve_codex_cli()

    def run(self, case: BenchmarkCase, input_path: Path) -> dict[str, Any]:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        schema_text = json.dumps(AGENT_OUTPUT_SCHEMA, indent=2)
        schema_artifact_path = self.artifact_dir / "agent_output.schema.json"
        schema_artifact_path.write_text(schema_text, encoding="utf-8")
        final_artifact_path = self.artifact_dir / f"{case.id}.agent.final.json"
        jsonl_path = self.artifact_dir / f"{case.id}.agent.events.jsonl"

        prefix = f"benchmark-builder-agent-{safe_path_component(case.id)}-"
        with tempfile.TemporaryDirectory(prefix=prefix) as temp_dir:
            run_workspace = Path(temp_dir).resolve()
            isolated_input_path = run_workspace / "input_data"
            copy_input_data(input_path, isolated_input_path)
            isolated_skill_path = copy_case_skill(case, run_workspace)
            schema_run_path = run_workspace / "slow_card_output.schema.json"
            schema_run_path.write_text(schema_text, encoding="utf-8")
            final_run_path = run_workspace / "agent.final.json"

            prompt = build_agent_prompt(
                case,
                isolated_input_path,
                visible_input_path=Path("input_data"),
                visible_skill_path=Path("skill") if isolated_skill_path is not None else None,
            )
            cmd = [
                str(self.codex_path),
                "--ask-for-approval",
                "never",
                *build_codex_provider_config_args(),
                "exec",
                "--json",
                "--skip-git-repo-check",
                "--ignore-rules",
                "--output-schema",
                str(schema_run_path),
                "--output-last-message",
                str(final_run_path),
                "--sandbox",
                "read-only",
                "-C",
                str(run_workspace),
            ]
            if self.model:
                cmd.extend(["--model", self.model])
            cmd.append(prompt)

            started = perf_counter()
            completed = subprocess.run(
                cmd,
                cwd=run_workspace,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
                env=build_codex_env(),
            )
            duration_ms = round((perf_counter() - started) * 1000)
            jsonl_path.write_text(completed.stdout, encoding="utf-8")

            raw_events = parse_jsonl(completed.stdout)
            token_usage = extract_token_usage(raw_events)
            if completed.returncode != 0:
                raise RuntimeError(
                    "codex exec failed for agent run "
                    f"{case.id} with exit code {completed.returncode}: {completed.stderr[-4000:]}"
                )
            final_answer = normalize_agent_final(read_structured_final(final_run_path, completed.stdout))
            final_artifact_path.write_text(
                json.dumps(final_answer, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            trace = TraceBuilder(
                case_id=case.id,
                prompt=case.prompt,
                agent={
                    **self.agent_info,
                    "model": self.model,
                    "cli_path": str(self.codex_path),
                    "isolation": "temp-workspace-copied-input",
                },
            )
            trace.add(
                "agent_run",
                command=safe_command_for_trace(cmd),
                cwd=str(run_workspace),
                input_data_path="input_data",
                input_data_isolation="copied",
                skill_path="skill" if isolated_skill_path is not None else None,
                skill_isolation="copied" if isolated_skill_path is not None else None,
                jsonl_path=display_path(jsonl_path, self.workspace),
                jsonl_event_count=len(raw_events),
                stderr_tail=completed.stderr[-2000:],
                duration_ms=duration_ms,
            )
            for event in normalize_codex_events(raw_events):
                trace.add(**event)
            trace.final_answer(final_answer)
            trace.finish(token_usage)
            trace.duration_ms = duration_ms
            return trace.to_dict()


class CodexCliJudge:
    judge_info = {
        "name": "codex-cli-judge",
        "runtime": "codex exec",
        "token_usage_mode": "codex-jsonl",
    }

    def __init__(
        self,
        *,
        workspace: Path,
        artifact_dir: Path,
        model: str | None = None,
        timeout_seconds: int = 900,
    ) -> None:
        self.workspace = workspace.resolve()
        self.artifact_dir = artifact_dir.resolve()
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.codex_path = resolve_codex_cli()

    def judge(
        self,
        case: BenchmarkCase,
        trace: dict[str, Any],
    ) -> dict[str, Any]:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        schema_text = json.dumps(JUDGE_OUTPUT_SCHEMA, indent=2)
        schema_artifact_path = self.artifact_dir / "judge_output.schema.json"
        schema_artifact_path.write_text(schema_text, encoding="utf-8")
        final_artifact_path = self.artifact_dir / f"{case.id}.judge.final.json"
        jsonl_path = self.artifact_dir / f"{case.id}.judge.events.jsonl"

        prompt = build_judge_prompt(case, trace)
        prefix = f"benchmark-builder-judge-{safe_path_component(case.id)}-"
        with tempfile.TemporaryDirectory(prefix=prefix) as temp_dir:
            run_workspace = Path(temp_dir).resolve()
            schema_run_path = run_workspace / "judge_output.schema.json"
            schema_run_path.write_text(schema_text, encoding="utf-8")
            final_run_path = run_workspace / "judge.final.json"

            cmd = [
                str(self.codex_path),
                "--ask-for-approval",
                "never",
                *build_codex_provider_config_args(),
                "exec",
                "--json",
                "--skip-git-repo-check",
                "--ignore-rules",
                "--output-schema",
                str(schema_run_path),
                "--output-last-message",
                str(final_run_path),
                "--sandbox",
                "read-only",
                "-C",
                str(run_workspace),
            ]
            if self.model:
                cmd.extend(["--model", self.model])
            cmd.append(prompt)

            started = perf_counter()
            completed = subprocess.run(
                cmd,
                cwd=run_workspace,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
                env=build_codex_env(),
            )
            duration_ms = round((perf_counter() - started) * 1000)
            jsonl_path.write_text(completed.stdout, encoding="utf-8")
            raw_events = parse_jsonl(completed.stdout)
            token_usage = extract_token_usage(raw_events)
            if completed.returncode != 0:
                raise RuntimeError(
                    "codex exec failed for judge run "
                    f"{case.id} with exit code {completed.returncode}: {completed.stderr[-4000:]}"
                )
            judge_result = normalize_judge_final(read_structured_final(final_run_path, completed.stdout))
            final_artifact_path.write_text(
                json.dumps(judge_result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            return {
                "case_id": case.id,
                "judge": {
                    **self.judge_info,
                    "model": self.model,
                    "cli_path": str(self.codex_path),
                    "isolation": "temp-workspace-no-repo",
                },
                **judge_result,
                "duration_ms": duration_ms,
                "token_usage": token_usage,
                "jsonl_path": display_path(jsonl_path, self.workspace),
            }


def resolve_codex_cli() -> Path:
    configured = os.environ.get("CODEX_CLI")
    candidates = [
        Path(configured).expanduser() if configured else None,
        Path(shutil.which("codex") or ""),
        Path("/Applications/Codex.app/Contents/Resources/codex"),
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    raise CodexCliUnavailableError("Codex CLI was not found. Install Codex CLI or set CODEX_CLI to its path.")


def build_codex_provider_config_args() -> list[str]:
    """Return per-run Codex config overrides for benchmark provider shims."""
    if not _deepseek_mode_enabled("CODEX_DEEPSEEK"):
        return []

    provider = os.environ.get("CODEX_DEEPSEEK_PROVIDER", "deepseek-proxy").strip() or "deepseek-proxy"
    base_url = os.environ.get("CODEX_DEEPSEEK_BASE_URL", "http://127.0.0.1:8787/v1").strip()
    bearer_token = os.environ.get("CODEX_DEEPSEEK_BEARER_TOKEN", "codex-local-proxy").strip()
    api_key_env = os.environ.get("CODEX_DEEPSEEK_API_KEY_ENV", "").strip()
    if not base_url:
        raise RuntimeError("CODEX_DEEPSEEK_BASE_URL must not be empty when CODEX_DEEPSEEK=1.")
    if not bearer_token and not api_key_env:
        raise RuntimeError("Set CODEX_DEEPSEEK_BEARER_TOKEN or CODEX_DEEPSEEK_API_KEY_ENV when CODEX_DEEPSEEK=1.")

    args = [
        "-c",
        f"model_provider={_toml_string(provider)}",
        "-c",
        f"model_providers.{provider}.name={_toml_string('DeepSeek via local proxy')}",
        "-c",
        f"model_providers.{provider}.base_url={_toml_string(base_url)}",
        "-c",
        f"model_providers.{provider}.wire_api={_toml_string('responses')}",
        "-c",
        f"model_providers.{provider}.requires_openai_auth=false",
        "-c",
        f"model_providers.{provider}.supports_websockets=false",
        "-c",
        'model_reasoning_effort="none"',
        "-c",
        'model_reasoning_summary="none"',
    ]
    if bearer_token:
        args.extend(
            [
                "-c",
                f"model_providers.{provider}.experimental_bearer_token={_toml_string(bearer_token)}",
            ]
        )
    else:
        args.extend(
            [
                "-c",
                f"model_providers.{provider}.env_key={_toml_string(api_key_env)}",
            ]
        )
    return args


def build_codex_env() -> dict[str, str]:
    env = os.environ.copy()
    if not _deepseek_mode_enabled("CODEX_DEEPSEEK"):
        return env

    return env


def _deepseek_mode_enabled(adapter_env: str) -> bool:
    return os.environ.get(adapter_env) == "1" or os.environ.get("BENCHMARK_DEEPSEEK") == "1"


def _toml_string(value: str) -> str:
    return json.dumps(value)


def build_agent_prompt(
    case: BenchmarkCase,
    input_path: Path,
    *,
    visible_input_path: Path | str | None = None,
    visible_skill_path: Path | str | None = None,
) -> str:
    data_guidance = build_input_data_guidance(input_path)
    prompt_input_path = visible_input_path or input_path
    skill_section = build_skill_section(visible_skill_path)
    universe = (
        "the input data directory and skill directory" if visible_skill_path is not None else "the input data directory"
    )
    return f"""You are the benchmarked agent.

Task:
{case.prompt}

Input data directory:
{prompt_input_path}

{skill_section}
Input data guidance:
{data_guidance}

Requirements:
- Inspect the local files you need under the input data directory.
- Treat {universe} as the entire benchmark universe.
- Do not inspect parent directories, benchmark source code, benchmark YAML files, run outputs, or any path outside the allowed benchmark directories.
- Prefer compact aggregate files over raw traces or databases.
- Never dump an entire large JSON/HTML/DB file into context; use targeted shell commands or small scripts to extract only the relevant rows/sections.
- Do not modify files.
- Return only JSON matching the provided schema.
- Do not include private chain-of-thought. Use reasoning_summary for a concise, auditable explanation.
"""


def build_skill_section(visible_skill_path: Path | str | None) -> str:
    if visible_skill_path is None:
        return ""
    return f"""Skill directory:
{visible_skill_path}

Skill requirements:
- Read and follow {visible_skill_path}/SKILL.md before analyzing the input data.
- You may inspect helper scripts under the skill directory when the skill calls for them.
- If the skill mentions an external command that is unavailable in this isolated run, use equivalent local evidence from input_data and state that limitation in the answer.
"""


def build_input_data_guidance(input_path: Path) -> str:
    if (input_path / "cluster_analysis_output" / "cluster_step_trace_time.csv").exists():
        return """This looks like an Ascend profiler bundle.
Start with:
- cluster_analysis_output/cluster_step_trace_time.csv
- mstt_advisor_*.html, especially the slow-rank section
- log/mstt_advisor_*.xlsx only if the HTML summary is insufficient

Avoid opening these large/raw files unless absolutely necessary:
- trace_view.json
- cluster_communication.json
- cluster_communication_matrix.json
- cluster.db
- mindstudio_insight_data*.db
"""
    if has_raw_ascend_profiler_data(input_path):
        return """This looks like raw Ascend profiler collection data.
There may be no cluster-level summary, no cluster.db, and no advisor HTML report.

Use the raw per-rank profiler directories directly:
- First identify rank ids from */profiler_info_*.json.
- Then compare */ASCEND_PROFILER_OUTPUT/step_trace_time.csv across all ranks.
- Focus on Step, Computing, Communication(Not Overlapped), Overlapped, Communication, Free, Stage, Bubble, and Preparing.
- A slow rank is usually the rank with the largest Stage time, or a clear outlier in Free/host idle, Computing, or non-overlapped Communication.
- Cross-check only small aggregate CSVs such as api_statistic.csv, op_statistic.csv, kernel_details.csv, and operator_details.csv when step_trace_time.csv is not enough.
- For communication suspicion, inspect small summaries first; use communication.json or communication_matrix.json with targeted extraction only.

Avoid these expensive raw artifacts unless absolutely necessary:
- trace_view.json
- mindstudio_insight_data*.db
- PROF_*/device_*/data/*
- PROF_*/host/data/*
- FRAMEWORK/torch.*

Do not look for benchmark source code, YAML ground truth, cluster_analysis_output, cluster.db, or mstt_advisor files; they are not part of this raw-data-only case.
"""
    if (input_path / "cluster.db").exists() and list(input_path.glob("mstt_advisor_*.html")):
        return """This looks like an Ascend profiler bundle with a cluster.db summary.
Start with targeted queries over cluster.db:
- step_statistic_info for per-rank compute, communication, free, and stage time
- communication_time_info and communication_bandwidth_info to check whether the anomaly is communication-related

Then cross-check mstt_advisor_*.html, especially the slow-rank section.
Only inspect per-rank ASCEND_PROFILER_OUTPUT/step_trace_time.csv files if the database and HTML summary are insufficient.

Avoid recursive full-file listing or raw trace dumps unless absolutely necessary:
- trace_view.json
- communication.json
- communication_matrix.json
- mindstudio_insight_data*.db
"""
    if (input_path / "metrics.csv").exists():
        return """This looks like a compact synthetic metrics bundle.
Start with metrics.csv, then use logs.txt and events.jsonl for corroborating evidence.
"""
    return "No known bundle type detected. List files first and inspect compact summaries before raw traces."


def has_raw_ascend_profiler_data(input_path: Path) -> bool:
    return any(input_path.glob("*_ascend_pt/profiler_info_*.json")) and any(
        input_path.glob("*_ascend_pt/ASCEND_PROFILER_OUTPUT/step_trace_time.csv")
    )


def safe_path_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or str(uuid4())


def copy_input_data(source: Path, destination: Path) -> None:
    source = source.resolve()
    if source.is_symlink():
        raise ValueError(f"Input data path must not be a symlink: {source}")
    if source.is_file():
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination / source.name)
        return
    if not source.is_dir():
        raise FileNotFoundError(f"Input data path does not exist: {source}")

    for root, dirs, files in os.walk(source, followlinks=False):
        root_path = Path(root)
        relative_root = root_path.relative_to(source)
        target_root = destination / relative_root
        target_root.mkdir(parents=True, exist_ok=True)

        dirs[:] = [dirname for dirname in dirs if not (root_path / dirname).is_symlink()]
        for filename in files:
            source_file = root_path / filename
            if source_file.is_symlink():
                continue
            shutil.copy2(source_file, target_root / filename)


def copy_case_skill(case: BenchmarkCase, run_workspace: Path) -> Path | None:
    source = case.resolve_skill_path()
    if source is None:
        return None
    if not source.exists():
        raise FileNotFoundError(f"Missing skill for {case.id}: {source}")
    destination = run_workspace / "skill"
    copy_input_data(source, destination)
    return destination


def display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def build_judge_prompt(
    case: BenchmarkCase,
    trace: dict[str, Any],
) -> str:
    judge_input = {
        "case": {
            "id": case.id,
            "prompt": case.prompt,
            "must_include": case.must_include,
            "scoring_prompt": case.scoring_prompt,
        },
        "trace_excerpt": trace_excerpt(trace),
        "final_answer": final_answer_from_trace(trace),
    }
    return f"""You are judging an AI agent benchmark run.

Judge only the visible trace and final answer. Do not reward hidden reasoning.

You have two jobs:
1. For every must_include item, decide whether the final answer semantically covers it.
   Allow equivalent wording, but mark covered=false when the answer omits or contradicts the item.
2. Use scoring_prompt to assign rubric_score from 0 to 5.

The final benchmark runner will set score=0 if any must_include item is not covered.

Return only JSON matching the provided schema.

Input:
{json.dumps(judge_input, ensure_ascii=False, indent=2)}
"""


def parse_jsonl(text: str) -> list[dict[str, Any]]:
    events = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def read_structured_final(path: Path, stdout: str) -> dict[str, Any]:
    if path.exists() and path.read_text(encoding="utf-8").strip():
        text = path.read_text(encoding="utf-8")
    else:
        text = stdout.strip().splitlines()[-1] if stdout.strip() else "{}"
    value = parse_structured_json_object(text)
    if not isinstance(value, dict):
        raise RuntimeError(f"Structured Codex final output is not an object: {value!r}")
    return value


def normalize_agent_final(value: dict[str, Any]) -> dict[str, Any]:
    if isinstance(value.get("answer"), str):
        normalized = dict(value)
        normalized.setdefault("evidence", [])
        normalized.setdefault("reasoning_summary", "")
        normalized.setdefault("confidence", 0.8)
        return normalized

    return {
        "answer": json.dumps(value, ensure_ascii=False, sort_keys=True),
        "evidence": _string_values(value),
        "reasoning_summary": (
            value.get("reasoning_summary")
            if isinstance(value.get("reasoning_summary"), str)
            else "Model returned a structured object; wrapped for benchmark scoring."
        ),
        "confidence": value.get("confidence") if isinstance(value.get("confidence"), int | float) else 0.8,
    }


def _string_values(value: Any) -> list[str]:
    results: list[str] = []
    if isinstance(value, str):
        if value.strip():
            results.append(value)
    elif isinstance(value, dict):
        for nested in value.values():
            results.extend(_string_values(nested))
    elif isinstance(value, list):
        for nested in value:
            results.extend(_string_values(nested))
    return results[:20]


def normalize_judge_final(value: dict[str, Any]) -> dict[str, Any]:
    if isinstance(value.get("must_include_results"), list):
        normalized = dict(value)
        normalized.setdefault("rubric_score", float(value.get("score", 0) or 0))
        normalized.setdefault("strengths", [])
        normalized.setdefault("weaknesses", [])
        normalized.setdefault(
            "must_include_pass",
            bool(normalized["must_include_results"])
            and all(bool(item.get("covered")) for item in normalized["must_include_results"] if isinstance(item, dict)),
        )
        return normalized

    raw_results = value.get("must_include")
    must_include_results: list[dict[str, Any]] = []
    if isinstance(raw_results, list):
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            name = item.get("item") or item.get("id") or item.get("text") or ""
            covered = bool(item.get("covered"))
            reason = item.get("reason")
            if not isinstance(reason, str):
                reason = "Covered by judge output." if covered else "Not covered by judge output."
            must_include_results.append(
                {
                    "item": str(name),
                    "covered": covered,
                    "reason": reason,
                }
            )
    elif isinstance(raw_results, dict):
        for name, covered_value in raw_results.items():
            covered = bool(covered_value)
            must_include_results.append(
                {
                    "item": str(name),
                    "covered": covered,
                    "reason": "Covered by judge output." if covered else "Not covered by judge output.",
                }
            )

    rubric_score = value.get("rubric_score", value.get("score", 0))
    if not isinstance(rubric_score, int | float):
        rubric_score = 0
    notes = value.get("meta", {}).get("notes") if isinstance(value.get("meta"), dict) else None
    return {
        "must_include_pass": bool(must_include_results) and all(bool(item["covered"]) for item in must_include_results),
        "must_include_results": must_include_results,
        "rubric_score": float(rubric_score),
        "strengths": value.get("strengths") if isinstance(value.get("strengths"), list) else ([notes] if notes else []),
        "weaknesses": value.get("weaknesses") if isinstance(value.get("weaknesses"), list) else [],
    }


def parse_structured_json_object(text: str) -> dict[str, Any]:
    candidates = [text.strip()]
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text.strip(), flags=re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1).strip())
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])

    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(value, dict):
            raise RuntimeError(f"Structured Codex final output is not an object: {value!r}")
        return value

    if last_error is not None:
        raise RuntimeError(f"Could not parse structured Codex final output: {text[:1000]}") from last_error
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Could not parse structured Codex final output: {text[:1000]}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Structured Codex final output is not an object: {value!r}")
    return value


def extract_token_usage(events: list[dict[str, Any]]) -> dict[str, Any]:
    candidates: list[dict[str, int]] = []
    for event in events:
        for usage in find_usage_dicts(event):
            normalized = normalize_usage(usage)
            if normalized is not None:
                candidates.append(normalized)
    if not candidates:
        return {
            "available": False,
            "source": "codex-jsonl-no-usage-event",
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    best = max(candidates, key=lambda item: item["total_tokens"])
    return {
        "available": True,
        "source": "codex-jsonl",
        **best,
    }


def find_usage_dicts(value: Any) -> list[dict[str, Any]]:
    found = []
    if isinstance(value, dict):
        keys = set(value)
        usage_keys = {
            "input_tokens",
            "prompt_tokens",
            "output_tokens",
            "completion_tokens",
            "total_tokens",
        }
        if keys & usage_keys:
            found.append(value)
        for child in value.values():
            found.extend(find_usage_dicts(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(find_usage_dicts(child))
    return found


def normalize_usage(value: dict[str, Any]) -> dict[str, int] | None:
    input_tokens = value.get("input_tokens", value.get("prompt_tokens"))
    output_tokens = value.get("output_tokens", value.get("completion_tokens"))
    total_tokens = value.get("total_tokens")
    try:
        input_int = int(input_tokens or 0)
        output_int = int(output_tokens or 0)
        total_int = int(total_tokens or input_int + output_int)
    except (TypeError, ValueError):
        return None
    if input_int == 0 and output_int == 0 and total_int == 0:
        return None
    return {
        "input_tokens": input_int,
        "output_tokens": output_int,
        "total_tokens": total_int,
    }


def normalize_codex_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for raw in events:
        event_type = str(raw.get("type") or raw.get("event") or "")
        item = raw.get("item")
        if isinstance(item, dict) and item.get("type") == "command_execution":
            normalized.append(normalize_command_execution_event(event_type, item, raw))
            continue

        text = json.dumps(raw, ensure_ascii=False)
        if "tool" in event_type.lower() or "exec" in text.lower() or "error" in event_type.lower():
            normalized.append(
                {
                    "event_type": "agent_event",
                    "raw_type": event_type,
                    "summary": trim_event_text(raw),
                }
            )
        elif "message" in event_type.lower() or "reasoning" in event_type.lower():
            normalized.append(
                {
                    "event_type": "agent_event",
                    "raw_type": event_type,
                    "summary": trim_event_text(raw),
                }
            )
    return normalized


def normalize_command_execution_event(
    event_type: str,
    item: dict[str, Any],
    raw: dict[str, Any],
) -> dict[str, Any]:
    item_id = item.get("id")
    command = item.get("command")
    status = item.get("status")
    if event_type.endswith("started") or status == "in_progress":
        return {
            "event_type": "tool_call",
            "raw_type": event_type,
            "tool": "command_execution",
            "item_id": item_id,
            "input": {
                "command": command,
            },
        }

    aggregated_output = str(item.get("aggregated_output") or "")
    return {
        "event_type": "tool_result",
        "raw_type": event_type,
        "tool": "command_execution",
        "item_id": item_id,
        "output": {
            "status": status,
            "exit_code": item.get("exit_code"),
            "aggregated_output": trim_text(aggregated_output),
            "aggregated_output_chars": len(aggregated_output),
            "aggregated_output_truncated": len(aggregated_output) > 4000,
        },
        "summary": trim_event_text(raw),
    }


def trim_event_text(event: dict[str, Any], limit: int = 2000) -> str:
    text = json.dumps(event, ensure_ascii=False, sort_keys=True)
    return text[:limit]


def trim_text(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit]


def safe_command_for_trace(cmd: list[str]) -> list[str]:
    safe = []
    skip_next = False
    for part in cmd:
        if skip_next:
            safe.append("<redacted>")
            skip_next = False
            continue
        safe.append(part)
        if part in {"--remote-auth-token-env"}:
            skip_next = True
    return safe


def trace_excerpt(trace: dict[str, Any]) -> dict[str, Any]:
    events = trace.get("events", [])
    compact_events = []
    for event in events:
        compact = {
            "type": event.get("type"),
            "tool": event.get("tool"),
            "content": event.get("content"),
            "answer": event.get("answer"),
            "summary": event.get("summary"),
        }
        compact_events.append({k: v for k, v in compact.items() if v is not None})
    return {
        "agent": trace.get("agent"),
        "duration_ms": trace.get("duration_ms"),
        "events": compact_events[-40:],
    }


def final_answer_from_trace(trace: dict[str, Any]) -> dict[str, Any]:
    for event in reversed(trace.get("events", [])):
        if event.get("type") == "final_answer" and isinstance(event.get("answer"), dict):
            return event["answer"]
    return {}
