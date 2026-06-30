from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SRC = PROJECT_ROOT / "tests" / "benchmark" / "src"


def _benchmark_module(name: str):
    src = str(BENCHMARK_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    return importlib.import_module(name)


def test_benchmark_smoke_case_runs_with_tracked_fixture(tmp_path: Path) -> None:
    run = _benchmark_module("run_benchmark").run
    config = PROJECT_ROOT / "tests" / "benchmark" / "benchmarks" / "mock_agent_smoke.yaml"
    out_dir = tmp_path / "out"

    report = run(
        config,
        out_dir,
        agent_kind="heuristic",
        judge_kind="heuristic",
        timeout_seconds=5,
    )

    assert report["case_count"] == 1
    assert report["failed_count"] == 0
    assert report["scores"][0]["case_id"] == "mock_agent_smoke"
    assert report["scores"][0]["score"] > 0
    assert (out_dir / "scores.json").exists()
    assert (out_dir / "traces" / "mock_agent_smoke.trace.json").exists()


def test_benchmark_records_failed_case_and_continues(tmp_path: Path) -> None:
    run = _benchmark_module("run_benchmark").run
    case_path = tmp_path / "cases" / "missing_input.yaml"
    case_path.parent.mkdir()
    case_path.write_text(
        """
id: missing_input
input_data_path: ./missing-data
prompt: Read the input data.
must_include:
  - expected answer
scoring_prompt: Score the answer.
""".strip(),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out"
    report = run(
        case_path,
        out_dir,
        agent_kind="heuristic",
        judge_kind="heuristic",
        timeout_seconds=5,
    )

    assert report["case_count"] == 1
    assert report["failed_count"] == 1
    assert report["average_score"] == 0.0
    assert report["scores"][0]["status"] == "failed"
    failure_path = Path(report["failed_cases"][0]["failure_path"])
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["case_id"] == "missing_input"
    assert failure["error"]["type"] == "FileNotFoundError"
    assert (out_dir / "scores.json").exists()


def test_must_include_results_only_match_explicit_item_keys(tmp_path: Path) -> None:
    run_benchmark = _benchmark_module("run_benchmark")
    schema = _benchmark_module("schema")
    case = schema.BenchmarkCase(
        id="case",
        input_data_path="input",
        skill_path=None,
        prompt="prompt",
        must_include=["expected item"],
        must_include_regex=[],
        must_tool_use=[],
        scoring_prompt="score",
        source_path=tmp_path / "case.yaml",
    )

    normalized = run_benchmark._normalize_must_include_results(
        case,
        {
            "must_include_results": [
                {
                    "item": "different item",
                    "covered": True,
                    "reason": "This must not be reused by position.",
                }
            ]
        },
    )

    assert normalized == [
        {
            "item": "expected item",
            "covered": False,
            "reason": "Judge did not return a result for this required item.",
        }
    ]


def test_load_suite_finds_nested_case_files(tmp_path: Path) -> None:
    schema = _benchmark_module("schema")
    case_path = tmp_path / "suite" / "domain" / "nested.yaml"
    case_path.parent.mkdir(parents=True)
    case_path.write_text(
        """
id: nested
input_data_path: ./input
prompt: Read the input.
must_include:
  - nested answer
scoring_prompt: Score the answer.
""".strip(),
        encoding="utf-8",
    )

    suite = schema.load_suite(tmp_path / "suite")

    assert [case.id for case in suite.cases] == ["nested"]


def test_string_list_fields_are_single_items_not_comma_split(tmp_path: Path) -> None:
    schema = _benchmark_module("schema")

    case = schema.BenchmarkCase.from_dict(
        {
            "id": "case",
            "input_data_path": "./input",
            "prompt": "prompt",
            "must_include": "alpha, beta",
            "scoring_prompt": "score",
        },
        tmp_path / "case.yaml",
    )

    assert case.must_include == ["alpha, beta"]


def test_codex_deepseek_mode_builds_provider_overrides(monkeypatch) -> None:
    codex_cli = _benchmark_module("third_party.codex_cli")

    monkeypatch.setenv("CODEX_DEEPSEEK", "1")

    args = codex_cli.build_codex_provider_config_args()
    env = codex_cli.build_codex_env()

    assert 'model_provider="deepseek-proxy"' in args
    assert 'model_providers.deepseek-proxy.base_url="http://127.0.0.1:8787/v1"' in args
    assert 'model_providers.deepseek-proxy.experimental_bearer_token="codex-local-proxy"' in args
    assert 'model_providers.deepseek-proxy.wire_api="responses"' in args
    assert "CODEX_DEEPSEEK" in env


def test_codex_structured_final_accepts_fenced_json() -> None:
    codex_cli = _benchmark_module("third_party.codex_cli")

    parsed = codex_cli.parse_structured_json_object('```json\n{"answer": "ok"}\n```')

    assert parsed == {"answer": "ok"}


def test_codex_agent_final_wraps_domain_json() -> None:
    codex_cli = _benchmark_module("third_party.codex_cli")

    normalized = codex_cli.normalize_agent_final(
        {
            "root_cause": "timeout spike",
            "impacted_service": "checkout-api",
            "recommended_action": "raise timeout",
        }
    )

    assert "timeout spike" in normalized["answer"]
    assert "checkout-api" in normalized["answer"]
    assert normalized["evidence"]


def test_codex_judge_final_accepts_short_must_include_shape() -> None:
    codex_cli = _benchmark_module("third_party.codex_cli")

    normalized = codex_cli.normalize_judge_final(
        {
            "must_include": [
                {"id": "checkout-api", "covered": True},
                {"id": "timeout budget", "covered": True},
            ],
            "score": 5,
        }
    )

    assert normalized["must_include_pass"] is True
    assert normalized["rubric_score"] == 5
    assert normalized["must_include_results"][0]["item"] == "checkout-api"


def test_codex_judge_final_accepts_must_include_dict() -> None:
    codex_cli = _benchmark_module("third_party.codex_cli")

    normalized = codex_cli.normalize_judge_final(
        {
            "must_include": {"rank3": True, "Free Time 异常": True},
            "rubric_score": 5,
        }
    )

    assert normalized["must_include_pass"] is True
    assert [item["item"] for item in normalized["must_include_results"]] == ["rank3", "Free Time 异常"]


def test_claude_deepseek_mode_maps_anthropic_env(monkeypatch) -> None:
    claude_cli = _benchmark_module("third_party.claude_cli")

    monkeypatch.setenv("BENCHMARK_DEEPSEEK", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "dummy")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "subscription-token")

    env = claude_cli.build_claude_env()

    assert env["ANTHROPIC_API_KEY"] == "dummy"
    assert env["ANTHROPIC_BASE_URL"] == "https://api.deepseek.com/anthropic"
    assert "ANTHROPIC_AUTH_TOKEN" not in env
