from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

from schema import BenchmarkCase
from trace import TraceBuilder  # pylint: disable=no-name-in-module


class MockBenchmarkAgent:
    """Deterministic local agent used only for harness smoke tests."""

    agent_info = {
        "name": "mock-benchmark-agent",
        "model": "heuristic-v0",
        "token_usage_mode": "estimated",
        "note": "Local smoke-test agent; not a valid benchmark participant.",
    }

    def run(self, case: BenchmarkCase, input_path: Path) -> dict[str, Any]:
        trace = TraceBuilder(case_id=case.id, prompt=case.prompt, agent=self.agent_info)
        trace.thought("Inspect the input directory shape, then produce a deterministic smoke-test answer.")

        trace.tool_call("summarize_input_data", {"path": str(input_path)})
        started = perf_counter()
        input_summary = self._summarize_input_data(input_path)
        trace.tool_result("summarize_input_data", input_summary, self._elapsed_ms(started))
        trace.observation(
            f"Input data contains {input_summary['file_count']} files; "
            "using sampled input text to exercise the judge contract."
        )

        trace.final_answer(self._build_final_answer(case, input_summary))
        trace.finish(self._estimate_agent_token_usage(trace.to_dict()))
        return trace.to_dict()

    def _summarize_input_data(self, input_path: Path) -> dict[str, Any]:
        if input_path.is_file():
            return {
                "root": str(input_path),
                "file_count": 1,
                "sample_files": [input_path.name],
            }
        if not input_path.is_dir():
            raise FileNotFoundError(f"Unsupported input data path: {input_path}")

        files = [
            path.relative_to(input_path).as_posix()
            for path in sorted(input_path.rglob("*"))
            if path.is_file() and not path.is_symlink()
        ]
        text_sample = ""
        for relative_file in self._prioritized_text_files(files):
            text_sample = self._sample_text(input_path / relative_file)
            if text_sample:
                break
        return {
            "root": str(input_path),
            "file_count": len(files),
            "sample_files": files[:20],
            "text_sample": text_sample,
        }

    def _sample_text(self, path: Path) -> str:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ""
        return " ".join(text.split())[:1000]

    def _prioritized_text_files(self, files: list[str]) -> list[str]:
        suffix_rank = {
            ".md": 0,
            ".txt": 1,
            ".csv": 2,
            ".jsonl": 3,
            ".json": 4,
        }
        preferred = [path for path in files if Path(path).suffix.lower() in suffix_rank]
        other = [path for path in files if path not in set(preferred)]
        return (
            sorted(
                preferred,
                key=lambda path: (suffix_rank[Path(path).suffix.lower()], path),
            )
            + other
        )

    def _build_final_answer(
        self,
        case: BenchmarkCase,
        input_summary: dict[str, Any],
    ) -> dict[str, Any]:
        answer = input_summary.get("text_sample") or f"Completed smoke-test answer for {case.id}."
        return {
            "answer": answer,
            "evidence": [
                f"Input file count: {input_summary['file_count']}",
                f"Sample files: {', '.join(input_summary['sample_files'][:5]) or 'none'}",
            ],
            "reasoning_summary": (
                "This deterministic agent is intended only to verify benchmark plumbing, not to measure model quality."
            ),
            "confidence": 0.5,
        }

    def _elapsed_ms(self, started: float) -> int:
        return round((perf_counter() - started) * 1000)

    def _estimate_agent_token_usage(self, trace: dict[str, Any]) -> dict[str, int]:
        prompt_chars = len(trace.get("prompt", ""))
        tool_chars = 0
        output_chars = 0
        for event in trace.get("events", []):
            if event.get("type") == "tool_result":
                tool_chars += len(json.dumps(event.get("output", ""), ensure_ascii=False))
            elif event.get("type") in {"observation", "final_answer"}:
                output_chars += len(json.dumps(event, ensure_ascii=False))
        input_tokens = max(1, round((prompt_chars + tool_chars) / 4))
        output_tokens = max(1, round(output_chars / 4))
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
