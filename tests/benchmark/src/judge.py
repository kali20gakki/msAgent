from __future__ import annotations

import json
from time import perf_counter
from typing import Any

from schema import BenchmarkCase


class MockLLMJudge:
    """Deterministic judge stub for smoke tests.

    Real judges use an LLM prompt for semantic coverage. This local judge keeps
    benchmark development repeatable and approximates coverage with substring
    matching over the final answer.
    """

    judge_info = {
        "name": "mock-llm-judge",
        "model": "heuristic-judge-v0",
        "token_usage_mode": "estimated",
        "coverage_mode": "substring-smoke-test",
    }

    def judge(
        self,
        case: BenchmarkCase,
        trace: dict[str, Any],
    ) -> dict[str, Any]:
        started = perf_counter()
        answer = self._final_answer(trace)
        answer_text = self._answer_text(answer)
        coverage = [
            {
                "item": item,
                "covered": self._contains(answer_text, item),
                "reason": self._coverage_reason(answer_text, item),
            }
            for item in case.must_include
        ]
        must_include_pass = all(item["covered"] for item in coverage)
        rubric_score = self._rubric_score(answer, must_include_pass)

        result = {
            "case_id": case.id,
            "judge": self.judge_info,
            "must_include_pass": must_include_pass,
            "must_include_results": coverage,
            "rubric_score": rubric_score,
            "strengths": self._strengths(answer, must_include_pass),
            "weaknesses": self._weaknesses(coverage),
        }
        result["duration_ms"] = round((perf_counter() - started) * 1000)
        result["token_usage"] = self._estimate_token_usage(case, trace, result)
        return result

    def _answer_text(self, answer: dict[str, Any]) -> str:
        raw_answer = answer.get("answer", "")
        if isinstance(raw_answer, str):
            return raw_answer
        return json.dumps(raw_answer, ensure_ascii=False)

    def _contains(self, answer_text: str, item: str) -> bool:
        return item.casefold() in answer_text.casefold()

    def _coverage_reason(self, answer_text: str, item: str) -> str:
        if self._contains(answer_text, item):
            return "The required item appears in the final answer text."
        return "The required item was not found by the local smoke-test matcher."

    def _rubric_score(self, answer: dict[str, Any], must_include_pass: bool) -> float:
        score = 1.0
        if isinstance(answer.get("answer"), str) and answer["answer"].strip():
            score += 1.5
        if isinstance(answer.get("evidence"), list) and answer["evidence"]:
            score += 1.0
        if isinstance(answer.get("reasoning_summary"), str) and answer["reasoning_summary"].strip():
            score += 1.0
        if must_include_pass:
            score += 0.5
        return round(min(5.0, score), 2)

    def _strengths(self, answer: dict[str, Any], must_include_pass: bool) -> list[str]:
        strengths = []
        if must_include_pass:
            strengths.append("Final answer covers all required must_include items.")
        if isinstance(answer.get("evidence"), list) and answer["evidence"]:
            strengths.append("Final answer includes evidence entries.")
        return strengths or ["Final answer was produced in the required JSON shape."]

    def _weaknesses(self, coverage: list[dict[str, Any]]) -> list[str]:
        missing = [item["item"] for item in coverage if not item["covered"]]
        if missing:
            return [f"Missing must_include item: {item}" for item in missing]
        return []

    def _final_answer(self, trace: dict[str, Any]) -> dict[str, Any]:
        finals = [event for event in trace.get("events", []) if event.get("type") == "final_answer"]
        if not finals:
            return {}
        answer = finals[-1].get("answer", {})
        return answer if isinstance(answer, dict) else {}

    def _estimate_token_usage(
        self,
        case: BenchmarkCase,
        trace: dict[str, Any],
        judge_result: dict[str, Any],
    ) -> dict[str, int]:
        judge_input = {
            "case": {
                "id": case.id,
                "prompt": case.prompt,
                "must_include": case.must_include,
                "scoring_prompt": case.scoring_prompt,
            },
            "trace": trace,
        }
        input_tokens = max(1, round(len(json.dumps(judge_input, ensure_ascii=False)) / 4))
        output_tokens = max(1, round(len(json.dumps(judge_result, ensure_ascii=False)) / 4))
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }


def normalized_judge_score(judge_result: dict[str, Any]) -> float:
    return max(0.0, min(1.0, float(judge_result.get("rubric_score", 0.0)) / 5.0))
