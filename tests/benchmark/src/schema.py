from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_TOOL_BUDGET = 20
DEFAULT_TOOL_PENALTY_PER_CALL = 0.02
DEFAULT_TOOL_PENALTY_FLOOR = 0.5


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        warnings.warn(
            "PyYAML is not installed. JSON and only simple case YAML are supported. Install with: pip install PyYAML",
            stacklevel=2,
        )
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return _load_simple_case_yaml(text, path)

    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping at top level: {path}")
    return loaded


def _load_simple_case_yaml(text: str, path: Path) -> dict[str, Any]:
    """Tiny fallback parser for the simple case YAML used by this project."""
    result: dict[str, Any] = {}
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            index += 1
            continue
        if line.startswith((" ", "\t")) or ":" not in line:
            raise ValueError(f"Unsupported YAML shape in {path}: {line!r}")

        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if value in {">", "|"}:
            block_lines: list[str] = []
            index += 1
            while index < len(lines) and (lines[index].startswith((" ", "\t")) or not lines[index].strip()):
                block_lines.append(lines[index].strip())
                index += 1
            result[key] = "\n".join(block_lines) if value == "|" else " ".join(block_lines)
            continue

        if value == "":
            list_items: list[str] = []
            index += 1
            while index < len(lines) and (lines[index].startswith((" ", "\t")) or not lines[index].strip()):
                item = lines[index].strip()
                if item.startswith("- "):
                    list_items.append(item[2:].strip().strip("\"'"))
                elif item:
                    raise ValueError(f"Unsupported YAML list item in {path}: {lines[index]!r}")
                index += 1
            result[key] = list_items
            continue

        if value == "[]":
            result[key] = []
            index += 1
            continue

        result[key] = value.strip("\"'")
        index += 1

    return result


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    input_data_path: str
    skill_path: str | None
    prompt: str
    must_include: list[str]
    must_include_regex: list[str]
    must_tool_use: list[str]
    scoring_prompt: str
    source_path: Path = field(repr=False)
    tool_budget: int = DEFAULT_TOOL_BUDGET
    tool_penalty_per_call: float = DEFAULT_TOOL_PENALTY_PER_CALL
    tool_penalty_floor: float = DEFAULT_TOOL_PENALTY_FLOOR

    @classmethod
    def from_dict(cls, raw: dict[str, Any], source_path: Path) -> "BenchmarkCase":
        missing = [key for key in ("input_data_path", "prompt", "must_include", "scoring_prompt") if key not in raw]
        if missing:
            raise ValueError(f"{source_path} is missing required field(s): {', '.join(missing)}")

        scoring_prompt = str(raw["scoring_prompt"]).strip()
        if not scoring_prompt:
            raise ValueError(f"{source_path} scoring_prompt must not be empty.")

        must_include_regex = _normalize_string_list(
            raw.get("must_include_regex", []),
            source_path,
            "must_include_regex",
        )
        _validate_regex_list(must_include_regex, source_path)

        return cls(
            id=str(raw.get("id") or source_path.stem),
            input_data_path=str(raw["input_data_path"]),
            skill_path=_normalize_optional_string(raw.get("skill_path")),
            prompt=str(raw["prompt"]),
            must_include=_normalize_string_list(raw["must_include"], source_path, "must_include"),
            must_include_regex=must_include_regex,
            must_tool_use=_normalize_string_list(
                raw.get("must_tool_use", []),
                source_path,
                "must_tool_use",
            ),
            scoring_prompt=scoring_prompt,
            source_path=source_path,
            tool_budget=_parse_tool_budget(raw.get("tool_budget"), source_path),
            tool_penalty_per_call=_parse_non_negative_float(
                raw.get("tool_penalty_per_call"),
                source_path,
                "tool_penalty_per_call",
                DEFAULT_TOOL_PENALTY_PER_CALL,
            ),
            tool_penalty_floor=_parse_unit_float(
                raw.get("tool_penalty_floor"),
                source_path,
                "tool_penalty_floor",
                DEFAULT_TOOL_PENALTY_FLOOR,
            ),
        )

    def resolve_input_path(self) -> Path:
        path = Path(self.input_data_path)
        if path.is_absolute():
            return path
        return (self.source_path.parent / path).resolve()

    def resolve_skill_path(self) -> Path | None:
        if self.skill_path is None:
            return None
        path = Path(self.skill_path)
        if path.is_absolute():
            return path
        return (self.source_path.parent / path).resolve()


@dataclass(frozen=True)
class BenchmarkSuite:
    name: str
    config_path: Path
    cases: list[BenchmarkCase]


def load_suite(config_path: str | Path) -> BenchmarkSuite:
    path = Path(config_path).resolve()
    if path.is_dir():
        case_files = sorted([*path.rglob("*.yaml"), *path.rglob("*.yml")])
        cases = [_load_case_file(case_file) for case_file in case_files]
        name = path.name
    else:
        cases = [_load_case_file(path)]
        name = path.stem

    if not cases:
        raise ValueError(f"No benchmark case YAML files found in {path}")

    return BenchmarkSuite(
        name=name,
        config_path=path,
        cases=cases,
    )


def _load_case_file(path: Path) -> BenchmarkCase:
    raw = _load_yaml_or_json(path)
    if "cases" in raw:
        raise ValueError(f"{path} contains a suite. Expected one case per YAML file.")
    return BenchmarkCase.from_dict(raw, path)


def _normalize_string_list(value: Any, source_path: Path, field_name: str) -> list[str]:
    if isinstance(value, str):
        if not value.strip():
            return []
        return [value.strip()]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raise ValueError(f"{source_path} {field_name} must be a list of strings.")


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_tool_budget(value: Any, source_path: Path) -> int:
    if value is None or value == "":
        return DEFAULT_TOOL_BUDGET
    try:
        budget = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source_path} tool_budget must be a non-negative integer.") from exc
    if budget < 0:
        raise ValueError(f"{source_path} tool_budget must be a non-negative integer.")
    return budget


def _parse_non_negative_float(value: Any, source_path: Path, field_name: str, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source_path} {field_name} must be a non-negative number.") from exc
    if parsed < 0:
        raise ValueError(f"{source_path} {field_name} must be a non-negative number.")
    return parsed


def _parse_unit_float(value: Any, source_path: Path, field_name: str, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source_path} {field_name} must be a number between 0 and 1.") from exc
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{source_path} {field_name} must be a number between 0 and 1.")
    return parsed


def _validate_regex_list(patterns: list[str], source_path: Path) -> None:
    for pattern in patterns:
        try:
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        except re.error as exc:
            raise ValueError(f"{source_path} must_include_regex contains invalid regex {pattern!r}: {exc}") from exc
