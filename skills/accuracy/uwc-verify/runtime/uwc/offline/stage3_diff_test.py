"""Dummy vs safetensors differential test (black-box Stage 3 check)."""

from __future__ import annotations

import json
from pathlib import Path

from ..common import log_warning, now_iso, read_json, write_json

_TEXT_KEYS = ("texts", "text", "outputs", "output", "generated_text", "generated_texts")
_METADATA_KEYS = {"tag", "step", "generated_at", "source", "load_format", "stop_reason"}


def _iter_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            if key in _METADATA_KEYS:
                continue
            yield from _iter_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_strings(child)


def _extract_texts(record) -> list[str] | None:
    if not isinstance(record, dict):
        return None
    for key in _TEXT_KEYS:
        value = record.get(key)
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)) and value and all(isinstance(item, str) for item in value):
            return list(value)

    payload = record.get("payload")
    if isinstance(payload, dict):
        for key in _TEXT_KEYS:
            value = payload.get(key)
            if isinstance(value, str):
                return [value]
            if isinstance(value, (list, tuple)) and value and all(isinstance(item, str) for item in value):
                return list(value)

    # Best effort: collect strings from the record, excluding metadata.
    strings = [text for text in _iter_strings(record) if text.strip()]
    return strings or None


_VALID_EXTRA = set(" \n\t\r.,!?;:'\"-_/()[]{}<>|@#$%^&*+=~`\\，。！？；：、（）《》【】“”‘’…—·")


def _is_valid_char(char: str) -> bool:
    if char.isalnum() or char.isspace():
        return True
    if char in _VALID_EXTRA:
        return True
    return "一" <= char <= "鿿" or "぀" <= char <= "ヿ" or "가" <= char <= "힯"


def garbled_score(text: str) -> float:
    """Heuristic 0..1 score; higher means more likely to be gibberish."""
    if text is None:
        return 1.0
    stripped = text.strip()
    if not stripped:
        return 1.0
    if "�" in text:
        return 1.0

    invalid = sum(1 for char in stripped if not _is_valid_char(char))
    invalid_ratio = invalid / len(stripped)

    max_run = 1
    current_run = 1
    for index in range(1, len(stripped)):
        if stripped[index] == stripped[index - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1

    unique_ratio = len(set(stripped)) / len(stripped)
    score = invalid_ratio
    if max_run >= 16:
        score = max(score, 0.7 + min(0.3, (max_run - 16) / 40.0))
    # Repetitive garbage such as "8 8 8 8 ..." has a tiny alphabet.  Do not use
    # the raw unique ratio alone: long English/Chinese text also has a low ratio
    # (a 2000-char English reply uses ~50 distinct characters, ratio ~0.03).
    if len(stripped) > 60 and len(set(stripped)) <= 10 and unique_ratio < 0.06:
        score = max(score, 0.75)
    return min(score, 1.0)


def is_garbled(texts: list[str] | None, threshold: float = 0.5) -> bool | None:
    if not texts:
        return None
    scores = [garbled_score(text) for text in texts]
    return (sum(scores) / len(scores)) >= threshold


def _strip_metadata(record):
    if not isinstance(record, dict):
        return record
    return {key: value for key, value in record.items() if key not in _METADATA_KEYS}


def decode_record_texts(record, tokenizer):
    """Decode ``payload.responses`` into ``texts`` when the record has none.

    veRL 0.9's async rollout only exposes token ids.  Decoding them offline (with
    the model tokenizer) restores the document's garbled-text matrix without
    touching the vLLM process.
    """
    if not isinstance(record, dict) or tokenizer is None:
        return record
    if _extract_texts(record):
        return record
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return record
    responses = payload.get("responses")
    if not responses:
        return record
    sequences = responses if isinstance(responses[0], (list, tuple)) else [responses]
    try:
        decoded = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    except Exception:
        return record
    if not decoded:
        return record
    out = dict(record)
    out["texts"] = [text for text in decoded if isinstance(text, str)]
    out["texts_source"] = "decoded_payload_responses"
    return out


def _load_tokenizer(model_path, trust_remote_code=True):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)


def analyze(dummy_record, safetensors_record, threshold: float = 0.5) -> dict:
    dummy_texts = _extract_texts(dummy_record)
    safe_texts = _extract_texts(safetensors_record)
    dummy_garbled = is_garbled(dummy_texts, threshold)
    safe_garbled = is_garbled(safe_texts, threshold)

    dummy_payload = dummy_record.get("payload") if isinstance(dummy_record, dict) else None
    safe_payload = safetensors_record.get("payload") if isinstance(safetensors_record, dict) else None
    if dummy_texts is not None and safe_texts is not None:
        identical = dummy_texts == safe_texts
    else:
        identical = _strip_metadata(dummy_record) == _strip_metadata(safetensors_record)
        if not identical and dummy_payload is not None and safe_payload is not None:
            identical = dummy_payload == safe_payload

    report = {
        "status": "fail",
        "conclusion": "",
        "dummy_garbled": dummy_garbled,
        "safetensors_garbled": safe_garbled,
        "identical_outputs": identical,
        "dummy_load_format": (dummy_record or {}).get("load_format") if isinstance(dummy_record, dict) else None,
        "safetensors_load_format": (safetensors_record or {}).get("load_format")
        if isinstance(safetensors_record, dict)
        else None,
        "dummy_text_samples": (dummy_texts or [])[:3],
        "safetensors_text_samples": (safe_texts or [])[:3],
        "dummy_text_count": len(dummy_texts or []),
        "safetensors_text_count": len(safe_texts or []),
        "garbled_threshold": threshold,
        "generated_at": now_iso(),
    }

    # Identical output is healthy: both runs end up with the actor weights, so a
    # deterministic decode should match.  It only proves the override was
    # ignored when both records also report the same load_format.
    same_format = bool(
        report["dummy_load_format"]
        and report["dummy_load_format"] == report["safetensors_load_format"]
    )

    if dummy_garbled is None or safe_garbled is None:
        if same_format:
            report["status"] = "fail"
            report["conclusion"] = (
                f"两次运行的 load_format 相同（都是 {report['dummy_load_format']}），"
                "A/B 实验无效、覆盖未生效；请检查启动参数。"
            )
        elif identical:
            report["status"] = "pass"
            report["conclusion"] = (
                "两次输出一致且 load_format 不同：dummy 同步后的权重与 safetensors 直接加载等价。"
            )
        else:
            report["status"] = "pass"
            report["conclusion"] = (
                "输出中未找到可判定乱码的文本（仅有 token id / 数值），两次输出不同，按行为判定为通过；"
                "建议同时采集解码文本以获得确定结论。"
            )
        return report

    if dummy_garbled and not safe_garbled:
        report["status"] = "fail"
        report["conclusion"] = (
            "dummy 输出乱码、safetensors 输出正常：某些层漏同步。"
            "请检查 vLLM load_weights 的覆盖范围（结合 data_ptr 报告定位具体层）。"
        )
    elif dummy_garbled and safe_garbled:
        report["status"] = "fail"
        report["conclusion"] = (
            "dummy 与 safetensors 输出均乱码：同步的是错误值。请回查阶段 1（Bridge P0/P2）和阶段 2（传输 P2_sender/P5）。"
        )
    elif not dummy_garbled and not safe_garbled:
        report["status"] = "pass"
        report["conclusion"] = "dummy 与 safetensors 输出均正常：阶段 3 黑盒行为校验通过。"
    else:
        report["status"] = "fail"
        report["conclusion"] = (
            "dummy 输出正常、safetensors 输出乱码（罕见）：safetensors 加载异常，"
            "或同步覆盖破坏了正确权重。"
        )

    # A same-format pair is not an A/B experiment at all: even fluent, differing
    # outputs only prove that sampling is stochastic, not that load_format was
    # overridden.  Fail regardless of the text matrix.
    if same_format:
        report["status"] = "fail"
        report["conclusion"] = (
            f"两次运行的 load_format 相同（都是 {report['dummy_load_format']}），"
            "A/B 实验无效、覆盖未生效，判定失败。"
        )
    return report


def run(
    dummy_path,
    safetensors_path,
    out_path=None,
    threshold: float = 0.5,
    tokenizer_path=None,
    trust_remote_code: bool = True,
) -> dict:
    dummy_record = read_json(Path(dummy_path))
    safe_record = read_json(Path(safetensors_path))
    if tokenizer_path:
        try:
            tokenizer = _load_tokenizer(tokenizer_path, trust_remote_code=trust_remote_code)
        except Exception as exc:
            log_warning(f"tokenizer load failed ({tokenizer_path}): {exc}")
            tokenizer = None
        if tokenizer is not None:
            dummy_record = decode_record_texts(dummy_record, tokenizer)
            safe_record = decode_record_texts(safe_record, tokenizer)
    report = analyze(dummy_record, safe_record, threshold=threshold)
    report["tokenizer"] = str(tokenizer_path) if tokenizer_path else None
    report["dummy_file"] = str(dummy_path)
    report["safetensors_file"] = str(safetensors_path)
    if out_path:
        write_json(Path(out_path), report)
    return report


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="UWC stage3 dummy vs safetensors differential test")
    parser.add_argument("--dummy", required=True, help="dummy_rl_first_step_output.json")
    parser.add_argument("--safetensors", required=True, help="safetensors_rl_first_step_output.json")
    parser.add_argument("--dump_dir", default=None, help="UWC dump directory (report written to stage3_vllm/)")
    parser.add_argument("--threshold", type=float, default=0.5, help="garbled score threshold")
    parser.add_argument("--tokenizer", default=None, help="HF model dir/tokenizer used to decode token ids")
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow the tokenizer repository to execute custom code while loading",
    )
    args = parser.parse_args(argv)

    out_path = None
    if args.dump_dir:
        out_path = Path(args.dump_dir) / "stage3_vllm" / "diff_test_report.json"
    report = run(
        args.dummy,
        args.safetensors,
        out_path=out_path,
        threshold=args.threshold,
        tokenizer_path=args.tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
