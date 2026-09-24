# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke test for ``score._resolve_output_dirs``.

AISBench ships results in two layouts depending on the config / version:

1. **Conventional** — ``{work_dir}/results/`` + ``{work_dir}/summary/``
   (older configs, or when ``output_dir`` is overridden to the work root).
2. **Default timestamped** — ``{work_dir}/outputs/default/<ts>/{results,summary}/``
   (stock AISBench config; multiple runs accumulate under different timestamps).

The resolver must transparently pick whichever exists. Run:

    python3 scripts/test_score_paths.py
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from score import ScoreError, _resolve_output_dirs  # noqa: E402


def _populate(results_dir: pathlib.Path, summary_dir: pathlib.Path, ts_name: str) -> None:
    (results_dir / "wan22").mkdir(parents=True, exist_ok=True)
    (results_dir / "wan22" / "vbench_subject_consistency.json").write_text(
        '{"accuracy": 80.5}', encoding="utf-8"
    )
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / f"summary_{ts_name}.txt").write_text(
        "vbench_quality: {'accuracy': 75.0}\n"
        "vbench_semantic: {'accuracy': 80.0}\n"
        "vbench_total: {'accuracy': 77.5}\n",
        encoding="utf-8",
    )


def test_conventional() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        _populate(td / "results", td / "summary", "20260101")
        r, s, src = _resolve_output_dirs(td)
        assert r == td / "results", r
        assert s == td / "summary", s
        assert src == "conventional"
        print(f"PASS: conventional layout -> {src}")


def test_timestamped_default() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        ts = td / "outputs" / "default" / "20260828_120000"
        _populate(ts / "results", ts / "summary", "20260828")
        r, s, src = _resolve_output_dirs(td)
        assert r == ts / "results", r
        assert s == ts / "summary", s
        assert src == "outputs/default/20260828_120000", src
        print(f"PASS: timestamped layout -> {src}")


def test_picks_latest_timestamp() -> None:
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        for ts_name in ("20260801_000000", "20260815_000000", "20260828_120000"):
            ts = td / "outputs" / "default" / ts_name
            _populate(ts / "results", ts / "summary", ts_name)
        r, _, src = _resolve_output_dirs(td)
        assert "20260828_120000" in str(r), f"expected latest, got {r}"
        assert src == "outputs/default/20260828_120000", src
        print(f"PASS: multiple timestamps -> picks latest {src}")


def test_conventional_preferred_when_both() -> None:
    """Conventional layout wins ties — fewer hops, easier to debug."""
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        _populate(td / "results", td / "summary", "20260101")
        ts = td / "outputs" / "default" / "20260101"
        _populate(ts / "results", ts / "summary", "20260101")
        _, _, src = _resolve_output_dirs(td)
        assert src == "conventional", src
        print(f"PASS: conventional wins over timestamped when both present ({src})")


def test_no_output_raises() -> None:
    with tempfile.TemporaryDirectory() as td:
        try:
            _resolve_output_dirs(pathlib.Path(td))
        except ScoreError as e:
            assert e.code == "AISBENCH_NO_OUTPUT", e.code
            print(f"PASS: no layout -> {e.code}")
            return
        raise AssertionError("expected ScoreError")


def test_empty_outputs_default_raises() -> None:
    """``outputs/default/`` exists but has no timestamped subdirs → still missing."""
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        (td / "outputs" / "default").mkdir(parents=True)
        try:
            _resolve_output_dirs(td)
        except ScoreError as e:
            assert e.code == "AISBENCH_NO_OUTPUT", e.code
            print(f"PASS: empty outputs/default -> {e.code}")
            return
        raise AssertionError("expected ScoreError")


def main() -> int:
    tests = [
        test_conventional,
        test_timestamped_default,
        test_picks_latest_timestamp,
        test_conventional_preferred_when_both,
        test_no_output_raises,
        test_empty_outputs_default_raises,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL: {t.__name__}: {e}")
            failed += 1
    if failed:
        print(f"\n{failed} test(s) failed.")
        return 1
    print(f"\nAll {len(tests)} path-resolution smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())