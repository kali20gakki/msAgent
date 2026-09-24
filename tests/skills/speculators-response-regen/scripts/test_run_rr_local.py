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
"""speculators-response-regen / run_rr_local.py 单元测试。

覆盖：
- _default_script() 的环境优先级（RR_SCRIPT > REPO 推导 > 空）
- main() 的各类退出（--script 不存在；锚点 A/B 缺失/重复 → WRAPPER_ERROR）
- main() 的成功注入路径（用 tmp 的假 script.py，无需真实 msmodelspec 仓）：
  锚点 A/B 注入、--data 本地分支/else 分支、-- 后参数转发与去前导 --、
  磁盘仓文件零改动、already-patched 跳过锚点 A 注入、重复执行幂等
"""

import ast
import os
import sys

import pytest

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        "skills",
        "speculators",
        "speculators-response-regen",
        "scripts",
    ),
)

import run_rr_local  # noqa: E402
from run_rr_local import _default_script  # noqa: E402


def _write_script(tmp_path, text):
    p = tmp_path / "script.py"
    p.write_text(text, encoding="utf-8")
    return p


def _make_fake(data_value="None", already_patched=False, anchor_a=True, anchor_b=1):
    """拼一个可运行的假 response_regeneration/script.py。

    结构等价真实脚本：main() 内含 parser.add_argument("--split"…)、
    dataset = load_dataset(...) 两处锚点与 args/dataset_id/load_dataset 桩，
    便于 run_rr_local 注入后编译执行并被观测（STUB_LOAD / Loading local data / ARGS_SEEN）。
    """
    parts = ["import sys\n\n\ndef main():\n"]
    parts.append("    class _P(object):\n        def add_argument(self, *a, **k):\n            return self\n")
    parts.append("    parser = _P()\n")
    if already_patched:
        # 源码里已含字面 "--data" -> run_rr_local 应跳过锚点 A 注入（幂等保护）
        parts.append(
            '    parser.add_argument(\n        "--data",\n        default=None,\n        help="already",\n    )\n'
        )
    if anchor_a:
        parts.append(
            '    parser.add_argument(\n        "--split",\n        default=None,\n        help="split help",\n    )\n'
        )
    parts.append("    class _Args(object):\n        pass\n")
    parts.append("    args = _Args()\n")
    parts.append("    args.data = %s\n" % data_value)
    parts.append('    dataset_id = "some_preset"\n')
    parts.append("    subset = None\n")
    parts.append('    split = "train"\n')
    parts.append('    def load_dataset(*a, **k):\n        print("STUB_LOAD", repr(k))\n        return None\n')
    for _ in range(anchor_b):
        parts.append("    dataset = load_dataset(dataset_id, name=subset, split=split, streaming=True)\n")
    parts.append('    print("ARGS_SEEN", sys.argv)\n\n\n')
    parts.append('if __name__ == "__main__":\n    main()\n')
    src = "".join(parts)
    assert src.count(run_rr_local._ANCHOR_A) == (1 if anchor_a else 0), "anchor A miscount"
    assert src.count(run_rr_local._ANCHOR_B) == anchor_b, "anchor B miscount"
    assert ('"--data"' in src) == already_patched, "--data presence mismatch"
    return src


def _run_main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", argv)
    run_rr_local.main()


# ---------- _default_script: env 优先级 ----------
def test_default_script_empty_env(monkeypatch):
    monkeypatch.delenv("RR_SCRIPT", raising=False)
    monkeypatch.delenv("REPO", raising=False)
    assert _default_script() == ""


def test_default_script_from_rr_script_env(monkeypatch):
    monkeypatch.setenv("RR_SCRIPT", "/a/b/script.py")
    monkeypatch.delenv("REPO", raising=False)
    assert _default_script() == "/a/b/script.py"


def test_default_script_from_repo_env(monkeypatch):
    monkeypatch.delenv("RR_SCRIPT", raising=False)
    monkeypatch.setenv("REPO", "/repo/root")
    expected = os.path.join("/repo/root", "scripts", "response_regeneration", "script.py")
    assert _default_script() == expected


def test_default_script_rr_script_wins_over_repo(monkeypatch):
    monkeypatch.setenv("RR_SCRIPT", "/rr.py")
    monkeypatch.setenv("REPO", "/repo")
    assert _default_script() == "/rr.py"


# ---------- main(): 退出路径 ----------
def test_main_missing_script_exits(tmp_path, monkeypatch):
    missing = tmp_path / "absent.py"
    with pytest.raises(SystemExit) as ei:
        _run_main(monkeypatch, ["run_rr_local.py", "--script", str(missing)])
    assert "script.py not found" in ei.value.code


def test_main_anchor_a_zero_exits(tmp_path, monkeypatch):
    # 无 "--data" 且 --split 锚点缺失(0 次) -> WRAPPER_ERROR
    p = _write_script(tmp_path, "just text without anchors\n")
    with pytest.raises(SystemExit) as ei:
        _run_main(monkeypatch, ["run_rr_local.py", "--script", str(p)])
    assert "WRAPPER_ERROR" in ei.value.code and "--split anchor not unique" in ei.value.code


def test_main_anchor_a_dup_exits(tmp_path, monkeypatch):
    # 无 "--data" 且 --split 锚点出现 2 次 -> WRAPPER_ERROR
    src = run_rr_local._ANCHOR_A + "\n" + run_rr_local._ANCHOR_A + "\n"
    p = _write_script(tmp_path, src)
    with pytest.raises(SystemExit) as ei:
        _run_main(monkeypatch, ["run_rr_local.py", "--script", str(p)])
    assert "WRAPPER_ERROR" in ei.value.code and "--split anchor not unique" in ei.value.code


def test_main_anchor_b_dup_exits(tmp_path, monkeypatch):
    # 锚点 A 正常注入后，dataset-load 锚点出现 2 次 -> WRAPPER_ERROR
    src = run_rr_local._ANCHOR_A + "\n" + run_rr_local._ANCHOR_B + "\n" + run_rr_local._ANCHOR_B + "\n"
    p = _write_script(tmp_path, src)
    with pytest.raises(SystemExit) as ei:
        _run_main(monkeypatch, ["run_rr_local.py", "--script", str(p)])
    assert "WRAPPER_ERROR" in ei.value.code and "dataset-load anchor not unique" in ei.value.code


def test_main_anchor_b_zero_exits(tmp_path, monkeypatch):
    # 源码已含 "--data"(跳过 A 检查) 但 dataset-load 锚点缺失 -> WRAPPER_ERROR
    p = _write_script(tmp_path, _make_fake(already_patched=True, anchor_a=False, anchor_b=0))
    with pytest.raises(SystemExit) as ei:
        _run_main(monkeypatch, ["run_rr_local.py", "--script", str(p)])
    assert "WRAPPER_ERROR" in ei.value.code and "dataset-load anchor not unique" in ei.value.code


# ---------- main(): 成功注入 + exec ----------
def test_main_injects_and_runs_hf_branch(tmp_path, monkeypatch, capsys):
    # args.data=None -> 走 else 分支执行原 HF load_dataset 行
    src = _make_fake(data_value="None")
    p = _write_script(tmp_path, src)
    _run_main(
        monkeypatch,
        ["run_rr_local.py", "--script", str(p), "--", "--dataset", "sharegpt"],
    )
    out = capsys.readouterr().out
    assert "STUB_LOAD" in out
    assert "Loading local data" not in out
    # 磁盘仓文件零改动
    assert p.read_text(encoding="utf-8") == src


def test_main_local_data_branch_taken(tmp_path, monkeypatch, capsys):
    # args.data 非 None -> 走注入的 --data(json loader) 分支
    src = _make_fake(data_value='"LOCAL.jsonl"')
    p = _write_script(tmp_path, src)
    _run_main(
        monkeypatch,
        ["run_rr_local.py", "--script", str(p), "--", "--dataset", "sharegpt"],
    )
    out = capsys.readouterr().out
    assert "Loading local data: LOCAL.jsonl" in out
    assert "data_files" in out and "LOCAL.jsonl" in out
    assert p.read_text(encoding="utf-8") == src  # 文件未被改写


def test_main_forwards_argv_strips_leading_dashes(tmp_path, monkeypatch, capsys):
    # 多个前导 "--" 被剥掉，剩余参数原样进 sys.argv
    p = _write_script(tmp_path, _make_fake(data_value="None"))
    _run_main(
        monkeypatch,
        [
            "run_rr_local.py",
            "--script",
            str(p),
            "--",
            "--",
            "--dataset",
            "sharegpt",
            "--limit",
            "5",
        ],
    )
    out = capsys.readouterr().out
    seen = next(line for line in out.splitlines() if line.startswith("ARGS_SEEN "))
    argv = ast.literal_eval(seen[len("ARGS_SEEN ") :])
    assert argv[0] == str(p)
    assert argv[1:] == ["--dataset", "sharegpt", "--limit", "5"]


def test_main_already_patched_skips_anchor_a(tmp_path, monkeypatch, capsys):
    # 源码已含 "--data" 且无 --split 锚点：若未跳过 A 注入会 WRAPPER_ERROR；
    # 正确行为：跳过并正常运行
    p = _write_script(tmp_path, _make_fake(already_patched=True, anchor_a=False, anchor_b=1))
    _run_main(monkeypatch, ["run_rr_local.py", "--script", str(p)])
    out = capsys.readouterr().out
    assert "STUB_LOAD" in out


def test_main_rerun_is_idempotent(tmp_path, monkeypatch, capsys):
    # 同一未改动的脚本重复执行两次均成功且不报错
    src = _make_fake(data_value="None")
    p = _write_script(tmp_path, src)
    argv = ["run_rr_local.py", "--script", str(p), "--", "--dataset", "sharegpt"]
    _run_main(monkeypatch, argv)
    first = capsys.readouterr().out
    assert "STUB_LOAD" in first
    _run_main(monkeypatch, argv)
    second = capsys.readouterr().out
    assert "STUB_LOAD" in second
    assert p.read_text(encoding="utf-8") == src
