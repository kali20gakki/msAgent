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
"""Non-invasive wrapper to run the msmodelspec repo's response_regeneration/script.py
with LOCAL jsonl data -- WITHOUT modifying any file in the msmodelspec repo.

It reads script.py source, injects local `--data` support in-memory (arg + a
json-loader branch for the single dataset load line), then execs it so the
official script's full behavior (concurrency/retry/progress/error file) runs
unchanged. The repo files on disk are never touched.

Usage:
  python3 run_rr_local.py \
      [--script /path/to/.../response_regeneration/script.py] \
      -- --dataset sharegpt --data <local.jsonl> --endpoint <verifier 服务>/v1/chat/completions \
         --model <verifier 模型路径> --outfile <out>.jsonl --limit N [--concurrency 8]

Everything after the bare `--` is forwarded verbatim to script.py's CLI.
"""

import argparse
import os
import sys


def _default_script() -> str:
    """Locate response_regeneration/script.py without any hardcoded path."""
    script = os.environ.get("RR_SCRIPT")
    if script:
        return script
    repo = os.environ.get("REPO")
    if repo:
        return os.path.join(repo, "scripts", "response_regeneration", "script.py")
    return ""


DEFAULT_SCRIPT = _default_script()

# --data argument block to inject before the --split argument (identical to the
# reference patch, applied only in memory).
_ARG_BLOCK = '''    parser.add_argument(
        "--data",
        default=None,
        help="Optional local JSONL (json loader) instead of HF preset; "
             "pair with --dataset sharegpt for local ShareGPT-like data",
    )
'''
_ANCHOR_A = '    parser.add_argument(\n        "--split",\n        default=None,'

# The single HF-preset dataset load line inside main(); we branch to a json
# loader when --data is given.
_ANCHOR_B = "    dataset = load_dataset(dataset_id, name=subset, split=split, streaming=True)"
_LOAD_NEW = (
    "    if args.data is not None:\n"
    "        dataset = load_dataset(\n"
    '            "json", data_files=args.data, split="train", streaming=True\n'
    "        )\n"
    "        dataset_id = args.data\n"
    '        split = "train"\n'
    '        print(f"Loading local data: {args.data}")\n'
    "    else:\n"
    "        dataset = load_dataset(dataset_id, name=subset, split=split, streaming=True)"
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--script", default=DEFAULT_SCRIPT, help="Path to the msmodelspec repo's response_regeneration script.py"
    )
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="forwarded script.py args after a bare '--'")
    args = ap.parse_args()

    path = args.script
    if not path or not os.path.exists(path):
        sys.exit(
            "script.py not found: %r\n"
            "Specify --script <msmodelspec 仓>/scripts/response_regeneration/script.py, "
            "or set REPO=<msmodelspec 仓根> (or RR_SCRIPT) so the path can be derived." % path
        )
    with open(path, encoding="utf-8") as f:
        src = f.read()

    # 内存注入 --data 支持(官方脚本仓内文件零改动)
    if '"--data"' not in src:
        if src.count(_ANCHOR_A) != 1:
            sys.exit(
                "WRAPPER_ERROR: --split anchor not unique in this script.py version; update run_rr_local.py anchors"
            )
        src = src.replace(_ANCHOR_A, _ARG_BLOCK + _ANCHOR_A, 1)
    if src.count(_ANCHOR_B) != 1:
        sys.exit("WRAPPER_ERROR: dataset-load anchor not unique; update run_rr_local.py")
    src = src.replace(_ANCHOR_B, _LOAD_NEW, 1)

    # 转发 -- 后的参数并 exec 运行官方脚本
    script_argv = args.extra
    while script_argv and script_argv[0] == "--":
        script_argv = script_argv[1:]
    sys.argv = [path] + script_argv

    ns = {"__name__": "__main__"}
    code = compile(src, path, "exec")
    exec(code, ns)  # script.py main() runs under __main__ with injected --data


if __name__ == "__main__":
    main()
