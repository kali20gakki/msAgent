#!/usr/bin/python3
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
import argparse
import logging
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BuildManager:
    """
    统一构建管理：依赖准备 → 构建 / 测试 → 归档。

    用法:
        python build.py                             安装 uv 并构建 wheel
        python build.py local                       使用本地 uv 构建 wheel
        python build.py --version/-v <version>      指定构建版本号
        python build.py --extra/-e KEY=VALUE        传递额外构建环境变量，可多次使用
        python build.py test                        安装 uv 并运行 UT 和 Skill 用例
        python build.py test local                  使用本地 uv 运行 UT 和 Skill 用例

    参数说明:
        - 参数: command    : 构建动作: 为空时为全构建, local 为使用本地 uv, test 为运行单元测试。
        - 参数: --version  : 构建版本号；不传时使用 version.info 中的 Version。
        - 参数: --extra    : 传递给构建脚本的环境变量，格式为 KEY=VALUE，可多次指定。
    """

    def __init__(self):
        self.project_root = Path(__file__).resolve().parent
        ap = argparse.ArgumentParser(description='Build the project and optionally run tests.')
        ap.add_argument(
            'command',
            nargs='*',
            default=[],
            choices=[[], 'local', 'test'],
            metavar='COMMAND',
            help='Build action: local and/or test',
        )
        ap.add_argument(
            '-v',
            '--version',
            type=str,
            help='Build version. Defaults to version.info when omitted.',
        )
        ap.add_argument(
            '-e',
            '--extra',
            metavar='KEY=VALUE',
            action='append',
            default=[],
            help='Extra build environment variables in KEY=VALUE format, can be specified multiple times',
        )
        self.args = ap.parse_args()

    def _execute_command(self, cmd, timeout_seconds=36000, cwd=None, env=None):
        logging.info("Running: %s", " ".join(cmd))
        subprocess.run(cmd, timeout=timeout_seconds, check=True, cwd=cwd, env=env)

    def _build_environment(self):
        env = os.environ.copy()
        if self.args.version:
            env["WHL_VERSION"] = self.args.version
        for opt in self.args.extra:
            key, _, val = opt.partition('=')
            if not key or not key.isidentifier() or not val:
                raise ValueError(f"Invalid --extra value {opt!r}; expected KEY=VALUE")
            env[key] = val
        return env

    def _archive_artifacts(self, src_dir, pattern):
        artifacts_dir = self.project_root / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        for src in Path(src_dir).glob(pattern):
            shutil.copy2(src, artifacts_dir)
            logging.info("Archived %s -> %s", src, artifacts_dir / src.name)

    def _build_whl(self, env):
        dist_dir = self.project_root / "dist"
        # 清理旧版本 whl
        shutil.rmtree(dist_dir, ignore_errors=True)

        self._execute_command(["bash", "scripts/build_whl.sh"], cwd=self.project_root, env=env)

        self._archive_artifacts(dist_dir, "*.whl")

    def _run_tests(self, env):
        self._execute_command(["bash", "scripts/run_ut.sh"], cwd=self.project_root, env=env)

    def run(self):
        os.chdir(self.project_root)

        env = self._build_environment()
        if "local" not in self.args.command:
            self._execute_command(
                [sys.executable, "-m", "pip", "install", "uv"],
                cwd=self.project_root,
            )

        if "test" in self.args.command:
            # -------------------- 单元测试 --------------------
            self._run_tests(env)
        else:
            # -------------------- 产品构建 --------------------
            logging.info("WHL_VERSION: %s", env.get("WHL_VERSION", "version.info"))
            for opt in self.args.extra:
                logging.info("--extra: %s", opt)
            self._build_whl(env)


if __name__ == "__main__":
    try:
        BuildManager().run()
    except Exception:
        logging.error("Unexpected error: %s", traceback.format_exc())
        sys.exit(1)
