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

"""本地 WHL 跨平台验证编排。"""

import os
from pathlib import Path

import nox


def _install_and_test(session: nox.Session, *, check_dependencies: bool = False) -> None:
    """安装待验证 WHL 及测试依赖后执行测试。"""
    session.chdir(Path(__file__).resolve().parent)
    whl_path = os.getenv("WHL_PATH", "").strip()
    if not whl_path:
        session.error("请设置 WHL_PATH 环境变量，指向待验证的 WHL 文件")

    whl = Path(whl_path).expanduser().resolve()
    if not whl.is_file():
        session.error(f"WHL_PATH 指向的文件不存在: {whl}")

    session.install(str(whl))
    if check_dependencies:
        session.run("pip", "check")
    session.install("-r", "requirements-test.txt")

    env_dict = {}
    llm_api_key = os.getenv("LLM_API_KEY")
    if llm_api_key is not None:
        env_dict["LLM_API_KEY"] = llm_api_key
    session.run("pytest", "test_case/", "-n", "auto", env=env_dict)


@nox.session(venv_backend="conda", python=["3.11"])
def test_conda(session: nox.Session) -> None:
    """在 Conda 环境中验证 WHL。"""
    _install_and_test(session, check_dependencies=True)


@nox.session(venv_backend="uv", python=["3.11"])
def test_uv(session: nox.Session) -> None:
    """在 uv 环境中验证 WHL。"""
    _install_and_test(session)
