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

"""本 Skill 测试的导入隔离。

host-cpu 下多个 Skill 都有 `report.py` / `workspace.py` / `preflight.py` 等同名模块，
它们的测试都用「往 sys.path 插 Skill 的 scripts 目录」的方式裸导入。同一 pytest
会话里先导入的 Skill 会把同名模块留在 sys.modules，后导入的 Skill 拿到别人的模块。
这里在收集本目录前把其它 Skill 的 scripts 模块从缓存里摘掉，并把自己的 scripts
目录插到 sys.path 首位，保证裸导入命中本 Skill 的实现。
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL_SCRIPTS = os.path.abspath(
    os.path.join(HERE, "..", "..", "..", "..", "skills", "host-cpu", "msagent-perf-cpu-config-audit", "scripts")
)
HOST_CPU_SKILLS = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "skills", "host-cpu"))


def _drop_foreign_skill_modules():
    for name, module in list(sys.modules.items()):
        location = getattr(module, "__file__", None)
        if not location:
            continue
        location = os.path.abspath(location)
        if location.startswith(HOST_CPU_SKILLS + os.sep) and not location.startswith(SKILL_SCRIPTS + os.sep):
            sys.modules.pop(name, None)


_drop_foreign_skill_modules()
if SKILL_SCRIPTS in sys.path:
    sys.path.remove(SKILL_SCRIPTS)
sys.path.insert(0, SKILL_SCRIPTS)
