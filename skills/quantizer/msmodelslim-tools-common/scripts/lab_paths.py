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
"""lab_practice / lab_calib path resolution (aligned with msmodelslim package layout)."""

from __future__ import annotations

from pathlib import Path

from msmodelslim.utils.security.path import get_valid_read_path


def get_lab_practice_dir() -> Path:
    import msmodelslim.lab_practice as lab_practice_pkg

    lab_practice_dir = Path(list(lab_practice_pkg.__path__)[0])
    lab_practice_dir = get_valid_read_path(str(lab_practice_dir), is_dir=True)
    return Path(lab_practice_dir)


def get_lab_calib_dir() -> Path:
    import msmodelslim.lab_calib as lab_calib_pkg

    lab_calib_dir = Path(list(lab_calib_pkg.__path__)[0])
    lab_calib_dir = get_valid_read_path(str(lab_calib_dir), is_dir=True)
    return Path(lab_calib_dir)
