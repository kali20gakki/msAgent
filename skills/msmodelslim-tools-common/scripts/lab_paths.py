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
