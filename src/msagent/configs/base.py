#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is part of the MindStudio project.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#    http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Base classes for versioned configurations."""

from pydantic import BaseModel


class VersionedConfig(BaseModel):
    """Base class for versioned configs with migration support."""

    @classmethod
    def get_latest_version(cls) -> str:
        """Return latest version for this config type. Must be overridden by subclasses."""
        raise NotImplementedError(f"{cls.__name__} must implement get_latest_version()")

    @classmethod
    def migrate(cls, data: dict, from_version: str) -> dict:
        """Migrate config data from older version."""
        return data
