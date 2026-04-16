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

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from msagent.cli.bootstrap.initializer import Initializer
from msagent.configs import CheckpointerConfig, CheckpointerProvider


@pytest.mark.asyncio
async def test_memory_checkpointer_factory_returns_memory_saver() -> None:
    initializer = Initializer()
    config = CheckpointerConfig(type=CheckpointerProvider.MEMORY)

    async with initializer._create_checkpointer(config, None) as checkpointer:
        assert isinstance(checkpointer, InMemorySaver)


@pytest.mark.asyncio
async def test_sqlite_checkpointer_factory_returns_async_sqlite_saver(tmp_path) -> None:
    initializer = Initializer()
    config = CheckpointerConfig(type=CheckpointerProvider.SQLITE)
    db_path = tmp_path / "checkpoints.db"

    async with initializer._create_checkpointer(config, str(db_path)) as checkpointer:
        assert isinstance(checkpointer, AsyncSqliteSaver)
