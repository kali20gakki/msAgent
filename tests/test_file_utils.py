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

from __future__ import annotations

from types import SimpleNamespace

from msagent.utils import file as file_module


def test_get_file_language_returns_detected_lexer_name(monkeypatch) -> None:
    monkeypatch.setattr(
        file_module,
        "get_lexer_for_filename",
        lambda path: SimpleNamespace(name="Python"),
    )

    assert file_module.get_file_language("example.py") == "Python"


def test_get_file_language_falls_back_to_plaintext_on_error(monkeypatch) -> None:
    def raise_lookup_error(_path: str) -> SimpleNamespace:
        raise RuntimeError("missing lexer")

    monkeypatch.setattr(file_module, "get_lexer_for_filename", raise_lookup_error)

    assert file_module.get_file_language("unknown.custom") == "Plaintext"
