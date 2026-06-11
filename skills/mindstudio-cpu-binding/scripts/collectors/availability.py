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

"""Shared availability tracker for collector modules.

Each collector reports `missing` fields (data we could not obtain at all),
`partial` fields (data degraded by fallback paths) and `errors` (an
underlying command or read failure tied to a component). The top-level
collect.py merges per-collector instances into the snapshot's
`availability` block.

Semantics
---------
- `complete` is True iff `missing` and `errors` are both empty. `partial`
  is informational only (a fallback was used but a value was produced).
- `add_missing` and `add_partial` deduplicate within a single instance
  and across merges.
- `add_error` records {component, message} entries; identical pairs are
  deduplicated on merge.
- `to_dict()` returns fresh lists so callers can mutate the result
  without affecting the underlying state.
"""

from __future__ import annotations

from typing import Any


class Availability:
    def __init__(self) -> None:
        self._missing: set[str] = set()
        self._partial: set[str] = set()
        self._errors: list[dict[str, str]] = []

    def add_missing(self, field: str) -> None:
        self._missing.add(field)

    def add_partial(self, field: str) -> None:
        self._partial.add(field)

    def add_error(self, component: str, message: str) -> None:
        entry = {"component": component, "message": message}
        if entry not in self._errors:
            self._errors.append(entry)

    def merge(self, other: Availability) -> None:
        self._missing.update(other._missing)
        self._partial.update(other._partial)
        for entry in other._errors:
            if entry not in self._errors:
                self._errors.append(dict(entry))

    def to_dict(self) -> dict[str, Any]:
        return {
            "complete": not self._missing and not self._errors,
            "missing": sorted(self._missing),
            "partial": sorted(self._partial),
            "errors": [dict(entry) for entry in self._errors],
        }
