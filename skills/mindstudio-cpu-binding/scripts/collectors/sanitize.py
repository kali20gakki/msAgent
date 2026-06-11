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

"""Sanitize process command lines before storing diagnostic artifacts."""

from __future__ import annotations

import re
import shlex

SENSITIVE_KEY_PATTERN = re.compile(
    r"(?:token|secret|passw(?:or)?d|passwd|api[-_]?key|authorization|credential|access[-_]?key)",
    re.IGNORECASE,
)
REDACTED = "REDACTED"


def sanitize_command(command: str) -> str:
    if not command:
        return command
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()

    sanitized: list[str] = []
    redact_next = False
    for token in tokens:
        if redact_next:
            sanitized.append(REDACTED)
            redact_next = False
            continue

        key, sep, _value = token.partition("=")
        if sep and SENSITIVE_KEY_PATTERN.search(key):
            sanitized.append(f"{key}={REDACTED}")
            continue

        sanitized.append(token)
        if _is_sensitive_key_token(token):
            redact_next = True

    return " ".join(sanitized)


def sanitize_cmdline(cmdline: str) -> str:
    return sanitize_command(cmdline.replace("\x00", " ").strip())


def _is_sensitive_key_token(token: str) -> bool:
    stripped = token.lstrip("-")
    return bool(stripped and SENSITIVE_KEY_PATTERN.search(stripped))
