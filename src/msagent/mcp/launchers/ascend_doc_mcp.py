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

"""Launcher for the Ascend docs MCP server.

The ``@opencxd/ascend-doc-mcp`` package is started through ``npx``. Running it
directly from a config entry is fragile on Windows (the MCP client does not
resolve ``npx.cmd`` through PATHEXT) and lets npm diagnostics pollute the
JSON-RPC stdout channel. This launcher:

- resolves the Node toolchain with ``.cmd``/``.exe`` candidates on Windows;
- selects an npm registry (``MSAGENT_NPM_REGISTRY`` first, then the registry the
  machine's own npm is configured with, then ``registry.npmmirror.com`` ->
  ``mirrors.huaweicloud.com/repository/npm`` -> ``registry.npmjs.org``) by
  reading the registry's HTTP metadata, so a broken node/npm toolchain cannot
  block mirror selection, with ``MSAGENT_NPM_REGISTRY_ONLY=1`` to force the
  explicit one;
- puts the resolved Node directory on the child PATH, so the npm/npx
  ``#!/usr/bin/env node`` shebang resolves even when Node lives only under
  ``~/.msagent/node``;
- isolates the npm cache into a writable directory;
- forwards only JSON-RPC lines from the child to stdout;
- stays quiet by default (errors only): informational launcher diagnostics and
  child-process stderr are shown only with ``MSAGENT_ASCEND_DOC_MCP_VERBOSE=1``
  so msagent session output is not polluted.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import gettempdir
from threading import Thread
from typing import Any, BinaryIO
from urllib.parse import quote
from uuid import uuid4

import httpx

PACKAGE_NAME = "@opencxd/ascend-doc-mcp"
DEFAULT_REGISTRIES = (
    "https://registry.npmmirror.com",
    # Huawei Cloud's npm mirror: usually reachable from corporate intranets
    # where registry.npmmirror.com and registry.npmjs.org are blocked.
    "https://mirrors.huaweicloud.com/repository/npm",
    "https://registry.npmjs.org",
)
TOOLCHAIN_COMMANDS = ("node", "npx")

# Node toolchain installed by scripts/install.sh into ~/.msagent/node (override
# with MSAGENT_NODE_HOME). The ascend-doc-mcp package pre-installed into
# ~/.msagent/ascend-doc-mcp is executed directly instead of through npx so the
# first run works offline (override with MSAGENT_ASCEND_DOC_MCP_PREFIX).
_NODE_HOME_ENV = "MSAGENT_NODE_HOME"
_LOCAL_INSTALL_ENV = "MSAGENT_ASCEND_DOC_MCP_PREFIX"


def _user_state_dir() -> Path:
    return Path(os.path.expanduser("~")) / ".msagent"


def _node_home() -> Path:
    configured = os.getenv(_NODE_HOME_ENV, "").strip()
    base = Path(configured).expanduser() if configured else _user_state_dir() / "node"
    return base


def _local_install_prefix() -> Path:
    configured = os.getenv(_LOCAL_INSTALL_ENV, "").strip()
    base = Path(configured).expanduser() if configured else _user_state_dir() / "ascend-doc-mcp"
    return base


def _is_windows() -> bool:
    """Return whether the launcher runs on native Windows.

    Kept behind a function so tests can simulate the Windows layout without
    mutating the shared ``os.name`` attribute (which breaks pathlib under
    pytest on the other platform).
    """
    return os.name == "nt"


def _tool_candidate_names(command: str) -> tuple[str, ...]:
    if _is_windows():
        return (f"{command}.cmd", f"{command}.exe", command)
    return (command,)


def _resolve_tool(command: str) -> str | None:
    """Resolve a Node tool, preferring the installer-managed Node home."""
    node_home = _node_home()
    if _is_windows():
        candidate_dirs = (node_home,)
    else:
        candidate_dirs = (node_home / "bin",)
    for directory in candidate_dirs:
        for name in _tool_candidate_names(command):
            candidate = directory / name
            if candidate.is_file():
                return str(candidate)
    for name in _tool_candidate_names(command):
        if shutil.which(name):
            # Historic behavior: return the candidate name (spawnable on PATH),
            # not shutil.which's resolved path.
            return name
    return None


@dataclass(frozen=True, slots=True)
class RegistrySelection:
    """Resolved npm registry and its selection diagnostics."""

    registry: str
    diagnostics: list[str]


def _stderr(message: str) -> None:
    print(message, file=sys.stderr)


def _verbose_enabled() -> bool:
    return os.getenv("MSAGENT_ASCEND_DOC_MCP_VERBOSE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _log_info(message: str) -> None:
    """Informational diagnostics: hidden unless verbose mode is enabled."""
    if _verbose_enabled():
        print(message, file=sys.stderr)


def _report_exit(return_code: int, *, verbose: bool) -> int:
    """Return the child exit code, explaining non-zero exits in quiet mode."""
    code = int(return_code)
    if code != 0 and not verbose:
        _stderr(
            "ascend-doc-mcp launcher error: MCP server exited with code "
            f"{code}; set MSAGENT_ASCEND_DOC_MCP_VERBOSE=1 for details"
        )
    return code


def _resolve_version() -> str:
    version = os.getenv("MSAGENT_ASCEND_DOC_MCP_VERSION", "").strip()
    return version or "latest"


def _check_toolchain() -> None:
    missing = [command for command in TOOLCHAIN_COMMANDS if _resolve_tool(command) is None]
    if missing:
        raise RuntimeError(f"Missing required Node.js tooling: {', '.join(missing)}")


def _tool_command(command: str) -> str:
    resolved = _resolve_tool(command)
    if resolved is None:
        raise RuntimeError(f"Missing required Node.js tooling: {command}")
    return resolved


def _metadata_spec(version: str) -> str:
    return f"{PACKAGE_NAME}@{version}"


def _node_bin_dir() -> Path | None:
    """Return the directory holding the Node executable used by the launcher.

    ``_resolve_tool`` may return a bare command name when Node is only found on
    PATH, so the located path is resolved back to its directory.
    """
    node = _resolve_tool("node")
    if node is None:
        return None
    located = shutil.which(node) or node
    candidate = Path(located)
    if candidate.parent == Path("."):
        return None
    return candidate.parent


def _expose_node_on_path(environment: dict[str, str]) -> None:
    """Put the resolved Node directory on the child PATH.

    ``npm`` and ``npx`` are ``#!/usr/bin/env node`` scripts, so they need
    ``node`` on PATH even when the launcher resolved it to an absolute path
    (for example the installer-provisioned ``~/.msagent/node``). Without this,
    npm fails with ``/usr/bin/env: 'node': No such file or directory`` on hosts
    where Node is not on the ambient PATH.
    """
    node_bin_dir = _node_bin_dir()
    if node_bin_dir is None:
        return
    path_key = next((key for key in environment if key.upper() == "PATH"), "PATH")
    entries = [entry for entry in environment.get(path_key, "").split(os.pathsep) if entry]
    normalized = os.path.normcase(str(node_bin_dir))
    if any(os.path.normcase(entry) == normalized for entry in entries):
        return
    environment[path_key] = os.pathsep.join([str(node_bin_dir), *entries])


def _npm_environment() -> dict[str, str]:
    """Use a writable npm cache when the user's global cache is unavailable."""
    environment = os.environ.copy()
    _expose_node_on_path(environment)
    configured_cache = os.getenv("MSAGENT_NPM_CACHE", "").strip()
    candidates = [
        Path(configured_cache) if configured_cache else None,
        (
            Path(os.getenv("LOCALAPPDATA", "")) / "msagent" / "npm-cache"
            if _is_windows() and os.getenv("LOCALAPPDATA")
            else None
        ),
        Path.home() / ".cache" / "msagent" / "npm-cache",
        Path(gettempdir()) / "msagent-npm-cache",
    ]
    for cache in candidates:
        if cache is None:
            continue
        try:
            cache.mkdir(parents=True, exist_ok=True)
            # mkdir can succeed even when Windows ACLs, antivirus, or a
            # redirected profile prevent npm from creating its temp files.
            probe = cache / f".msagent-write-test-{uuid4().hex}"
            probe.write_bytes(b"")
            probe.unlink()
            environment["npm_config_cache"] = str(cache)
            return environment
        except OSError:
            try:
                probe.unlink()
            except (UnboundLocalError, OSError):
                pass
            continue
    return environment


def _probe_registry(registry: str, *, version: str, timeout_seconds: float = 20.0) -> str:
    """Return the package version this registry serves for ``version``.

    Reads the registry's HTTP metadata (``<registry>/<pkg>/<tag>``) instead of
    shelling out to ``npm view``: mirror selection then works even when node/npm
    are missing or broken, and no Node toolchain is needed just to pick a
    registry. The Node toolchain is only required to actually run the package.
    """
    spec = _metadata_spec(version)
    # Percent-encode the package name the way npm itself requests scoped
    # packages (``/@scope%2Fname``): the scope separator is data, not a path
    # separator, and a registry or proxy that routes on path depth would
    # otherwise see two segments. ``@`` stays literal -- it is a legal path
    # character and every registry serves it that way.
    package_path = quote(PACKAGE_NAME, safe="@")
    url = f"{registry.rstrip('/')}/{package_path}/{version}"
    try:
        response = httpx.get(
            url,
            timeout=timeout_seconds,
            follow_redirects=True,
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(f"metadata request failed for {spec}: {exc}") from exc

    try:
        parsed: Any = response.json()
    except ValueError as exc:
        raise RuntimeError(f"npm metadata for {spec} was not valid JSON: {exc}") from exc

    if isinstance(parsed, dict):
        value = parsed.get("version")
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise RuntimeError(f"npm metadata for {spec} did not include a version")

    if isinstance(parsed, str) and parsed.strip():
        return parsed.strip()

    raise RuntimeError(f"npm metadata for {spec} was not a version string")


def _configured_npm_registry() -> str | None:
    """Registry npm itself is configured to use, when it is not a public default.

    Corporate intranets usually point npm at an internal mirror (``.npmrc``
    pushed by IT), so honoring that configuration makes on-demand fetches work
    on an intranet without hardcoding any internal host here. Any failure or
    timeout is ignored: the caller falls back to the probed defaults.
    """
    npm = _resolve_tool("npm")
    if npm is None:
        return None
    try:
        completed = subprocess.run(
            [npm, "config", "get", "registry"],
            capture_output=True,
            check=False,
            text=True,
            timeout=15.0,
            env=_npm_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    value = (completed.stdout or "").strip()
    if not value or value.lower() in {"undefined", "null"}:
        return None
    normalized = value.rstrip("/")
    if any(normalized == registry.rstrip("/") for registry in DEFAULT_REGISTRIES):
        return None
    return value


def _registry_candidates() -> list[str]:
    preferred = os.getenv("MSAGENT_NPM_REGISTRY", "").strip()
    registry_only = os.getenv("MSAGENT_NPM_REGISTRY_ONLY", "").strip() == "1"

    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
        if registry_only:
            return candidates

    for registry in DEFAULT_REGISTRIES:
        if registry not in candidates:
            candidates.append(registry)
    return candidates


def select_registry(*, version: str | None = None) -> RegistrySelection:
    """Choose an npm registry by probing package metadata."""
    resolved_version = version or _resolve_version()
    registry_only = os.getenv("MSAGENT_NPM_REGISTRY_ONLY", "").strip() == "1"
    preferred_registry = os.getenv("MSAGENT_NPM_REGISTRY", "").strip()
    if registry_only and not preferred_registry:
        raise RuntimeError("MSAGENT_NPM_REGISTRY_ONLY=1 requires MSAGENT_NPM_REGISTRY to be set")

    candidates = _registry_candidates()
    if not registry_only:
        configured = _configured_npm_registry()
        if configured and configured not in candidates:
            # Explicit env var first, then the machine's own npm configuration
            # (that is where intranet mirrors live), then the public defaults.
            candidates.insert(1 if preferred_registry else 0, configured)

    diagnostics: list[str] = []
    last_error: str | None = None

    for registry in candidates:
        try:
            discovered_version = _probe_registry(registry, version=resolved_version)
        except Exception as exc:  # pragma: no cover - surfaced in diagnostics
            last_error = f"{registry}: {exc}"
            diagnostics.append(last_error)
            continue

        diagnostics.append(
            f"Selected npm registry for {PACKAGE_NAME}@{resolved_version}: {registry} "
            f"(metadata version: {discovered_version})"
        )
        return RegistrySelection(registry=registry, diagnostics=diagnostics)

    details = "; ".join(diagnostics)
    raise RuntimeError(
        f"Unable to resolve a working npm registry for {PACKAGE_NAME}@{resolved_version}"
        + (f": {details}" if details else "")
    )


def build_npx_command(*, registry: str, version: str | None = None) -> list[str]:
    """Build the npx command used to spawn the MCP server."""
    resolved_version = version or _resolve_version()
    return [
        _tool_command("npx"),
        "--yes",
        "--registry",
        registry,
        _metadata_spec(resolved_version),
    ]


def _resolve_local_run_command() -> list[str] | None:
    """Resolve a directly executable command for a local package install.

    scripts/install.sh / install.ps1 pre-install ``@opencxd/ascend-doc-mcp``
    into ``~/.msagent/ascend-doc-mcp`` (``MSAGENT_ASCEND_DOC_MCP_PREFIX``
    overrides). When present, the launcher runs ``node <package-bin>`` instead
    of fetching through npx, so the first use works offline.
    """
    pkg_dir = _local_install_prefix() / "node_modules" / "@opencxd" / "ascend-doc-mcp"
    package_json = pkg_dir / "package.json"
    if not package_json.is_file():
        return None
    try:
        metadata = json.loads(package_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    bin_field = metadata.get("bin")
    if isinstance(bin_field, str):
        bin_rel = bin_field
    elif isinstance(bin_field, dict):
        bin_rel = bin_field.get("ascend-doc-mcp")
        if not isinstance(bin_rel, str):
            # Fall back to the first declared bin entry.
            bin_rel = next(iter(bin_field.values()), None)
    else:
        return None
    if not isinstance(bin_rel, str) or not bin_rel.strip():
        return None

    script = (pkg_dir / bin_rel.strip()).resolve()
    if not script.is_file():
        return None
    node_bin = _resolve_tool("node")
    if node_bin is None:
        return None
    return [node_bin, str(script)]


def _is_jsonrpc_message(line: bytes) -> bool:
    """Return whether a stdout line is a JSON-RPC 2.0 message."""
    try:
        message = json.loads(line.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(message, dict) and message.get("jsonrpc") == "2.0"


def _forward_mcp_stdout(stdout: BinaryIO, protocol_stdout: BinaryIO, diagnostics: BinaryIO) -> None:
    """Keep upstream diagnostics out of the MCP protocol stream."""
    for line in stdout:
        target = protocol_stdout if _is_jsonrpc_message(line) else diagnostics
        try:
            target.write(line)
            target.flush()
        except BrokenPipeError:
            return


@contextmanager
def _child_error_sink(*, verbose: bool):
    """Yield a discard sink when quiet, or None when verbose (use stderr)."""
    if verbose:
        yield None
        return
    with open(os.devnull, "wb") as devnull:
        yield devnull


def _run_mcp_server(command: list[str]) -> int:
    """Run the child while separating JSON-RPC from accidental stdout logs.

    Child stderr (and non-JSON-RPC stdout lines) are forwarded only in verbose
    mode; otherwise they are discarded so msagent's session output stays clean.
    """
    default_stderr = getattr(sys.stderr, "buffer", sys.stderr)
    with _child_error_sink(verbose=_verbose_enabled()) as error_sink:
        child_stderr = error_sink if error_sink is not None else default_stderr
        with subprocess.Popen(
            command,
            stdin=getattr(sys.stdin, "buffer", sys.stdin),
            stdout=subprocess.PIPE,
            stderr=child_stderr,
            env=_npm_environment(),
        ) as child:
            if child.stdout is None:
                raise RuntimeError("npx MCP server did not provide a stdout pipe")

            protocol_stdout = getattr(sys.stdout, "buffer", sys.stdout)
            diagnostics = error_sink if error_sink is not None else default_stderr
            forwarder = Thread(
                target=_forward_mcp_stdout,
                args=(child.stdout, protocol_stdout, diagnostics),
                name="ascend-doc-mcp-stdout-forwarder",
                daemon=True,
            )
            forwarder.start()
            try:
                return_code = child.wait()
            finally:
                forwarder.join()
            return int(return_code)


def main() -> int:
    """Start the Ascend docs MCP server (local install first, npx as fallback).

    Informational diagnostics are hidden unless MSAGENT_ASCEND_DOC_MCP_VERBOSE=1.
    """
    verbose = _verbose_enabled()
    if _resolve_tool("node") is None:
        _stderr("ascend-doc-mcp launcher error: Missing required Node.js tooling: node")
        _stderr(
            "ascend-doc-mcp launcher error: re-run the msagent installer to "
            "provision Node, or install Node.js >= 22 yourself."
        )
        return 127

    local_command = _resolve_local_run_command()
    if local_command is not None:
        node_bin, script = local_command
        _log_info(f"ascend-doc-mcp launcher: running local install {Path(script).parent} (node: {node_bin})")
        try:
            return _report_exit(_run_mcp_server(local_command), verbose=verbose)
        except Exception as exc:
            _stderr(f"ascend-doc-mcp launcher error while starting local MCP server: {exc}")
            return 127

    # Only npx is required here: registry selection uses plain HTTP metadata,
    # so a broken or missing npm no longer blocks the on-demand path.
    missing = [command for command in ("npx",) if _resolve_tool(command) is None]
    if missing:
        _stderr(f"ascend-doc-mcp launcher error: Missing required Node.js tooling: {', '.join(missing)}")
        return 127

    version = _resolve_version()
    _log_info(f"ascend-doc-mcp launcher: probing npm registries for {PACKAGE_NAME}@{version}")

    try:
        selection = select_registry(version=version)
    except Exception as exc:
        _stderr(f"ascend-doc-mcp launcher error: {exc}")
        return 127

    for line in selection.diagnostics:
        _log_info(f"ascend-doc-mcp launcher: {line}")

    command = build_npx_command(registry=selection.registry, version=version)
    _log_info(f"ascend-doc-mcp launcher: exec {' '.join(command)}")

    try:
        return _report_exit(_run_mcp_server(command), verbose=verbose)
    except Exception as exc:
        _stderr(f"ascend-doc-mcp launcher error while starting MCP server: {exc}")
        return 127


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
