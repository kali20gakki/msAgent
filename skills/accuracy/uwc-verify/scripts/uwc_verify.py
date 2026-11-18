#!/usr/bin/env python3
"""uwc-verify: automate the 5 UWC weight-consistency checks for veRL + mcore + vLLM.

Commands:
  doctor   检查环境/配置/目标源码是否就绪
  prepare  向 veRL 源码注入 best-effort 采集钩子（幂等，支持 --dry-run/--restore）
  run      按阶段启动采集命令
  compare  离线比对 P0/P2、P2_sender/P5、data_ptr、G0/G1、dummy/safetensors
  report   生成 report.md / report.html / report.json
  all      prepare + run + compare + report
  demo     用合成数据跑通离线链路并生成演示报告
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_DIR = SKILL_ROOT / "runtime"
if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is a msprobe dependency
    yaml = None

import uwc  # noqa: E402
from uwc.common import now_iso, read_json, write_json  # noqa: E402
from uwc.offline import stage3_diff_test, uwc_compare, uwc_summary  # noqa: E402

PATCH_BEGIN = "# >>> UWC-SKILL-PATCH v1 (managed; uwc-verify restore removes this) >>>"
PATCH_END = "# <<< UWC-SKILL-PATCH v1 <<<"
PATCH_BLOCK = (
    "\n\n"
    + PATCH_BEGIN
    + "\n"
    + "try:\n"
    + "    from verl.utils.uwc.hooks import install_module\n"
    + "    install_module(__name__, globals())\n"
    + "except Exception as _uwc_exc:\n"
    + '    print(f"[UWC] hook install skipped: {_uwc_exc}", flush=True)\n'
    + PATCH_END
    + "\n"
)

CORE_TARGETS = [
    "verl/workers/engine/megatron/transformer_impl.py",
    "verl/workers/rollout/vllm_rollout/utils.py",
    "verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py",
    "verl/workers/engine_workers.py",
    "verl/workers/rollout/vllm_rollout/vllm_rollout.py",
    # veRL 0.9 async rollout: generation goes through this server, not through
    # ServerAdapter.generate_sequences (checked for the stage-5 diff dump).
    "verl/workers/rollout/vllm_rollout/vllm_async_server.py",
]
OPTIONAL_TARGETS = [
    "verl/experimental/agent_loop/agent_loop.py",
]
RUNTIME_MARKER = ".uwc_skill_runtime"
MANIFEST_NAME = ".uwc_skill_manifest.json"

USE_SHM_PATTERN = re.compile(
    r"(?m)^(?P<indent>[ \t]*)self\.use_shm[ \t]*=[ \t]*not[ \t]+is_support_ipc\(\)[ \t]*$"
)
USE_SHM_REPLACEMENT = (
    '{indent}self.use_shm = __import__("os").environ.get("VERL_FORCE_USE_SHM", "0") == "1" '
    "or not is_support_ipc()"
)
# Matches the line written by ``USE_SHM_REPLACEMENT`` so ``--restore`` can
# still undo an edit whose manifest record was lost (e.g. a second ``prepare``
# took the "already patched" branch before the manifest carried edits forward).
USE_SHM_PATCHED_PATTERN = re.compile(
    r"(?m)^(?P<indent>[ \t]*)self\.use_shm[ \t]*=[ \t]*"
    r'__import__\("os"\)\.environ\.get\("VERL_FORCE_USE_SHM", "0"\) == "1"[ \t]*'
    r"or[ \t]+not[ \t]+is_support_ipc\(\)[ \t]*$"
)

DEFAULT_STOP_WHEN = {
    "stage1": ["stage1_bridge/P2_hf_exported_step*_rank*.pt"],
    "stage2": ["stage2_transfer/P5_receiver_step*_rank*.pt"],
    "stage3_ptr": ["stage3_vllm/ptr_before_after_step*_rank*.json"],
    "stage3_dummy": ["stage3_vllm/dummy_rl_first_step_output.json"],
    "stage3_safetensors": ["stage3_vllm/safetensors_rl_first_step_output.json"],
}

DEFAULT_TIMEOUTS = {
    "stage1": 600,
    "stage2": 1200,
    "stage3_ptr": 1200,
    "stage3_dummy": 1800,
    "stage3_safetensors": 1800,
    "forward": 900,
    "default": 900,
}


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def _default_config() -> dict:
    return {
        "verl": {
            "path": None,
            "cwd": None,
            "python": sys.executable,
            "set_pythonpath": True,
        },
        "model": {
            "hf_path": None,
            "config_json": None,
        },
        "dump_dir": "/tmp/uwc_dump",
        "stages": [1, 2, 3, 4, 5],
        "commands": {},
        "stop_when": dict(DEFAULT_STOP_WHEN),
        "timeouts": dict(DEFAULT_TIMEOUTS),
        "env": {},
        "compare": {
            "stage1_atol": 1e-3,
            "stage1_rtol": 0.0,
            "stage1_exact": False,
            "stage2_exact": True,
            "qkv_layout": "contiguous",
            "gate_up_layout": "contiguous",
            "avg_cos": 0.99,
            "min_cos": 0.95,
            "diff_threshold": 0.5,
        },
        "forward": {
            "input_ids": [1, 2, 3, 4, 5],
            "device": None,
            "dtype": "fp32",
            "use_builtin": True,
        },
        "report": {
            "title": "UWC 训推权重一致性验证报告",
            "out_dir": None,
        },
        "patch": {
            "use_shm": True,
            "optional_targets": True,
        },
    }


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _expand(value, variables):
    if isinstance(value, str):
        for key, replacement in variables.items():
            if replacement is not None:
                value = value.replace("${" + key + "}", str(replacement))
        return value
    if isinstance(value, list):
        return [_expand(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item, variables) for key, item in value.items()}
    return value


def load_config(path: str | Path | None) -> tuple[dict, Path | None]:
    if path is None:
        candidate = Path.cwd() / "uwc.yaml"
        path = candidate if candidate.is_file() else None
    config = _default_config()
    config_path = None
    if path:
        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"config not found: {config_path}")
        text = config_path.read_text(encoding="utf-8")
        if config_path.suffix in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("PyYAML is required to read YAML config; use JSON or install pyyaml")
            payload = yaml.safe_load(text) or {}
        else:
            payload = json.loads(text)
        config = _deep_merge(config, payload)

    # Environment overrides.
    if os.environ.get("UWC_VERL_PATH"):
        config["verl"]["path"] = os.environ["UWC_VERL_PATH"]
    if os.environ.get("UWC_DUMP_DIR"):
        config["dump_dir"] = os.environ["UWC_DUMP_DIR"]
    if os.environ.get("UWC_MODEL_PATH"):
        config["model"]["hf_path"] = os.environ["UWC_MODEL_PATH"]

    variables = {
        "dump_dir": config.get("dump_dir"),
        "verl.path": (config.get("verl") or {}).get("path"),
        "model.hf_path": (config.get("model") or {}).get("hf_path"),
        "config_dir": str(config_path.parent) if config_path else None,
    }
    config = _expand(config, variables)

    # Derive paths that were not set explicitly.
    model = config.get("model") or {}
    if model.get("hf_path") and not model.get("config_json"):
        candidate = Path(model["hf_path"]) / "config.json"
        model["config_json"] = str(candidate) if candidate.is_file() else None
    if not (config.get("verl") or {}).get("cwd"):
        if config.get("verl", {}).get("path"):
            config["verl"]["cwd"] = config["verl"]["path"]
    return config, config_path


def _dump_dir(config: dict) -> Path:
    path = Path(config.get("dump_dir") or "/tmp/uwc_dump").expanduser().resolve()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        # A dump directory that cannot be created must abort loudly; silently
        # continuing would let later stages report "skip" without telling the
        # user their output never had a place to land.
        raise SystemExit(f"uwc-verify: dump_dir {path} is not usable: {exc}") from exc
    return path


def _verl_root(config: dict) -> Path | None:
    value = (config.get("verl") or {}).get("path")
    if not value:
        return None
    root = Path(value).expanduser().resolve()
    return root if root.exists() else None


def _model_config(config: dict):
    model = config.get("model") or {}
    return model.get("config_json") or None


# ---------------------------------------------------------------------------
# prepare / restore
# ---------------------------------------------------------------------------


def _strip_patch(text: str) -> str:
    """Remove the appended block, restoring the file's trailing newline."""
    pattern = re.compile(
        r"\n*"
        + re.escape(PATCH_BEGIN)
        + r".*?"
        + re.escape(PATCH_END)
        + r"\n?",
        re.DOTALL,
    )
    # ``prepare`` rstrips the source before appending, so exactly one newline
    # belongs to the original file.
    return pattern.sub("\n", text)


def _copy_runtime(verl_root: Path, force: bool, dry_run: bool) -> dict:
    target = verl_root / "verl" / "utils" / "uwc"
    marker = target / RUNTIME_MARKER
    source = RUNTIME_DIR / "uwc"
    info = {"target": str(target), "source": str(source), "action": "copy"}
    if target.exists() and not marker.is_file() and not force:
        raise RuntimeError(
            f"{target} already exists and is not managed by uwc-verify; "
            "refusing to overwrite. Re-run with --force to replace it."
        )
    if target.exists() and not force and marker.is_file():
        info["action"] = "update"
    if dry_run:
        info["action"] = "dry-run:" + info["action"]
        return info

    if target.exists() and (force or marker.is_file()):
        shutil.rmtree(target)
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (target / RUNTIME_MARKER).write_text(
        f"managed by uwc-verify {uwc.__version__}\n", encoding="utf-8"
    )
    return info


def _apply_use_shm(text: str, path: Path) -> tuple[str, dict | None]:
    if USE_SHM_PATTERN.search(text):
        match = USE_SHM_PATTERN.search(text)
        original_line = match.group(0)
        replacement = USE_SHM_REPLACEMENT.format(indent=match.group("indent"))
        patched = text[: match.start()] + replacement + text[match.end() :]
        return patched, {"path": str(path), "original_line": original_line, "patched_line": replacement}
    return text, None


def _remove_use_shm(text: str, edit: dict) -> str:
    patched_line = edit.get("patched_line")
    original_line = edit.get("original_line")
    if patched_line and patched_line in text:
        return text.replace(patched_line, original_line, 1)
    # Fallback for manifests written before edits were carried across repeated
    # ``prepare`` runs: rebuild the canonical original line.
    return USE_SHM_PATCHED_PATTERN.sub(
        lambda match: f"{match.group('indent')}self.use_shm = not is_support_ipc()",
        text,
    )


def prepare(
    config: dict,
    dry_run: bool = False,
    restore: bool = False,
    force: bool = False,
) -> int:
    verl_root = _verl_root(config)
    if verl_root is None:
        print("ERROR: verl.path is not set or does not exist. Set verl.path in uwc.yaml.", file=sys.stderr)
        return 2
    if not (verl_root / "verl").is_dir():
        print(f"ERROR: {verl_root} does not contain a 'verl/' package directory.", file=sys.stderr)
        return 2

    manifest_path = verl_root / MANIFEST_NAME
    manifest = {}
    if manifest_path.is_file():
        try:
            manifest = read_json(manifest_path)
        except Exception:
            manifest = {}

    targets = list(CORE_TARGETS)
    if (config.get("patch") or {}).get("optional_targets", True):
        targets.extend(OPTIONAL_TARGETS)

    if restore:
        if not manifest:
            print(f"WARNING: no manifest at {manifest_path}; only stripping patch blocks.")
        # Revert textual edits first.
        for edit in manifest.get("edits", []) or []:
            path = Path(edit.get("path", ""))
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            new_text = _remove_use_shm(text, edit)
            if new_text != text and not dry_run:
                path.write_text(new_text, encoding="utf-8")
                print(f"reverted use_shm edit: {path}")
        # Strip appended blocks.
        for relative in manifest.get("targets", targets) or targets:
            path = verl_root / relative
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            # Also undo a use_shm edit whose manifest record is missing (older
            # manifests, or a prepare that saw "already patched").
            new_text = _remove_use_shm(text, {})
            new_text = _strip_patch(new_text)
            if new_text != text:
                if dry_run:
                    print(f"[dry-run] would strip UWC block: {path}")
                else:
                    path.write_text(new_text, encoding="utf-8")
                    print(f"stripped UWC block: {path}")
        runtime_dir = verl_root / "verl" / "utils" / "uwc"
        if (runtime_dir / RUNTIME_MARKER).is_file():
            if dry_run:
                print(f"[dry-run] would remove {runtime_dir}")
            else:
                shutil.rmtree(runtime_dir)
                print(f"removed runtime: {runtime_dir}")
        if manifest_path.is_file() and not dry_run:
            manifest_path.unlink()
            print(f"removed manifest: {manifest_path}")
        return 0

    # Check targets.
    found = []
    missing = []
    for relative in targets:
        path = verl_root / relative
        if path.is_file():
            found.append(path)
        else:
            missing.append(relative)

    print(f"verl root: {verl_root}")
    print(f"targets found: {len(found)}")
    for path in found:
        print(f"  - {path.relative_to(verl_root)}")
    if missing:
        print("targets not found (skipped, may be a different veRL version):")
        for relative in missing:
            print(f"  - {relative}")

    runtime_info = _copy_runtime(verl_root, force=force, dry_run=dry_run)
    print(f"runtime: {runtime_info['action']} {runtime_info['target']}")

    edits = []
    manifest_targets = []
    # Re-running ``prepare`` must not drop the edit records written by the
    # first run: otherwise ``--restore`` can no longer undo the use_shm edit.
    previous_edits = {
        edit.get("path"): edit
        for edit in (manifest.get("edits") or [])
        if isinstance(edit, dict) and edit.get("path")
    }
    for path in found:
        text = path.read_text(encoding="utf-8")
        cleaned = _strip_patch(text)
        new_text = cleaned
        if (config.get("patch") or {}).get("use_shm", True):
            new_text, edit = _apply_use_shm(new_text, path)
            if edit:
                edits.append(edit)
                print(f"  use_shm patched: {path.relative_to(verl_root)}")
            elif "VERL_FORCE_USE_SHM" in new_text:
                previous_edit = previous_edits.get(str(path))
                if previous_edit:
                    edits.append(previous_edit)
                    print(f"  use_shm already patched: {path.relative_to(verl_root)} (edit carried over)")
                else:
                    print(f"  use_shm already patched: {path.relative_to(verl_root)} (no previous edit record)")
            elif "use_shm" in new_text:
                print(f"  WARNING: use_shm anchor not found in {path.relative_to(verl_root)} (runtime hook will try)")
        new_text = new_text.rstrip() + PATCH_BLOCK
        # Always validate syntax before writing.
        try:
            compile(new_text, str(path), "exec")
        except SyntaxError as exc:
            print(f"ERROR: patched file would not compile: {path}: {exc}", file=sys.stderr)
            return 3
        manifest_targets.append(str(path.relative_to(verl_root)))
        if dry_run:
            print(f"[dry-run] would patch: {path.relative_to(verl_root)}")
        else:
            path.write_text(new_text, encoding="utf-8")
            print(f"patched: {path.relative_to(verl_root)}")

    if not dry_run:
        manifest = {
            "schema": "uwc-skill-manifest/v1",
            "created_at": now_iso(),
            "verl_root": str(verl_root),
            "runtime": str(verl_root / "verl" / "utils" / "uwc"),
            "targets": manifest_targets,
            "edits": edits,
        }
        write_json(manifest_path, manifest)
        print(f"manifest: {manifest_path}")
    else:
        print("[dry-run] no files were changed")
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _base_env(config: dict) -> dict:
    env = dict(os.environ)
    env.update({str(key): str(value) for key, value in (config.get("env") or {}).items()})
    env["UWC_DUMP_DIR"] = str(_dump_dir(config))
    model = config.get("model") or {}
    if model.get("hf_path"):
        env.setdefault("UWC_MODEL_PATH", str(model["hf_path"]))
    verl_root = _verl_root(config)
    if verl_root and (config.get("verl") or {}).get("set_pythonpath", True):
        existing = env.get("PYTHONPATH", "")
        parts = [str(verl_root)] + ([existing] if existing else [])
        env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _command_for(config: dict, key: str) -> str | None:
    commands = config.get("commands") or {}
    value = commands.get(key)
    if value is None:
        return None
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _stop_files(dump_dir: Path, patterns, since: float | None = None) -> list[Path]:
    """Return files matching ``patterns``.

    ``since`` filters by mtime: a file left over from an earlier run must not
    stop the current run before it even produced anything.  ``run`` passes the
    process start time; ``reuse`` passes ``None`` to accept any existing file.
    """
    matches = []
    for pattern in patterns or []:
        for path in sorted(dump_dir.glob(pattern)):
            if since is not None:
                try:
                    if path.stat().st_mtime < since:
                        continue
                except OSError:
                    continue
            matches.append(path)
    return matches


def _terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=15)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        proc.kill()
    proc.wait(timeout=10)


def _process_alive(pid: int) -> bool:
    """Return False for missing or zombie processes (unreaped zombies keep /proc entries)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    try:
        state = stat.rsplit(")", 1)[1].split()[0]
    except (IndexError, ValueError):
        return False
    return state != "Z"


_STAGE_ENV_KEYS = (
    "UWC_DUMP_DIR",
    "UWC_STAGE1",
    "UWC_STAGE2",
    "UWC_STAGE3",
    "UWC_PTR_CHECK",
    "UWC_DIFF_DUMP",
    "UWC_DIFF_TAG",
    "UWC_LOAD_FORMAT",
)


def _cleanup_stage_processes(env: dict) -> int:
    """Kill stage processes that outlive the command that owns them.

    veRL starts Ray actors in their own sessions, so terminating the stage's
    process group leaves ``ray::WorkerDict`` / vLLM workers behind.  Those
    orphans keep holding NPU memory and the *next* stage OOMs.  Match on the
    stage-specific ``UWC_*`` environment markers so only this run's processes
    are touched (no name-based killing).
    """
    markers = [
        f"{key}={env[key]}".encode()
        for key in _STAGE_ENV_KEYS
        if isinstance(env.get(key), str)
    ]
    if not markers:
        return 0

    victims: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid() or not _process_alive(pid):
            continue
        try:
            parts = (entry / "environ").read_bytes().split(b"\0")
        except OSError:
            continue
        if all(marker in parts for marker in markers):
            victims.append(pid)

    for pid in victims:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.time() + 10
    while time.time() < deadline and any(_process_alive(pid) for pid in victims):
        time.sleep(0.5)
    for pid in victims:
        if _process_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    deadline = time.time() + 10
    while time.time() < deadline and any(_process_alive(pid) for pid in victims):
        time.sleep(0.5)
    return len(victims)


def run_process(
    command: str,
    env: dict,
    cwd: str | Path | None,
    log_path: Path,
    timeout: int,
    stop_patterns,
    dump_dir: Path,
) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    stale = _stop_files(dump_dir, stop_patterns)
    if stale:
        print(
            f"[run] ignoring {len(stale)} pre-existing stop file(s) "
            f"(e.g. {stale[0].name}); waiting for a fresh dump"
        )
    with open(log_path, "wb") as log_handle:
        proc = subprocess.Popen(
            command,
            shell=True,
            executable="/bin/bash",
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        stopped_by = None
        timed_out = False
        while True:
            if proc.poll() is not None:
                break
            if stop_patterns and _stop_files(dump_dir, stop_patterns, since=started):
                stopped_by = "dump_found"
                break
            if time.time() - started > timeout:
                timed_out = True
                break
            time.sleep(1)
        if proc.poll() is None:
            _terminate_process(proc)
    leftover = _cleanup_stage_processes(env)
    if leftover:
        print(f"[run] cleaned up {leftover} leftover process(es) from the stage")
    return {
        "command": command,
        "cwd": str(cwd) if cwd else None,
        "log": str(log_path),
        "returncode": proc.returncode,
        "duration_sec": round(time.time() - started, 1),
        "stopped_by": stopped_by,
        "timed_out": timed_out,
    }


def _stage_env(config: dict, stage: str) -> dict:
    env = _base_env(config)
    mapping = {
        "stage1": {"UWC_ENABLE": "1", "UWC_STAGE1": "1", "UWC_STAGE2": "0", "UWC_STAGE3": "0"},
        "stage2": {"UWC_ENABLE": "1", "UWC_STAGE1": "0", "UWC_STAGE2": "1", "UWC_STAGE3": "0"},
        "stage3_ptr": {
            "UWC_ENABLE": "1",
            "UWC_STAGE1": "0",
            "UWC_STAGE2": "0",
            "UWC_STAGE3": "1",
            "UWC_PTR_CHECK": "1",
            "UWC_DIFF_DUMP": "0",
        },
        "stage3_dummy": {
            "UWC_ENABLE": "1",
            "UWC_STAGE1": "0",
            "UWC_STAGE2": "0",
            "UWC_STAGE3": "1",
            "UWC_PTR_CHECK": "0",
            "UWC_DIFF_DUMP": "1",
            "UWC_DIFF_TAG": "dummy",
            "UWC_LOAD_FORMAT": "dummy",
        },
        "stage3_safetensors": {
            "UWC_ENABLE": "1",
            "UWC_STAGE1": "0",
            "UWC_STAGE2": "0",
            "UWC_STAGE3": "1",
            "UWC_PTR_CHECK": "0",
            "UWC_DIFF_DUMP": "1",
            "UWC_DIFF_TAG": "safetensors",
            "UWC_LOAD_FORMAT": "safetensors",
        },
    }
    env.update(mapping[stage])
    # ``patch.use_shm`` used to only rewrite the text anchor, which veRL 0.9 no
    # longer matches. Export the runtime switch too: veRL 0.9's ServerAdapter
    # reads VERL_FORCE_USE_SHM and passes the flag down to the sender/receiver.
    if (config.get("patch") or {}).get("use_shm", True):
        env.setdefault("VERL_FORCE_USE_SHM", "1")
    # Opt-in fault R5: force both stage-5 runs to record the same
    # load_format, proving the "A/B override ineffective" verdict.
    if _fault_stage5_same_format() and stage in ("stage3_dummy", "stage3_safetensors"):
        env["UWC_LOAD_FORMAT"] = "dummy"
    return env


def _fault_stage5_same_format() -> bool:
    """Opt-in fault R5: both stage-5 runs must load the same format."""
    return os.environ.get("UWC_FAULT_STAGE5_SAME_FORMAT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _force_dummy_load_format(command: str) -> str:
    """Rewrite a leading ``LOAD_FORMAT=...`` assignment in a shell command."""
    return re.sub(r"(^|\s)LOAD_FORMAT=\S+", r"\1LOAD_FORMAT=dummy", command)


def _timeout(config: dict, key: str) -> int:
    return int((config.get("timeouts") or {}).get(key, (config.get("timeouts") or {}).get("default", 900)))


def _stop_patterns(config: dict, key: str):
    return (config.get("stop_when") or {}).get(key, DEFAULT_STOP_WHEN.get(key, []))


def _forward_command(config: dict, backend: str) -> str | None:
    explicit = _command_for(config, f"forward_{backend}")
    if explicit:
        return explicit
    forward = config.get("forward") or {}
    if not forward.get("use_builtin", True):
        return None
    model_path = (config.get("model") or {}).get("hf_path")
    if not model_path:
        return None
    python = (config.get("verl") or {}).get("python") or sys.executable
    input_ids = ",".join(str(item) for item in forward.get("input_ids", [1, 2, 3, 4, 5]))
    device = forward.get("device")
    device_arg = f" --device {device}" if device else ""
    return (
        f"{python} -m verl.utils.uwc.offline.forward_verify --backend {backend} "
        f"--local_path {model_path} --input_ids {input_ids} --dtype {forward.get('dtype', 'fp32')} "
        f"--dump_dir {_dump_dir(config)}{device_arg}"
    )


def run_stages(config: dict, stages: list[str], dry_run: bool = False, reuse: bool = False) -> dict:
    dump_dir = _dump_dir(config)
    cwd = (config.get("verl") or {}).get("cwd") or (config.get("verl") or {}).get("path")
    results = {}
    runs_path = dump_dir / "runs.json"
    if runs_path.is_file():
        try:
            results = read_json(runs_path)
        except Exception:
            results = {}

    for stage in stages:
        if stage == "4":
            for backend in ("hf", "mcore"):
                key = f"forward_{backend}"
                command = _forward_command(config, backend)
                if not command:
                    results[key] = {"status": "skip", "reason": "forward command not configured"}
                    print(f"[{key}] SKIP: forward command not configured")
                    continue
                env = _base_env(config)
                logits = dump_dir / "stage1_bridge" / (
                    "G0_logits_gt.pt" if backend == "hf" else "G1_logits_mcore.pt"
                )
                if reuse and logits.is_file():
                    results[key] = {"status": "reused", "reason": f"existing {logits.name}"}
                    print(f"[{key}] REUSE: existing {logits.name} found")
                    continue
                if dry_run:
                    print(f"[dry-run] {key} env UWC_DUMP_DIR={env['UWC_DUMP_DIR']}")
                    print(f"[dry-run] {key} cmd: {command}")
                    results[key] = {"status": "dry-run", "command": command}
                    continue
                result = run_process(
                    command,
                    env,
                    cwd,
                    dump_dir / "logs" / f"{key}_{int(time.time())}.log",
                    _timeout(config, "forward"),
                    [],
                    dump_dir,
                )
                result["status"] = "ok" if result["returncode"] == 0 else "fail"
                results[key] = result
                print(f"[{key}] {result['status']} (rc={result['returncode']}, log={result['log']})")
            continue

        if stage in {"1", "2", "3"}:
            # One command per check: picking "1" must not also run checks 2/3.
            command_key, env_key = {
                "1": ("stage1", "stage1"),
                "2": ("stage2", "stage2"),
                "3": ("stage3_ptr", "stage3_ptr"),
            }[stage]
            entries = [(command_key, env_key)]
        elif stage == "5":
            entries = [
                ("stage3_dummy", "stage3_dummy"),
                ("stage3_safetensors", "stage3_safetensors"),
            ]
        else:
            results[stage] = {"status": "skip", "reason": f"unknown stage {stage}"}
            continue
        for command_key, env_key in entries:
            command = _command_for(config, command_key)
            if not command:
                results[command_key] = {"status": "skip", "reason": f"commands.{command_key} not configured"}
                print(f"[{command_key}] SKIP: commands.{command_key} not configured")
                continue
            if stage == "5" and _fault_stage5_same_format():
                command = _force_dummy_load_format(command)
                print(f"[{command_key}] FAULT: forcing LOAD_FORMAT=dummy in the command")
            stop_patterns = _stop_patterns(config, env_key)
            if reuse and _stop_files(dump_dir, stop_patterns):
                results[command_key] = {"status": "reused", "stopped_by": "existing_dump"}
                print(f"[{command_key}] REUSE: existing dump found")
                continue
            env = _stage_env(config, env_key)
            if dry_run:
                print(f"[dry-run] {command_key} env: " + " ".join(f"{k}={v}" for k, v in sorted(env.items()) if k.startswith("UWC_") or k == "VERL_FORCE_USE_SHM"))
                print(f"[dry-run] {command_key} cmd: {command}")
                results[command_key] = {"status": "dry-run", "command": command, "stop_when": stop_patterns}
                continue
            result = run_process(
                command,
                env,
                cwd,
                dump_dir / "logs" / f"{command_key}_{int(time.time())}.log",
                _timeout(config, env_key),
                stop_patterns,
                dump_dir,
            )
            result["status"] = "ok" if (result["stopped_by"] or result["returncode"] == 0) else "fail"
            results[command_key] = result
            print(
                f"[{command_key}] {result['status']} (rc={result['returncode']}, "
                f"stopped_by={result['stopped_by']}, log={result['log']})"
            )

    if not dry_run:
        write_json(dump_dir / "runs.json", results)
    return results


# ---------------------------------------------------------------------------
# compare / report
# ---------------------------------------------------------------------------


def compare(config: dict) -> dict:
    dump_dir = _dump_dir(config)
    compare_config = config.get("compare") or {}
    # Stage 5 dumps token ids only (veRL 0.9 async rollout).  Decode them with
    # the model tokenizer so the garbled-text matrix stays usable.
    tokenizer_path = compare_config.get("tokenizer") or (config.get("model") or {}).get("hf_path")
    summary = uwc_compare.compare_all(
        dump_dir=dump_dir,
        model_config=_model_config(config),
        stage1_atol=float(compare_config.get("stage1_atol", 1e-3)),
        stage1_rtol=float(compare_config.get("stage1_rtol", 0.0)),
        stage1_exact=bool(compare_config.get("stage1_exact", False)),
        stage2_exact=bool(compare_config.get("stage2_exact", True)),
        qkv_layout=compare_config.get("qkv_layout", "contiguous"),
        gate_up_layout=compare_config.get("gate_up_layout", "contiguous"),
        avg_cos=float(compare_config.get("avg_cos", 0.99)),
        min_cos=float(compare_config.get("min_cos", 0.95)),
        diff_threshold=float(compare_config.get("diff_threshold", 0.5)),
        tokenizer_path=tokenizer_path,
        trust_remote_code=bool(compare_config.get("trust_remote_code", True)),
    )
    return summary


def make_report(config: dict, demo: bool = False) -> dict:
    dump_dir = _dump_dir(config)
    extra = {
        "model": (config.get("model") or {}).get("hf_path"),
        "verl_path": (config.get("verl") or {}).get("path"),
    }
    verl_root = _verl_root(config)
    if verl_root and (verl_root / ".git").exists():
        try:
            extra["verl_commit"] = subprocess.check_output(
                ["git", "-C", str(verl_root), "rev-parse", "--short", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            pass
    report_config = config.get("report") or {}
    report = uwc_summary.build_report(
        dump_dir,
        title=report_config.get("title"),
        demo=demo,
        extra=extra,
    )
    out_dir = report_config.get("out_dir") or None
    paths = uwc_summary.write_report(report, out_dir=out_dir, dump_dir=dump_dir)
    return {"report": report, "paths": paths}


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------


def _demo_texts(scenario: str = "pass"):
    if scenario == "fail":
        return (
            "� � � 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8",
            "北京是中国的首都，也是一座历史悠久的城市。",
        )
    # Both outputs are normal (stage-3 pass) but differ in content, proving the
    # dummy path was actually replaced by safetensors weights.
    return (
        "北京是中国的首都。",
        "北京是中国的首都，也是一座历史悠久的城市。",
    )


def generate_demo_data(out_dir: Path, scenario: str = "pass") -> dict:
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage1_bridge").mkdir(exist_ok=True)
    (out_dir / "stage2_transfer").mkdir(exist_ok=True)
    (out_dir / "stage3_vllm").mkdir(exist_ok=True)

    generator = torch.Generator().manual_seed(20261113)
    shapes = {
        "model.embed_tokens.weight": (64, 16),
        "model.layers.0.self_attn.q_proj.weight": (16, 16),
        "model.layers.0.self_attn.k_proj.weight": (8, 16),
        "model.layers.0.self_attn.v_proj.weight": (8, 16),
        "model.layers.0.self_attn.o_proj.weight": (16, 16),
        "model.layers.0.mlp.gate_proj.weight": (32, 16),
        "model.layers.0.mlp.up_proj.weight": (32, 16),
        "model.layers.0.mlp.down_proj.weight": (16, 32),
        "model.norm.weight": (16,),
        "lm_head.weight": (64, 16),
    }
    p0 = {
        name: torch.randn(shape, generator=generator, dtype=torch.float32) * 0.02
        for name, shape in shapes.items()
    }
    noise = torch.Generator().manual_seed(7)
    p2 = {name: tensor + torch.randn(tensor.shape, generator=noise) * 1e-6 for name, tensor in p0.items()}
    sender = [(name, tensor.clone()) for name, tensor in p2.items()]
    receiver = [(name, tensor.clone()) for name, tensor in p2.items()]

    torch.save(p0, out_dir / "stage1_bridge" / "P0_hf_original_step0_rank0.pt")
    torch.save(p2, out_dir / "stage1_bridge" / "P2_hf_exported_step0_rank0.pt")
    torch.save(sender, out_dir / "stage2_transfer" / "P2_sender_step0_rank0.pt")
    torch.save(receiver, out_dir / "stage2_transfer" / "P5_receiver_step0_rank0.pt")

    logits = torch.randn(1, 6, 48, generator=generator) * 0.5
    logits_g1 = logits + torch.randn(logits.shape, generator=torch.Generator().manual_seed(11)) * 1e-4
    torch.save(
        {"backend": "hf", "input_ids": [1, 2, 3, 4, 5, 6], "logits": logits},
        out_dir / "stage1_bridge" / "G0_logits_gt.pt",
    )
    torch.save(
        {"backend": "mcore", "input_ids": [1, 2, 3, 4, 5, 6], "logits": logits_g1},
        out_dir / "stage1_bridge" / "G1_logits_mcore.pt",
    )

    ptr_report = {
        "status": "pass",
        "step": 0,
        "rank": 0,
        "checked_layers": len(shapes),
        "synced_layers": len(shapes),
        "not_synced_layers": 0,
        "missing_layers": 0,
        "not_synced": [],
        "details": [
            {
                "name": name,
                "ptr_before": 100000 + index * 4096,
                "ptr_after": 200000 + index * 4096,
                "synced": True,
            }
            for index, name in enumerate(shapes)
        ],
    }
    write_json(out_dir / "stage3_vllm" / "ptr_before_after_step0_rank0.json", ptr_report)

    dummy_text, normal = _demo_texts(scenario)
    write_json(
        out_dir / "stage3_vllm" / "dummy_rl_first_step_output.json",
        {"tag": "dummy", "step": 0, "generated_at": now_iso(), "texts": [dummy_text], "payload": {}},
    )
    write_json(
        out_dir / "stage3_vllm" / "safetensors_rl_first_step_output.json",
        {"tag": "safetensors", "step": 0, "generated_at": now_iso(), "texts": [normal], "payload": {}},
    )

    write_json(
        out_dir / "model_config.json",
        {
            "hidden_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
        },
    )
    return {
        "keys": len(shapes),
        "p0": str(out_dir / "stage1_bridge" / "P0_hf_original_step0_rank0.pt"),
        "p2": str(out_dir / "stage1_bridge" / "P2_hf_exported_step0_rank0.pt"),
        "sender": str(out_dir / "stage2_transfer" / "P2_sender_step0_rank0.pt"),
        "receiver": str(out_dir / "stage2_transfer" / "P5_receiver_step0_rank0.pt"),
    }


def demo(out_dir: str | Path, scenario: str = "pass") -> int:
    out_dir = Path(out_dir).expanduser().resolve()
    print(f"generating synthetic UWC data in {out_dir} (scenario={scenario})")
    info = generate_demo_data(out_dir, scenario=scenario)
    print(json.dumps(info, indent=2, ensure_ascii=False))
    config = _default_config()
    config["dump_dir"] = str(out_dir)
    config["model"]["config_json"] = str(out_dir / "model_config.json")
    compare(config)
    result = make_report(config, demo=True)
    overall = result["report"]["overall"]
    print(json.dumps({"overall": overall, "paths": result["paths"]}, indent=2, ensure_ascii=False))
    return 0 if overall["status"] == ("pass" if scenario == "pass" else "fail") else 1


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def doctor(config: dict) -> int:
    checks = []

    def check(name, ok, detail="", critical=True):
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail), "critical": critical})

    verl_root = _verl_root(config)
    check("verl.path", verl_root is not None, verl_root or "not set / not found")
    if verl_root:
        check("verl package", (verl_root / "verl").is_dir(), verl_root / "verl")
        found = [relative for relative in CORE_TARGETS if (verl_root / relative).is_file()]
        check("core targets", len(found) == len(CORE_TARGETS), f"{len(found)}/{len(CORE_TARGETS)} found", critical=False)
        runtime = verl_root / "verl" / "utils" / "uwc" / RUNTIME_MARKER
        check("runtime installed", runtime.is_file(), runtime, critical=False)
        manifest = verl_root / MANIFEST_NAME
        check("manifest", manifest.is_file(), manifest, critical=False)

    dump_dir = _dump_dir(config)
    check("dump_dir writable", os.access(dump_dir, os.W_OK), dump_dir)

    model = config.get("model") or {}
    check("model.hf_path", bool(model.get("hf_path")) and Path(model["hf_path"]).exists(), model.get("hf_path") or "not set", critical=False)
    check("model config.json", bool(_model_config(config)), _model_config(config) or "not set", critical=False)

    try:
        import torch

        check("torch", True, torch.__version__)
        npu = getattr(torch, "npu", None)
        check("NPU available", bool(npu is not None and npu.is_available()), "torch.npu", critical=False)
    except Exception as exc:
        check("torch", False, exc, critical=False)

    commands = config.get("commands") or {}
    for key in ("stage1", "stage2", "stage3_ptr", "stage3_dummy", "stage3_safetensors"):
        check(f"commands.{key}", bool(commands.get(key)), commands.get(key) or "not set", critical=False)

    for item in checks:
        mark = "OK  " if item["ok"] else ("FAIL" if item["critical"] else "WARN")
        print(f"[{mark}] {item['name']}: {item['detail']}")
    critical_failures = [item for item in checks if item["critical"] and not item["ok"]]
    return 0 if not critical_failures else 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_stages(value: str) -> list[str]:
    if value in {"all", None, ""}:
        return ["1", "2", "3", "4", "5"]
    stages = []
    for token in str(value).replace(";", ",").split(","):
        token = token.strip()
        if token:
            stages.append(token)
    return stages


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="uwc-verify",
        description="veRL 训推权重一致性（UWC）5 项自动验证",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("doctor", "prepare", "run", "compare", "report", "all"):
        command_parser = sub.add_parser(name)
        command_parser.add_argument("--config", default=None, help="uwc.yaml path")

    prepare_parser = sub.choices["prepare"]
    prepare_parser.add_argument("--dry-run", action="store_true")
    prepare_parser.add_argument("--restore", action="store_true")
    prepare_parser.add_argument("--force", action="store_true")

    run_parser = sub.choices["run"]
    run_parser.add_argument("--stage", default="all", help="1,2,3,4,5 / all")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--reuse", action="store_true", help="已有 dump 时跳过运行")

    all_parser = sub.choices["all"]
    all_parser.add_argument("--stage", default="all")
    all_parser.add_argument("--dry-run", action="store_true")
    all_parser.add_argument("--reuse", action="store_true")
    all_parser.add_argument("--skip-prepare", action="store_true")

    demo_parser = sub.add_parser("demo")
    demo_parser.add_argument("--out", default="/tmp/uwc_demo")
    demo_parser.add_argument("--scenario", choices=["pass", "fail"], default="pass")

    args = parser.parse_args(argv)

    if args.command == "demo":
        return demo(args.out, scenario=args.scenario)

    config, config_path = load_config(args.config)
    if config_path:
        print(f"config: {config_path}")

    if args.command == "doctor":
        return doctor(config)
    if args.command == "prepare":
        return prepare(config, dry_run=args.dry_run, restore=args.restore, force=args.force)
    if args.command == "run":
        results = run_stages(config, _parse_stages(args.stage), dry_run=args.dry_run, reuse=args.reuse)
        print(json.dumps({key: value.get("status", value) for key, value in results.items()}, indent=2, ensure_ascii=False))
        return 0
    if args.command == "compare":
        summary = compare(config)
        print(json.dumps(
            {"stage1": summary["stage1"]["status"], "stage2": summary["stage2"]["status"], "stage3": summary["stage3"]["status"]},
            indent=2,
            ensure_ascii=False,
        ))
        return 0
    if args.command == "report":
        result = make_report(config)
        print(json.dumps({"overall": result["report"]["overall"], "paths": result["paths"]}, indent=2, ensure_ascii=False))
        return 0 if result["report"]["overall"]["status"] in {"pass", "partial"} else 1
    if args.command == "all":
        if not args.skip_prepare:
            code = prepare(config)
            if code != 0:
                return code
        run_stages(config, _parse_stages(args.stage), dry_run=args.dry_run, reuse=args.reuse)
        if args.dry_run:
            print("[dry-run] skipping compare/report")
            return 0
        compare(config)
        result = make_report(config)
        print(json.dumps({"overall": result["report"]["overall"], "paths": result["paths"]}, indent=2, ensure_ascii=False))
        return 0 if result["report"]["overall"]["status"] in {"pass", "partial"} else 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
