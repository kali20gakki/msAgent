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
"""Validate installed Adapter interfaces before anti-outlier processor gates."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Mapping


ALGORITHM_INTERFACES = {
    "quarot": "QuaRotInterface",
    "flex_smooth_quant": "FlexSmoothQuantInterface",
    "flex_awq_ssz": "FlexSmoothQuantInterface",
    "iter_smooth": "IterSmoothInterface",
}
SMOOTH_INTERFACES = {"FlexSmoothQuantInterface", "IterSmoothInterface"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(model_path: str | Path) -> str:
    """Use the same checkpoint identity format as the anti-outlier gate."""
    resolved = Path(model_path).expanduser().resolve()
    config_path = resolved / "config.json"
    files = [
        {"path": str(item.relative_to(resolved)), "sha256": _sha256_file(item)}
        for item in sorted(resolved.rglob("*"))
        if item.is_file() and item.suffix in (".safetensors", ".bin", ".pt", ".pth", ".ckpt")
    ]
    payload = {
        "path": str(resolved),
        "config_sha256": _sha256_file(config_path) if config_path.is_file() else "missing",
        "weight_files": files,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def validate_interface(adapter: Any, interface: type, interface_name: str) -> dict[str, Any]:
    """Check an already installed interface without running a processor."""
    errors: list[str] = []
    mapping_count: int | None = None
    if not isinstance(adapter, interface):
        errors.append(f"Adapter does not implement {interface_name}")
    elif interface_name in SMOOTH_INTERFACES:
        getter = getattr(adapter, "get_adapter_config_for_subgraph", None)
        if not callable(getter):
            errors.append("get_adapter_config_for_subgraph is missing")
        else:
            try:
                mappings = getter()
                if not isinstance(mappings, list) or not mappings:
                    errors.append("get_adapter_config_for_subgraph must return a nonempty list")
                else:
                    mapping_count = len(mappings)
                    for index, item in enumerate(mappings):
                        mapping = getattr(item, "mapping", None)
                        source = getattr(mapping, "source", None)
                        targets = getattr(mapping, "targets", None)
                        if not getattr(item, "subgraph_type", None) or not isinstance(targets, list) or not targets:
                            errors.append(f"mapping {index} has no subgraph type or targets")
                        if source is not None and not isinstance(source, str):
                            errors.append(f"mapping {index} has invalid source")
            except Exception as exc:
                errors.append(f"mapping inspection failed: {type(exc).__name__}: {exc}")
    else:
        for method_name in ("get_ln_fuse_map", "get_bake_names", "get_rotate_map"):
            method = getattr(adapter, method_name, None)
            if not callable(method):
                errors.append(f"{method_name} is missing")
            elif getattr(method, "__func__", method) is getattr(interface, method_name, None):
                errors.append(f"{method_name} still uses the interface stub")
            elif method_name == "get_rotate_map" and len(inspect.signature(method).parameters) != 1:
                errors.append("get_rotate_map must accept block_size")
    return {"status": "PASS" if not errors else "FAIL", "passed": not errors,
            "mapping_count": mapping_count, "errors": errors}


def verify_source_install(source_root: Path, module_name: str, installed_file: Path) -> dict[str, Any]:
    """Prove the imported Adapter matches its source-tree implementation."""
    root = source_root.expanduser().resolve()
    installed = installed_file.expanduser().resolve()
    errors = []
    if not module_name.startswith("msmodelslim."):
        errors.append("Adapter module is outside msmodelslim")
    if any(part in ("site-packages", "dist-packages") for part in root.parts):
        errors.append("source root must not be a Python installation directory")
    source = root.joinpath(*module_name.split(".")).with_suffix(".py")
    if not source.is_file():
        errors.append(f"Adapter source file is missing: {source}")
    if not installed.is_file():
        errors.append(f"installed Adapter file is missing: {installed}")
    if not errors and _sha256_file(source) != _sha256_file(installed):
        errors.append("installed Adapter differs from the source-tree file; reinstall msModelSlim")
    return {"source_path": str(source), "installed_adapter_path": str(installed),
            "source_matches_installed": not errors, "errors": errors}


def validate_installed_adapter(
    adapter: Any,
    interface_types: Mapping[str, type],
    algorithms: list[str],
    model_path: str | Path,
    output_dir: str | Path,
    *,
    installed_package_path: str,
    source_check: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Write one checkpoint-bound result per distinct interface."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    identity = checkpoint_identity(model_path)
    results = {}
    for name in dict.fromkeys(ALGORITHM_INTERFACES[algorithm] for algorithm in algorithms):
        result = validate_interface(adapter, interface_types[name], name)
        if source_check is not None and not source_check["source_matches_installed"]:
            result["errors"].extend(source_check["errors"])
            result["status"] = "FAIL"
            result["passed"] = False
        result.update({
            "interface": name,
            "checkpoint_identity": identity,
            "adapter_class": f"{type(adapter).__module__}:{type(adapter).__qualname__}",
            "installed_package_path": installed_package_path,
            "source_check": dict(source_check) if source_check is not None else None,
        })
        path = destination / f"interface_validation.{name}.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        results[name] = {"path": str(path.resolve()), **result}
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="msModelSlim checkout root used for installation")
    parser.add_argument("--algorithm", action="append", choices=tuple(ALGORITHM_INTERFACES))
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    import msmodelslim
    from msmodelslim.model.interface_hub import (
        FlexSmoothQuantInterface, IterSmoothInterface, QuaRotInterface,
    )
    from msmodelslim.model.plugin_factory.plugin_model_factory import PluginModelFactory

    adapter = PluginModelFactory().create(args.model_type, args.model_path, args.trust_remote_code)
    interfaces = {
        "QuaRotInterface": QuaRotInterface,
        "FlexSmoothQuantInterface": FlexSmoothQuantInterface,
        "IterSmoothInterface": IterSmoothInterface,
    }
    results = validate_installed_adapter(
        adapter, interfaces, args.algorithm or list(ALGORITHM_INTERFACES),
        args.model_path, args.output_dir, installed_package_path=str(Path(msmodelslim.__file__).resolve()),
        source_check=verify_source_install(
            args.source_root, type(adapter).__module__, Path(inspect.getfile(type(adapter)))
        ),
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0 if all(result["status"] == "PASS" for result in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
