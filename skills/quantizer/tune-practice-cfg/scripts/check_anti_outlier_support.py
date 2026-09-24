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
"""List anti-outlier algorithms exposed by the installed model Adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


ALGORITHM_INTERFACES = {
    "quarot": "QuaRotInterface",
    "flex_smooth_quant": "FlexSmoothQuantInterface",
    "flex_awq_ssz": "FlexSmoothQuantInterface",
    "iter_smooth": "IterSmoothInterface",
    "smooth_quant": "SmoothQuantInterface",
    "oasq": "OASQInterface",
}


def list_supported_algorithms(
    adapter: Any, interface_types: Mapping[str, type]
) -> dict[str, Any]:
    """Return selectable names using the installed Adapter's interface contract."""
    supported = []
    unsupported = []
    for algorithm, interface_name in ALGORITHM_INTERFACES.items():
        interface = interface_types.get(interface_name)
        (supported if isinstance(interface, type) and isinstance(adapter, interface) else unsupported).append(
            algorithm
        )
    return {"ok": True, "supported_algorithms": supported, "unsupported_algorithms": unsupported}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    from msmodelslim.model import interface_hub
    from msmodelslim.model.plugin_factory.plugin_model_factory import PluginModelFactory

    adapter = PluginModelFactory().create(args.model_type, args.model_path, args.trust_remote_code)
    interface_types = {
        name: getattr(interface_hub, name, None)
        for name in set(ALGORITHM_INTERFACES.values())
    }
    result = list_supported_algorithms(adapter, interface_types)
    result["adapter_class"] = f"{type(adapter).__module__}:{type(adapter).__qualname__}"
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
