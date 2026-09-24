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

import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "skills/quantizer/msmodelslim-anti-outlier-adapt/scripts/validate_anti_outlier_interfaces.py"
)
SPEC = importlib.util.spec_from_file_location("validate_anti_outlier_interfaces", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FlexInterface:
    pass


class IterInterface:
    pass


class Mapping:
    source = "model.layers.0.input_layernorm"
    targets = ["model.layers.0.self_attn.q_proj"]


class AdapterConfig:
    subgraph_type = "norm-linear"
    mapping = Mapping()


class FlexAdapter(FlexInterface):
    def get_adapter_config_for_subgraph(self):
        return [AdapterConfig()]


def test_shared_flex_interface_is_checked_once(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    results = MODULE.validate_installed_adapter(
        FlexAdapter(),
        {"FlexSmoothQuantInterface": FlexInterface},
        ["flex_smooth_quant", "flex_awq_ssz"],
        model_path,
        tmp_path / "output",
        installed_package_path="/installed/msmodelslim/__init__.py",
    )
    assert list(results) == ["FlexSmoothQuantInterface"]
    assert results["FlexSmoothQuantInterface"]["status"] == "PASS"
    saved = json.loads(Path(results["FlexSmoothQuantInterface"]["path"]).read_text(encoding="utf-8"))
    assert saved["checkpoint_identity"] == MODULE.checkpoint_identity(model_path)
    assert saved["mapping_count"] == 1


def test_missing_interface_fails_before_processor(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    results = MODULE.validate_installed_adapter(
        FlexAdapter(),
        {"IterSmoothInterface": IterInterface},
        ["iter_smooth"],
        model_path,
        tmp_path / "output",
        installed_package_path="/installed/msmodelslim/__init__.py",
    )
    assert results["IterSmoothInterface"]["status"] == "FAIL"
    assert "does not implement" in results["IterSmoothInterface"]["errors"][0]


def test_source_install_check_rejects_direct_site_packages_edits(tmp_path):
    root = tmp_path / "source"
    source = root / "msmodelslim/model/demo/adapter.py"
    source.parent.mkdir(parents=True)
    source.write_text("source version", encoding="utf-8")
    installed = tmp_path / "site-packages/msmodelslim/model/demo/adapter.py"
    installed.parent.mkdir(parents=True)
    installed.write_text("modified installed version", encoding="utf-8")
    result = MODULE.verify_source_install(root, "msmodelslim.model.demo.adapter", installed)
    assert result["source_matches_installed"] is False
    assert "reinstall" in result["errors"][0]

    installed.write_text("source version", encoding="utf-8")
    assert (
        MODULE.verify_source_install(root, "msmodelslim.model.demo.adapter", installed)["source_matches_installed"]
        is True
    )
