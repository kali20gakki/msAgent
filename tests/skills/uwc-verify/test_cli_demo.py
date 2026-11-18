"""End-to-end CLI demo test: synthetic data -> 5 comparisons -> report."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
SKILL_ROOT = REPO_ROOT / "skills" / "accuracy" / "uwc-verify"
sys.path.insert(0, str(SKILL_ROOT / "runtime"))
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

try:
    import torch  # noqa: F401
except Exception as exc:  # pragma: no cover
    torch = None
    TORCH_ERROR = exc
else:
    TORCH_ERROR = None


def _load_uwc_verify():
    path = SKILL_ROOT / "scripts" / "uwc_verify.py"
    spec = importlib.util.spec_from_file_location("uwc_verify_demo_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(torch is None, f"torch unavailable: {TORCH_ERROR}")
class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.uwc_verify = _load_uwc_verify()

    def test_pass_scenario(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = self.uwc_verify.demo(tmp, scenario="pass")
            self.assertEqual(rc, 0)
            report = json.loads((Path(tmp) / "report.json").read_text())
            self.assertEqual(report["overall"]["status"], "pass")
            self.assertEqual(report["overall"]["counts"]["pass"], 5)
            for name in ("report.md", "report.html"):
                self.assertTrue((Path(tmp) / name).is_file())

    def test_fail_scenario_flags_only_diff_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = self.uwc_verify.demo(tmp, scenario="fail")
            self.assertEqual(rc, 0)
            report = json.loads((Path(tmp) / "report.json").read_text())
            self.assertEqual(report["overall"]["status"], "fail")
            failed = [item["id"] for item in report["items"] if item["status"] == "fail"]
            self.assertEqual(failed, ["dummy_vs_safetensors"])


class RunProcessTests(unittest.TestCase):
    """A dump from an earlier run must not stop the current run immediately."""

    @classmethod
    def setUpClass(cls):
        cls.uwc_verify = _load_uwc_verify()

    def test_stale_stop_file_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            dump = Path(tmp)
            (dump / "stage3_vllm").mkdir()
            stale = dump / "stage3_vllm" / "dummy_rl_first_step_output.json"
            stale.write_text("{}")
            result = self.uwc_verify.run_process(
                "sleep 3",
                dict(**__import__("os").environ),
                tmp,
                dump / "log.txt",
                timeout=30,
                stop_patterns=["stage3_vllm/dummy_rl_first_step_output.json"],
                dump_dir=dump,
            )
            self.assertIsNone(result["stopped_by"])
            self.assertGreaterEqual(result["duration_sec"], 2.5)

    def test_fresh_stop_file_stops_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            dump = Path(tmp)
            (dump / "stage3_vllm").mkdir()
            result = self.uwc_verify.run_process(
                "sleep 2; : > stage3_vllm/dummy_rl_first_step_output.json; sleep 30",
                dict(**__import__("os").environ),
                tmp,
                dump / "log.txt",
                timeout=60,
                stop_patterns=["stage3_vllm/dummy_rl_first_step_output.json"],
                dump_dir=dump,
            )
            self.assertEqual(result["stopped_by"], "dump_found")
            self.assertLess(result["duration_sec"], 15)

    def test_leftover_stage_processes_are_cleaned_up(self):
        """Ray actors outlive the killed process group; the next stage must not OOM."""
        import subprocess
        import time

        marker = {
            "UWC_DUMP_DIR": tempfile.mkdtemp(prefix="uwc_leftover_"),
            "UWC_STAGE1": "1",
        }
        child = subprocess.Popen(  # pylint: disable=consider-using-with
            ["sleep", "120"],
            env=dict(os.environ, **marker),
            start_new_session=True,
        )
        try:
            time.sleep(0.2)
            cleaned = self.uwc_verify._cleanup_stage_processes(marker)
            self.assertGreaterEqual(cleaned, 1)
            child.wait(timeout=20)
            self.assertIsNotNone(child.poll())
        finally:
            if child.poll() is None:
                child.kill()


class Stage5FaultTests(unittest.TestCase):
    """Opt-in fault R5: both runs must load dummy so the A/B check fails."""

    @classmethod
    def setUpClass(cls):
        cls.uwc_verify = _load_uwc_verify()

    def setUp(self):
        self._saved = os.environ.pop("UWC_FAULT_STAGE5_SAME_FORMAT", None)

    def tearDown(self):
        os.environ.pop("UWC_FAULT_STAGE5_SAME_FORMAT", None)
        if self._saved is not None:
            os.environ["UWC_FAULT_STAGE5_SAME_FORMAT"] = self._saved

    def test_command_rewrite(self):
        command = "LOAD_FORMAT=safetensors\nbash run.sh --x 1"
        self.assertEqual(
            self.uwc_verify._force_dummy_load_format(command),
            "LOAD_FORMAT=dummy\nbash run.sh --x 1",
        )
        self.assertEqual(self.uwc_verify._force_dummy_load_format("bash run.sh"), "bash run.sh")

    def test_env_override_for_both_stage5_runs(self):
        os.environ["UWC_FAULT_STAGE5_SAME_FORMAT"] = "1"
        self.assertTrue(self.uwc_verify._fault_stage5_same_format())
        with tempfile.TemporaryDirectory() as tmp:
            config = {"dump_dir": tmp}
            for stage in ("stage3_dummy", "stage3_safetensors"):
                env = self.uwc_verify._stage_env(config, stage)
                self.assertEqual(env["UWC_LOAD_FORMAT"], "dummy", stage)

    def test_no_override_by_default(self):
        self.assertFalse(self.uwc_verify._fault_stage5_same_format())
        with tempfile.TemporaryDirectory() as tmp:
            env = self.uwc_verify._stage_env({"dump_dir": tmp}, "stage3_safetensors")
            self.assertEqual(env["UWC_LOAD_FORMAT"], "safetensors")


class Stage5RecordTests(unittest.TestCase):
    """The record's load_format must come from the stage mapping.

    A shell that exports ``LOAD_FORMAT=dummy`` (the run script's default) used
    to leak into the record because the collector fell back to that variable.
    """

    @classmethod
    def setUpClass(cls):
        cls.uwc_verify = _load_uwc_verify()
        from uwc.collectors.stage3_vllm_collector import _DUMPED_TAGS, dump_generation_output

        cls.dump_generation_output = staticmethod(dump_generation_output)
        cls.dumped_tags = _DUMPED_TAGS

    def setUp(self):
        self.dumped_tags.clear()

    def test_safetensors_stage_ignores_stale_shell_load_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"LOAD_FORMAT": "dummy"}, clear=False):
                env = self.uwc_verify._stage_env({"dump_dir": tmp}, "stage3_safetensors")
            self.assertEqual(env["UWC_LOAD_FORMAT"], "safetensors")
            with mock.patch.dict(os.environ, env, clear=True):
                path = self.dump_generation_output({"responses": [[1, 2, 3]]}, dump_dir=tmp)
            record = json.loads(Path(path).read_text(encoding="utf-8"))
            self.assertEqual(record["tag"], "safetensors")
            self.assertEqual(record["load_format"], "safetensors")


if __name__ == "__main__":
    unittest.main(verbosity=2)
