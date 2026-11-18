"""Offline comparator / report tests for uwc-verify (run inside a torch container)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[3]
SKILL_ROOT = REPO_ROOT / "skills" / "accuracy" / "uwc-verify"
sys.path.insert(0, str(SKILL_ROOT / "runtime"))

try:
    import torch
except Exception as exc:  # pragma: no cover
    torch = None
    TORCH_ERROR = exc
else:
    TORCH_ERROR = None

from uwc import comparator, logits_comparator, name_normalizer  # noqa: E402
from uwc.offline import stage3_diff_test, uwc_compare, uwc_summary  # noqa: E402


@unittest.skipIf(torch is None, f"torch unavailable: {TORCH_ERROR}")
class ComparatorTests(unittest.TestCase):
    def test_atol_pass_and_fail(self):
        a = torch.zeros(4)
        b = torch.full((4,), 5e-4)
        self.assertTrue(comparator.compare_tensors(a, b, atol=1e-3)["passed"])
        self.assertFalse(comparator.compare_tensors(a, b, atol=1e-5)["passed"])

    def test_exact(self):
        a = torch.tensor([1.0, 2.0])
        b = a.clone()
        self.assertTrue(comparator.compare_tensors(a, b, exact=True)["passed"])
        b[0] = 1.0000001
        self.assertFalse(comparator.compare_tensors(a, b, exact=True)["passed"])

    def test_shape_mismatch(self):
        result = comparator.compare_tensors(torch.zeros(2), torch.zeros(3))
        self.assertFalse(result["passed"])
        self.assertIn("shape mismatch", result["reason"])

    def test_missing_and_extra_keys(self):
        result = comparator.compare_weight_dicts({"a": torch.zeros(2)}, {"b": torch.zeros(2)})
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["missing_count"], 1)
        self.assertEqual(result["extra_count"], 1)

    def test_nan_handling(self):
        a = torch.tensor([float("nan"), 1.0])
        b = torch.tensor([float("nan"), 1.0])
        self.assertTrue(comparator.compare_tensors(a, b, exact=True)["passed"])
        c = torch.tensor([0.0, 1.0])
        self.assertFalse(comparator.compare_tensors(a, c, exact=True)["passed"])


@unittest.skipIf(torch is None, f"torch unavailable: {TORCH_ERROR}")
class NameNormalizerTests(unittest.TestCase):
    def test_prefix_strip(self):
        self.assertEqual(name_normalizer.canonical_name("model.model.layers.0.w"), "model.layers.0.w")
        self.assertEqual(name_normalizer.canonical_name("module.model.layers.0.w"), "model.layers.0.w")
        self.assertEqual(name_normalizer.canonical_name("model.layers.0.w"), "model.layers.0.w")

    def test_split_qkv_contiguous(self):
        config = {"hidden_size": 8, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4}
        q = torch.arange(8, dtype=torch.float32)
        k = torch.arange(4, dtype=torch.float32) + 100
        v = torch.arange(4, dtype=torch.float32) + 200
        fused = torch.cat([q, k, v])
        normalised, meta = name_normalizer.normalize_weights(
            {"model.layers.0.self_attn.qkv_proj.weight": fused}, config
        )
        self.assertIn("model.layers.0.self_attn.q_proj.weight", normalised)
        self.assertTrue(torch.equal(normalised["model.layers.0.self_attn.q_proj.weight"], q))
        self.assertTrue(torch.equal(normalised["model.layers.0.self_attn.k_proj.weight"], k))
        self.assertTrue(torch.equal(normalised["model.layers.0.self_attn.v_proj.weight"], v))
        self.assertEqual(meta["unresolved_combined"], [])

    def test_split_gate_up(self):
        gate = torch.arange(6, dtype=torch.float32)
        up = gate + 100
        normalised, _ = name_normalizer.normalize_weights(
            {"model.layers.0.mlp.gate_up_proj.weight": torch.cat([gate, up])}, {}
        )
        self.assertTrue(torch.equal(normalised["model.layers.0.mlp.gate_proj.weight"], gate))
        self.assertTrue(torch.equal(normalised["model.layers.0.mlp.up_proj.weight"], up))

    def test_unresolved_without_config(self):
        normalised, meta = name_normalizer.normalize_weights(
            {"model.layers.0.self_attn.qkv_proj.weight": torch.zeros(12)}, {}
        )
        self.assertIn("model.layers.0.self_attn.qkv_proj.weight", normalised)
        self.assertEqual(len(meta["unresolved_combined"]), 1)


@unittest.skipIf(torch is None, f"torch unavailable: {TORCH_ERROR}")
class LogitsTests(unittest.TestCase):
    def test_identical_logits(self):
        a = torch.randn(1, 4, 16, generator=torch.Generator().manual_seed(1))
        report = logits_comparator.compare_logits(a, a.clone())
        self.assertEqual(report["status"], "pass")
        self.assertAlmostEqual(report["min_cos"], 1.0, places=5)

    def test_orthogonal_logits(self):
        a = torch.zeros(1, 4, 16)
        a[0, :, 0] = 1.0
        b = torch.zeros(1, 4, 16)
        b[0, :, 1] = 1.0
        report = logits_comparator.compare_logits(a, b)
        self.assertEqual(report["status"], "fail")

    def test_shape_mismatch(self):
        report = logits_comparator.compare_logits(torch.zeros(1, 4, 16), torch.zeros(1, 4, 8))
        self.assertEqual(report["status"], "fail")
        self.assertIn("shape mismatch", report["reason"])


class BridgeApiTests(unittest.TestCase):
    """stage-4 must work with both Megatron-Bridge generations."""

    def test_new_megatron_bridge_api(self):
        from uwc.offline import forward_verify

        calls = {}

        class FakeBridge:
            @classmethod
            def from_hf_pretrained(cls, path, **kwargs):
                calls["from"] = (path, kwargs)
                return cls()

            def to_megatron_model(self, **kwargs):
                calls["build"] = kwargs
                return ["chunk0"]

            def load_hf_weights(self, model, path):
                calls["load"] = (model, path)

        bridge, model = forward_verify._bridge_build_and_load(FakeBridge, "/hf/model", "bf16", True)
        self.assertEqual(calls["from"], ("/hf/model", {"trust_remote_code": True}))
        self.assertEqual(calls["build"]["load_weights"], False)
        self.assertEqual(calls["build"]["hf_path"], "/hf/model")
        self.assertTrue(calls["build"]["bf16"])
        self.assertFalse(calls["build"]["fp16"])
        self.assertEqual(calls["load"], (["chunk0"], "/hf/model"))
        self.assertEqual(model, ["chunk0"])

    def test_legacy_mbridge_api(self):
        from uwc.offline import forward_verify

        calls = {}

        class FakeBridge:
            @classmethod
            def from_hf_pretrained(cls, path, **kwargs):
                calls["from"] = (path, kwargs)
                return cls()

            def get_model(self, **kwargs):
                calls["build"] = kwargs
                return "model"

            def load_weights(self, model, path):
                calls["load"] = (model, path)

        _bridge, model = forward_verify._bridge_build_and_load(FakeBridge, "/hf/model", "fp32", False)
        self.assertEqual(calls["build"], {"fp16": False, "bf16": False})
        self.assertEqual(calls["load"], ("model", "/hf/model"))
        self.assertEqual(model, "model")

    def test_load_bridge_prefers_megatron_bridge(self):
        from uwc.offline import forward_verify

        try:
            import megatron.bridge  # noqa: F401
        except ImportError:
            self.skipTest("megatron.bridge not installed")
        self.assertIs(forward_verify._load_bridge(), megatron.bridge.AutoBridge)


class DiffTestTests(unittest.TestCase):
    def _record(self, texts, load_format=None):
        return {"tag": "t", "step": 0, "texts": texts, "payload": {}, "load_format": load_format}

    def test_dummy_garbled_safe_normal_fails(self):
        report = stage3_diff_test.analyze(
            self._record(["� � � 8 8 8 8 8 8 8 8 8 8 8 8"]),
            self._record(["北京是中国的首都，也是一座历史悠久的城市。"]),
        )
        self.assertEqual(report["status"], "fail")
        self.assertTrue(report["dummy_garbled"])
        self.assertFalse(report["safetensors_garbled"])
        self.assertFalse(report["identical_outputs"])

    def test_long_normal_english_is_not_garbled(self):
        """Long natural text has a small alphabet; the unique-ratio rule must not fire."""
        text = (
            "<think>\nOkay, so I need to find all natural numbers a > 1 such that every "
            "prime divisor of a^6 - 1 also divides at least one of a^3 - 1 or a^2 - 1. "
            "Hmm, let me start by factoring a^6 - 1 = (a^3 - 1)(a^3 + 1) and then check "
            "each prime divisor for the two smaller expressions. This takes a while but "
            "the argument is straightforward once the factorisation is written down.\n"
        ) * 4
        self.assertLess(stage3_diff_test.garbled_score(text), 0.5)
        report = stage3_diff_test.analyze(
            self._record([text], load_format="dummy"),
            self._record([text], load_format="safetensors"),
        )
        self.assertEqual(report["status"], "pass", report)

    def test_both_normal_passes(self):
        report = stage3_diff_test.analyze(
            self._record(["北京是中国的首都。"]),
            self._record(["北京是中国的首都，也是一座历史悠久的城市。"]),
        )
        self.assertEqual(report["status"], "pass")
        self.assertFalse(report["identical_outputs"])

    def test_both_garbled_fails(self):
        report = stage3_diff_test.analyze(
            self._record(["� � � 8 8 8 8 8 8 8 8 8 8"]),
            self._record(["� � � 9 9 9 9 9 9 9 9 9 9"]),
        )
        self.assertEqual(report["status"], "fail")

    def test_identical_normal_outputs_pass_when_formats_differ(self):
        """Same weights after sync => deterministic decode matches; that is healthy."""
        report = stage3_diff_test.analyze(
            self._record(["北京是中国的首都。"], load_format="dummy"),
            self._record(["北京是中国的首都。"], load_format="safetensors"),
        )
        self.assertEqual(report["status"], "pass", report)
        self.assertTrue(report["identical_outputs"])

    def test_identical_outputs_same_format_fail(self):
        """Both runs loading the same format means the override never took effect."""
        report = stage3_diff_test.analyze(
            self._record(["北京是中国的首都。"], load_format="dummy"),
            self._record(["北京是中国的首都。"], load_format="dummy"),
        )
        self.assertEqual(report["status"], "fail", report)

    def test_different_outputs_same_format_fail(self):
        """Differing outputs with the same format only prove sampling is random."""
        report = stage3_diff_test.analyze(
            self._record(["北京是中国的首都。"], load_format="dummy"),
            self._record(["上海是中国的城市。"], load_format="dummy"),
        )
        self.assertEqual(report["status"], "fail", report)

    def test_token_only_different_outputs_same_format_fail(self):
        report = stage3_diff_test.analyze(
            {"payload": {"responses": [[1, 2, 3]]}, "load_format": "dummy"},
            {"payload": {"responses": [[4, 5, 6]]}, "load_format": "dummy"},
        )
        self.assertEqual(report["status"], "fail", report)

    def test_token_only_identical_outputs_pass(self):
        report = stage3_diff_test.analyze(
            {"payload": {"responses": [[1, 2, 3]]}, "load_format": "dummy"},
            {"payload": {"responses": [[1, 2, 3]]}, "load_format": "safetensors"},
        )
        self.assertEqual(report["status"], "pass", report)
        self.assertIsNone(report["dummy_garbled"])

    def test_stop_reason_is_not_treated_as_generated_text(self):
        """``TokenOutput`` carries stop_reason="completed"; that is not a text."""
        report = stage3_diff_test.analyze(
            {"payload": {"responses": [[1, 2, 3]], "stop_reason": "completed"}},
            {"payload": {"responses": [[4, 5, 6]], "stop_reason": "completed"}},
        )
        self.assertIsNone(report["dummy_garbled"])
        self.assertIsNone(report["safetensors_garbled"])
        self.assertFalse(report["identical_outputs"])
        self.assertEqual(report["status"], "pass")

    def test_token_only_different_outputs(self):
        report = stage3_diff_test.analyze(
            {"payload": {"responses": [[1, 2, 3]]}},
            {"payload": {"responses": [[4, 5, 6]]}},
        )
        self.assertEqual(report["status"], "pass")
        self.assertIsNone(report["dummy_garbled"])

    def test_decode_record_texts_restores_garbled_matrix(self):
        class FakeTokenizer:
            def batch_decode(self, sequences, skip_special_tokens=True):
                self.seen = sequences
                return ["北京是中国的首都。"] * len(sequences)

        tokenizer = FakeTokenizer()
        record = {"payload": {"responses": [[1, 2, 3]], "stop_reason": "completed"}}
        decoded = stage3_diff_test.decode_record_texts(record, tokenizer)
        self.assertEqual(decoded["texts"], ["北京是中国的首都。"])
        self.assertEqual(decoded["texts_source"], "decoded_payload_responses")
        self.assertEqual(tokenizer.seen, [[1, 2, 3]])

        report = stage3_diff_test.analyze(
            dict(decoded, load_format="dummy"),
            dict(decoded, load_format="safetensors"),
        )
        self.assertFalse(report["dummy_garbled"])
        self.assertTrue(report["identical_outputs"])
        self.assertEqual(report["status"], "pass")

    def test_decode_record_texts_keeps_existing_texts(self):
        class FailTokenizer:
            def batch_decode(self, *args, **kwargs):
                raise AssertionError("must not decode twice")

        record = {"texts": ["北京是中国的首都。"], "payload": {"responses": [[1]]}}
        self.assertIs(stage3_diff_test.decode_record_texts(record, FailTokenizer()), record)


@unittest.skipIf(torch is None, f"torch unavailable: {TORCH_ERROR}")
class FaultInjectionTests(unittest.TestCase):
    """``UWC_FAULT_*`` must be inert unless explicitly requested."""

    _KEYS = (
        "UWC_FAULT_STAGE1",
        "UWC_FAULT_STAGE2",
        "UWC_FAULT_STAGE3_SKIP",
        "UWC_FAULT_STAGE4",
        "UWC_FAULT_STAGE5_SAME_FORMAT",
    )

    def setUp(self):
        from uwc import fault

        self.fault = fault
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        for key in self._KEYS:
            os.environ.pop(key, None)

    def test_every_stage_is_off_by_default(self):
        self.assertEqual(self.fault.stage1_spec(), [])
        self.assertEqual(self.fault.stage2_spec(), [])
        self.assertEqual(self.fault.stage3_skip_patterns(), [])
        self.assertEqual(self.fault.stage4_spec(), [])
        self.assertFalse(self.fault.stage5_same_format())

    def test_spec_parsing(self):
        os.environ["UWC_FAULT_STAGE1"] = "q_proj:0.5, down_proj"
        self.assertEqual(self.fault.stage1_spec(), [("q_proj", 0.5), ("down_proj", 0.01)])

    def test_perturb_weights_changes_only_matching_tensor(self):
        os.environ["UWC_FAULT_STAGE2"] = "down_proj:0.5"
        good = torch.zeros(4)
        bad = torch.zeros(4)
        weights = [("layer.q_proj", good), ("layer.down_proj", bad)]
        changed = self.fault.perturb_weights(
            weights,
            self.fault.stage2_spec(),
            2,
        )
        self.assertEqual(changed, ["layer.down_proj"])
        # The sent tensors are replaced by perturbed copies; the originals
        # (which may alias model parameters) must not be written through.
        self.assertTrue(torch.equal(good, torch.zeros(4)))
        self.assertTrue(torch.equal(bad, torch.zeros(4)))
        self.assertIs(weights[0][1], good)
        self.assertIsNot(weights[1][1], bad)
        self.assertFalse(torch.equal(weights[1][1], torch.zeros(4)))

    def test_perturb_models_handles_pipeline_chunks(self):
        class Chunk:
            def __init__(self, name):
                self._name = name

            def named_parameters(self):
                return [(self._name, torch.zeros(4))]

        os.environ["UWC_FAULT_STAGE1"] = "layers.0:0.25"
        changed = self.fault.perturb_models(
            [Chunk("layers.0.mlp.down_proj"), Chunk("layers.1.mlp.down_proj")],
            self.fault.stage1_spec(),
            1,
        )
        self.assertEqual(changed, ["layers.0.mlp.down_proj"])

    def test_perturb_requires_grad_parameter(self):
        """Model parameters are leaves requiring grad; the fault must still apply."""
        param = torch.nn.Parameter(torch.zeros(4))
        os.environ["UWC_FAULT_STAGE4"] = "w:0.5"
        changed = self.fault.perturb_named([("w", param)], self.fault.stage4_spec(), 4)
        self.assertEqual(changed, ["w"])
        self.assertFalse(torch.equal(param.detach(), torch.zeros(4)))

    def test_stage4_full_perturbation_shifts_every_element(self):
        """One element does not move the cosine; stage 4 must shift the tensor."""
        param = torch.nn.Parameter(torch.zeros(4))
        os.environ["UWC_FAULT_STAGE4"] = "w:0.5"
        changed = self.fault.perturb_named([("w", param)], self.fault.stage4_spec(), 4, full=True)
        self.assertEqual(changed, ["w"])
        self.assertTrue(torch.equal(param.detach(), torch.full((4,), 0.5)))

    def test_drop_weights_list(self):
        weights = [
            ("a.q_proj", torch.zeros(2)),
            ("a.down_proj", torch.zeros(2)),
            ("b.down_proj", torch.zeros(2)),
        ]
        dropped = self.fault.drop_weights(weights, ["down_proj"])
        self.assertEqual(dropped, ["a.down_proj", "b.down_proj"])
        self.assertEqual([name for name, _ in weights], ["a.q_proj"])

    def test_drop_weights_dict_and_no_match(self):
        weights = {"a.q_proj": torch.zeros(2), "a.down_proj": torch.zeros(2)}
        self.assertEqual(self.fault.drop_weights(weights, ["embed"]), [])
        self.assertEqual(len(weights), 2)
        self.assertEqual(self.fault.drop_weights(weights, ["down_proj"]), ["a.down_proj"])
        self.assertEqual(list(weights), ["a.q_proj"])


@unittest.skipIf(torch is None, f"torch unavailable: {TORCH_ERROR}")
class ReportTests(unittest.TestCase):
    def test_full_pipeline_with_synthetic_dump(self):
        with tempfile.TemporaryDirectory() as tmp:
            dump = Path(tmp)
            (dump / "stage1_bridge").mkdir(parents=True)
            (dump / "stage2_transfer").mkdir()
            (dump / "stage3_vllm").mkdir()

            p0 = {"model.layers.0.w": torch.zeros(4), "lm_head.weight": torch.ones(4)}
            p2 = {name: tensor.clone() for name, tensor in p0.items()}
            torch.save(p0, dump / "stage1_bridge" / "P0_hf_original_step0_rank0.pt")
            torch.save(p2, dump / "stage1_bridge" / "P2_hf_exported_step0_rank0.pt")
            torch.save([("model.layers.0.w", torch.zeros(4))], dump / "stage2_transfer" / "P2_sender_step0_rank0.pt")
            torch.save([("model.layers.0.w", torch.zeros(4))], dump / "stage2_transfer" / "P5_receiver_step0_rank0.pt")
            logits = torch.randn(1, 3, 8, generator=torch.Generator().manual_seed(2))
            torch.save({"logits": logits}, dump / "stage1_bridge" / "G0_logits_gt.pt")
            torch.save({"logits": logits.clone()}, dump / "stage1_bridge" / "G1_logits_mcore.pt")
            (dump / "stage3_vllm" / "ptr_before_after_step0_rank0.json").write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "checked_layers": 1,
                        "synced_layers": 1,
                        "not_synced_layers": 0,
                        "missing_layers": 0,
                        "not_synced": [],
                        "details": [],
                    }
                )
            )
            (dump / "stage3_vllm" / "dummy_rl_first_step_output.json").write_text(
                json.dumps({"texts": ["北京是中国的首都。"]})
            )
            (dump / "stage3_vllm" / "safetensors_rl_first_step_output.json").write_text(
                json.dumps({"texts": ["北京是中国的首都，也是一座历史悠久的城市。"]})
            )

            uwc_compare.compare_all(dump_dir=dump)
            report = uwc_summary.build_report(dump, demo=True)
            self.assertEqual(report["overall"]["status"], "pass")
            self.assertEqual(report["overall"]["counts"]["pass"], 5)
            paths = uwc_summary.write_report(report, out_dir=dump)
            for key in ("json", "markdown", "html"):
                self.assertTrue(Path(paths[key]).is_file(), key)
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
            self.assertIn("dummy vs safetensors", markdown)
            self.assertIn("PASS", markdown)

    def _build_dump(self, dump: Path) -> dict:
        (dump / "stage1_bridge").mkdir(parents=True)
        (dump / "stage2_transfer").mkdir()
        (dump / "stage3_vllm").mkdir()
        p0 = {"model.layers.0.w": torch.zeros(4), "lm_head.weight": torch.ones(4)}
        p2 = {name: tensor.clone() for name, tensor in p0.items()}
        torch.save(p0, dump / "stage1_bridge" / "P0_hf_original_step0_rank0.pt")
        torch.save(p2, dump / "stage1_bridge" / "P2_hf_exported_step0_rank0.pt")
        torch.save(
            [("model.layers.0.w", torch.zeros(4))],
            dump / "stage2_transfer" / "P2_sender_step0_rank0.pt",
        )
        torch.save(
            [("model.layers.0.w", torch.zeros(4))],
            dump / "stage2_transfer" / "P5_receiver_step0_rank0.pt",
        )
        logits = torch.randn(1, 3, 8, generator=torch.Generator().manual_seed(2))
        torch.save({"logits": logits}, dump / "stage1_bridge" / "G0_logits_gt.pt")
        torch.save({"logits": logits.clone()}, dump / "stage1_bridge" / "G1_logits_mcore.pt")
        (dump / "stage3_vllm" / "ptr_before_after_step0_rank0.json").write_text(
            json.dumps(
                {
                    "status": "pass",
                    "checked_layers": 1,
                    "synced_layers": 1,
                    "not_synced_layers": 0,
                    "missing_layers": 0,
                    "not_synced": [],
                }
            )
        )
        (dump / "stage3_vllm" / "dummy_rl_first_step_output.json").write_text(
            json.dumps({"texts": ["北京是中国的首都。"], "load_format": "dummy"})
        )
        (dump / "stage3_vllm" / "safetensors_rl_first_step_output.json").write_text(
            json.dumps({"texts": ["北京是中国的首都。"], "load_format": "safetensors"})
        )
        return uwc_compare.compare_all(dump_dir=dump)

    def test_report_rejects_evidence_changed_after_compare(self):
        """A dump edited after compare must not keep its PASS verdict."""
        with tempfile.TemporaryDirectory() as tmp:
            dump = Path(tmp)
            self._build_dump(dump)
            self.assertEqual(uwc_summary.build_report(dump)["overall"]["status"], "pass")

            report_mtime = (dump / "stage1" / "report.json").stat().st_mtime
            p0 = dump / "stage1_bridge" / "P0_hf_original_step0_rank0.pt"
            os.utime(p0, (report_mtime + 10, report_mtime + 10))

            stale = uwc_summary.build_report(dump)
            bridge = [item for item in stale["items"] if item["id"] == "bridge_p0_p2"][0]
            self.assertEqual(bridge["status"], "error", bridge)
            self.assertIn("compare 之后", bridge["reason"])
            self.assertEqual(stale["overall"]["status"], "fail")

    def test_report_rejects_missing_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            dump = Path(tmp)
            self._build_dump(dump)
            (dump / "stage1_bridge" / "P2_hf_exported_step0_rank0.pt").unlink()
            report = uwc_summary.build_report(dump)
            bridge = [item for item in report["items"] if item["id"] == "bridge_p0_p2"][0]
            self.assertEqual(bridge["status"], "error", bridge)


@unittest.skipIf(torch is None, f"torch unavailable: {TORCH_ERROR}")
class CompareCliTests(unittest.TestCase):
    """The CLI must not forward model-only kwargs to stages that do not take them."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.dump = Path(self.tmp.name)
        (self.dump / "stage2_transfer").mkdir(parents=True)
        (self.dump / "stage3_vllm").mkdir()
        torch.save(
            [("model.layers.0.w", torch.zeros(4))],
            self.dump / "stage2_transfer" / "P2_sender_step0_rank0.pt",
        )
        torch.save(
            [("model.layers.0.w", torch.zeros(4))],
            self.dump / "stage2_transfer" / "P5_receiver_step0_rank0.pt",
        )
        (self.dump / "stage3_vllm" / "ptr_before_after_step0_rank0.json").write_text(
            json.dumps(
                {
                    "status": "pass",
                    "checked_layers": 1,
                    "synced_layers": 1,
                    "not_synced_layers": 0,
                    "missing_layers": 0,
                    "not_synced": [],
                }
            )
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_stage2_without_model_config(self):
        self.assertEqual(uwc_compare.main(["--dump_dir", str(self.dump), "--stage", "2"]), 0)
        report = json.loads((self.dump / "stage2" / "report.json").read_text())
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["transfer"]["status"], "pass")

    def test_stage3_without_model_config(self):
        self.assertEqual(uwc_compare.main(["--dump_dir", str(self.dump), "--stage", "3"]), 0)


@unittest.skipIf(torch is None, f"torch unavailable: {TORCH_ERROR}")
class ReviewFollowupTests(unittest.TestCase):
    """Regression tests for MR review findings (trust/rank/checkpoint loading)."""

    def test_tokenizer_trust_flag_is_forwarded(self):
        from uwc.offline import stage3_diff_test as diff_test

        captured = {}

        class _AutoTokenizer:
            @staticmethod
            def from_pretrained(path, **kwargs):
                captured["path"] = path
                captured.update(kwargs)
                return object()

        fake_transformers = types.SimpleNamespace(AutoTokenizer=_AutoTokenizer)
        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            diff_test._load_tokenizer("/model", trust_remote_code=False)

        self.assertEqual(captured["path"], "/model")
        self.assertIs(captured["trust_remote_code"], False)

    def test_forward_cli_forwards_no_trust_remote_code(self):
        from uwc.offline import forward_verify

        captured = {}

        def fake_run_hf(local_path, input_ids, device=None, dtype="fp32", trust_remote_code=True):
            captured["trust_remote_code"] = trust_remote_code
            return {"logits": []}

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"UWC_DUMP_DIR": tmp}):
            with mock.patch.object(forward_verify, "run_hf", fake_run_hf):
                with mock.patch.object(forward_verify, "_save", lambda payload, out: str(out)):
                    rc = forward_verify.main(["--backend", "hf", "--local_path", "/model", "--no-trust_remote_code"])

        self.assertEqual(rc, 0)
        self.assertIs(captured["trust_remote_code"], False)

    def test_checkpoint_loader_prefers_weights_only_true(self):
        from uwc.collectors import stage1_bridge_collector

        calls = []

        class _FakeTorch:
            @staticmethod
            def load(_path, **kwargs):
                calls.append(kwargs.get("weights_only"))
                if kwargs.get("weights_only"):
                    raise RuntimeError("unsupported global")
                return {"w": torch.zeros(1)}

        with mock.patch.object(stage1_bridge_collector, "_torch", return_value=_FakeTorch()):
            payload = stage1_bridge_collector._load_torch_file(Path("/tmp/unused.pt"))

        self.assertEqual(calls, [True, False])
        self.assertIn("w", payload)

    def test_non_main_rank_skips_sync_and_materialisation(self):
        from uwc.collectors import stage2_transfer_collector

        with mock.patch.dict(os.environ, {"RANK": "1", "LOCAL_RANK": "1"}, clear=False):
            with mock.patch.object(stage2_transfer_collector, "sync_device", side_effect=AssertionError("sync_device")):
                with mock.patch.object(
                    stage2_transfer_collector, "_materialise", side_effect=AssertionError("_materialise")
                ):
                    self.assertIsNone(
                        stage2_transfer_collector.collect_p2_sender({"w": 1}, dump_dir="/tmp/unused", step=0)
                    )
                    self.assertIsNone(stage2_transfer_collector.collect_p5({"w": 1}, dump_dir="/tmp/unused", step=0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
