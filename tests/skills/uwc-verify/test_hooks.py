"""Integration tests: patch a fake veRL tree, fire the runtime hooks, then restore."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SKILL_ROOT = REPO_ROOT / "skills" / "accuracy" / "uwc-verify"
sys.path.insert(0, str(SKILL_ROOT / "runtime"))
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

try:
    import torch
except Exception as exc:  # pragma: no cover
    torch = None
    TORCH_ERROR = exc
else:
    TORCH_ERROR = None


def _load_uwc_verify():
    path = SKILL_ROOT / "scripts" / "uwc_verify.py"
    spec = importlib.util.spec_from_file_location("uwc_verify_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FAKE_FILES = {
    "verl/__init__.py": "",
    "verl/utils/__init__.py": "",
    "verl/workers/__init__.py": "",
    "verl/workers/engine/__init__.py": "",
    "verl/workers/engine/megatron/__init__.py": "",
    "verl/workers/rollout/__init__.py": "",
    "verl/workers/rollout/vllm_rollout/__init__.py": "",
    "verl/workers/engine/megatron/transformer_impl.py": '''
        import os

        import torch
        import torch.nn as nn


        class FakeBridge:
            def load_weights(self, models, weights_path):
                model = models[0] if isinstance(models, (list, tuple)) else models
                payload = torch.load(
                    os.path.join(weights_path, "pytorch_model.bin"),
                    map_location="cpu",
                    weights_only=False,
                )
                for name, tensor in payload.items():
                    setattr(model, name.replace(".", "_"), nn.Parameter(tensor.clone()))
                return True

            def export_weights(self, models):
                model = models[0] if isinstance(models, (list, tuple)) else models
                for name in ("model.layers.0.w", "lm_head.weight"):
                    yield name, getattr(model, name.replace(".", "_")).detach().clone()


        class FakeEngine:
            def __init__(self, model_config=None):
                self.model_config = model_config or {}
                self.bridge = None
                self.module = None

            def _build_tf_config(self):
                self.bridge = FakeBridge()

            def initialize(self):
                self._build_tf_config()
                self.module = nn.Linear(2, 2, bias=False)
                self.bridge.load_weights(self.module, self.model_config["local_path"])

            def get_per_tensor_param(self, base_sync_done=False, **kwargs):
                def generator():
                    for item in self.bridge.export_weights(self.module):
                        yield item

                return generator(), None
    ''',
    "verl/workers/rollout/vllm_rollout/utils.py": '''
        import torch
        import torch.nn as nn


        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2, bias=False)


        class _Runner:
            def __init__(self):
                self.model = _Model()


        class vLLMColocateWorkerExtension:
            def __init__(self):
                self.model_runner = _Runner()
                self.inplace = False

            def _update_weights(self, weights, peft_config=None, base_sync_done=False):
                for name, tensor in weights:
                    if name != "linear.weight":
                        continue
                    if self.inplace:
                        with torch.no_grad():
                            self.model_runner.model.linear.weight.copy_(torch.as_tensor(tensor))
                    else:
                        self.model_runner.model.linear.weight = nn.Parameter(
                            torch.as_tensor(tensor).clone()
                        )
                return True

            def update_weights_from_ipc(self, buckets, peft_config=None, base_sync_done=False, use_shm=False):
                """veRL 0.9 entry point: one call per sync round, one bucket per callback."""
                for bucket in buckets:
                    self._update_weights(bucket, peft_config, base_sync_done)
                return True
    ''',
    "verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py": '''
        class BucketedWeightSender:
            def __init__(self):
                self.use_shm = True
                self.bucket_size = 2
                self.zmq_handle = None

            async def send(self, weights):
                sent = []
                async for item in weights:
                    sent.append(item)
                return sent

            async def async_send_weights(self, weights):
                """veRL 0.9+ name for the same transfer entry point."""
                sent = []
                async for item in weights:
                    sent.append(item)
                return sent
    ''',
    "verl/workers/rollout/vllm_rollout/vllm_async_server.py": '''
        class _TokenOutput:
            def __init__(self, token_ids, log_probs=None, stop_reason=None):
                self.token_ids = token_ids
                self.log_probs = log_probs
                self.stop_reason = stop_reason


        class vLLMHttpServer:
            """veRL 0.9 async rollout entry: generate() returns a TokenOutput."""

            def __init__(self, replica_rank=0):
                self.replica_rank = replica_rank

            async def generate(self, prompt_ids, sampling_params, request_id):
                return _TokenOutput([11, 12, 13], log_probs=[-0.1, -0.2, -0.3], stop_reason="completed")
    ''',
    "verl/workers/rollout/vllm_rollout/vllm_rollout.py": '''
        def is_support_ipc():
            return False


        class ServerAdapter:
            tokenizer = None

            def __init__(self):
                self.use_shm = not is_support_ipc()

            def generate_sequences(self, prompts=None):
                return {"texts": ["北京是中国的首都。"]}
    ''',
    "verl/workers/engine_workers.py": '''
        class EngineWorker:
            async def update_weights(self, *args, **kwargs):
                return "ok"
    ''',
}


def _write_fake_tree(root: Path) -> dict[str, str]:
    originals = {}
    for relative, source in FAKE_FILES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        text = textwrap.dedent(source).lstrip("\n")
        path.write_text(text, encoding="utf-8")
        originals[relative] = text
    return originals


@unittest.skipIf(torch is None, f"torch unavailable: {TORCH_ERROR}")
class HookIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="uwc_hooks_"))
        cls.verl_root = cls.tmp / "fake_verl"
        cls.model_dir = cls.tmp / "hf_model"
        cls.dump_dir = cls.tmp / "dump"
        cls.model_dir.mkdir(parents=True)
        cls.dump_dir.mkdir(parents=True)

        cls.p0 = {
            "model.layers.0.w": torch.arange(4, dtype=torch.float32),
            "lm_head.weight": torch.arange(4, dtype=torch.float32) + 10,
        }
        torch.save(cls.p0, cls.model_dir / "pytorch_model.bin")

        cls.originals = _write_fake_tree(cls.verl_root)

        cls.uwc_verify = _load_uwc_verify()
        rc = cls.uwc_verify.prepare({"verl": {"path": str(cls.verl_root)}})
        assert rc == 0, f"prepare returned {rc}"

        os.environ.update(
            {
                "UWC_ENABLE": "1",
                "UWC_DUMP_DIR": str(cls.dump_dir),
                "UWC_STEP": "0",
                "RANK": "0",
                "LOCAL_RANK": "0",
                "UWC_STAGE1": "1",
                "UWC_STAGE2": "1",
                "UWC_STAGE3": "1",
                "UWC_PTR_CHECK": "1",
                "UWC_DIFF_DUMP": "1",
                "UWC_DIFF_TAG": "dummy",
                "VERL_FORCE_USE_SHM": "1",
            }
        )
        sys.path.insert(0, str(cls.verl_root))
        for name in list(sys.modules):
            if name == "verl" or name.startswith("verl."):
                del sys.modules[name]

        cls.transformer_impl = importlib.import_module("verl.workers.engine.megatron.transformer_impl")
        cls.vllm_utils = importlib.import_module("verl.workers.rollout.vllm_rollout.utils")
        cls.bucket = importlib.import_module("verl.workers.rollout.vllm_rollout.bucketed_weight_transfer")
        cls.vllm_rollout = importlib.import_module("verl.workers.rollout.vllm_rollout.vllm_rollout")
        cls.async_server = importlib.import_module("verl.workers.rollout.vllm_rollout.vllm_async_server")

        cls.engine = cls.transformer_impl.FakeEngine({"local_path": str(cls.model_dir)})
        cls.engine.initialize()
        cls.params, _ = cls.engine.get_per_tensor_param()

    @classmethod
    def tearDownClass(cls):
        for name in list(sys.modules):
            if name == "verl" or name.startswith("verl."):
                del sys.modules[name]
        if str(cls.verl_root) in sys.path:
            sys.path.remove(str(cls.verl_root))
        for key in (
            "UWC_ENABLE",
            "UWC_DUMP_DIR",
            "UWC_STEP",
            "UWC_STAGE1",
            "UWC_STAGE2",
            "UWC_STAGE3",
            "UWC_PTR_CHECK",
            "UWC_DIFF_DUMP",
            "UWC_DIFF_TAG",
            "VERL_FORCE_USE_SHM",
        ):
            os.environ.pop(key, None)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_01_stage1_p0_p2_dumped(self):
        p0_path = self.dump_dir / "stage1_bridge" / "P0_hf_original_step0_rank0.pt"
        p2_path = self.dump_dir / "stage1_bridge" / "P2_hf_exported_step0_rank0.pt"
        self.assertTrue(p0_path.is_file(), f"missing {p0_path}")
        self.assertTrue(p2_path.is_file(), f"missing {p2_path}")
        p0 = torch.load(p0_path, weights_only=False)
        p2 = torch.load(p2_path, weights_only=False)
        self.assertEqual(set(p0), set(self.p0))
        self.assertIn("model.layers.0.w", p2)
        self.assertTrue(torch.equal(p0["model.layers.0.w"], p2["model.layers.0.w"]))

    def test_02_stage2_p2_sender_dumped(self):
        path = self.dump_dir / "stage2_transfer" / "P2_sender_step0_rank0.pt"
        self.assertTrue(path.is_file(), f"missing {path}")
        items = dict(torch.load(path, weights_only=False))
        self.assertIn("model.layers.0.w", items)
        self.assertIsInstance(self.params, list)
        self.assertIn("model.layers.0.w", dict(self.params))

    def test_02b_stage2_materialization_keeps_tensor_references(self):
        """Cloning the whole export set doubles device memory (~8 GB for a 4B model)."""
        from uwc import hooks as uwc_hooks

        tensor = torch.zeros(4)
        os.environ["UWC_STEP"] = "9"
        try:
            out = uwc_hooks._materialize_per_tensor([("probe", tensor)], None, base_sync_done=True)
        finally:
            os.environ["UWC_STEP"] = "0"
        self.assertEqual(out[0][0], "probe")
        self.assertIs(out[0][1], tensor)

    def test_03_stage2_p5_and_ptr_pass(self):
        extension = self.vllm_utils.vLLMColocateWorkerExtension()
        weights = [("linear.weight", torch.ones(2, 2))]
        p5 = self.dump_dir / "stage2_transfer" / "P5_receiver_step0_rank0.pt"
        extension.update_weights_from_ipc([weights], None, False)
        self.assertTrue(p5.is_file(), f"missing {p5}")
        received = dict(torch.load(p5, weights_only=False))
        self.assertTrue(torch.equal(received["linear.weight"], torch.ones(2, 2)))

        import json

        ptr = json.loads((self.dump_dir / "stage3_vllm" / "ptr_before_after_step0_rank0.json").read_text())
        self.assertEqual(ptr["status"], "pass", ptr)
        self.assertEqual(ptr["checked_layers"], 1)
        self.assertEqual(ptr["synced_layers"], 1)

    def test_03b_p5_accumulates_across_buckets(self):
        """P5 must be the union of every bucket in a sync round, flushed once at the end."""
        extension = self.vllm_utils.vLLMColocateWorkerExtension()
        p5 = self.dump_dir / "stage2_transfer" / "P5_receiver_step0_rank0.pt"
        p5.unlink(missing_ok=True)
        os.environ["UWC_PTR_CHECK"] = "0"
        try:
            extension.update_weights_from_ipc(
                [
                    [("a.weight", torch.zeros(2, 2))],
                    [("b.weight", torch.ones(2, 2))],
                ],
                None,
                False,
            )
        finally:
            os.environ["UWC_PTR_CHECK"] = "1"
        self.assertTrue(p5.is_file(), "P5 should be written at the end of the round")
        received = dict(torch.load(p5, weights_only=False))
        self.assertEqual(set(received), {"a.weight", "b.weight"})
        self.assertTrue(torch.equal(received["b.weight"], torch.ones(2, 2)))

    def test_03c_inplace_copy_counts_as_synced(self):
        """vLLM copies into the existing buffer: ptr stays, content changes -> synced."""
        extension = self.vllm_utils.vLLMColocateWorkerExtension()
        with torch.no_grad():
            extension.model_runner.model.linear.weight.zero_()
        extension.inplace = True
        os.environ["UWC_STEP"] = "2"
        try:
            extension.update_weights_from_ipc([[("linear.weight", torch.ones(2, 2))]], None, False)
        finally:
            os.environ["UWC_STEP"] = "0"
        import json

        report = json.loads((self.dump_dir / "stage3_vllm" / "ptr_before_after_step2_rank0.json").read_text())
        self.assertEqual(report["status"], "pass", report)
        self.assertEqual(report["ptr_changed_layers"], 0)
        self.assertEqual(report["content_changed_layers"], 1)
        self.assertEqual(report["synced_layers"], 1)

    def test_03d_noop_round_is_not_synced(self):
        """A round that writes identical values must still be flagged."""
        extension = self.vllm_utils.vLLMColocateWorkerExtension()
        with torch.no_grad():
            extension.model_runner.model.linear.weight.zero_()
        extension.inplace = True
        os.environ["UWC_STEP"] = "3"
        try:
            extension.update_weights_from_ipc([[("linear.weight", torch.zeros(2, 2))]], None, False)
        finally:
            os.environ["UWC_STEP"] = "0"
        import json

        report = json.loads((self.dump_dir / "stage3_vllm" / "ptr_before_after_step3_rank0.json").read_text())
        self.assertEqual(report["status"], "fail", report)
        self.assertEqual(report["not_synced"], ["linear.weight"])

    def test_04_legacy_noop_write_is_failure(self):
        """Legacy path: unchanged pointer *and* unchanged content means no sync."""
        os.environ["UWC_STEP"] = "1"
        try:
            extension = self.vllm_utils.vLLMColocateWorkerExtension()
            with torch.no_grad():
                extension.model_runner.model.linear.weight.zero_()
            extension.inplace = True
            extension._update_weights([("linear.weight", torch.zeros(2, 2))], None, False)
        finally:
            os.environ["UWC_STEP"] = "0"
        import json

        report = json.loads((self.dump_dir / "stage3_vllm" / "ptr_before_after_step1_rank0.json").read_text())
        self.assertEqual(report["status"], "fail", report)
        self.assertEqual(report["method"], "ptr_or_content_sample")
        self.assertEqual(report["not_synced_layers"], 1)
        self.assertEqual(report["not_synced"], ["linear.weight"])

    def test_04b_legacy_inplace_content_change_counts_as_synced(self):
        """Legacy path must use content sampling: in-place copy keeps data_ptr."""
        os.environ["UWC_STEP"] = "5"
        try:
            extension = self.vllm_utils.vLLMColocateWorkerExtension()
            with torch.no_grad():
                extension.model_runner.model.linear.weight.zero_()
            extension.inplace = True
            extension._update_weights([("linear.weight", torch.ones(2, 2))], None, False)
        finally:
            os.environ["UWC_STEP"] = "0"
        import json

        report = json.loads((self.dump_dir / "stage3_vllm" / "ptr_before_after_step5_rank0.json").read_text())
        self.assertEqual(report["status"], "pass", report)
        self.assertEqual(report["ptr_changed_layers"], 0)
        self.assertEqual(report["content_changed_layers"], 1)
        self.assertEqual(report["synced_layers"], 1)

    def test_05_bucket_send_tee_keeps_weights(self):
        import asyncio

        async def items():
            yield ("a", torch.zeros(1))
            yield ("b", torch.ones(1))

        sender = self.bucket.BucketedWeightSender()
        sent = asyncio.run(sender.send(items()))
        self.assertEqual([name for name, _ in sent], ["a", "b"])

    def test_05b_async_send_weights_tee_keeps_weights(self):
        """veRL 0.9 renamed the sender entry point to ``async_send_weights``."""
        import asyncio

        async def items():
            yield ("c", torch.zeros(1))
            yield ("d", torch.ones(1))

        sender = self.bucket.BucketedWeightSender()
        sent = asyncio.run(sender.async_send_weights(items()))
        self.assertEqual([name for name, _ in sent], ["c", "d"])

    def test_06_generation_dump_and_force_shm(self):
        adapter = self.vllm_rollout.ServerAdapter()
        self.assertTrue(adapter.use_shm, "VERL_FORCE_USE_SHM should force use_shm=True")
        adapter.generate_sequences(None)
        import json

        path = self.dump_dir / "stage3_vllm" / "dummy_rl_first_step_output.json"
        self.assertTrue(path.is_file(), f"missing {path}")
        record = json.loads(path.read_text())
        self.assertIn("北京是中国的首都。", record["texts"])

    def test_08_async_server_generation_dump(self):
        """veRL 0.9 async rollouts must still produce the stage-5 diff dump."""
        import asyncio
        import json

        os.environ["UWC_DIFF_TAG"] = "safetensors"
        try:
            server = self.async_server.vLLMHttpServer(replica_rank=0)
            result = asyncio.run(server.generate([1, 2, 3], {}, "req-0"))
        finally:
            os.environ["UWC_DIFF_TAG"] = "dummy"
        self.assertEqual(list(result.token_ids), [11, 12, 13])
        path = self.dump_dir / "stage3_vllm" / "safetensors_rl_first_step_output.json"
        self.assertTrue(path.is_file(), f"missing {path}")
        payload = json.loads(path.read_text())["payload"]
        self.assertEqual(payload["responses"], [[11, 12, 13]])
        self.assertEqual(payload["log_probs"], [[-0.1, -0.2, -0.3]])

    def test_08b_nonzero_replica_does_not_dump(self):
        import asyncio

        os.environ["UWC_DIFF_TAG"] = "replica1"
        try:
            server = self.async_server.vLLMHttpServer(replica_rank=1)
            asyncio.run(server.generate([1], {}, "req-1"))
        finally:
            os.environ["UWC_DIFF_TAG"] = "dummy"
        path = self.dump_dir / "stage3_vllm" / "replica1_rl_first_step_output.json"
        self.assertFalse(path.exists())

    def test_09_stage2_fault_corrupts_sent_weights_only(self):
        """R2: P2_sender is dumped before the fault, P5 sees the corrupted tensor."""

        p2_sender = self.dump_dir / "stage2_transfer" / "P2_sender_step0_rank0.pt"
        p5 = self.dump_dir / "stage2_transfer" / "P5_receiver_step0_rank0.pt"
        os.environ["UWC_FAULT_STAGE2"] = "model.layers.0.w:0.5"
        try:
            params, _ = self.engine.get_per_tensor_param()
        finally:
            os.environ.pop("UWC_FAULT_STAGE2", None)

        sent = dict(params)
        dumped = dict(torch.load(p2_sender, weights_only=False))
        self.assertTrue(torch.equal(dumped["model.layers.0.w"], self.p0["model.layers.0.w"]))
        self.assertFalse(torch.equal(sent["model.layers.0.w"], self.p0["model.layers.0.w"]))

        extension = self.vllm_utils.vLLMColocateWorkerExtension()
        extension.update_weights_from_ipc([params], None, False)
        received = dict(torch.load(p5, weights_only=False))
        self.assertTrue(torch.equal(received["model.layers.0.w"], sent["model.layers.0.w"]))

    def test_10_stage3_fault_skip_is_reported_as_not_synced(self):
        """R3: dropped tensors are recorded in P5 but never applied by vLLM."""
        extension = self.vllm_utils.vLLMColocateWorkerExtension()
        with torch.no_grad():
            extension.model_runner.model.linear.weight.zero_()
        extension.inplace = True
        os.environ["UWC_STEP"] = "4"
        os.environ["UWC_FAULT_STAGE3_SKIP"] = "linear.weight"
        try:
            extension.update_weights_from_ipc([[("linear.weight", torch.ones(2, 2))]], None, False)
        finally:
            os.environ.pop("UWC_FAULT_STAGE3_SKIP", None)
            os.environ["UWC_STEP"] = "0"

        p5 = dict(torch.load(self.dump_dir / "stage2_transfer" / "P5_receiver_step4_rank0.pt", weights_only=False))
        self.assertIn("linear.weight", p5, "P5 must still record what was received")
        import json

        report = json.loads((self.dump_dir / "stage3_vllm" / "ptr_before_after_step4_rank0.json").read_text())
        self.assertEqual(report["status"], "fail", report)
        self.assertEqual(report["not_synced"], ["linear.weight"])

    def test_11_stage1_fault_makes_p2_diverge_from_p0(self):
        """R1: corrupting the mcore model after load shows up as P2 != P0."""
        # prepare copies runtime/uwc into the fake tree, so the installed hooks
        # live in verl.utils.uwc.hooks and have their own _STATE.
        hooks = importlib.import_module("verl.utils.uwc.hooks")

        saved_state = dict(hooks._STATE)
        hooks._STATE["stage1_p0_done"] = False
        hooks._STATE["stage1_p2_done"] = False
        os.environ["UWC_FAULT_STAGE1"] = "layers_0:0.5"
        try:
            engine = self.transformer_impl.FakeEngine({"local_path": str(self.model_dir)})
            engine.initialize()
            params, _ = engine.get_per_tensor_param()
        finally:
            os.environ.pop("UWC_FAULT_STAGE1", None)
            hooks._STATE.update(saved_state)

        p0 = torch.load(
            self.dump_dir / "stage1_bridge" / "P0_hf_original_step0_rank0.pt",
            weights_only=False,
        )
        p2 = torch.load(
            self.dump_dir / "stage1_bridge" / "P2_hf_exported_step0_rank0.pt",
            weights_only=False,
        )
        self.assertTrue(torch.equal(p0["model.layers.0.w"], self.p0["model.layers.0.w"]))
        self.assertFalse(torch.equal(p2["model.layers.0.w"], p0["model.layers.0.w"]))
        self.assertTrue(torch.equal(dict(params)["model.layers.0.w"], p2["model.layers.0.w"]))

    def test_07_prepare_is_idempotent(self):
        target = self.verl_root / "verl/workers/engine/megatron/transformer_impl.py"
        before = target.read_text().count(self.uwc_verify.PATCH_BEGIN)
        self.assertEqual(before, 1)
        rc = self.uwc_verify.prepare({"verl": {"path": str(self.verl_root)}})
        self.assertEqual(rc, 0)
        after = target.read_text().count(self.uwc_verify.PATCH_BEGIN)
        self.assertEqual(after, 1)


@unittest.skipIf(torch is None, f"torch unavailable: {TORCH_ERROR}")
class PrepareRestoreTests(unittest.TestCase):
    def test_restore_reverts_everything(self):
        tmp = Path(tempfile.mkdtemp(prefix="uwc_restore_"))
        try:
            root = tmp / "fake_verl"
            originals = _write_fake_tree(root)
            uwc_verify = _load_uwc_verify()

            self.assertEqual(uwc_verify.prepare({"verl": {"path": str(root)}}), 0)
            patched_file = root / "verl/workers/rollout/vllm_rollout/vllm_rollout.py"
            patched_text = patched_file.read_text()
            self.assertIn("UWC-SKILL-PATCH", patched_text)
            self.assertIn("VERL_FORCE_USE_SHM", patched_text)
            self.assertTrue((root / "verl/utils/uwc/hooks.py").is_file())
            self.assertTrue((root / uwc_verify.MANIFEST_NAME).is_file())

            self.assertEqual(uwc_verify.prepare({"verl": {"path": str(root)}}, restore=True), 0)
            for relative, original in originals.items():
                self.assertEqual(
                    (root / relative).read_text(),
                    original,
                    f"{relative} was not restored byte-for-byte",
                )
            self.assertFalse((root / "verl/utils/uwc").exists())
            self.assertFalse((root / uwc_verify.MANIFEST_NAME).exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_double_prepare_then_restore_is_byte_identical(self):
        """A second prepare must not lose the use_shm edit record."""
        tmp = Path(tempfile.mkdtemp(prefix="uwc_restore2_"))
        try:
            root = tmp / "fake_verl"
            originals = _write_fake_tree(root)
            uwc_verify = _load_uwc_verify()
            self.assertEqual(uwc_verify.prepare({"verl": {"path": str(root)}}), 0)
            self.assertEqual(uwc_verify.prepare({"verl": {"path": str(root)}}), 0)
            manifest = json.loads((root / uwc_verify.MANIFEST_NAME).read_text())
            self.assertTrue(
                any("vllm_rollout.py" in str(edit.get("path")) for edit in manifest["edits"]),
                manifest["edits"],
            )
            self.assertEqual(uwc_verify.prepare({"verl": {"path": str(root)}}, restore=True), 0)
            for relative, original in originals.items():
                self.assertEqual((root / relative).read_text(), original, relative)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_restore_falls_back_when_manifest_lost_the_edit(self):
        """An old manifest without the edit record must still restore exactly."""
        tmp = Path(tempfile.mkdtemp(prefix="uwc_restore3_"))
        try:
            root = tmp / "fake_verl"
            originals = _write_fake_tree(root)
            uwc_verify = _load_uwc_verify()
            self.assertEqual(uwc_verify.prepare({"verl": {"path": str(root)}}), 0)
            manifest_path = root / uwc_verify.MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text())
            manifest["edits"] = []
            manifest_path.write_text(json.dumps(manifest))
            self.assertEqual(uwc_verify.prepare({"verl": {"path": str(root)}}, restore=True), 0)
            for relative, original in originals.items():
                self.assertEqual((root / relative).read_text(), original, relative)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_dry_run_changes_nothing(self):
        tmp = Path(tempfile.mkdtemp(prefix="uwc_dryrun_"))
        try:
            root = tmp / "fake_verl"
            originals = _write_fake_tree(root)
            uwc_verify = _load_uwc_verify()
            self.assertEqual(uwc_verify.prepare({"verl": {"path": str(root)}}, dry_run=True), 0)
            for relative, original in originals.items():
                self.assertEqual((root / relative).read_text(), original)
            self.assertFalse((root / "verl/utils/uwc").exists())
            self.assertFalse((root / uwc_verify.MANIFEST_NAME).exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_run_dry_run_exposes_env_and_stop_patterns(self):
        tmp = Path(tempfile.mkdtemp(prefix="uwc_run_"))
        try:
            root = tmp / "fake_verl"
            _write_fake_tree(root)
            uwc_verify = _load_uwc_verify()
            config = {
                "verl": {"path": str(root)},
                "dump_dir": str(tmp / "dump"),
                "commands": {
                    "stage1": "true",
                    "stage2": "true",
                    "stage3_ptr": "true",
                    "stage3_dummy": "true",
                    "stage3_safetensors": "true",
                    "forward_hf": "true",
                    "forward_mcore": "true",
                },
            }
            results = uwc_verify.run_stages(config, ["1", "2", "3", "4", "5"], dry_run=True)
            self.assertEqual(results["stage1"]["status"], "dry-run")
            self.assertEqual(results["stage3_ptr"]["status"], "dry-run")
            self.assertEqual(results["stage3_dummy"]["status"], "dry-run")
            self.assertEqual(results["forward_hf"]["status"], "dry-run")
            self.assertTrue(results["stage1"]["stop_when"])
            self.assertEqual(results["stage1"]["command"], "true")
            self.assertFalse((tmp / "dump" / "runs.json").exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_run_reuse_skips_existing_forward_logits(self):
        tmp = Path(tempfile.mkdtemp(prefix="uwc_run_reuse_"))
        try:
            root = tmp / "fake_verl"
            _write_fake_tree(root)
            uwc_verify = _load_uwc_verify()
            bridge = tmp / "dump" / "stage1_bridge"
            bridge.mkdir(parents=True)
            (bridge / "G0_logits_gt.pt").write_bytes(b"g0")
            (bridge / "G1_logits_mcore.pt").write_bytes(b"g1")
            config = {
                "verl": {"path": str(root)},
                "dump_dir": str(tmp / "dump"),
                "model": {"hf_path": str(tmp / "hf")},
                "commands": {"forward_hf": "false", "forward_mcore": "false"},
            }
            results = uwc_verify.run_stages(config, ["4"], reuse=True)
            self.assertEqual(results["forward_hf"]["status"], "reused", results)
            self.assertEqual(results["forward_mcore"]["status"], "reused", results)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_run_dry_run_stage_selection(self):
        tmp = Path(tempfile.mkdtemp(prefix="uwc_run_stage_"))
        try:
            root = tmp / "fake_verl"
            _write_fake_tree(root)
            uwc_verify = _load_uwc_verify()
            config = {
                "verl": {"path": str(root)},
                "dump_dir": str(tmp / "dump"),
                "commands": {
                    "stage1": "true",
                    "stage2": "true",
                    "stage3_ptr": "true",
                    "stage3_dummy": "true",
                    "stage3_safetensors": "true",
                },
            }
            self.assertEqual(set(uwc_verify.run_stages(config, ["1"], dry_run=True)), {"stage1"})
            self.assertEqual(set(uwc_verify.run_stages(config, ["2"], dry_run=True)), {"stage2"})
            self.assertEqual(set(uwc_verify.run_stages(config, ["3"], dry_run=True)), {"stage3_ptr"})
            self.assertEqual(
                set(uwc_verify.run_stages(config, ["5"], dry_run=True)),
                {"stage3_dummy", "stage3_safetensors"},
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_stage_env_wires_force_shm(self):
        """patch.use_shm must reach the child process, not just the text patch."""
        uwc_verify = _load_uwc_verify()
        base = {"verl": {"path": "/nonexistent"}}
        env = uwc_verify._stage_env({**base, "patch": {"use_shm": True}}, "stage2")
        self.assertEqual(env["VERL_FORCE_USE_SHM"], "1")
        env = uwc_verify._stage_env({**base, "patch": {"use_shm": False}}, "stage2")
        self.assertNotIn("VERL_FORCE_USE_SHM", env)
        # An explicit env entry always wins over the patch default.
        env = uwc_verify._stage_env(
            {**base, "patch": {"use_shm": True}, "env": {"VERL_FORCE_USE_SHM": "0"}},
            "stage2",
        )
        self.assertEqual(env["VERL_FORCE_USE_SHM"], "0")

    def test_doctor_flags_missing_verl(self):
        tmp = Path(tempfile.mkdtemp(prefix="uwc_doctor_"))
        try:
            uwc_verify = _load_uwc_verify()
            code = uwc_verify.doctor({"verl": {"path": str(tmp / "missing")}, "dump_dir": str(tmp / "dump")})
            self.assertNotEqual(code, 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
