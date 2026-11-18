"""Runtime hooks installed by appending a tiny block to veRL source files.

The appended block is::

    try:
        from verl.utils.uwc.hooks import install_module
        install_module(__name__, globals())
    except Exception as exc:
        print(f"[UWC] hook install skipped: {exc}")

Everything here is best-effort: a failure in a collector must never change or
break the training / weight-sync logic.
"""

from __future__ import annotations

import functools
import inspect
import os
import sys
import threading

from . import common, fault
from .collectors import stage1_bridge_collector as stage1
from .collectors import stage2_transfer_collector as stage2
from .collectors import stage3_vllm_collector as stage3

_STATE = {
    "stage1_p0_done": False,
    "stage1_p2_done": False,
    "installed_modules": set(),
    "warned_shard": False,
    "warned_shm": False,
}
_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# generic wrapper helpers
# ---------------------------------------------------------------------------


def _bind(original, *args, **kwargs):
    try:
        return inspect.signature(original).bind_partial(*args, **kwargs)
    except (TypeError, ValueError):
        return None


def _arg(bound, *names, default=None):
    if bound is None:
        return default
    for name in names:
        if name in bound.arguments:
            return bound.arguments[name]
    return default


def _classes_defined_in(module):
    result = []
    for value in vars(module).values():
        if isinstance(value, type) and getattr(value, "__module__", None) == module.__name__:
            result.append(value)
    return result


def _wrap_attr(owner, attribute, wrapper_builder):
    original = vars(owner).get(attribute)
    if original is None or getattr(original, "_uwc_wrapped", False):
        return False
    try:
        wrapped = wrapper_builder(original)
    except Exception:
        return False
    if wrapped is None:
        return False
    wrapped._uwc_wrapped = True
    try:
        setattr(owner, attribute, wrapped)
    except Exception:
        return False
    return True


def _wrap_class_method(cls, name, builder):
    if name not in cls.__dict__:
        return False
    return _wrap_attr(cls, name, builder)


def _log_exception(context, exc):
    try:
        common.log_warning(f"{context}: {exc}")
    except Exception:
        pass


def _device_label(tensor):
    try:
        return str(tensor.device)
    except Exception:
        return "unknown"


def _tensor_line(name, tensor):
    try:
        flat = tensor.float()
        return (
            f"{name}: device={_device_label(tensor)}, dtype={tensor.dtype}, "
            f"shape={list(tensor.shape)}, mean={flat.mean().item():.6f}, std={flat.std().item():.6f}"
        )
    except Exception as exc:
        return f"{name}: <stats failed: {exc}>"


# ---------------------------------------------------------------------------
# stage 1: Bridge load -> export
# ---------------------------------------------------------------------------


def _engine_model_path(engine):
    config = getattr(engine, "model_config", None)
    if config is None:
        return None
    path = getattr(config, "local_path", None)
    if path:
        return path
    try:
        return config.get("local_path")
    except Exception:
        return None


def _install_bridge_wrappers(engine):
    bridge = getattr(engine, "bridge", None)
    if bridge is None:
        return
    bridge_cls = type(bridge)
    for name in ("load_weights", "load_hf_weights"):
        method = getattr(bridge_cls, name, None)
        if callable(method) and not getattr(method, "_uwc_wrapped", False):
            try:
                wrapped = _make_load_wrapper(method)
                wrapped._uwc_wrapped = True
                setattr(bridge_cls, name, wrapped)
            except Exception as exc:
                _log_exception(f"cannot wrap bridge.{name}", exc)


def _make_load_wrapper(original):
    @functools.wraps(original)
    def wrapper(bridge_self, *args, **kwargs):
        if not common.stage_enabled(1):
            return original(bridge_self, *args, **kwargs)

        bound = _bind(original, bridge_self, *args, **kwargs)
        models = _arg(bound, "models", "model", default=None)
        path = _arg(bound, "weights_path", "local_path", "path", "hf_path", default=None)
        if models is None and args:
            models = args[0]
        if path is None and len(args) > 1:
            path = args[1]
        model_list = models if isinstance(models, (list, tuple)) else ([models] if models is not None else None)
        step = common.get_step()
        dump_dir = common.get_dump_dir()

        if not _STATE["stage1_p0_done"] and path:
            try:
                if stage1.collect_p0(bridge_self, path, dump_dir, step=step):
                    _STATE["stage1_p0_done"] = True
            except Exception as exc:
                _log_exception("stage1 P0 collect failed", exc)

        result = original(bridge_self, *args, **kwargs)

        # Opt-in fault R1: corrupt the freshly loaded mcore weights so the
        # exported P2 disagrees with P0.  Never active unless requested.
        specs = fault.stage1_spec()
        if specs and model_list:
            try:
                fault.perturb_models(model_list, specs, 1)
            except Exception as exc:
                _log_exception("stage1 fault injection", exc)

        if not _STATE["stage1_p2_done"] and model_list:
            try:
                if stage1.collect_p2(bridge_self, model_list, dump_dir, step=step):
                    _STATE["stage1_p2_done"] = True
            except Exception as exc:
                _log_exception("stage1 P2 collect failed", exc)
        return result

    return wrapper


def _stage1_fallback(engine):
    """Export P2 after init if the bridge load call was missed by the wrapper."""
    if not common.stage_enabled(1) or _STATE["stage1_p2_done"]:
        return
    bridge = getattr(engine, "bridge", None)
    module = getattr(engine, "module", None)
    if bridge is None or module is None:
        return
    path = _engine_model_path(engine)
    step = common.get_step()
    dump_dir = common.get_dump_dir()
    if not _STATE["stage1_p0_done"] and path:
        try:
            if stage1.collect_p0(bridge, path, dump_dir, step=step):
                _STATE["stage1_p0_done"] = True
        except Exception as exc:
            _log_exception("stage1 P0 fallback failed", exc)
    try:
        if stage1.collect_p2(bridge, module, dump_dir, step=step):
            _STATE["stage1_p2_done"] = True
    except Exception as exc:
        _log_exception("stage1 P2 fallback failed", exc)


def _wrap_after_build_tf_config(original):
    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        try:
            _install_bridge_wrappers(self)
        except Exception as exc:
            _log_exception("install bridge wrappers", exc)
        return result

    return wrapper


def _wrap_after_init(original):
    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        try:
            _install_bridge_wrappers(self)
            _stage1_fallback(self)
        except Exception as exc:
            _log_exception("post-init UWC hook", exc)
        return result

    return wrapper


def _wrap_after_initialize(original):
    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        try:
            _install_bridge_wrappers(self)
            _stage1_fallback(self)
        except Exception as exc:
            _log_exception("post-initialize UWC hook", exc)
        return result

    return wrapper


def _materialize_per_tensor(params, engine, base_sync_done=False):
    if params is None:
        return params
    if isinstance(params, dict):
        items = list(params.items())
    elif hasattr(params, "items"):
        items = list(params.items())
    else:
        items = list(params)

    materialised = []
    for name, tensor in items:
        # Keep the tensor reference instead of cloning: cloning every export
        # tensor doubles the device footprint (~8 GB for Qwen3-4B bf16) and
        # OOMs when vLLM is colocated.  The dump copies each tensor to CPU one
        # at a time, and the R2 fault replaces only the matched entry.
        materialised.append((str(name), tensor))

    dump_dir = common.get_dump_dir()
    step = common.get_step()
    try:
        stage2.collect_p2_sender(materialised, dump_dir, step=step)
    except Exception as exc:
        _log_exception("stage2 P2_sender collect failed", exc)

    if not base_sync_done:
        try:
            path = _engine_model_path(engine)
            for index, (name, tensor) in enumerate(materialised):
                if index >= 3:
                    break
                common.log(f"export diag {_tensor_line(name, tensor)}")
            stage1.diag_export_weights(materialised, path, dump_dir=dump_dir, step=step)
        except Exception as exc:
            _log_exception("export diagnostic failed", exc)

    # Opt-in fault R2: P2_sender has been dumped already, so corrupting the
    # materialised tensors now makes P5 disagree with the sender dump.
    try:
        fault.perturb_weights(materialised, fault.stage2_spec(), 2)
    except Exception as exc:
        _log_exception("stage2 fault injection", exc)
    return materialised


def _wrap_per_tensor_param(original):
    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if not common.stage_enabled(2):
            return result
        try:
            bound = _bind(original, self, *args, **kwargs)
            base_sync_done = bool(_arg(bound, "base_sync_done", default=False))
            if isinstance(result, tuple) and len(result) == 2:
                params, peft_config = result
                return _materialize_per_tensor(params, self, base_sync_done), peft_config
            return _materialize_per_tensor(result, self, base_sync_done)
        except Exception as exc:
            _log_exception("stage2 get_per_tensor_param hook", exc)
            return result

    return wrapper


def _wrap_shard_param(original):
    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        if common.stage_enabled(2) and not _STATE["warned_shard"]:
            _STATE["warned_shard"] = True
            common.log_warning(
                "weight sync is using the shard/delta path; UWC stage2 only captures "
                "the per-tensor path. Force the per-tensor sync mode to verify P2_sender vs P5."
            )
        return original(self, *args, **kwargs)

    return wrapper


def _install_transformer_impl(module):
    for cls in _classes_defined_in(module):
        _wrap_class_method(cls, "_build_tf_config", _wrap_after_build_tf_config)
        _wrap_class_method(cls, "initialize", _wrap_after_initialize)
        _wrap_class_method(cls, "__init__", _wrap_after_init)
        _wrap_class_method(cls, "get_per_tensor_param", _wrap_per_tensor_param)
        _wrap_class_method(cls, "get_per_tensor_param_shard", _wrap_shard_param)
        _wrap_class_method(cls, "get_per_tensor_param_delta_shard", _wrap_shard_param)


# ---------------------------------------------------------------------------
# stage 2: receiver P5 + stage 3 data_ptr
# ---------------------------------------------------------------------------


def _find_model(obj):
    """Locate the vLLM runner model (an nn.Module) from a worker extension."""
    if obj is None:
        return None
    direct_paths = [
        ("model_runner", "model"),
        ("model",),
        ("module",),
        ("engine", "module"),
        ("llm_engine", "model_executor", "driver_worker", "model_runner", "model"),
    ]
    for path in direct_paths:
        current = obj
        for attribute in path:
            current = getattr(current, attribute, None)
            if current is None:
                break
        if current is not None and hasattr(current, "named_parameters"):
            return current

    # Limited BFS for version-specific wrappers.
    queue = [(obj, 0)]
    seen = set()
    while queue:
        current, depth = queue.pop(0)
        if depth >= 3 or id(current) in seen:
            continue
        seen.add(id(current))
        if hasattr(current, "named_parameters") and not isinstance(current, type):
            try:
                next(iter(current.named_parameters()))
                return current
            except StopIteration:
                continue
            except Exception:
                pass
        for attribute in ("model_runner", "model", "module", "engine", "llm_engine", "model_executor"):
            child = getattr(current, attribute, None)
            if child is not None and not isinstance(child, (str, bytes, int, float)):
                queue.append((child, depth + 1))
    return None


def _make_update_weights_wrapper(original, is_method):
    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        obj = args[0] if is_method and args else None
        bound = _bind(original, *args, **kwargs)
        weights = _arg(bound, "weights", default=None)
        if weights is None:
            if is_method and len(args) > 1:
                weights = args[1]
            elif not is_method and args:
                weights = args[0]
            else:
                weights = kwargs.get("weights")

        dump_dir = common.get_dump_dir()
        step = common.get_step()
        model = _find_model(obj) if obj is not None else None
        ptr_snapshot = None

        if common.stage_enabled(2) and weights is not None:
            # veRL 0.9 delivers weights bucket by bucket and reuses the IPC
            # buffer, so copy to CPU now and flush the full set once the whole
            # sync round finishes (see ``_p5_flush``).
            try:
                if common.is_main_rank():
                    for name, tensor in _iter_named(weights):
                        _P5_ACC[str(name)] = common.to_cpu(tensor)
            except Exception as exc:
                _log_exception("stage2 P5 accumulate failed", exc)

        if (
            not _ROUND_ACTIVE
            and common.stage_enabled(3)
            and common.env_bool("UWC_PTR_CHECK")
            and model is not None
        ):
            # Legacy veRL (<=0.8) calls ``_update_weights`` once with the full
            # weight set.  veRL 0.9 wraps each round in
            # ``update_weights_from_ipc``; there the round-level snapshot in
            # ``_make_sync_round_wrapper`` is the correct (and only correct)
            # scope, because one call only carries one bucket.
            #
            # Snapshot pointer *and* sampled content: vLLM's ``load_weights``
            # writes in place, so a pointer-only check would flag every
            # successfully synced layer.
            try:
                ptr_snapshot = stage3.snapshot_params(model)
            except Exception as exc:
                _log_exception("stage3 ptr snapshot failed", exc)

        # Opt-in fault R3: P5 has been recorded above; dropping tensors here
        # means vLLM never applies them although they arrived intact.
        try:
            fault.drop_weights(weights, fault.stage3_skip_patterns())
        except Exception as exc:
            _log_exception("stage3 fault injection", exc)

        result = original(*args, **kwargs)

        if ptr_snapshot:
            try:
                stage3.compare_ptr_snapshots(model, ptr_snapshot, dump_dir, step=step)
            except Exception as exc:
                _log_exception("stage3 ptr compare failed", exc)
        return result

    return wrapper


_P5_ACC: dict[str, object] = {}


def _iter_named(weights):
    if isinstance(weights, dict):
        return list(weights.items())
    if hasattr(weights, "items"):
        return list(weights.items())
    return list(weights)


def _p5_flush():
    if not _P5_ACC:
        return
    try:
        stage2.collect_p5(
            list(_P5_ACC.items()), common.get_dump_dir(), step=common.get_step()
        )
    except Exception as exc:
        _log_exception("stage2 P5 flush failed", exc)
    finally:
        _P5_ACC.clear()


_ROUND_ACTIVE = False


def _make_sync_round_wrapper(original):
    """Reset/flush the per-round accumulators around ``update_weights_from_ipc``.

    ``update_weights_from_ipc`` is the veRL 0.9 entry point for one complete
    weight-sync round; ``_update_weights`` is then called once per bucket.  P5
    must be the union of all buckets, and the data_ptr check must compare the
    model before the first bucket with its state after the last one.
    """

    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        global _ROUND_ACTIVE
        ptr_requested = common.stage_enabled(3) and common.env_bool("UWC_PTR_CHECK")
        if not common.stage_enabled(2) and not ptr_requested:
            return original(self, *args, **kwargs)
        _P5_ACC.clear()
        model = None
        snapshot = None
        if ptr_requested:
            try:
                model = _find_model(self)
                snapshot = stage3.snapshot_params(model)
            except Exception as exc:
                _log_exception("stage3 ptr snapshot failed", exc)
        _ROUND_ACTIVE = True
        try:
            return original(self, *args, **kwargs)
        finally:
            _ROUND_ACTIVE = False
            _p5_flush()
            if snapshot:
                try:
                    stage3.compare_ptr_snapshots(
                        model, snapshot, common.get_dump_dir(), step=common.get_step()
                    )
                except Exception as exc:
                    _log_exception("stage3 ptr compare failed", exc)

    return wrapper


def _install_update_weights(module):
    for name in ("_update_weights",):
        if callable(vars(module).get(name)):
            _wrap_attr(module, name, lambda original: _make_update_weights_wrapper(original, False))
    for cls in _classes_defined_in(module):
        _wrap_class_method(cls, "_update_weights", lambda original: _make_update_weights_wrapper(original, True))
        _wrap_class_method(cls, "update_weights_from_ipc", _make_sync_round_wrapper)


# ---------------------------------------------------------------------------
# stage 2 diagnostics: bucketed sender / engine worker
# ---------------------------------------------------------------------------


async def _aiter_any(iterable):
    if hasattr(iterable, "__aiter__"):
        async for item in iterable:
            yield item
    else:
        for item in iterable:
            yield item


async def _tee_weights(iterable):
    count = 0
    async for item in _aiter_any(iterable):
        try:
            if count < 3:
                name, tensor = item
                common.log(f"stage2 sender diag {_tensor_line(name, tensor)}")
            count += 1
        except Exception:
            pass
        yield item


def _make_send_wrapper(original):
    if inspect.iscoroutinefunction(original):

        @functools.wraps(original)
        async def async_wrapper(self, weights, *args, **kwargs):
            if common.stage_enabled(2):
                try:
                    common.log(
                        f"stage2 sender use_shm={getattr(self, 'use_shm', None)}, "
                        f"bucket_size={getattr(self, 'bucket_size', None)}, "
                        f"zmq={getattr(self, 'zmq_handle', None)}"
                    )
                    weights = _tee_weights(weights)
                except Exception as exc:
                    _log_exception("stage2 sender diagnostics", exc)
            return await original(self, weights, *args, **kwargs)

        return async_wrapper

    @functools.wraps(original)
    def sync_wrapper(self, weights, *args, **kwargs):
        if common.stage_enabled(2):
            try:
                common.log(
                    f"stage2 sender use_shm={getattr(self, 'use_shm', None)}, "
                    f"bucket_size={getattr(self, 'bucket_size', None)}, "
                    f"zmq={getattr(self, 'zmq_handle', None)}"
                )
            except Exception:
                pass
        return original(self, weights, *args, **kwargs)

    return sync_wrapper


def _install_bucket_send(module):
    for cls in _classes_defined_in(module):
        # veRL <= 0.8 exposes ``send``; 0.9+ renamed it to ``async_send_weights``.
        for name in ("send", "async_send_weights"):
            _wrap_class_method(cls, name, _make_send_wrapper)


def _make_engine_update_wrapper(original):
    if inspect.iscoroutinefunction(original):

        @functools.wraps(original)
        async def async_wrapper(*args, **kwargs):
            if common.stage_enabled(2):
                common.log("engine update_weights begin")
            result = await original(*args, **kwargs)
            if common.stage_enabled(2):
                common.log("engine update_weights end")
            return result

        return async_wrapper

    @functools.wraps(original)
    def sync_wrapper(*args, **kwargs):
        return original(*args, **kwargs)

    return sync_wrapper


def _install_engine_workers(module):
    for cls in _classes_defined_in(module):
        _wrap_class_method(cls, "update_weights", _make_engine_update_wrapper)
    if callable(vars(module).get("update_weights")):
        _wrap_attr(module, "update_weights", _make_engine_update_wrapper)


# ---------------------------------------------------------------------------
# stage 3: first RL-step generation dump + VERL_FORCE_USE_SHM fallback
# ---------------------------------------------------------------------------


def _make_generate_wrapper(original):
    if inspect.iscoroutinefunction(original):

        @functools.wraps(original)
        async def async_wrapper(self, *args, **kwargs):
            result = await original(self, *args, **kwargs)
            try:
                tokenizer = getattr(self, "tokenizer", None)
                stage3.maybe_dump_first_generation(result, tokenizer=tokenizer)
            except Exception as exc:
                _log_exception("stage3 generation dump", exc)
            return result

        return async_wrapper

    @functools.wraps(original)
    def sync_wrapper(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        try:
            tokenizer = getattr(self, "tokenizer", None)
            stage3.maybe_dump_first_generation(result, tokenizer=tokenizer)
        except Exception as exc:
            _log_exception("stage3 generation dump", exc)
        return result

    return sync_wrapper


def _wrap_force_shm(original):
    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        try:
            if common.env_bool("VERL_FORCE_USE_SHM") and hasattr(self, "use_shm"):
                self.use_shm = True
                if not _STATE["warned_shm"]:
                    _STATE["warned_shm"] = True
                    common.log("VERL_FORCE_USE_SHM=1: forced use_shm=True after init")
        except Exception as exc:
            _log_exception("force shm", exc)
        return result

    return wrapper


def _install_vllm_rollout(module):
    for cls in _classes_defined_in(module):
        _wrap_class_method(cls, "generate_sequences", _make_generate_wrapper)
        _wrap_class_method(cls, "__init__", _wrap_force_shm)


def _make_async_server_generate_wrapper(original):
    """Dump the first async rollout output (veRL 0.9 ``vLLMHttpServer.generate``).

    The async rollout path never calls ``ServerAdapter.generate_sequences``, so
    without this hook the dummy/safetensors differential has no data.  Only
    replica 0 writes, so both runs record a comparable request stream.
    """
    if not inspect.iscoroutinefunction(original):
        return _make_generate_wrapper(original)

    @functools.wraps(original)
    async def wrapper(self, *args, **kwargs):
        result = await original(self, *args, **kwargs)
        try:
            if getattr(self, "replica_rank", 0) == 0:
                stage3.maybe_dump_first_generation(result)
        except Exception as exc:
            _log_exception("stage3 async generation dump", exc)
        return result

    return wrapper


def _install_vllm_async_server(module):
    for cls in _classes_defined_in(module):
        _wrap_class_method(cls, "generate", _make_async_server_generate_wrapper)


def _install_generation_dump(module):
    for cls in _classes_defined_in(module):
        _wrap_class_method(cls, "generate_sequences", _make_generate_wrapper)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


_ROLE_HANDLERS = (
    ("workers.engine.megatron.transformer_impl", _install_transformer_impl),
    ("workers.rollout.vllm_rollout.utils", _install_update_weights),
    ("workers.rollout.vllm_rollout.bucketed_weight_transfer", _install_bucket_send),
    # Must precede the plain vllm_rollout entry only in spirit; the module
    # names differ, but keep the async server first for readability.
    ("workers.rollout.vllm_rollout.vllm_async_server", _install_vllm_async_server),
    ("workers.rollout.vllm_rollout.vllm_rollout", _install_vllm_rollout),
    ("workers.engine_workers", _install_engine_workers),
    ("experimental.agent_loop.agent_loop", _install_generation_dump),
)


def install_module(module_name, module_globals=None):
    """Install the hooks for one veRL module.  Called from the appended block."""
    if not common.env_bool("UWC_ENABLE"):
        return
    module = sys.modules.get(module_name)
    if module is None and module_globals is not None:
        module = sys.modules.get(module_globals.get("__name__"))
    if module is None:
        common.log_warning(f"hook install skipped: module {module_name} not in sys.modules")
        return

    with _LOCK:
        if module_name in _STATE["installed_modules"]:
            return
        _STATE["installed_modules"].add(module_name)

    for marker, handler in _ROLE_HANDLERS:
        if marker in module_name:
            try:
                handler(module)
                common.log(f"hooks installed for {module_name}")
            except Exception as exc:
                _log_exception(f"hook install failed for {module_name}", exc)
            return
    common.log(f"no UWC hook role matched module {module_name}")
