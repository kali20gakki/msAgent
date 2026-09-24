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

#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
DiT（扩散/多模态生成）模型适配器模板。

适用：`MultimodalSDModelslimV1QuantService` + `MultimodalPipelineInterface`（重构路径）。
覆盖单 DiT（HunyuanVideo / FLUX / SD3 / Sana / HunyuanDiT / CogViewX）和双专家 DiT（如 Wan2.2）。
双专家场景的 `base + scenario + expert_sub` 三层结构见 `assets/dit/skeleton.md`。

12 类通用陷阱（meta tensor / 显存碎片 / enable_dump 短路 / parse_args 单次 / quant_overrides /
双 expert key / 子适配器接口继承 / ExitStack / task 串扰 / 样本校验 / KV cache / block_keyword）
见 `references/dit/pitfalls.md`，落盘前必查。
"""

from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any, ClassVar, Dict, Generator, List, Optional, Tuple, Type

import torch
from pydantic import BaseModel, ConfigDict
from torch import nn

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.core.const import DeviceType
from msmodelslim.infra.dataset_loader.vlm_dataset_loader import VlmCalibSample
from msmodelslim.model.base import BaseModelAdapter
from msmodelslim.model.common.layer_wise_forward import (
    TransformersForwardBreak,
    generated_decoder_layer_visit_func_with_keyword,
)
from msmodelslim.model.interface_hub import (
    ModelInfoInterface,
    MultimodalPipelineInterface,
)
from msmodelslim.utils.cache import load_cached_data_for_models, to_device
from msmodelslim.utils.exception import InvalidModelError, SchemaValidateError
from msmodelslim.utils.logging import get_logger, logger_setter


@logger_setter()
class DitModelAdapter(
    BaseModelAdapter,
    ModelInfoInterface,
    MultimodalPipelineInterface,
):
    """DiT（扩散/多模态生成）模型适配器模板。

    关键点：
    - 不依赖 KV cache：`enable_kv_cache` 为 no-op。
    - 校准输入含 timestep / text_embeds / latents 等条件信号，由 `inference_dump_calib_data`
      通过浮点推理 dump。
    - `generate_model_visit` / `generate_model_forward` 严格按 block 顺序一一对应，
      block 列表由 `self.block_keyword`（自动探测自模型类名）过滤 `named_modules` 得到。
    - `init_model` 返回 `Dict[str, nn.Module]`，与 `prepare_calib_data` 的 key 对齐。
    """

    def __init__(self, model_type: str, model_path: Path, trust_remote_code: bool = False):
        super().__init__(model_type, model_path, trust_remote_code)
        # 由 _load_pipeline 装配；占位以避免子类访问时 AttributeError
        self.pipeline = None
        self.transformer = None
        self.model_args = None
        # 由 _resolve_block_keyword 在 init_model 中自动探测；不在此硬编码模型特定值
        self.block_keyword: Optional[str] = None

    # ==================== ModelInfoInterface ====================
    def get_model_pedigree(self) -> str:
        return "dit"

    def get_model_type(self) -> str:
        return self.model_type

    # ==================== MultimodalPipelineInterface ====================
    class InferenceConfig(BaseModel):
        """`extra="forbid"`。字段名必须与目标推理仓 `parse_args` 的 `--key` 一一对应。

        推理仓字段速查见 `references/interface_reference.md §5.5`（HunyuanVideo / Wan 系 / FLUX 对比表）。
        """

        model_config = ConfigDict(extra="forbid")
        # HunyuanVideo 参考字段；子类按目标推理仓重写
        model_resolution: Optional[str] = None
        video_size: Optional[Tuple[int, int]] = None
        video_length: Optional[int] = None
        infer_steps: Optional[int] = None
        seed: Optional[int] = None
        neg_prompt: Optional[str] = None
        cfg_scale: Optional[float] = None
        embedded_cfg_scale: Optional[float] = None
        num_videos: Optional[int] = None
        flow_shift: Optional[float] = None
        batch_size: Optional[int] = None

    def get_inference_config_class(self) -> Type[BaseModel]:
        """返回本适配器对应的 InferenceConfig Pydantic 配置类。"""
        return self.InferenceConfig

    def configure_runtime(self, inference_config: Any) -> None:
        """把已校验的 inference_config 落到 self.model_args（单次 parse_args）。

        子类按目标推理仓 CLI 桥接：白名单校验 → 构造 argv（含 quant_overrides）→ parse_args。
        失败模式见 pitfalls.md §D / §E。
        """
        # Placeholder: 子类按原推理仓 CLI 桥接
        self.model_args = inference_config
        get_logger().info(
            "DitModelAdapter.configure_runtime invoked; subclass must bridge to "
            "the original inference repo's parse_args."
        )

    def _fixed_quant_runtime_overrides(self) -> Dict[str, Any]:
        """通用：分布式并行 + cache/fusion 关闭。字段名按目标 DiT 实际支持的调整。"""
        return {
            "ulysses_degree": 1, "ring_degree": 1,
            "vae_parallel": False, "use_cache": False, "use_cache_double": False,
            "use_attentioncache": False,
        }

    def _calib_data_when_dump_disabled(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """P3 短路：dump_config.enable_dump=False 时直接返回空 calib_data（DiT data-free 默认路径）。"""
        return {n: None for n in models}

    def inference_dump_calib_data(
        self,
        dataset: Any = None,
        inference_config: Any = None,
    ) -> None:
        """执行浮点推理 dump 校准数据。本方法仅在 `enable_dump=True` 时被调用（见
        `prepare_calib_data` 短路）；data-free 全动态量化（W8A8 动态）正是通过
        `enable_dump: false` + `_calib_data_when_dump_disabled` 短路实现，本方法不触发。

        子类按原推理仓 sampler.predict / pipeline(...) 重放浮点推理。
        """
        raise NotImplementedError(
            "DitModelAdapter.inference_dump_calib_data 仅为模板占位；"
            "请按原推理仓 sampler.predict / pipeline(...) 重放浮点推理以 dump 校准数据。"
        )

    def prepare_calib_data(
        self,
        models: Dict[str, Any],
        dump_config: Any,
        save_path: Path,
        dataset: Any,
        inference_config: Any,
    ) -> Dict[str, Any]:
        """按 expert_name 构造/加载 calib_data 缓存。返回 dict 的 key 必须与 `init_model` 一致。

        - `enable_dump=False` → 短路返回 `{expert: None}`（见 pitfalls.md §C）
        - `enable_dump=True` → pth 缓存命中则加载，未命中则调 `inference_dump_calib_data` 生成
        """
        if not dump_config.enable_dump:
            return self._calib_data_when_dump_disabled(models)

        base_dir = dump_config.dump_data_dir if dump_config.dump_data_dir else save_path
        pth_file_path_list: Dict[str, str] = {}
        for expert_name in models:
            pth_file_path_list[expert_name] = str(
                Path(base_dir).joinpath(f"calib_data_{self.model_type}_{expert_name}.pth")
            )
        calib_data = load_cached_data_for_models(
            pth_file_path_list=pth_file_path_list,
            generate_func=lambda: self.inference_dump_calib_data(
                dataset=dataset,
                inference_config=inference_config,
            ),
            models=models,
            dump_config=dump_config,
        )
        get_logger().info("prepare calib_data from %s success", base_dir)
        return calib_data

    def release_auxiliary_models(self) -> None:
        """P2：dump 完释放 text_encoder / vae / image_encoder 等量化无关子模块 + empty_cache。

        任何 DiT 通用（详见 pitfalls.md §B 显存碎片化）。子模型若有额外的"量化无关子模块"，
        在此一并释放。
        """
        for pipeline_attr in ("pipeline",):
            pipeline = getattr(self, pipeline_attr, None)
            if pipeline is None:
                continue
            for sub_name in ("text_encoder", "vae", "image_encoder", "safety_checker"):
                sub = getattr(pipeline, sub_name, None)
                if sub is None:
                    continue
                inner = getattr(sub, "model", None)
                if isinstance(inner, nn.Module):
                    inner.cpu()
                del pipeline.__dict__[sub_name]
        if hasattr(torch, "npu") and hasattr(torch.npu, "empty_cache"):
            torch.npu.empty_cache()

    def quantization_context(self) -> AbstractContextManager:
        """量化运行上下文（autocast + no_grad + no_sync）。多 expert 用 ExitStack 见 skeleton.md。"""
        @contextmanager
        def _ctx():
            @contextmanager
            def _noop_no_sync():
                yield

            no_sync = getattr(self, "no_sync", _noop_no_sync)
            with torch.autocast(device_type="npu", dtype=torch.bfloat16), torch.no_grad(), no_sync():
                yield

        return _ctx()

    def get_expert_adapter(self, expert_name: str) -> "DitModelAdapter":
        """按 expert_name 返回子适配器；单 DiT 默认返回 self。双专家见 skeleton.md。"""
        _ = expert_name
        return self

    # ==================== BaseModelAdapter 5 个基础接口 ====================
    def validate_calib_samples(self, samples: List[VlmCalibSample]) -> List[VlmCalibSample]:
        """校准样本场景校验：默认 text 必填、image 禁止（T2V 规则）。子类按任务覆写。"""
        for idx, sample in enumerate(samples):
            if not isinstance(sample.text, str) or not sample.text.strip():
                raise SchemaValidateError(
                    f"{self.model_type} sample[{idx}] requires non-empty text",
                    action="Provide text in dataset entries (index.jsonl / VlmCalibSample.text).",
                )
            if sample.image is not None:
                raise SchemaValidateError(
                    f"{self.model_type} sample[{idx}] must not include image",
                    action="T2V calibration is text-only by default; override for I2V/TI2V.",
                )
        return samples

    def handle_dataset(
        self,
        dataset: Any,
        device: DeviceType = DeviceType.NPU,
    ) -> List[Any]:
        """dump 前仅做场景校验，不做模型 forward。dump 由 `inference_dump_calib_data` 触发。"""
        _ = device
        if dataset is None:
            return []
        if isinstance(dataset, VlmCalibSample):
            return self.validate_calib_samples([dataset])
        if isinstance(dataset, list) and dataset and isinstance(dataset[0], VlmCalibSample):
            return self.validate_calib_samples(dataset)
        if not isinstance(dataset, list):
            raise SchemaValidateError(
                "handle_dataset expects dataset to be a list, got %s" % type(dataset).__name__
            )
        return dataset

    def init_model(self, device: DeviceType = DeviceType.NPU) -> Dict[str, nn.Module]:
        """加载 DiT 主干并返回 Dict（单 DiT `{'': self.transformer}`；双专家见 skeleton.md）。"""
        _ = device
        self._load_pipeline()
        self._setup_cache()
        if self.block_keyword is None:
            self.block_keyword = self._resolve_block_keyword(self.transformer)
        return {"": self.transformer}

    def generate_model_visit(
        self,
        model: nn.Module,
    ) -> Generator[ProcessRequest, Any, None]:
        """按 block 顺序遍历，使用 `self.block_keyword` 过滤 `named_modules`。

        与 `generate_model_forward` 严格一一对应。
        """
        if not self.block_keyword:
            raise RuntimeError(
                "DitModelAdapter.block_keyword not resolved; call init_model first."
            )
        return generated_decoder_layer_visit_func_with_keyword(model, keyword=self.block_keyword)

    def generate_model_forward(
        self,
        model: nn.Module,
        inputs: Any,
    ) -> Generator[ProcessRequest, Any, None]:
        """分段前向；与 `generate_model_visit` 严格一一对应。"""
        if not self.block_keyword:
            raise RuntimeError(
                "DitModelAdapter.block_keyword not resolved; call init_model first."
            )

        transformer_blocks: List[Tuple[str, nn.Module]] = [
            (name, module)
            for name, module in model.named_modules()
            if self.block_keyword in module.__class__.__name__.lower()
        ]
        if not transformer_blocks:
            raise InvalidModelError(
                f"No module matched block_keyword={self.block_keyword!r} in {type(model).__name__}",
                action="Check self.block_keyword against inference_repo's modeling_*.py block class names.",
            )

        first_block_input: Optional[Tuple[Tuple[Any, ...], Dict[str, Any]]] = None

        def break_hook(module: nn.Module, hook_args: Tuple[Any, ...], hook_kwargs: Dict[str, Any]):
            nonlocal first_block_input
            first_block_input = (hook_args, hook_kwargs)
            raise TransformersForwardBreak()

        hooks = [transformer_blocks[0][1].register_forward_pre_hook(break_hook, with_kwargs=True)]

        try:
            if isinstance(inputs, (list, tuple)):
                model(*inputs)
            elif isinstance(inputs, dict):
                model(**inputs)
            else:
                model(inputs)
        except TransformersForwardBreak:
            pass
        finally:
            for hook in hooks:
                hook.remove()

        if first_block_input is None:
            raise InvalidModelError(
                "Can't get first block input.",
                action="Please check the model and input.",
            )

        first_block_input = to_device(first_block_input, "cpu")
        current_inputs = first_block_input

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        for name, block in transformer_blocks:
            args, kwargs = current_inputs
            outputs = yield ProcessRequest(name, block, args, kwargs)
            hidden_states = outputs
            current_inputs = ((hidden_states,), current_inputs[1])

    def enable_kv_cache(self, model: nn.Module, need_kv_cache: bool) -> None:
        """DiT 推理不依赖 KV cache；保持 no-op。"""
        _ = model, need_kv_cache
        self._logger.info(
            "DitModelAdapter.enable_kv_cache is a no-op (need_kv_cache=%s); "
            "diffusion inference does not rely on KV cache.",
            need_kv_cache,
        )
        return None

    # ==================== block_keyword 自动探测 ====================
    def _resolve_block_keyword(self, model: nn.Module) -> str:
        """从已加载模型的 module 类名自动探测 block_keyword（子串匹配）。

        若自动探测失败（模型结构特殊），子类应在 `_load_pipeline` 后直接赋值
        `self.block_keyword`（值需来自 inference_repo 的 modeling_*.py block 类名）。详见 pitfalls.md §L。
        """
        from collections import Counter

        _trivial = frozenset({
            "module", "identity", "linear", "layernorm", "rmsnorm", "groupnorm",
            "conv2d", "conv1d", "conv3d", "parameterdict", "parameterlist",
            "sequential", "modulelist", "tanh", "silu", "gelu", "relu", "sigmoid",
            "dropout", "embedding", "container", "scaled_dot", "diagonal",
            "modulation", "frequencyembedder", "timestepembedder", "rope",
        })

        counter: Counter = Counter()
        for _, module in model.named_modules():
            cls_name = module.__class__.__name__.lower()
            if cls_name in _trivial:
                continue
            counter[cls_name] += 1

        # block 类通常重复出现（>=2）；优先选类名含 "block" 的
        block_like = [name for name, _ in counter.most_common() if counter[name] >= 2 and "block" in name]
        if not block_like:
            block_like = [name for name, cnt in counter.most_common() if cnt >= 2]
        if not block_like:
            raise InvalidModelError(
                f"Cannot auto-discover block_keyword from {type(model).__name__}; "
                f"no repeated non-trivial block classes found.",
                action="Read inference_repo's modeling_*.py to find block class name, "
                       "then set self.block_keyword manually in init_model.",
            )

        keyword = self._longest_common_substring(block_like)
        # 回退：LCS 太短或为空时用最高频类名（子串匹配自身仍成立）
        if len(keyword) < 4:
            keyword = block_like[0]

        self._logger.info(
            "Auto-discovered block_keyword=%r from repeated block classes: %s",
            keyword, block_like[:5],
        )
        return keyword

    @staticmethod
    def _longest_common_substring(strings: List[str]) -> str:
        """返回多个字符串的最长公共子串；单元素时返回该串本身。"""
        if not strings:
            return ""
        if len(strings) == 1:
            return strings[0]
        reference = strings[0]
        best = ""
        for i in range(len(reference)):
            for j in range(i + len(best) + 1, len(reference) + 1):
                candidate = reference[i:j]
                if all(candidate in s for s in strings[1:]):
                    if len(candidate) > len(best):
                        best = candidate
        return best

    # ==================== 私有运行时装配（子类实现） ====================
    # CLI list/tuple 字段常量：与目标推理仓 argparse 中 nargs="+" 的字段对齐；
    # 仅此集合内的 list/tuple 字段才会在 _namespace_to_argv 中展开为多 token argv。
    # HunyuanVideo 参考：frozenset({"video_size"})。
    _CLI_LIST_FIELDS: ClassVar[frozenset[str]] = frozenset()

    def _load_pipeline(self) -> None:
        """加载原推理仓 Pipeline（transformer / vae / text encoder 等）。子类必实现。

        需注入 `inference_repo` 到 `sys.path`；单卡：`ulysses_degree=1` / `ring_degree=1` /
        `vae_parallel=False`；写出 `self.transformer` 与 `self.pipeline`。
        """
        raise NotImplementedError(
            "DitModelAdapter._load_pipeline 仅为模板占位；"
            "请按 inference_repo 加载 pipeline 并写出 self.transformer / self.pipeline。"
        )

    def _setup_cache(self) -> None:
        """装配 block 级 attention_cache（DiT block.forward 依赖）。无需求可保持空实现。"""
        return None
