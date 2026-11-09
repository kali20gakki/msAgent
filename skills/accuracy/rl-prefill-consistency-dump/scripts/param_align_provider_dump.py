#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 5.1 配套工具: 从训练侧实际 bridge 实例化 provider, dump 实际生效 config JSON。

训练侧"实际生效"配置 = 训练侧 bridge(Megatron-Bridge / MindSpeed-Bridge 子类)实例的
provider 属性集合。bridge 字段名写错时(如 mHC: 写 enable_hyper_connections 而
MindSpeed 读 enable_mhc_connections), provider 对应属性是 dataclass 默认值——
dump 出来就能被 param_align_check.py 检出与推理侧不一致。

用法:
  python3 param_align_provider_dump.py <hf_model_path> <out_provider_config.json> \
    [--bridge DeepSeekV4FlashBridge] [--bridge-src <bridge 源码根> ...]

--bridge: 训练侧实际使用的 bridge 类名, 默认 DeepSeekV4FlashBridge(MindSpeed 训练链)。
          纯 Megatron-Bridge 场景可传 DeepSeekV4Bridge; 新模型按现场 bridge 名传。
--bridge-src: 训练侧 bridge 源码根目录(可多次传入), 插入 sys.path 以定位
              mindspeed_bridge / megatron.bridge 模块。不传时沿用默认容器路径
              /workspace/Megatron-Bridge/src 与 /workspace/MindSpeed-Bridge
              (特定容器布局, 环境不符会报裸 ModuleNotFoundError, 此时请用 --bridge-src 指定)。

输出: provider 上所有可 JSON 序列化的标量/list/dict 属性(与 param_align_check.py
      --mapping 的 train 字段点分路径配合; 点分路径可深取嵌套 dict 值)。

环境适配: 本脚本会补 Megatron-LM 的 nvrx 版本探测(见 _nvrx_patch), 仅在导入链
阶段生效, 不影响实际训练/推理行为。
"""
import argparse
import json
import sys


def _nvrx_patch():
    """环境适配: Megatron-LM 的 nvrx 版本探测在缺 __version__ 或版本过低时崩。
    仅影响导入链里的探测函数(是否启用 nvrx 异步), 不影响实际训练/推理行为。"""
    try:
        import nvidia_resiliency_ext as nvrx
        nvrx.__version__ = "0.6.0"
    except Exception:
        pass


def _serializable(v):
    if v is None or isinstance(v, (int, float, bool, str)):
        return True, v
    if isinstance(v, (list, tuple)):
        ok, items = True, []
        for x in v:
            s, y = _serializable(x)
            if not s:
                return False, None
            items.append(y)
        return True, items
    if isinstance(v, dict):
        ok, items = True, {}
        for k, x in v.items():
            s, y = _serializable(x)
            if not s:
                return False, None
            items[str(k)] = y
        return True, items
    return False, None


def _collect_provider(provider):
    out = {}
    for k in dir(provider):
        if k.startswith("_"):
            continue
        try:
            v = getattr(provider, k)
        except Exception:
            continue  # 副作用 property(如 meta_model)访问失败/触发构建, 跳过
        if callable(v):
            continue
        ok, y = _serializable(v)
        if ok:
            out[k] = y
    return out


def _resolve_bridge(name):
    if name == "DeepSeekV4FlashBridge":
        from mindspeed_bridge.models.deepseek_v4_flash.deepseek_v4_flash_bridge import (
            DeepSeekV4FlashBridge,
        )
        return DeepSeekV4FlashBridge
    if name == "DeepSeekV4Bridge":
        from megatron.bridge.models.deepseek.deepseek_v4_bridge import DeepSeekV4Bridge
        return DeepSeekV4Bridge
    import importlib
    mod_name, cls_name = name.rsplit(".", 1)
    return getattr(importlib.import_module(mod_name), cls_name)


def main():
    ap = argparse.ArgumentParser(description="阶段 5.1 配套: dump 训练侧实际生效 provider config")
    ap.add_argument("hf_model_path", help="HF 模型目录(含 config.json)")
    ap.add_argument("out", help="输出 provider config JSON 路径")
    ap.add_argument("--bridge", default="DeepSeekV4FlashBridge",
                    help="训练侧实际 bridge 类名(默认 DeepSeekV4FlashBridge)")
    ap.add_argument("--bridge-src", action="append", default=[],
                    help="训练侧 bridge 源码根目录, 可多次传入(默认容器布局路径, 环境不符请传)")
    args = ap.parse_args()

    _nvrx_patch()
    if args.bridge_src:
        for p in args.bridge_src:
            sys.path.insert(0, p)
    else:
        print("提示: 未传 --bridge-src, 沿用默认容器路径 /workspace/Megatron-Bridge/src 与 "
              "/workspace/MindSpeed-Bridge(特定容器布局); 环境不符时请用 --bridge-src 指定源码根目录",
              file=sys.stderr)
        sys.path.insert(0, "/workspace/Megatron-Bridge/src")
        sys.path.insert(0, "/workspace/MindSpeed-Bridge")

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(args.hf_model_path)

    class _FakePreTrained:
        """轻量 hf_pretrained 壳: provider_bridge 通常只用 .config;
        若实际 bridge 还需要更多属性, 按现场补(field 名以运行时报错为准)。"""

        def __init__(self, cfg):
            self.config = cfg

    B = _resolve_bridge(args.bridge)
    provider = B().provider_bridge(_FakePreTrained(config))
    out = _collect_provider(provider)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"written: {args.out} ({len(out)} serializable attrs)")
    # mHC 关键字段仅打印实际存在的字段:provider 可能不提供 mHC(非该模型),硬编码访问会
    # 在输出文件已写好之后抛 AttributeError,造成假失败。
    for key in ("enable_mhc_connections", "mhc_num_residual_streams", "mhc_sinkhorn_iterations"):
        if hasattr(provider, key):
            print(f"mHC 关键字段: {key}={getattr(provider, key)}")


if __name__ == "__main__":
    main()
