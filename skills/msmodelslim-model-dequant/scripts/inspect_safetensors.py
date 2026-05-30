#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import argparse
import fnmatch
import json
import os
from collections import defaultdict

from safetensors import safe_open


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect tensor shape and dtype from safetensors by wildcard pattern."
    )
    parser.add_argument(
        "-m",
        "--model-path",
        required=True,
        help="Model directory containing model.safetensors.index.json",
    )
    parser.add_argument(
        "-p",
        "--pattern",
        default="*",
        help="Wildcard pattern for tensor names, e.g. 'model.layers.*.mlp*.weight*'",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit output rows, 0 means no limit",
    )
    return parser.parse_args()


def load_weight_map(model_path: str):
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        model_index = json.load(f)
    return model_index.get("weight_map", {})


def main():
    args = parse_args()
    weight_map = load_weight_map(args.model_path)
    if not weight_map:
        print("No weight_map found in model.safetensors.index.json")
        return

    filtered_names = [name for name in weight_map if fnmatch.fnmatch(name, args.pattern)]
    if not filtered_names:
        print(f"No tensors matched pattern: {args.pattern}")
        return

    file_to_names = defaultdict(list)
    for name in filtered_names:
        file_to_names[weight_map[name]].append(name)

    rows = []
    for file_name, names in sorted(file_to_names.items()):
        file_path = os.path.join(args.model_path, file_name)
        with safe_open(file_path, framework="pt", device="cpu") as f:
            available = set(f.keys())
            for name in sorted(names):
                if name not in available:
                    rows.append((name, "N/A", "N/A", file_name))
                    continue
                tensor = f.get_tensor(name)
                rows.append((name, str(tuple(tensor.shape)), str(tensor.dtype), file_name))

    if args.limit > 0:
        rows = rows[: args.limit]

    print(f"Matched tensors: {len(rows)} (pattern={args.pattern})")
    print("tensor_name\tshape\tdtype\tfile")
    for name, shape, dtype, file_name in rows:
        print(f"{name}\t{shape}\t{dtype}\t{file_name}")


if __name__ == "__main__":
    main()
