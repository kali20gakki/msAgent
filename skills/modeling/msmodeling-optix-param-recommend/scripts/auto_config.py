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
"""Auto configuration script - automatically modifies config.toml based on scenarios"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional


# Scenario configuration templates
SCENARIOS = {
    "quick-test": {
        "description": "Quick test scenario",
        "n_particles": 5,
        "iters": 3,
        "ttft_penalty": 1,
        "tpot_penalty": 1,
        "ttft_slo": 2.0,
        "tpot_slo": 0.1,
        "sample_size": 100000,
        "target_field_ranges": {
            "max_batch_size": {"min": 10, "max": 100},
            "max_num_seqs": {"min": 8, "max": 32},
        },
    },
    "standard": {
        "description": "Standard optimization scenario",
        "n_particles": 8,
        "iters": 4,
        "ttft_penalty": 1,
        "tpot_penalty": 1,
        "ttft_slo": 2.0,
        "tpot_slo": 0.05,
        "sample_size": 100000,
        "target_field_ranges": {
            "max_batch_size": {"min": 10, "max": 400},
            "max_num_seqs": {"min": 8, "max": 64},
        },
    },
    "deep-optimize": {
        "description": "Deep optimization scenario",
        "n_particles": 30,
        "iters": 20,
        "ttft_penalty": 3,
        "tpot_penalty": 3,
        "ttft_slo": 1.0,
        "tpot_slo": 0.03,
        "sample_size": 300000,
        "target_field_ranges": {
            "max_batch_size": {"min": 10, "max": 1000},
            "max_num_seqs": {"min": 8, "max": 128},
        },
    },
    "ttft-priority": {
        "description": "First token latency priority",
        "n_particles": 15,
        "iters": 10,
        "ttft_penalty": 10,
        "tpot_penalty": 1,
        "ttft_slo": 0.5,
        "tpot_slo": 0.1,
        "sample_size": 150000,
        "target_field_ranges": {
            "max_batch_size": {"min": 10, "max": 200},
            "max_prefill_batch_size": {"min": 0.3, "max": 0.7},
        },
    },
    "tpot-priority": {
        "description": "Non-first token latency priority",
        "n_particles": 15,
        "iters": 10,
        "ttft_penalty": 1,
        "tpot_penalty": 10,
        "ttft_slo": 2.0,
        "tpot_slo": 0.02,
        "sample_size": 150000,
        "target_field_ranges": {
            "max_batch_size": {"min": 50, "max": 400},
            "max_prefill_batch_size": {"min": 0.1, "max": 0.3},
        },
    },
    "throughput": {
        "description": "Throughput priority",
        "n_particles": 10,
        "iters": 5,
        "ttft_penalty": 1,
        "tpot_penalty": 1,
        "ttft_slo": 5.0,
        "tpot_slo": 0.2,
        "success_rate_penalty": 10,
        "sample_size": 100000,
        "target_field_ranges": {
            "max_batch_size": {"min": 100, "max": 1000},
            "max_num_seqs": {"min": 64, "max": 256},
        },
    },
}


def parse_time_budget(time_str: str) -> int:
    """Parse time budget string, return minutes"""
    if not time_str:
        return None

    time_str = time_str.lower().strip()

    if time_str.endswith('h'):
        return int(time_str[:-1]) * 60
    elif time_str.endswith('m'):
        return int(time_str[:-1])
    elif time_str.endswith('d'):
        return int(time_str[:-1]) * 24 * 60
    else:
        try:
            return int(time_str)
        except ValueError:
            return None


def calculate_optimal_params(time_budget_minutes: int, single_test_minutes: int = 10) -> Dict[str, int]:
    """Calculate optimal n_particles and iters based on time budget

    Note: Each seed runs the service and test twice (warmup + formal test),
    so actual single group time = 2 × single_test_minutes
    """
    # Actual time per seed = 2 × (service startup + benchmark)
    actual_single_test_minutes = single_test_minutes * 2

    # Reserve 20% buffer time
    effective_budget = int(time_budget_minutes * 0.8)

    # Calculate total number of groups that can be run
    total_groups = effective_budget // actual_single_test_minutes

    # Recommended config: iters is about 1/2 of n_particles
    # n_particles * iters ≈ total_groups
    # Let iters = n_particles / 2
    # Then n_particles * (n_particles / 2) ≈ total_groups
    # n_particles ≈ sqrt(2 * total_groups)

    import math

    n_particles = min(int(math.sqrt(2 * total_groups)), 50)
    iters = max(min(n_particles // 2, 20), 3)
    n_particles = max(min(n_particles, 100), 5)

    print(f"Time budget: {time_budget_minutes} minutes")
    print(f"Single test estimate: {single_test_minutes} minutes")
    print(f"Actual group time (x2): {actual_single_test_minutes} minutes")
    print(f"Total groups possible: ~{total_groups}")

    return {"n_particles": n_particles, "iters": iters}


def update_config_value(content: str, key: str, value: Any) -> str:
    """Update a single value in the config file"""
    # Handle different types of values
    if isinstance(value, str):
        escaped_value = value.replace('\\', '\\\\').replace('"', '\\"')
        value_str = f'"{escaped_value}"'
    elif isinstance(value, bool):
        value_str = str(value).lower()
    elif isinstance(value, (int, float)):
        value_str = str(value)
    else:
        value_str = str(value)

    # Match key = value or key=value format
    pattern = rf'^{re.escape(key)}\s*=\s*.+$'
    replacement = f'{key} = {value_str}'

    if re.search(pattern, content, re.MULTILINE):
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    else:
        # If key doesn't exist, add to the beginning of the file
        content = f'{key} = {value_str}\n{content}'

    return content


def update_target_field(content: str, field_name: str, updates: Dict[str, Any]) -> str:
    """Update parameters in target_field"""
    # Find [[xxx.target_field]] block
    pattern = rf'(\[\[.*?\.target_field\]\][^\[]*?name\s*=\s*["\']{re.escape(field_name)}["\'][^\[]*?)'

    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print(f"⚠ Parameter not found: {field_name}")
        return content

    block = match.group(1)
    new_block = block

    for key, value in updates.items():
        if isinstance(value, str):
            escaped_value = value.replace('\\', '\\\\').replace('"', '\\"')
            value_str = f'"{escaped_value}"'
        elif isinstance(value, bool):
            value_str = str(value).lower()
        else:
            value_str = str(value)

        # Update value within block
        key_pattern = rf'^{re.escape(key)}\s*=\s*.+$'
        key_replacement = f'{key} = {value_str}'

        if re.search(key_pattern, new_block, re.MULTILINE):
            new_block = re.sub(key_pattern, key_replacement, new_block, flags=re.MULTILINE)

    content = content.replace(block, new_block)
    return content


def apply_scenario_config(content: str, scenario: str, args) -> str:
    """Apply scenario configuration"""
    config = SCENARIOS[scenario].copy()

    # Adjust parameters based on time budget
    if args.time_budget:
        time_minutes = parse_time_budget(args.time_budget)
        if time_minutes:
            optimal = calculate_optimal_params(time_minutes)
            config.update(optimal)
            print(
                f"Adjusted based on time budget {args.time_budget}: n_particles={optimal['n_particles']}, iters={optimal['iters']}"
            )

    # Apply command line override parameters
    if args.ttft_slo is not None:
        config["ttft_slo"] = args.ttft_slo
    if args.tpot_slo is not None:
        config["tpot_slo"] = args.tpot_slo

    # Update basic parameters
    basic_params = [
        "n_particles",
        "iters",
        "ttft_penalty",
        "tpot_penalty",
        "ttft_slo",
        "tpot_slo",
        "success_rate_penalty",
        "sample_size",
    ]

    for param in basic_params:
        if param in config:
            content = update_config_value(content, param, config[param])

    # Update target_field ranges
    if "target_field_ranges" in config:
        for field_name, ranges in config["target_field_ranges"].items():
            content = update_target_field(content, field_name, ranges)

    # Update pso_top_k for deep-optimize scenario
    if scenario == "deep-optimize":
        pso_top_k = getattr(args, "pso_top_k", None) or 4
        content = update_config_value(content, "pso_top_k", pso_top_k)
        print(f"  - pso_top_k: {pso_top_k} (recommended for deep-optimize: 3-5)")

    # Update engine related configuration
    if args.engine == "vllm":
        content = update_vllm_config(content, args)
    elif args.engine == "mindie":
        content = update_mindie_config(content, args)

    # Update benchmark tool configuration
    if args.benchmark:
        content = update_benchmark_config(content, args)

    return content


def update_vllm_config(content: str, args) -> str:
    """Update VLLM related configuration"""
    if args.model:
        content = update_config_section(content, "vllm.command", "model", args.model)
    if args.served_name:
        content = update_config_section(content, "vllm.command", "served_model_name", args.served_name)
    if args.host:
        content = update_config_section(content, "vllm.command", "host", args.host)
    if args.port:
        content = update_config_section(content, "vllm.command", "port", args.port)

    # Update corresponding parameters for vllm_benchmark
    if args.model:
        content = update_config_section(content, "vllm_benchmark.command", "model", args.model)
    if args.served_name:
        content = update_config_section(content, "vllm_benchmark.command", "served_model_name", args.served_name)
    if args.host:
        content = update_config_section(content, "vllm_benchmark.command", "host", args.host)
    if args.port:
        content = update_config_section(content, "vllm_benchmark.command", "port", str(args.port))

    return content


def update_mindie_config(content: str, args) -> str:
    """Update MindIE related configuration"""
    # MindIE mainly configures through target_field
    return content


def update_benchmark_config(content: str, args) -> str:
    """Update benchmark tool configuration"""
    if args.benchmark == "vllm_benchmark":
        # Set default others for vllm_benchmark if not already configured
        others_pattern = r'\[vllm_benchmark\.command\](.*?)(?=\[|$)'
        match = re.search(others_pattern, content, re.DOTALL)
        if match:
            section = match.group(1)
            # Check if others is empty or still has placeholder
            others_match = re.search(r'^others\s*=\s*"([^"]*)"', section, re.MULTILINE)
            if others_match and (not others_match.group(1) or others_match.group(1) == ""):
                content = update_config_section(
                    content, "vllm_benchmark.command", "others",
                    "--random-input-len 3000 --random-output-len 1000"
                )
                print("  📝 Set vllm_benchmark default others: --random-input-len 3000 --random-output-len 1000")
    return content


def get_section_value(content: str, section: str, key: str) -> Optional[str]:
    """Read existing value of a key from a config section. Returns None if not found."""
    section_pattern = rf'\[{re.escape(section)}\](.*?)(?=\[|$)'
    match = re.search(section_pattern, content, re.DOTALL)
    if not match:
        return None
    key_pattern = rf'^{re.escape(key)}\s*=\s*["\']?([^"\'\n]+)["\']?\s*$'
    key_match = re.search(key_pattern, match.group(1), re.MULTILINE)
    return key_match.group(1).strip() if key_match else None


def update_config_section(content: str, section: str, key: str, value: Any) -> str:
    """Update value in specific config section"""
    # Find config section
    section_pattern = rf'\[{re.escape(section)}\](.*?)(?=\[|$)'
    match = re.search(section_pattern, content, re.DOTALL)

    if not match:
        return content

    section_content = match.group(1)

    # Update value within section
    if isinstance(value, str):
        escaped_value = value.replace('\\', '\\\\').replace('"', '\\"')
        value_str = f'"{escaped_value}"'
    else:
        value_str = str(value)

    key_pattern = rf'^{re.escape(key)}\s*=\s*.+$'
    key_replacement = f'{key} = {value_str}'

    if re.search(key_pattern, section_content, re.MULTILINE):
        new_section = re.sub(key_pattern, key_replacement, section_content, flags=re.MULTILINE)
        content = content.replace(section_content, new_section)
    else:
        # Key doesn't exist, add to end of section
        new_section = section_content.rstrip() + f'\n{key} = {value_str}\n'
        content = content.replace(section_content, new_section)

    return content


def generate_target_field_block(
    name: str,
    config_position: str,
    dtype: str,
    min_val=None,
    max_val=None,
    value=None,
    dtype_param=None,
    enum_values=None,
    factories_config=None,
) -> str:
    """Generate target_field configuration block

    Args:
        name: Parameter name
        config_position: Config position (usually "env")
        dtype: Parameter type
        min_val: Minimum value (search parameter)
        max_val: Maximum value (search parameter)
        value: Fixed value (fixed parameter)
        dtype_param: Type parameter (ratio/factories/times)
        enum_values: Enum values list
        factories_config: factories config JSON

    Returns:
        TOML formatted config block string
    """
    lines = ['[[target_field]]']
    lines.append(f'name = "{name}"')
    lines.append(f'config_position = "{config_position}"')

    # Generate different fields based on type
    if dtype == "enum" and enum_values:
        lines.append('dtype = "enum"')

        # Handle enum values: detect JSON objects and add shell single quotes
        import json

        try:
            enum_list = json.loads(enum_values)
            processed_values = []
            for v in enum_list:
                if isinstance(v, str) and '{' in v and '}' in v:
                    # Contains JSON object, need to wrap JSON part with single quotes
                    # Example: --config {"key": "value"} -> --config '{"key": "value"}'
                    import re

                    # Match JSON object part and wrap with single quotes
                    processed = re.sub(r'(\{.*\})', r"'\1'", v)
                    processed_values.append(processed)
                else:
                    processed_values.append(v)
            # Re-serialize to TOML format
            toml_value = json.dumps(processed_values, ensure_ascii=False)
            lines.append(f'dtype_param = {toml_value}')
        except (json.JSONDecodeError, TypeError):
            lines.append(f'dtype_param = {enum_values}')

        # For string enums, must specify value, otherwise Pydantic will use default float 0.0 causing type error
        if value is not None:
            # Handle value containing JSON format
            if isinstance(value, str) and '{' in value and '}' in value:
                import re

                value = re.sub(r'(\{.*\})', r"'\1'", value)
            lines.append(f'value = {value if isinstance(value, int) else repr(value)}')
        else:
            # Parse enum_values and use first non-empty value as default
            try:
                enum_list = json.loads(enum_values)
                if enum_list:
                    # Prefer first non-empty value as default
                    default_val = None
                    for v in enum_list:
                        if v:  # Non-empty value
                            default_val = v
                            break
                    # If all values are empty, use first value
                    if default_val is None:
                        default_val = enum_list[0]
                    # Handle JSON format values
                    if isinstance(default_val, str) and '{' in default_val and '}' in default_val:
                        import re

                        default_val = re.sub(r'(\{.*\})', r"'\1'", default_val)
                    if isinstance(default_val, str):
                        lines.append(f'value = {repr(default_val)}')
                    else:
                        lines.append(f'value = {default_val}')
            except (json.JSONDecodeError, TypeError):
                pass
    elif dtype == "ratio":
        lines.append(f'min = {min_val if min_val is not None else 0}')
        lines.append(f'max = {max_val if max_val is not None else 1}')
        lines.append('dtype = "ratio"')
        if dtype_param:
            lines.append(f'dtype_param = "{dtype_param}"')
        if value is not None:
            lines.append(f'value = {value}')
    elif dtype == "factories":
        lines.append('min = 0')
        lines.append('max = 0')
        lines.append('dtype = "factories"')
        if factories_config:
            import json

            if isinstance(factories_config, str):
                config_dict = json.loads(factories_config)
            else:
                config_dict = factories_config
            lines.append(
                f'dtype_param = {{target_name = "{config_dict.get("target_name")}", product = {config_dict.get("product")}, dtype = "{config_dict.get("dtype", "int")}"}}'
            )
        if value is not None:
            lines.append(f'value = {value}')
    elif dtype == "times":
        lines.append('dtype = "times"')
        if dtype_param:
            import json

            if isinstance(dtype_param, str):
                config_dict = json.loads(dtype_param)
            else:
                config_dict = dtype_param
            lines.append(
                f'dtype_param = {{target_name = "{config_dict.get("target_name")}", product = {config_dict.get("product")}, dtype = "{config_dict.get("dtype", "int")}"}}'
            )
        if value is not None:
            lines.append(f'value = {value}')
    elif dtype == "range":
        # range type: requires min, max, dtype_param (step)
        if min_val is None or max_val is None:
            raise ValueError("range type must specify --min and --max")
        if dtype_param is None:
            raise ValueError("range type must specify --dtype-param as step")
        lines.append(f'min = {min_val}')
        lines.append(f'max = {max_val}')
        lines.append('dtype = "range"')
        lines.append(f'dtype_param = {dtype_param}')
        if value is not None:
            lines.append(f'value = {value}')
    else:
        # int, float, bool, str
        if min_val is not None:
            lines.append(f'min = {min_val}')
        if max_val is not None:
            lines.append(f'max = {max_val}')
        lines.append(f'dtype = "{dtype}"')
        if value is not None:
            if dtype == "str":
                escaped_value = str(value).replace('\\', '\\\\').replace('"', '\\"')
                lines.append(f'value = "{escaped_value}"')
            elif dtype == "bool":
                lines.append(f'value = {str(value).lower()}')
            else:
                lines.append(f'value = {value}')

    return '\n'.join(lines)


def find_last_target_field_position(content: str, engine: str) -> int:
    """Find end position of the last [[engine.target_field]] block for the engine

    Returns:
        End position of last target_field block, -1 if not found
    """
    # Exact match [[engine.target_field]], not match [[engine_xxx.target_field]]
    pattern = rf'\[\[{re.escape(engine)}\.target_field\]\]'
    matches = list(re.finditer(pattern, content))

    if not matches:
        return -1

    # Find last match
    last_match = matches[-1]

    # From last [[engine.target_field]], find end position of this block
    # Block ends at next [[xxx]] or [xxx] or # --- comment separator or end of file
    start_pos = last_match.start()
    remaining = content[start_pos:]

    # Find next config section start ([[xxx]] or [xxx] or # --- comment separator)
    next_section = re.search(r'\n(?=\[\[|\[(?!\[)|# -+)', remaining[1:])  # Skip current [[

    if next_section:
        return start_pos + 1 + next_section.start()
    else:
        # No next section, return end of file position (excluding trailing newlines)
        return len(content.rstrip()) + 1


def find_engine_command_end(content: str, engine: str) -> int:
    """Find end position of [engine.command] section (before next section or comment separator)

    Returns:
        Section end position, -1 if not found
    """
    # Find [engine.command] section
    pattern = rf'^\[{re.escape(engine)}\.command\]'
    match = re.search(pattern, content, re.MULTILINE)
    if not match:
        return -1

    start_pos = match.start()
    remaining = content[start_pos:]

    # Find next section separator ([[xxx]]/[xxx] or comment separator line # ----)
    next_section = re.search(r'\n(?=\[\[|\[(?!\[)|# -+)', remaining[1:])

    if next_section:
        return start_pos + 1 + next_section.start()
    else:
        return len(content.rstrip())


def find_engine_section_position(content: str, engine: str) -> int:
    """Find position of [engine] or [engine.command] section

    Returns:
        Section start position, -1 if not found
    """
    # Exact match [engine] or [engine.xxx], avoid matching [engine_benchmark]
    pattern = rf'^\[{re.escape(engine)}(?:\.[^\]]+)?\]'
    match = re.search(pattern, content, re.MULTILINE)
    return match.start() if match else -1


def find_target_field_by_name(content: str, engine: str, param_name: str) -> Optional[tuple]:
    """Find target_field block with specified parameter name

    Returns:
        (start_pos, end_pos) tuple, None if not found
    """
    # Find target_field block containing the parameter
    pattern = rf'(\[\[{re.escape(engine)}\.target_field\]\][^\[]*?name\s*=\s*["\']{re.escape(param_name)}["\'][^\[]*?)'
    match = re.search(pattern, content, re.DOTALL)

    if match:
        start_pos = match.start()
        block = match.group(1)

        # Find block end position
        next_block = re.search(r'\n(?=\[\[|\[(?!\[))', content[start_pos + len(block) :])
        if next_block:
            end_pos = start_pos + len(block) + next_block.start()
        else:
            end_pos = len(content.rstrip()) + 1

        return (start_pos, end_pos)

    return None


def add_target_field(content: str, engine: str, block: str, force: bool = False) -> str:
    """Add target_field configuration block to config file

    Args:
        content: Config file content
        engine: Engine name (vllm, mindie, etc.)
        block: Configuration block content
        force: Whether to force overwrite existing parameter

    Returns:
        Updated config content
    """
    # Replace [[target_field]] with [[engine.target_field]]
    engine_block = block.replace('[[target_field]]', f'[[{engine}.target_field]]')

    # Extract parameter name
    name_match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', engine_block)
    if name_match:
        param_name = name_match.group(1)

        # Check if parameter already exists
        existing = find_target_field_by_name(content, engine, param_name)
        if existing:
            if force:
                # Force mode: delete old block, add new block
                start_pos, end_pos = existing
                content = content[:start_pos] + engine_block + '\n' + content[end_pos:]
                print(f"  ✓ Overwritten existing parameter: {param_name}")
                return content
            else:
                # Non-force mode: check if there are fields to update
                print(f"  ⚠ Parameter {param_name} already exists in config, use --force to overwrite")
                # Should update existing block instead of returning directly
                start_pos, end_pos = existing
                existing_block = content[start_pos:end_pos]
                updated_block = update_existing_target_field(existing_block, engine_block)
                content = content[:start_pos] + updated_block + content[end_pos:]
                print(f"  ✓ Updated existing parameter block: {param_name}")
                return content

    # Strategy: find position after the last [[engine.target_field]] block, insert after it
    last_field_pos = find_last_target_field_position(content, engine)

    if last_field_pos > 0:
        # Insert after last target_field block
        content = content[:last_field_pos] + '\n' + engine_block + '\n' + content[last_field_pos:]
    else:
        # No existing target_field found, insert after [engine.command] section
        command_end_pos = find_engine_command_end(content, engine)
        if command_end_pos > 0:
            # Insert at end of [engine.command] section (before comment separator)
            content = content[:command_end_pos] + '\n' + engine_block + '\n' + content[command_end_pos:]
        else:
            # Find [engine] section
            section_pos = find_engine_section_position(content, engine)
            if section_pos >= 0:
                remaining = content[section_pos:]
                next_line = remaining.find('\n')
                if next_line >= 0:
                    after_header = section_pos + next_line + 1
                    next_section = re.search(r'^(?=\[|# -+)', content[after_header:], re.MULTILINE)
                    if next_section:
                        insert_pos = after_header + next_section.start()
                    else:
                        insert_pos = len(content.rstrip())
                    content = content[:insert_pos] + '\n' + engine_block + '\n' + content[insert_pos:]
                else:
                    content = content.rstrip() + '\n\n' + engine_block + '\n'
            else:
                # Insert at end of file
                content = content.rstrip() + '\n\n' + engine_block + '\n'

    return content


def update_existing_target_field(existing_block: str, new_block: str) -> str:
    """Update existing target_field block

    Args:
        existing_block: Existing config block content
        new_block: New config block content

    Returns:
        Updated config block
    """
    # Parse fields from new block
    new_fields = {}
    for line in new_block.split('\n'):
        line = line.strip()
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            new_fields[key] = value

    # Update fields in existing block
    result = existing_block
    for key, new_value in new_fields.items():
        if key == 'name':
            # name field is not updated
            continue

        # Build replacement pattern
        pattern = rf'^{re.escape(key)}\s*=\s*.+$'
        replacement = f'{key} = {new_value}'

        if re.search(pattern, result, re.MULTILINE):
            result = re.sub(pattern, replacement, result, flags=re.MULTILINE)
        else:
            # Field doesn't exist, add to end of block
            result = result.rstrip() + '\n' + f'{key} = {new_value}'

    return result


def update_others_with_param(
    content: str, engine: str, param_name: str, cli_arg: str = None, force: bool = False
) -> str:
    """Update parameter reference in others

    Args:
        content: Config file content
        engine: Engine name (vllm, mindie, etc.)
        param_name: Parameter name (e.g., TEST)
        cli_arg: CLI argument name (e.g., --test), if None, auto-generated from param_name
                  If empty string "", only add $PARAM_NAME (for enum values containing full parameters)
        force: Whether to force update existing parameter reference

    Returns:
        Updated config content
    """
    env_ref = f"${param_name}"

    # Handle cli_arg
    if cli_arg is None:
        # Default: convert parameter name to lowercase and add -- prefix
        cli_arg = f"--{param_name.lower().replace('_', '-')}"
        param_str = f"{cli_arg} {env_ref}"
    elif cli_arg == "":
        # Empty string means only add variable reference (for enum values containing full parameters)
        param_str = env_ref
    else:
        param_str = f"{cli_arg} {env_ref}"

    # Find others in [engine.command] section
    section_pattern = rf'(\[{re.escape(engine)}\.command\].*?)(others\s*=\s*")(.*?)("\s*)(?=\n|$)'
    match = re.search(section_pattern, content, re.DOTALL)

    if match:
        prefix = match.group(1)
        others_key = match.group(2)
        others_value = match.group(3)
        suffix = match.group(4)

        # Check if parameter reference already exists
        if env_ref in others_value:
            if not force:
                print(f"  ⚠ {env_ref} already in others, skip (use --force to force normalize)")
                return content

            # Force mode: remove existing same-name reference, then append normalized format
            ref_pattern = rf'(?<!\S)(?:--[^\s"]+\s+)?{re.escape(env_ref)}(?!\S)'
            normalized = re.sub(ref_pattern, '', others_value)
            normalized = re.sub(r'\s+', ' ', normalized).strip()
            new_others_value = f"{normalized} {param_str}".strip()
            print(f"  ✓ Normalized parameter in [{engine}.command].others: {param_str}")
        else:
            # Add parameter to end of others
            new_others_value = f"{others_value.rstrip()} {param_str}".strip()
            print(f"  ✓ Added to [{engine}.command].others: {param_str}")

        new_section = prefix + others_key + new_others_value + suffix

        # Replace original matched content
        content = content[: match.start()] + new_section + content[match.end() :]
    else:
        print(f"  ⚠ [{engine}.command].others not found, please manually add {param_str}")

    return content


def add_search_param(content: str, args) -> str:
    """Add search parameter"""
    block = generate_target_field_block(
        name=args.param_name,
        config_position=args.config_position or "env",
        dtype=args.dtype,
        min_val=args.min,
        max_val=args.max,
        value=args.value,
        dtype_param=args.dtype_param,
        enum_values=args.enum_values,
        factories_config=args.factories_config,
    )
    content = add_target_field(content, args.engine, block, force=getattr(args, 'force', False))

    # Auto add parameter reference in others
    if hasattr(args, 'cli_arg') and args.cli_arg is not None:
        content = update_others_with_param(content, args.engine, args.param_name, args.cli_arg, force=getattr(args, 'force', False))
    else:
        content = update_others_with_param(content, args.engine, args.param_name, force=getattr(args, 'force', False))

    return content


def add_fixed_param(content: str, args) -> str:
    """Add fixed parameter"""
    block = generate_target_field_block(
        name=args.param_name, config_position=args.config_position or "env", dtype=args.dtype, value=args.value
    )
    return add_target_field(content, args.engine, block, force=getattr(args, 'force', False))


def set_vllm_command(content: str, args) -> str:
    """Configure VLLM command parameters"""
    updates = {}
    common_updates = {}
    if args.model:
        updates['model'] = args.model
        common_updates['model'] = args.model
    if args.served_name:
        updates['served_model_name'] = args.served_name
        common_updates['served_model_name'] = args.served_name
    if args.host:
        updates['host'] = args.host
        common_updates['host'] = args.host
    if args.port:
        updates['port'] = str(args.port)
        common_updates['port'] = str(args.port)
    if args.others:
        updates['others'] = args.others

    for key, value in updates.items():
        content = update_config_section(content, "vllm.command", key, value)

    # Sync model, served_model_name, host, port to vllm_benchmark as well
    for key, value in common_updates.items():
        content = update_config_section(content, "vllm_benchmark.command", key, value)

    if common_updates:
        print("  ⚡ Synced model/served_model_name/host/port to vllm_benchmark.command")

    return content


def set_ais_bench_config(content: str, args) -> str:
    """Configure ais_bench benchmark parameters"""
    updates = {}
    if args.models:
        updates['models'] = args.models
    if args.datasets:
        updates['datasets'] = args.datasets
    if args.mode:
        updates['mode'] = args.mode
    if args.ais_num_prompts:
        updates['num_prompts'] = args.ais_num_prompts
    else:
        existing = get_section_value(content, "ais_bench.command", "num_prompts")
        if existing and existing not in ("0", ""):
            print(f"  ⚠ Keeping existing num_prompts: {existing}")
        else:
            updates['num_prompts'] = 150
            print("  📝 Using default num_prompts: 150")

    for key, value in updates.items():
        content = update_config_section(content, "ais_bench.command", key, value)

    return content


def set_vllm_benchmark_config(content: str, args) -> str:
    """Configure vllm_benchmark benchmark parameters"""
    updates = {}
    common_updates = {}
    if args.model:
        updates['model'] = args.model
        common_updates['model'] = args.model
    if args.served_name:
        updates['served_model_name'] = args.served_name
        common_updates['served_model_name'] = args.served_name
    if args.host:
        updates['host'] = args.host
        common_updates['host'] = args.host
    if args.port:
        updates['port'] = str(args.port)
        common_updates['port'] = str(args.port)
    if args.dataset_name:
        updates['dataset_name'] = args.dataset_name
    else:
        existing = get_section_value(content, "vllm_benchmark.command", "dataset_name")
        if existing and existing not in ("", "dataset_name"):
            print(f"  ⚠ Keeping existing dataset_name: {existing}")
        else:
            updates['dataset_name'] = "random"
            print("  📝 Using default dataset_name: random")
    if args.vllm_num_prompts:
        updates['num_prompts'] = args.vllm_num_prompts
    else:
        existing = get_section_value(content, "vllm_benchmark.command", "num_prompts")
        if existing and existing not in ("0", ""):
            print(f"  ⚠ Keeping existing num_prompts: {existing}")
        else:
            updates['num_prompts'] = 100
            print("  📝 Using default num_prompts: 100")
    if args.others:
        updates['others'] = args.others
    else:
        # No default for workload-specific params; user must provide based on actual business
        print("  ⚠ --others 未指定，请根据业务负载显式指定 benchmark 参数。")
        print("    示例: --others '--random-input-len 3500 --random-output-len 1500'")

    for key, value in updates.items():
        content = update_config_section(content, "vllm_benchmark.command", key, value)

    # Sync model, served_model_name, host, port back to vllm.command as well
    for key, value in common_updates.items():
        content = update_config_section(content, "vllm.command", key, value)

    if common_updates:
        print("  ⚡ Synced model/served_model_name/host/port to vllm.command")

    return content


def main():
    parser = argparse.ArgumentParser(
        description="Auto configure msmodeling optix config.toml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Set scenario
  %(prog)s --scenario standard --engine vllm

  # Add search parameter
  %(prog)s --add-search-param --engine vllm --param-name MAX_BATCH_SIZE \\
      --min 10 --max 400 --dtype int --value 100

  # Add fixed parameter
  %(prog)s --add-fixed-param --engine vllm --param-name MODEL_PATH \\
      --value "/model" --dtype str

  # Configure VLLM command
  %(prog)s --set-vllm-command --model /path/to/model --served-name my-model

  # Configure ais_bench
  %(prog)s --set-ais-bench --models /path/to/models.yaml --datasets /path/to/datasets.yaml
        """,
    )

    # Operation mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--scenario", choices=list(SCENARIOS.keys()), help="Optimization scenario template")
    mode_group.add_argument("--add-search-param", action="store_true", help="Add search parameter (with range)")
    mode_group.add_argument("--add-fixed-param", action="store_true", help="Add fixed parameter (without range)")
    mode_group.add_argument("--set-vllm-command", action="store_true", help="Configure VLLM command parameters")
    mode_group.add_argument("--set-ais-bench", action="store_true", help="Configure ais_bench benchmark parameters")
    mode_group.add_argument(
        "--set-vllm-benchmark", action="store_true", help="Configure vllm_benchmark benchmark parameters"
    )

    # Common parameters
    parser.add_argument(
        "--engine",
        choices=["mindie", "vllm"],
        default="mindie",
        help="Inference framework (default: mindie)",
    )
    parser.add_argument(
        "--config-path",
        default="optix/config.toml",
        help="Config file path (default: optix/config.toml)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing parameters")

    # Scenario related parameters
    parser.add_argument("--benchmark", choices=["ais_bench", "vllm_benchmark"], help="Benchmark tool")
    parser.add_argument("--host", default="127.0.0.1", help="Service host")
    parser.add_argument("--port", type=str, default="8000", help="Service port")
    parser.add_argument("--ttft-slo", type=float, help="First token latency limit (seconds)")
    parser.add_argument("--tpot-slo", type=float, help="Non-first token latency limit (seconds)")
    parser.add_argument("--time-budget", help="Time budget (e.g.: 8h, 30m)")
    parser.add_argument(
        "--pso-top-k",
        type=int,
        choices=[3, 4, 5],
        metavar="3-5",
        help="Set pso_top_k for deep-optimize scenario (recommended: 3-5). Auto-applied when using --scenario deep-optimize.",
    )

    # Model parameters
    parser.add_argument("--model", help="Model path")
    parser.add_argument("--served-name", help="Model served name")

    # Parameter configuration related
    parser.add_argument("--param-name", help="Parameter name")
    parser.add_argument("--config-position", default="env", help="Config position (default: env)")
    parser.add_argument(
        "--dtype",
        choices=["int", "float", "bool", "str", "enum", "ratio", "factories", "times", "range"],
        default="int",
        help="Parameter type",
    )
    parser.add_argument("--min", type=float, help="Minimum value")
    parser.add_argument("--max", type=float, help="Maximum value")
    parser.add_argument("--value", help="Fixed value")
    parser.add_argument("--dtype-param", help="Type parameter (ratio/factories/times)")
    parser.add_argument("--enum-values", help="Enum values list JSON (e.g.: [1,2,4,8])")
    parser.add_argument("--factories-config", help="factories config JSON")
    parser.add_argument("--cli-arg", help="CLI argument name (e.g., --test), used to add reference in others")

    # VLLM command parameters
    parser.add_argument("--others", help="Other VLLM parameters")

    # ais_bench parameters
    parser.add_argument("--models", help="AISBench models config path")
    parser.add_argument("--datasets", help="AISBench datasets config path")
    parser.add_argument(
        "--mode", choices=["perf", "concurrency", "throughput"], default="perf", help="AISBench run mode"
    )
    parser.add_argument("--ais-num-prompts", type=int, dest="ais_num_prompts", help="AISBench number of prompts")

    # vllm_benchmark parameters
    parser.add_argument("--dataset-name", help="vllm_benchmark dataset name")
    parser.add_argument("--vllm-num-prompts", type=int, dest="vllm_num_prompts", help="vllm_benchmark prompt count")

    args = parser.parse_args()

    # Read config file
    config_path = Path(args.config_path)
    if not config_path.exists():
        print(f"✗ Config file does not exist: {config_path}")
        # Try to find in current directory
        alt_path = Path.cwd() / args.config_path
        if alt_path.exists():
            config_path = alt_path
            print(f"Found config file: {config_path}")
        else:
            sys.exit(1)

    print(f"Reading config file: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Execute different operations based on mode
    if args.scenario:
        print(f"\nApply scenario: {SCENARIOS[args.scenario]['description']}")
        new_content = apply_scenario_config(content, args.scenario, args)

        # Show scenario change summary
        print("\nConfig changes:")
        print(f"  - n_particles: {SCENARIOS[args.scenario].get('n_particles')}")
        print(f"  - iters: {SCENARIOS[args.scenario].get('iters')}")
        print(f"  - engine: {args.engine}")
        if args.scenario == "deep-optimize":
            pso_top_k = getattr(args, "pso_top_k", None) or 4
            print(f"  - pso_top_k: {pso_top_k} (fine-tuning seed: 3-5 recommended)")

    elif args.add_search_param:
        if not args.param_name:
            print("✗ --param-name is required")
            sys.exit(1)
        if args.value is None:
            print("✗ --value is required (default parameter value, for enum type should be one of the enum values)")
            sys.exit(1)
        print(f"\nAdd search parameter: {args.param_name}")
        new_content = add_search_param(content, args)
        print(f"  - Type: {args.dtype}")
        print(f"  - Default value: {args.value}")
        if args.min is not None and args.max is not None:
            print(f"  - Range: {args.min} ~ {args.max}")

    elif args.add_fixed_param:
        if not args.param_name:
            print("✗ --param-name is required")
            sys.exit(1)
        print(f"\nAdd fixed parameter: {args.param_name}")
        new_content = add_fixed_param(content, args)
        print(f"  - Type: {args.dtype}")
        print(f"  - Value: {args.value}")

    elif args.set_vllm_command:
        print("\nConfigure VLLM command parameters:")
        new_content = set_vllm_command(content, args)
        if args.model:
            print(f"  - model: {args.model}")
        if args.served_name:
            print(f"  - served_model_name: {args.served_name}")
        if args.host:
            print(f"  - host: {args.host}")
        if args.port:
            print(f"  - port: {args.port}")

    elif args.set_ais_bench:
        print("\nConfigure ais_bench benchmark parameters:")
        new_content = set_ais_bench_config(content, args)
        if args.models:
            print(f"  - models: {args.models}")
        if args.datasets:
            print(f"  - datasets: {args.datasets}")
        if args.mode:
            print(f"  - mode: {args.mode}")
        if args.ais_num_prompts:
            print(f"  - num_prompts: {args.ais_num_prompts}")
    elif args.set_vllm_benchmark:
        print("\nConfigure vllm_benchmark benchmark parameters:")
        new_content = set_vllm_benchmark_config(content, args)
        if args.model:
            print(f"  - model: {args.model}")
        if args.served_name:
            print(f"  - served_model_name: {args.served_name}")
        if args.host:
            print(f"  - host: {args.host}")
        if args.port:
            print(f"  - port: {args.port}")
        if args.dataset_name:
            print(f"  - dataset_name: {args.dataset_name}")
        if args.vllm_num_prompts:
            print(f"  - num_prompts: {args.vllm_num_prompts}")
        if args.others:
            print(f"  - others: {args.others}")
    else:
        print("✗ Please specify operation mode")
        sys.exit(1)

    if args.dry_run:
        print("\n[DRY RUN] Preview modified config:")
        print("=" * 50)
        # Show first 50 lines
        lines = new_content.split('\n')[:50]
        for line in lines:
            print(line)
        if len(new_content.split('\n')) > 50:
            print("... (showing first 50 lines only)")
        print("=" * 50)
        print("Not actually written to file")
    else:
        # Backup original file
        backup_path = config_path.with_suffix('.toml.bak')
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\nOriginal config backed up: {backup_path}")

        # Write new config
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"✓ Config updated: {config_path}")

        print("\nNext step:")
        print(f"  msmodeling optix -e {args.engine}")


if __name__ == "__main__":
    main()
