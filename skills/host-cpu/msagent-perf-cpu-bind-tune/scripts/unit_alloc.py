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

"""unit 内角色分核：compute / comm / io / main。

角色来自线程 comm 分类（roles.py），按连续核集切分：
- 通信/NPU runtime 独立核；
- ZMQ/tokenizer/dataloader 等 IO 独立核；
- 主线程保留 1 核；
- 其余为计算主体。

绑定语义（Linux 只有线程级亲和）：
- main：taskset/亲和绑定 tid==pid（主线程）；
- comm/io：执行时枚举 /proc/<pid>/task，按 comm 归类后细绑；
- compute：默认绑定到 compute 核集（与 OMP_NUM_THREADS 口径一致，OMP_PROC_BIND
  在该核集内自绑）；仅当外部显式标 inherit 时才跳过。
"""

import roles
import topology

# ---------------------------------------------------------------------------
# 角色核数阈值（A2 8×910B2 实机线程画像校准：vLLM worker 量级约
# compute 47 / comm 24 / io 4 / main 15 线程）。
# 换机型或新框架画像后在此统一调整，勿在分配逻辑内散落魔数；
# 对应 RFC §3.5 的「线程画像校准」待办。
# ---------------------------------------------------------------------------
COMM_HEAVY_THREADS = 12    # comm 线程数 ≥ 该值 → 2 核，否则 1 核
IO_HEAVY_THREADS = 8       # io 线程数 ≥ 该值 → 2 核，否则 1 核
MAIN_HEAVY_THREADS = 20    # main 线程数 ≥ 该值且 unit 核数充足 → 2 核
MAIN_HEAVY_MIN_UNIT = 12   # main 给 2 核所需的 unit 最小核数
# 事前/线程未观测时按框架已知行为预留角色核：
# 推理引擎与训练框架必有通信面；分词/数据面仅推理与 RL 调度存在。
FRAMEWORK_RESERVED_COMM = ("vllm", "sglang", "megatron", "mindspeed", "verl")
FRAMEWORK_RESERVED_IO = ("vllm", "sglang", "verl")


def _count_role_cores(roles_count, framework, n):
    """按线程画像与框架预留计算 comm / io / main 的初始核数。"""
    comm_threads = roles_count.get("comm", 0)
    io_threads = roles_count.get("io", 0)
    main_threads = roles_count.get("main", 0)

    comm_cores = 2 if comm_threads >= COMM_HEAVY_THREADS else (1 if comm_threads > 0 else 0)
    if framework in FRAMEWORK_RESERVED_COMM and comm_cores == 0:
        comm_cores = 1
    io_cores = 2 if io_threads >= IO_HEAVY_THREADS else (1 if io_threads > 0 else 0)
    if framework in FRAMEWORK_RESERVED_IO and io_cores == 0:
        io_cores = 1
    main_cores = 2 if (main_threads >= MAIN_HEAVY_THREADS and n >= MAIN_HEAVY_MIN_UNIT) else 1
    return comm_cores, io_cores, main_cores


def _shrink_role_cores(comm_cores, io_cores, main_cores, n):
    """按优先级收缩角色核数，返回收缩后的 comm / io / main / compute。"""
    # 核数不足时按优先级收缩：main -> io -> comm -> compute 至少 1
    while n > 0 and (comm_cores + io_cores + main_cores) >= n:
        if main_cores > 1:
            main_cores -= 1
        elif io_cores > 1:
            io_cores -= 1
        elif comm_cores > 1:
            comm_cores -= 1
        elif io_cores > 0:
            io_cores -= 1
        elif comm_cores > 0:
            comm_cores -= 1
        else:
            main_cores = 0
            break

    compute_cores = n - comm_cores - io_cores - main_cores
    if compute_cores < 1:
        compute_cores = 1
        if main_cores > 0:
            main_cores -= 1
        elif io_cores > 0:
            io_cores -= 1
        elif comm_cores > 0:
            comm_cores -= 1
    return comm_cores, io_cores, main_cores, compute_cores


def _slice_role_cpus(cpus, comm_cores, io_cores, main_cores, compute_cores):
    """按 compute / comm / io / main 顺序切分连续核集。"""
    n = len(cpus)
    idx = 0
    slices = {}
    for name, count in (("compute", compute_cores), ("comm", comm_cores),
                        ("io", io_cores), ("main", main_cores)):
        if count <= 0:
            slices[name] = []
            continue
        slices[name] = cpus[idx:idx + count]
        idx += count
    # 浮点误差兜底：剩余核并入 compute
    if idx < n and slices.get("compute") is not None:
        slices["compute"].extend(cpus[idx:])
    return slices


def _build_role_map(slices, roles_count):
    """角色核集转 cpus / cores / threads 描述。"""
    role_map = {}
    for name, role_cpus in slices.items():
        role_map[name] = {
            "cpus": topology.cpus_to_str(role_cpus),
            "cores": len(role_cpus),
            "threads": roles_count.get(name, 0),
        }
    return role_map


def _thread_bindings(slices, main_cpuset, compute_cpuset):
    """生成执行期线程绑定规则：main / comm / io / compute。"""
    return [
        {
            "role": "main",
            "match_comm": [],
            "cpuset": main_cpuset,
            "apply": "exec",
            "note": "主线程（tid == pid）",
        },
        {
            "role": "comm",
            "match_comm": roles.keywords("comm"),
            "cpuset": slices.get("comm") and topology.cpus_to_str(slices["comm"]) or "",
            "apply": "exec" if slices.get("comm") else "skip",
            "note": "",
        },
        {
            "role": "io",
            "match_comm": roles.keywords("io"),
            "cpuset": slices.get("io") and topology.cpus_to_str(slices["io"]) or "",
            "apply": "exec" if slices.get("io") else "skip",
            "note": "",
        },
        {
            "role": "compute",
            "match_comm": roles.keywords("compute"),
            "cpuset": compute_cpuset,
            "apply": "exec" if compute_cpuset else "skip",
            "note": "绑定到 compute 核集；OMP_PROC_BIND=close 在其内自绑，与 OMP_NUM_THREADS 核数一致",
        },
    ]


def allocate(unit, cpus):
    cpus = sorted(cpus)
    n = len(cpus)
    roles_count = unit.get("thread_roles") or {}
    framework = unit.get("framework") or "unknown"

    comm_cores, io_cores, main_cores = _count_role_cores(roles_count, framework, n)
    comm_cores, io_cores, main_cores, compute_cores = _shrink_role_cores(
        comm_cores, io_cores, main_cores, n)
    slices = _slice_role_cpus(cpus, comm_cores, io_cores, main_cores, compute_cores)
    # 极端收缩下 main 为 0：主线程跟随 compute 首核（先回填，保证 roles 与 thread_binding 一致）
    if not slices.get("main") and cpus:
        slices["main"] = list(slices.get("compute") or cpus)[:1]

    role_map = _build_role_map(slices, roles_count)

    unit["cpuset_cpus"] = cpus
    unit["cpuset"] = topology.cpus_to_str(cpus)
    main_cpus = slices.get("main") or []
    compute_cpus = slices.get("compute") or []
    main_cpuset = topology.cpus_to_str(main_cpus)
    compute_cpuset = topology.cpus_to_str(compute_cpus)
    work_cpus = sorted(set(compute_cpus) | set(main_cpus))
    unit["main_cpuset"] = main_cpuset
    unit["compute_cpuset"] = compute_cpuset
    unit["work_cpuset"] = topology.cpus_to_str(work_cpus) if work_cpus else unit["cpuset"]
    unit["thread_binding"] = _thread_bindings(slices, main_cpuset, compute_cpuset)
    unit["roles"] = role_map
    unit["env"] = {
        "OMP_NUM_THREADS": str(max(1, role_map["compute"]["cores"])),
        "OMP_PROC_BIND": "close",
        "OMP_PLACES": "cores",
    }
    return unit
