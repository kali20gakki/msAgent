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

"""HTML 可视化报告：NUMA 拓扑 / unit→核分配 / 偏差 / 验证。

只读 plan + topology，不重新判定；结论数字来自 plan。
"""

import html
import os
import shlex
import string
import time

import topology


_ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources")


def _load_asset(name):
    """读取 resources/ 下的前端资源；去掉文件末尾换行，保持与内联版本一致的输出。"""
    with open(os.path.join(_ASSETS_DIR, name), "r", encoding="utf-8") as f:
        return f.read().rstrip("\n")


def _e(value):
    """HTML 转义：所有来自 /proc / plan 的动态字段（comm/主机名/设备名等）
    必须经本函数后再拼入 HTML，防止进程名中的尖括号/引号破坏页面结构。"""
    return html.escape(str(value if value is not None else ""))


def _e_conclusion(text):
    """结论文本转义，同时保留结论自带的 <b> 强调标签。"""
    return _e(text).replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")


def _cpus(text):
    return topology.parse_cpulist(text)


def _role_map(plan):
    out = {}
    for unit in plan.get("units") or []:
        for role, item in (unit.get("roles") or {}).items():
            for cpu in _cpus(item.get("cpus")):
                out[cpu] = (unit.get("unit_id"), role, unit.get("framework"), unit.get("pid"))
    return out


def _zone_cpus(plan, node):
    z = (plan.get("zones") or {}).get(str(node)) or {}
    return z


MODE_ALIAS = topology.LOCALITY_ALIAS


def _mode(value):
    return MODE_ALIAS.get(value, value)


def _host_meta(topo, plan):
    """主机与采集元信息：明确单机视角与数据来源，不虚构集群信息。"""
    meta = topo.get("meta") or {}
    ts = meta.get("ts_ns")
    ts_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts / 1e9)) if ts else None
    numa_src = {"sysfs": "sysfs", "single_node_fallback": "单 node 降级（sysfs 不可见）"}.get(
        meta.get("numa_source"), meta.get("numa_source"))
    mode = {"host": "宿主机", "container": "容器", "cluster": "集群"}.get(
        plan.get("mode"), plan.get("mode"))
    items = [
        ("主机", meta.get("host") or plan.get("host")),
        ("部署形态", mode),
        ("互联形态", {"ub": "UB（灵衢）超节点", "hccs_supernode": "HCCS 超节点",
                    "hccs": "HCCS（卡间直连）", "pcie": "PCIe"}.get(
            plan.get("fabric") or meta.get("fabric"), plan.get("fabric") or meta.get("fabric"))),
        ("求解模式", {"topo_affinity": "topo_affinity（按 NPU 亲和）", "global_slice": "global_slice（CPU 池均分）"}.get(
            _mode(plan.get("locality_mode") or meta.get("locality_mode")),
            _mode(plan.get("locality_mode") or meta.get("locality_mode")))),
        ("亲和粒度", {"per_npu": "每卡独立", "group": "按组共享（多卡同一核区间）", "none": "无（与各 node 等距）"}.get(
            plan.get("affinity_granularity") or meta.get("affinity_granularity"),
            plan.get("affinity_granularity") or meta.get("affinity_granularity"))),
        ("NPU 粒度", _npu_granularity(meta)),
        ("内核", meta.get("kernel")),
        ("架构", meta.get("machine")),
        ("拓扑来源", numa_src),
        ("采集时间", ts_text),
    ]
    chips = "".join(
        '<span class="kv"><i>%s</i>%s</span>' % (_e(k), _e(v)) for k, v in items if v
    )
    return '<div class="meta">%s</div>' % chips


def _npu_granularity(meta):
    """NPU 粒度：仅双 die 机型标注封装内 die 对锁定同 node；单 die 不占位。"""
    pkg = meta.get("package_count")
    die = meta.get("die_count")
    try:
        pkg, die = int(pkg), int(die)
    except (TypeError, ValueError):
        return None
    if pkg <= 0 or die <= 0:
        return None
    per = max(1, die // pkg)
    if per > 1:
        return "%d 封装 × %d die（同封装 die 对锁定同 node）" % (pkg, per)
    return None


def _node_card(node, topo, plan):
    nid = node["node"]
    npus = [n for n in topo.get("npu") or [] if n.get("numa_node") == nid]
    nics = [n for n in topo.get("nic") or [] if n.get("numa_node") == nid]
    storage = [s for s in topo.get("storage") or [] if s.get("numa_node") == nid]
    units = [u for u in plan.get("units") or [] if u.get("numa_node") == nid]
    z = _zone_cpus(plan, nid)
    # 超节点形态下 NPU 无 node 归属，按本 node 承接的 unit 反推展示哪些 NPU 在此
    npu_ids = [n.get("id") for n in npus]
    for u in units:
        p = u.get("primary_npu")
        if p is not None and p not in npu_ids:
            npu_ids.append(p)
    devices = "".join(
        ['<span class="chip npu" title="NPU">%s</span>' % _e("NPU%s" % i) for i in npu_ids]
        + ['<span class="chip nic" title="网卡">%s</span>' % _e(n.get("name")) for n in nics]
        + ['<span class="chip storage" title="存储">%s</span>' % _e(s.get("name")) for s in storage]
    ) or '<span class="muted">无</span>'
    parts = [
        '<div class="node"><div class="head"><strong>Node %d</strong><span>%s GB · %d 核</span></div>' % (
            nid, _e(node.get("mem_gb")), len(node.get("cpus") or [])
        ),
        '<div class="row"><span class="k">设备</span>%s</div>' % devices,
        '<div class="row"><span class="k">业务核</span>'
        '<span class="chip cpu" title="可分配给 NPU 进程与线程的核">%s</span>'
        '<span class="muted">中断 %s · 隔离 %s</span></div>' % (
            _e(z.get("business") or "-"), _e(z.get("system_irq") or "-"), _e(z.get("interference") or "-")
        ),
        '<div class="row"><span class="k">单元</span>%s</div>' % ("".join(
            '<span class="chip unit" title="该 node 承接的求解单元（NPU%s 对应进程）">unit %s</span>' % (
                _e(u.get("primary_npu")), _e(u.get("unit_id"))
            ) for u in units
        ) or '<span class="muted">无</span>'),
        '</div>',
    ]
    return "".join(parts)


def _strip(node, role_map, plan):
    nid = node["node"]
    z = _zone_cpus(plan, nid)
    irq = set(_cpus(z.get("system_irq")))
    inter = set(_cpus(z.get("interference")))
    cpus = node.get("cpus") or []
    cols = min(32, max(8, len(cpus)))
    cells = []
    for cpu in cpus:
        info = role_map.get(cpu)
        if info:
            uid, role, fw, pid = info
            cls = role
            title = "CPU %d · unit %s · %s" % (cpu, uid, role)
            if pid is not None:
                title += " · pid %s" % pid
        elif cpu in irq:
            cls = "system"
            title = "CPU %d · system_irq" % cpu
        elif cpu in inter:
            cls = "interference"
            title = "CPU %d · interference" % cpu
        else:
            cls = "idle"
            title = "CPU %d · idle" % cpu
        cells.append('<div class="core %s" title="%s">%d</div>' % (cls, _e(title), cpu))
    return ('<div class="strip-block"><div class="strip-head"><strong>Node %d</strong>'
            '<span>业务 %s · 系统 %s · 干扰 %s</span></div>'
            '<div class="strip" style="grid-template-columns:repeat(%d,1fr)">%s</div></div>') % (
        nid, _e(z.get("business")), _e(z.get("system_irq")), _e(z.get("interference")),
        cols, "".join(cells)
    )


def _affinity_matrix(topo, plan):
    """设备 × node 亲和矩阵：本地 / 远端（相对时延）/ 不可观测。"""
    nodes = topo.get("nodes") or []
    if not nodes:
        return '<div class="unknownbox">未采集到 NUMA node（sysfs 不可见）</div>'
    nids = [n["node"] for n in nodes]
    dist = {n["node"]: (n.get("distance") or []) for n in nodes}

    head = ['<tr><th class="dev">设备</th><th class="dim">亲和</th>']
    for n in nodes:
        head.append('<th>Node%s<br><span class="muted">%s</span></th>' % (_e(n["node"]), _e(n.get("cpulist"))))
    head.append("</tr>")

    groups = [
        ("NPU", [(("NPU%s" % n.get("id")), n.get("numa_node"), n.get("affinity_source"),
                  n.get("affinity")) for n in (topo.get("npu") or [])]),
        ("网卡", [((n.get("name")), n.get("numa_node"), n.get("numa_source"), n.get("pci"))
                  for n in (topo.get("nic") or [])]),
        ("存储", [((s.get("name")), s.get("numa_node"), s.get("numa_source"), s.get("pci"))
                  for s in (topo.get("storage") or [])]),
    ]

    body = []
    mode = _mode(plan.get("locality_mode") or (topo.get("meta") or {}).get("locality_mode"))
    for title, items in groups:
        if not items:
            continue
        body.append('<tr><td class="dev" colspan="%d" style="background:#f8fafc;font-weight:800;font-size:10px;color:#475569">%s</td></tr>'
                    % (len(nids) + 2, _e(title)))
        for name, node, source, extra in items:
            equi = node is None and mode == "global_slice" and title == "NPU"
            if equi:
                aff = '<span class="chip cpu">与各 node 等距</span>'
            elif node is None:
                aff = '<span class="tag view_limited">不可观测</span>'
            else:
                aff = '<span class="chip cpu">node%s</span>' % _e(node)
            cells = ['<td class="dev"><b>%s</b>%s</td><td class="dim">%s</td>' % (
                _e(name),
                ('<br><span class="muted">%s</span>' % _e(extra)) if extra else "",
                aff,
            )]
            for nid in nids:
                if equi:
                    cells.append('<td class="cell equi" title="超节点/UB 形态：NPU 与各 node 距离相同">等距</td>')
                elif node is None:
                    cells.append('<td class="cell unknown" title="%s">?</td>' % _e(_src(source)))
                elif node == nid:
                    cells.append('<td class="cell local">本地</td>')
                else:
                    d = dist.get(node) or []
                    dd = d[nid] if 0 <= nid < len(d) else None
                    cells.append('<td class="cell remote">%s</td>' % (("远 %s" % dd) if dd is not None else "远端"))
            body.append("<tr>%s</tr>" % "".join(cells))

    if not body:
        return '<div class="unknownbox">未采集到 NPU / 网卡 / 存储设备</div>'
    legend = ('<div class="legend"><span><i style="background:var(--green-soft)"></i>本地 node</span>'
              '<span><i style="background:#f1f5f9"></i>远端（数字为相对时延）</span>'
              '<span><i style="background:var(--blue-soft)"></i>与各 node 等距（超节点/UB）</span>'
              '<span><i style="background:#fef3c7"></i>不可观测</span></div>')
    return ('%s<table class="matrix"><thead>%s</thead><tbody>%s</tbody></table>') % (
        legend, "".join(head), "".join(body)
    )


def _distance_table(topo):
    """node × node 相对时延矩阵（sysfs distance）。"""
    nodes = topo.get("nodes") or []
    if len(nodes) < 2:
        return ""
    nids = [n["node"] for n in nodes]
    dist = {n["node"]: (n.get("distance") or []) for n in nodes}
    head = ['<tr><th class="dev">node</th>'] + ['<th>%s</th>' % _e(n) for n in nids] + ["</tr>"]
    rows = []
    for a in nids:
        cells = ['<td class="dev"><b>node%s</b></td>' % _e(a)]
        for b in nids:
            vals = dist.get(a) or []
            dd = vals[b] if 0 <= b < len(vals) else None
            if a == b:
                cls = "diag"
            elif dd is not None and dd <= 12:
                cls = "near"
            elif dd is not None and dd > 20:
                cls = "far"
            else:
                cls = ""
            cells.append('<td class="v %s">%s</td>' % (cls, "" if dd is None else _e(dd)))
        rows.append("<tr>%s</tr>" % "".join(cells))
    return '<table class="matrix dist"><thead>%s</thead><tbody>%s</tbody></table>' % (
        "".join(head), "".join(rows)
    )


SOURCE_LABEL = {
    "npu_smi_topo": "npu-smi topo",
    "pci_local_cpulist": "PCI local_cpulist",
    "pci_numa": "PCI numa_node",
    "device_numa_node": "device/numa_node",
    "none": "无可用来源",
    "unknown": "无可用来源",
}


def _src(source):
    return SOURCE_LABEL.get(source or "", source or "无可用来源")


def _unknown_card(topo, plan=None):
    """亲和不可观测的设备清单：明确标注原因，避免被读成『无设备』。"""
    mode = _mode((plan or {}).get("locality_mode") or (topo.get("meta") or {}).get("locality_mode"))
    missing = []
    for n in topo.get("npu") or []:
        if n.get("numa_node") is None and mode != "global_slice":
            missing.append(("NPU%s" % n.get("id"), n.get("affinity_source") or "unknown",
                            "npu-smi info -t topo 无 CPU Affinity 列且 PCI numa_node 不可读"))
    for n in topo.get("nic") or []:
        if n.get("numa_node") is None:
            missing.append((n.get("name"), n.get("numa_source") or "unknown", "sysfs 未提供 device/numa_node"))
    for s in topo.get("storage") or []:
        if s.get("numa_node") is None:
            missing.append((s.get("name"), s.get("numa_source") or "unknown", "sysfs 未提供 device/numa_node"))
    if not missing:
        return ""
    items = "".join(
        "<li><b>%s</b> · %s</li>" % (_e(name), _e(reason))
        for name, src, reason in missing[:20]
    )
    more = "" if len(missing) <= 20 else "<li>其余 %d 项略</li>" % (len(missing) - 20)
    return ('<div class="unknownbox"><b>%d 项设备无法判定所属 node</b>（矩阵中记为 <code>?</code>，'
            '不参与本地/远端判定，也不影响 NPU 与网卡的判定）：<ul style="margin:6px 0 0;padding-left:18px">%s%s</ul></div>'
            ) % (len(missing), items, more)


def _runbook(plan):
    """§06 可复制执行 runbook：预演 -> 执行 -> 验证 -> 回滚。

    命令全部引用 plan 文件路径，不含 pid 等易变内容（apply 执行时才枚举
    /proc/<pid>/task）；pre_run 形态改为输出按单元的环境变量。
    """
    apply_cfg = plan.get("apply") or {}
    apply_cmd = (apply_cfg.get("command") or "").strip()
    rollback_cmd = (apply_cfg.get("rollback") or "").strip()
    verify_path = (plan.get("verify_script_path") or "").strip()
    lines = []
    if plan.get("form") == "in_run" and apply_cmd:
        lines.append("# 1) 预演：列出将执行的绑定动作，不修改配置")
        lines.append(apply_cmd + " --dry-run")
        lines.append("# 2) 执行：由管理员确认后运行（跨用户场景加 sudo 或以进程所属用户执行）")
        lines.append(apply_cmd)
        if verify_path:
            lines.append("# 3) 验证：全部通过表示配置已生效")
            lines.append("sh %s" % shlex.quote(verify_path))
        if rollback_cmd:
            lines.append("# 4) 回滚：恢复执行前的亲和设置")
            lines.append(rollback_cmd)
        lines.append("# 进程重启后需重新生成方案")
    else:
        # 事前形态：按 unit 一行给出环境变量（避免每 unit 两行 export 造成长列表）
        rows = []
        for u in plan.get("units") or []:
            env = u.get("env") or {}
            if not env:
                continue
            rows.append((
                str(u.get("unit_id")),
                "NPU%s" % (u.get("primary_npu") if u.get("primary_npu") is not None else "-"),
                "node%s" % (u.get("numa_node") if u.get("numa_node") is not None else "-"),
                " ".join("%s=%s" % (k, v) for k, v in env.items()),
            ))
        if not rows:
            lines.append("# 无环境变量配置")
        else:
            w = [max(len(r[i]) for r in rows + [("unit", "NPU", "node", "")]) for i in range(3)]
            lines.append("# 未发现运行中的 NPU 进程；启动业务时按 unit 注入以下环境变量")
            lines.append("%-*s  %-*s  %-*s  %s" % (
                w[0], "unit", w[1], "NPU", w[2], "node", "环境变量"))
            for r in rows:
                lines.append("%-*s  %-*s  %-*s  %s" % (w[0], r[0], w[1], r[1], w[2], r[2], r[3]))
    return "\n".join(lines)


STATE_TAG = {"未绑核": "high", "跨 node": "high", "已绑核": "solve",
             "待启动": "info", "未观测": "view_limited"}

DEVIATION_LABEL = {
    "affinity_outside_npu_range": "亲和越出 NPU 推荐范围",
    "process_node_vs_npu_node": "进程 node 与 NPU node 不一致",
    "numa_maps_unavailable": "numa_maps 不可读",
    "last_cpu_node_mismatch": "最近运行核与 NPU node 不一致",
    "thread_node_drift": "线程跨 node",
    "cross_user_control": "跨用户权限",
    "nic_node_mismatch": "网卡不在 NPU 所在 node",
    "storage_node_mismatch": "存储不在 NPU 所在 node",
    "multi_npu_cross_node": "多卡跨 node",
    "die_pair_same_node": "同封装 die 跨 node",
}
HANDLING_LABEL = {"solve": "纳入方案", "handoff": "转出", "view_limited": "受限",
                  "note": "提示", "observe": "观察"}
SEVERITY_LABEL = {"high": "高", "medium": "中", "low": "低", "info": "信息"}


def _unit_state(unit, total_cpus):
    """按当前亲和判定绑核状态，供 §03 状态列与结论合并使用。

    未绑核 = 亲和覆盖全部核（未做任何收敛）；跨 node = 已绑核但与 NPU 所在
    node 不一致；已绑核 = 亲和已落在目标 node 内；待启动 = 无运行进程。
    """
    if not unit.get("pids") and unit.get("pid") is None:
        return "待启动"
    affinity = unit.get("current_affinity")
    if not affinity:
        return "未观测"
    if total_cpus and len(_cpus(affinity)) >= total_cpus:
        return "未绑核"
    current, target = unit.get("current_node"), unit.get("numa_node")
    if current is not None and target is not None and current != target:
        return "跨 node"
    return "已绑核"


def _conclusions(plan, topo):
    """结论摘要：同类问题合并成一条，说明「未绑核建议绑核 / 已绑核提醒问题」。"""
    units = plan.get("units") or []
    if not units:
        return ["未发现运行中的 NPU 进程，无法给出绑核建议。"]
    total = sum(len(n.get("cpus") or []) for n in topo.get("nodes") or [])
    buckets = {}
    for u in units:
        buckets.setdefault(_unit_state(u, total), []).append(u)
    items = []

    if (plan.get("locality_mode") or "") == "global_slice":
        items.append("该机 NPU 与各 node 等距（超节点形态），按封装分组把 CPU 池均分到各 node，"
                     "同一 node 内的切片连续且互不重叠。")

    unbound = buckets.get("未绑核") or []
    if unbound:
        names = "、".join("unit %s（NPU%s→node%s）" % (
            u.get("unit_id"), u.get("primary_npu"), u.get("numa_node")) for u in unbound)
        items.append("<b>%d 个单元未绑核</b>：亲和覆盖全部 %d 核，建议按 NPU 所在 node 收敛绑定 —— %s。"
                     % (len(unbound), total, names))

    cross = buckets.get("跨 node") or []
    if cross:
        names = "、".join("unit %s（node%s→node%s）" % (
            u.get("unit_id"), u.get("current_node"), u.get("numa_node")) for u in cross)
        items.append("<b>%d 个单元已绑核但跨 node</b>：与 NPU 所在 node 不一致，需重新绑核 —— %s。"
                     % (len(cross), names))

    ok = buckets.get("已绑核") or []
    if ok:
        items.append("%d 个单元已绑核且落在 NPU 所在 node，沿用现有绑定即可。" % len(ok))

    pending = buckets.get("待启动") or []
    if pending:
        items.append("%d 个单元当前无运行进程，待业务启动后按方案绑核。" % len(pending))

    unobserved = buckets.get("未观测") or []
    if unobserved:
        names = "、".join("unit %s" % u.get("unit_id") for u in unobserved)
        items.append("<b>%d 个单元状态不可观测</b>：缺少当前亲和证据，结论仅覆盖可观测单元 —— %s。"
                     % (len(unobserved), names))

    kinds = {}
    for u in units:
        for d in u.get("deviations") or []:
            kinds.setdefault(d.get("kind"), []).append(u)
    if kinds.get("nic_node_mismatch"):
        items.append("网卡不在 NPU 所在 node（%d 个单元），已转 config-audit 处理中断亲和。"
                     % len(kinds["nic_node_mismatch"]))
    if kinds.get("storage_node_mismatch"):
        items.append("存储不在 NPU 所在 node（%d 个单元），模型与数据加载存在跨 node 访存。"
                     % len(kinds["storage_node_mismatch"]))
    if kinds.get("numa_maps_unavailable"):
        items.append("numa_maps 不可读，远端访存占比无法判定，以上结论基于亲和与最近运行核。")

    hbm = (plan.get("summary") or {}).get("unknown_hbm_npus") or []
    if hbm:
        items.append("%s 有显存占用但无可见进程（%s），绑核前请确认是否为容器内进程。" % (
            "、".join("NPU%s" % h.get("npu") for h in hbm),
            "、".join("%.1f GB" % ((h.get("hbm_used_mb") or 0) / 1024.0) for h in hbm)))
    return items or ["未发现需要调整的绑核问题。"]


SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}


def _dev_groups(plan):
    """偏差按（类型, 严重度, 处理）合并，避免逐 unit 重复同一问题。"""
    groups = {}
    order = []
    for u in plan.get("units") or []:
        for d in u.get("deviations") or []:
            key = (d.get("kind"), d.get("severity"), d.get("handling"))
            if key not in groups:
                groups[key] = {"kind": d.get("kind"), "severity": d.get("severity"),
                               "handling": d.get("handling"), "units": [],
                               "evidence": d.get("evidence")}
                order.append(key)
            groups[key]["units"].append(u.get("unit_id"))
    rows = [groups[k] for k in order]
    rows.sort(key=lambda g: (SEVERITY_ORDER.get(g["severity"], 9), str(g["kind"])))
    return rows


def _build_header_labels(plan):
    """页头标签：形态、部署形态、可执行状态、执行模式与预期收益。"""
    exec_mode = (plan.get("privilege") or {}).get("execution_mode") or "unknown"
    return {
        "form": "事中（in_run）" if plan.get("form") == "in_run" else "事前（pre_run）",
        "mode": {"host": "宿主机", "container": "容器"}.get(plan.get("mode"), plan.get("mode") or "-"),
        "badge": "plan 可执行" if plan.get("executable") else (
            "预规划（pre_run）" if plan.get("form") == "pre_run" else "需提权执行"),
        "exec_mode": {"self": "当前用户执行", "delegated": "需代办执行", "plan_only": "仅出方案"}.get(
            exec_mode, exec_mode),
        "effect": {"high": "高", "medium": "中", "low": "低"}.get(
            (plan.get("expected_effect") or {}).get("level"), "-"),
    }


def _build_kpis(units, deviations, plan, effect_label):
    """概览 KPI 四项：求解单元 / 亲和偏差 / 未解决项 / 预期收益。"""
    return [
        ("%d" % len(units), "求解单元", ""),
        ("%d" % len(deviations), "亲和偏差", "hot" if deviations else "ok"),
        ("%d" % len(plan.get("unresolved") or []), "未解决项",
         "ok" if not plan.get("unresolved") else "hot"),
        (effect_label, "预期收益", "ok"),
    ]


def _render_kpi_cards(kpis):
    """KPI 概览卡片 HTML 片段。"""
    return "".join('<div class="kpi %s"><div class="v">%s</div><div class="l">%s</div></div>'
                   % (cls, _e(v), _e(label)) for v, label, cls in kpis)


def _render_unit_rows(units, total_cpus):
    """§02 进程 → 目标核分配表体。"""
    unit_rows = []
    for u in units:
        state = _unit_state(u, total_cpus)
        warn = ' class="warn"' if state in ("未绑核", "跨 node") else ""
        pids = u.get("pids") or ([u.get("pid")] if u.get("pid") is not None else [])
        proc = _e(u.get("process_name") or "-")
        if pids:
            proc += '<br><span class="muted">pid %s</span>' % _e("/".join(str(p) for p in pids))
        cur_node = "node%s" % u.get("current_node") if u.get("current_node") is not None else "未绑定"
        tgt_node = "node%s" % u.get("numa_node") if u.get("numa_node") is not None else "未定"
        cpuset = _e(u.get("cpuset")) if u.get("cpuset") else '<span class="tag medium">未生成</span>'
        if u.get("main_cpuset"):
            cpuset += '<br><span class="muted">main %s</span>' % _e(u.get("main_cpuset"))
        note = ""
        if u.get("capacity_gap"):
            note = '<br><span class="muted">较当前少 %s 核</span>' % _e(u.get("capacity_gap"))
        unit_rows.append(
            "<tr%s><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>"
            "<td><span class='tag %s'>%s</span>%s</td></tr>" % (
                warn, _e(u.get("unit_id")), proc, _e(u.get("primary_npu")),
                "%s · %s" % (_e(cur_node), _e(u.get("current_affinity") or "-")),
                "%s · %s" % (_e(tgt_node), cpuset),
                STATE_TAG.get(state, "low"), _e(state), note,
            )
        )
    return "".join(unit_rows) or '<tr><td colspan="6" class="muted">未发现运行中的 NPU 进程</td></tr>'


def _render_deviation_rows(plan):
    """§03 偏差处理表体：按（类型, 严重度, 处理）合并后成行。"""
    dev_rows = []
    for g in _dev_groups(plan):
        ids = g["units"]
        shown = "、".join("unit %s" % i for i in ids[:6])
        if len(ids) > 6:
            shown += " 等 %d 个" % len(ids)
        dev_rows.append(
            "<tr><td title=\"%s\">%s</td><td>%s</td><td><span class='tag %s'>%s</span></td>"
            "<td><span class='tag %s'>%s</span></td></tr>" % (
                _e(g["kind"]), _e(DEVIATION_LABEL.get(g["kind"], g["kind"])), _e(shown),
                _e(g["severity"]), _e(SEVERITY_LABEL.get(g["severity"], g["severity"])),
                _e(g["handling"]), _e(HANDLING_LABEL.get(g["handling"], g["handling"]))
            )
        )
    return "".join(dev_rows) or '<tr><td colspan="4" class="muted">未发现偏差</td></tr>'


def _render_detail_rows(deviations):
    """§05 逐单元偏差证据表体。"""
    dev_detail_rows = []
    for u, d in deviations:
        kind, handling = d.get("kind"), d.get("handling")
        dev_detail_rows.append(
            "<tr><td>unit %s</td><td title=\"%s\">%s</td><td><span class='tag %s'>%s</span></td>"
            "<td><span class='tag %s'>%s</span></td><td>%s</td></tr>" % (
                _e(u.get("unit_id")), _e(kind), _e(DEVIATION_LABEL.get(kind, kind)),
                _e(d.get("severity")), _e(SEVERITY_LABEL.get(d.get("severity"), d.get("severity"))),
                _e(handling), _e(HANDLING_LABEL.get(handling, handling)), _e(d.get("evidence"))
            )
        )
    return "".join(dev_detail_rows) or '<tr><td colspan="5" class="muted">无</td></tr>'


def _render_bind_rows(units):
    """§05 线程绑定规则表体。"""
    bind_rows = []
    for u in units:
        for b in u.get("thread_binding") or []:
            bind_rows.append(
                "<tr><td>unit %s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (
                    _e(u.get("unit_id")), _e(b.get("role")), _e(b.get("apply")),
                    _e(", ".join(b.get("match_comm") or []) or "-"), _e(b.get("cpuset") or "-")
                )
            )
    return "".join(bind_rows) or '<tr><td colspan="5" class="muted">未发现运行中的 NPU 进程</td></tr>'


def _build_exec_hint(plan, priv, exec_mode_label):
    """§04 执行提示：按事前 / 事中形态给出执行模式与权限说明。"""
    targets = priv.get("target_uids") or []
    target_text = "/".join(str(t) for t in targets) or "-"
    numa_text = "可读" if priv.get("can_read_numa_maps") else "不可读"
    if plan.get("form") == "in_run":
        return ("执行模式：%s · 当前用户：%s（uid %s） · 目标进程用户：%s · numa_maps：%s。"
                "按序执行；执行前逐项校验进程属主，跨用户需 sudo 或以进程所属用户执行。") % (
            exec_mode_label, priv.get("current_user"), priv.get("current_uid"),
            target_text, numa_text)
    return "执行模式：%s · 当前用户：%s（uid %s）。未发现运行中的 NPU 进程，启动业务时注入下列环境变量。" % (
        exec_mode_label, priv.get("current_user"), priv.get("current_uid"))


def _render_page(parts):
    """页面骨架与样式外置在 resources/，这里只做占位符替换。"""
    template = string.Template(_load_asset("report_template.html"))
    values = dict(parts)
    values["css"] = _load_asset("report.css")
    return template.substitute(values)


def render(plan, topo):
    role_map = _role_map(plan)
    units = plan.get("units") or []
    priv = plan.get("privilege") or {}
    labels = _build_header_labels(plan)
    deviations = []
    for u in units:
        for d in u.get("deviations") or []:
            deviations.append((u, d))

    kpis = _build_kpis(units, deviations, plan, labels["effect"])
    notes_list = plan.get("notes") or []
    total_cpus = sum(len(n.get("cpus") or []) for n in topo.get("nodes") or [])
    return _render_page({
        "host": _e(plan.get("host")),
        "mode_label": _e(labels["mode"]),
        "concl": "".join("<li>%s</li>" % _e_conclusion(s) for s in _conclusions(plan, topo)),
        "exec_badge": _e(labels["badge"]),
        "form_label": _e(labels["form"]),
        "exec_mode_label": _e(labels["exec_mode"]),
        "kpis": _render_kpi_cards(kpis),
        "host_meta": _host_meta(topo, plan),
        "node_cards": "".join(_node_card(n, topo, plan) for n in topo.get("nodes") or []),
        "unit_rows": _render_unit_rows(units, total_cpus),
        "dev_rows": _render_deviation_rows(plan),
        "exec_hint": _e(_build_exec_hint(plan, priv, labels["exec_mode"])),
        "cmd_text": _e(_runbook(plan)),
        "verify_script": _e(plan.get("verify_script")),
        "verify_metrics": _e(", ".join((plan.get("verify") or {}).get("effect", {}).get("metrics") or [])),
        "affinity_matrix": _affinity_matrix(topo, plan),
        "unknown_card": _unknown_card(topo, plan),
        "distance_table": _distance_table(topo),
        "strips": "".join(_strip(n, role_map, plan) for n in topo.get("nodes") or []),
        "bind_rows": _render_bind_rows(units),
        "dev_count": len(deviations),
        "notes_count": len(notes_list),
        "dev_detail_rows": _render_detail_rows(deviations),
        "notes": "".join("<li>%s</li>" % _e(n) for n in notes_list) or "<li>无</li>",
        "plan_id": _e(plan.get("plan_id")),
    })


def write(plan, topo, path):
    html_text = render(plan, topo)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_text)
    return path
