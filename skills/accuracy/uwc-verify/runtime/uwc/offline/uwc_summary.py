"""Aggregate UWC stage reports into the final 5-item verification report."""

from __future__ import annotations

import argparse
import html
import json
import platform
from pathlib import Path

from ..common import get_dump_dir, now_iso, read_json, write_json


ITEM_DEFINITIONS = [
    ("bridge_p0_p2", "Bridge 转换：P0 vs P2", "stage1", "export"),
    ("transport_p2sender_p5", "传输完整性：P2_sender vs P5", "stage2", "transfer"),
    ("data_ptr", "推理侧加载：data_ptr 指针偏移", "stage3", "ptr"),
    ("bridge_g0_g1", "前向一致性：G0 vs G1", "stage1", "forward"),
    ("dummy_vs_safetensors", "dummy vs safetensors 差分", "stage3", "diff_test"),
]


def _status_label(status: str) -> str:
    return {
        "pass": "PASS",
        "fail": "FAIL",
        "skip": "SKIP",
        "error": "ERROR",
    }.get(status, str(status).upper())


def _load_stage_reports(dump_dir: Path) -> dict:
    reports = {}
    for stage in (1, 2, 3):
        path = dump_dir / f"stage{stage}" / "report.json"
        if path.is_file():
            try:
                reports[stage] = read_json(path)
            except Exception as exc:
                reports[stage] = {"status": "error", "error": f"cannot read {path}: {exc}"}
        else:
            reports[stage] = {"status": "skip", "reason": f"{path} not found"}
    return reports


def _metrics(item_id: str, section: dict) -> dict:
    if item_id == "bridge_p0_p2":
        return {
            "P0 keys": section.get("reference_keys"),
            "P2 keys": section.get("candidate_keys"),
            "compared": section.get("compared_keys"),
            "mismatched weights": section.get("mismatched_keys"),
            "max_abs_diff": section.get("max_abs_diff"),
            "threshold": section.get("threshold"),
        }
    if item_id == "transport_p2sender_p5":
        return {
            "P2_sender keys": section.get("reference_keys"),
            "P5 keys": section.get("candidate_keys"),
            "compared": section.get("compared_keys"),
            "mismatched weights": section.get("mismatched_keys"),
            "max_abs_diff": section.get("max_abs_diff"),
            "threshold": section.get("threshold"),
        }
    if item_id == "data_ptr":
        return {
            "checked layers": section.get("checked_layers"),
            "synced layers": section.get("synced_layers"),
            "not synced": section.get("not_synced_layers"),
            "missing after load": section.get("missing_layers"),
            "report files": len(section.get("files", []) or []),
        }
    if item_id == "bridge_g0_g1":
        return {
            "avg_cos": section.get("avg_cos"),
            "min_cos": section.get("min_cos"),
            "max_cos": section.get("max_cos"),
            "max_abs_diff": section.get("max_abs_diff"),
            "positions": section.get("num_positions"),
            "thresholds": section.get("thresholds"),
        }
    if item_id == "dummy_vs_safetensors":
        return {
            "dummy garbled": section.get("dummy_garbled"),
            "safetensors garbled": section.get("safetensors_garbled"),
            "identical outputs": section.get("identical_outputs"),
            "conclusion": section.get("conclusion"),
        }
    return {}


def _evidence(item_id: str, section: dict, dump_dir: Path) -> list[str]:
    if item_id == "bridge_p0_p2":
        return [path for path in (section.get("P0_file"), section.get("P2_file")) if path]
    if item_id == "transport_p2sender_p5":
        return [path for path in (section.get("P2_sender_file"), section.get("P5_file")) if path]
    if item_id == "data_ptr":
        return [str(path) for path in section.get("files", []) or []]
    if item_id == "bridge_g0_g1":
        return [path for path in (section.get("G0_file"), section.get("G1_file")) if path]
    if item_id == "dummy_vs_safetensors":
        return [
            path
            for path in (
                section.get("dummy_file"),
                section.get("safetensors_file"),
                str(dump_dir / "stage3_vllm" / "diff_test_report.json"),
            )
            if path
        ]
    return []


def _details(item_id: str, section: dict) -> dict:
    if item_id in {"bridge_p0_p2", "transport_p2sender_p5"}:
        return {
            "missing_keys": section.get("missing_keys", []),
            "extra_keys": section.get("extra_keys", []),
            "worst": section.get("worst", []),
            "reasons": section.get("reasons", []),
            "missing_count": section.get("missing_count"),
            "extra_count": section.get("extra_count"),
        }
    if item_id == "data_ptr":
        return {
            "not_synced": section.get("not_synced", []),
            "errors": section.get("errors", []),
            "reason": section.get("reason"),
        }
    if item_id == "bridge_g0_g1":
        return {
            "shape_g0": section.get("shape_g0"),
            "shape_g1": section.get("shape_g1"),
            "reason": section.get("reason"),
        }
    if item_id == "dummy_vs_safetensors":
        return {
            "dummy samples": section.get("dummy_text_samples", []),
            "safetensors samples": section.get("safetensors_text_samples", []),
            "reason": section.get("reason"),
        }
    return {}


def _stage_report_mtimes(dump_dir: Path) -> dict:
    mtimes = {}
    for stage in (1, 2, 3):
        path = dump_dir / f"stage{stage}" / "report.json"
        try:
            mtimes[stage] = path.stat().st_mtime
        except OSError:
            continue
    return mtimes


def _stale_evidence(evidence, report_mtime, tolerance: float = 1.0) -> list[str]:
    """Evidence modified after its compare report means the verdict is stale.

    ``diff_test_report.json`` is produced by ``compare`` itself, not by the
    run, so it is excluded.  The 1 s tolerance absorbs filesystem granularity
    between writing an evidence file and the report that summarises it.
    """
    if report_mtime is None:
        return []
    stale = []
    for entry in evidence or []:
        path = Path(str(entry))
        if path.name == "diff_test_report.json":
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            stale.append(str(path))
            continue
        if mtime > report_mtime + tolerance:
            stale.append(str(path))
    return stale


def build_report(dump_dir: str | Path | None = None, title: str | None = None, demo: bool = False, extra: dict | None = None) -> dict:
    dump_dir = Path(dump_dir) if dump_dir else get_dump_dir()
    stage_reports = _load_stage_reports(dump_dir)
    stage_mtimes = _stage_report_mtimes(dump_dir)
    items = []
    for item_id, name, stage_key, section_key in ITEM_DEFINITIONS:
        stage_number = int(stage_key.replace("stage", ""))
        stage_report = stage_reports.get(stage_number, {})
        section = stage_report.get(section_key, {}) if isinstance(stage_report, dict) else {}
        if not isinstance(section, dict):
            section = {}
        status = section.get("status", stage_report.get("status", "skip"))
        if status not in {"pass", "fail", "skip", "error"}:
            status = "error" if status else "skip"
        reason = section.get("reason") or stage_report.get("reason") or section.get("error") or stage_report.get("error")
        evidence = _evidence(item_id, section, dump_dir)
        # Never present a verdict for a dump that changed after compare: the
        # numbers in the stage report no longer describe the evidence.
        stale = _stale_evidence(evidence, stage_mtimes.get(stage_number))
        if stale:
            status = "error"
            reason = (
                f"证据文件在 compare 之后被修改或缺失（{stale[0]}），"
                "当前结论已失效；请重新执行 compare 后再生成报告。"
            )
        items.append(
            {
                "id": item_id,
                "index": len(items) + 1,
                "name": name,
                "stage": stage_number,
                "status": status,
                "status_label": _status_label(status),
                "reason": reason,
                "metrics": _metrics(item_id, section),
                "evidence": evidence,
                "details": _details(item_id, section),
                "raw": section,
            }
        )

    counts = {"pass": 0, "fail": 0, "skip": 0, "error": 0}
    for item in items:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    if counts["fail"] or counts["error"]:
        overall = "fail"
    elif counts["skip"]:
        overall = "partial" if counts["pass"] else "skip"
    else:
        overall = "pass"

    environment = {
        "dump_dir": str(dump_dir),
        "generated_at": now_iso(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "demo": bool(demo),
    }
    try:
        import torch

        environment["torch"] = torch.__version__
        npu = getattr(torch, "npu", None)
        environment["npu_available"] = bool(npu is not None and npu.is_available())
    except Exception:
        environment["torch"] = None
    if extra:
        environment.update(extra)

    report = {
        "schema": "uwc-report/v1",
        "title": title or "UWC 训推权重一致性验证报告",
        "generated_at": now_iso(),
        "overall": {"status": overall, "counts": counts, "total": len(items)},
        "environment": environment,
        "items": items,
        "stage_reports": {str(key): value for key, value in stage_reports.items()},
    }
    return report


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def render_markdown(report: dict) -> str:
    lines = []
    title = report.get("title", "UWC 权重一致性验证报告")
    lines.append(f"# {title}")
    lines.append("")
    if report.get("environment", {}).get("demo"):
        lines.append("> **演示数据**：本报告由 `uwc-verify demo` 使用合成权重/输出生成，用于展示报告格式与判定逻辑，不代表真实模型验证结果。")
        lines.append("")
    overall = report.get("overall", {})
    counts = overall.get("counts", {})
    lines.append(
        f"**总体结论：{_status_label(overall.get('status', 'skip'))}** "
        f"（通过 {counts.get('pass', 0)} / 失败 {counts.get('fail', 0)} / "
        f"跳过 {counts.get('skip', 0)} / 错误 {counts.get('error', 0)}，共 {overall.get('total', 0)} 项）"
    )
    lines.append("")
    env = report.get("environment", {})
    lines.append("| 项目 | 值 |")
    lines.append("|---|---|")
    for key in ("dump_dir", "generated_at", "host", "platform", "python", "torch", "npu_available"):
        if key in env:
            lines.append(f"| {key} | {_fmt(env[key])} |")
    lines.append("")

    lines.append("## 验证项汇总")
    lines.append("")
    lines.append("| # | 验证项 | 结果 | 关键指标 |")
    lines.append("|---|---|---|---|")
    for item in report.get("items", []):
        metric_text = "; ".join(f"{key}={_fmt(value)}" for key, value in item.get("metrics", {}).items() if value is not None)
        if not metric_text:
            metric_text = item.get("reason") or "-"
        lines.append(f"| {item['index']} | {item['name']} | **{item['status_label']}** | {metric_text} |")
    lines.append("")

    for item in report.get("items", []):
        lines.append(f"## {item['index']}. {item['name']} — {item['status_label']}")
        lines.append("")
        if item.get("reason"):
            lines.append(f"原因：{item['reason']}")
            lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|---|---|")
        for key, value in item.get("metrics", {}).items():
            lines.append(f"| {key} | {_fmt(value)} |")
        lines.append("")
        details = item.get("details", {})
        for key, value in details.items():
            if not value:
                continue
            lines.append(f"**{key}**")
            lines.append("")
            if isinstance(value, list):
                for entry in value[:20]:
                    lines.append(f"- `{_fmt(entry)}`")
                if len(value) > 20:
                    lines.append(f"- ... ({len(value) - 20} more)")
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    lines.append(f"- {sub_key}: `{_fmt(sub_value)}`")
            else:
                lines.append(f"`{_fmt(value)}`")
            lines.append("")
        if item.get("evidence"):
            lines.append("**证据文件**")
            lines.append("")
            for path in item["evidence"]:
                lines.append(f"- `{path}`")
            lines.append("")

    lines.append("## 复现命令")
    lines.append("")
    lines.append("```bash")
    lines.append("# 1) 注入采集代码（幂等，可 --dry-run）")
    lines.append("python3 scripts/uwc_verify.py prepare --config uwc.yaml")
    lines.append("# 2) 按阶段运行采集")
    lines.append("python3 scripts/uwc_verify.py run --config uwc.yaml --stage all")
    lines.append("# 3) 离线比对 + 生成报告")
    lines.append("python3 scripts/uwc_verify.py compare --config uwc.yaml")
    lines.append("python3 scripts/uwc_verify.py report  --config uwc.yaml")
    lines.append("```")
    lines.append("")
    lines.append("> 判定口径：阶段 1/2 依据文档中的 atol / exact 容差；G0 vs G1 要求 avg_cos>0.99 且 min_cos>=0.95；data_ptr 检查按同步轮前后对比，ptr 或内容采样发生变化即视为已同步（vLLM 原地 copy_ 时 ptr 不变）；dummy vs safetensors 按文档的 2×2 矩阵，两次输出相同且 load_format 也相同才判定为覆盖未生效。")
    lines.append("")
    return "\n".join(lines)


_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
       margin: 0; background: #f6f7f9; color: #1f2933; }
.wrap { max-width: 1100px; margin: 0 auto; padding: 32px 24px 64px; }
h1 { font-size: 26px; margin: 0 0 8px; }
h2 { font-size: 18px; margin: 28px 0 12px; border-left: 4px solid #2f6fed; padding-left: 10px; }
.meta { color: #616e7c; font-size: 13px; margin-bottom: 16px; }
.badge { display: inline-block; padding: 4px 10px; border-radius: 999px; font-weight: 700; font-size: 13px; }
.pass { background: #e3f9e5; color: #1f7a33; }
.fail { background: #ffe3e3; color: #b42318; }
.skip { background: #eef1f4; color: #52606d; }
.error { background: #fff3cd; color: #8a5a00; }
table { width: 100%; border-collapse: collapse; background: #fff; box-shadow: 0 1px 2px rgba(0,0,0,.06);
        border-radius: 8px; overflow: hidden; margin: 8px 0 16px; }
th, td { text-align: left; padding: 10px 12px; border-bottom: 1px solid #eef1f4; font-size: 13px; vertical-align: top; }
th { background: #f0f4f8; font-weight: 600; }
tr:last-child td { border-bottom: none; }
pre, code { font-family: "SFMono-Regular", Consolas, monospace; font-size: 12px; }
pre { background: #0f172a; color: #e2e8f0; padding: 14px; border-radius: 8px; overflow-x: auto; }
.item { background: #fff; border-radius: 10px; padding: 18px 20px; margin: 14px 0; box-shadow: 0 1px 2px rgba(0,0,0,.06); }
.item h3 { margin: 0 0 8px; font-size: 16px; }
.muted { color: #616e7c; }
.demo { background: #fff8e1; border: 1px solid #f2c94c; padding: 12px 14px; border-radius: 8px; margin-bottom: 16px; }
"""

_HTML_HEAD = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
</head>
<body><div class="wrap">
"""

_HTML_TAIL = "</div></body></html>\n"


def _badge(status: str) -> str:
    return f'<span class="badge {html.escape(status)}">{html.escape(_status_label(status))}</span>'


def render_html(report: dict) -> str:
    title = report.get("title", "UWC 权重一致性验证报告")
    out = [_HTML_HEAD.format(title=html.escape(title), css=_CSS)]
    out.append(f"<h1>{html.escape(title)}</h1>")
    env = report.get("environment", {})
    out.append(
        '<div class="meta">'
        f"dump_dir: {html.escape(str(env.get('dump_dir', '')))} &nbsp;|&nbsp; "
        f"generated_at: {html.escape(str(env.get('generated_at', '')))} &nbsp;|&nbsp; "
        f"python: {html.escape(str(env.get('python', '')))} &nbsp;|&nbsp; "
        f"torch: {html.escape(str(env.get('torch', '')))}"
        "</div>"
    )
    if env.get("demo"):
        out.append('<div class="demo">演示数据：本报告由 <code>uwc-verify demo</code> 使用合成权重/输出生成，用于展示报告格式与判定逻辑，不代表真实模型验证结果。</div>')

    overall = report.get("overall", {})
    counts = overall.get("counts", {})
    out.append(
        f"<p>总体结论：{_badge(overall.get('status', 'skip'))} "
        f"<span class='muted'>通过 {counts.get('pass', 0)} / 失败 {counts.get('fail', 0)} / "
        f"跳过 {counts.get('skip', 0)} / 错误 {counts.get('error', 0)}</span></p>"
    )

    out.append("<h2>验证项汇总</h2><table><tr><th>#</th><th>验证项</th><th>结果</th><th>关键指标</th></tr>")
    for item in report.get("items", []):
        metric_text = "; ".join(
            f"{html.escape(str(key))}={html.escape(_fmt(value))}"
            for key, value in item.get("metrics", {}).items()
            if value is not None
        ) or html.escape(str(item.get("reason") or "-"))
        out.append(
            f"<tr><td>{item['index']}</td><td>{html.escape(item['name'])}</td>"
            f"<td>{_badge(item['status'])}</td><td>{metric_text}</td></tr>"
        )
    out.append("</table>")

    for item in report.get("items", []):
        out.append('<div class="item">')
        out.append(f"<h3>{item['index']}. {html.escape(item['name'])} {_badge(item['status'])}</h3>")
        if item.get("reason"):
            out.append(f"<p class='muted'>{html.escape(str(item['reason']))}</p>")
        out.append("<table><tr><th>指标</th><th>值</th></tr>")
        for key, value in item.get("metrics", {}).items():
            out.append(f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(_fmt(value))}</td></tr>")
        out.append("</table>")
        for key, value in item.get("details", {}).items():
            if not value:
                continue
            out.append(f"<p><strong>{html.escape(key)}</strong></p><pre>{html.escape(_fmt(value))}</pre>")
        if item.get("evidence"):
            out.append("<p><strong>证据文件</strong></p><ul>")
            for path in item["evidence"]:
                out.append(f"<li><code>{html.escape(str(path))}</code></li>")
            out.append("</ul>")
        out.append("</div>")

    out.append(
        "<h2>复现命令</h2><pre>"
        "python3 scripts/uwc_verify.py prepare --config uwc.yaml\n"
        "python3 scripts/uwc_verify.py run --config uwc.yaml --stage all\n"
        "python3 scripts/uwc_verify.py compare --config uwc.yaml\n"
        "python3 scripts/uwc_verify.py report --config uwc.yaml"
        "</pre>"
    )
    out.append(_HTML_TAIL)
    return "".join(out)


def write_report(report: dict, out_dir: str | Path | None = None, dump_dir: str | Path | None = None) -> dict:
    out_dir = Path(out_dir) if out_dir else (Path(dump_dir) if dump_dir else Path(report["environment"]["dump_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    html_path = out_dir / "report.html"
    write_json(json_path, report)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path), "html": str(html_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="UWC summary report")
    parser.add_argument("--dump_dir", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args(argv)

    report = build_report(args.dump_dir, title=args.title, demo=args.demo)
    paths = write_report(report, out_dir=args.out_dir, dump_dir=args.dump_dir)
    print(json.dumps({"overall": report["overall"], "paths": paths}, indent=2, ensure_ascii=False))
    return 0 if report["overall"]["status"] in {"pass", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
