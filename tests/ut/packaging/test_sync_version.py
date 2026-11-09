from __future__ import annotations

from pathlib import Path

from scripts.sync_version import _sync_version_info


def test_sync_version_info_updates_version_and_preserves_build_metadata(tmp_path: Path) -> None:
    version_info = tmp_path / "version.info"
    version_info.write_text(
        "# generated metadata\n[PACKAGE]\nName=mindstudio-agent\nVersion=26.1.3\nCommit=abc1234\nDate=2026-11-06\n",
        encoding="utf-8",
    )

    _sync_version_info(version_info, "26.3.0")

    content = version_info.read_text(encoding="utf-8")
    assert content == (
        "# generated metadata\n[PACKAGE]\nName=mindstudio-agent\nVersion=26.3.0\nCommit=abc1234\nDate=2026-11-06\n"
    )


def test_sync_version_info_does_not_rewrite_when_version_is_unchanged(tmp_path: Path) -> None:
    version_info = tmp_path / "version.info"
    original = "[PACKAGE]\nVersion=26.3.0\n"
    version_info.write_text(original, encoding="utf-8")

    _sync_version_info(version_info, "26.3.0")

    assert version_info.read_text(encoding="utf-8") == original
