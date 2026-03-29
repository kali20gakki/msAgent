from __future__ import annotations

import base64
from pathlib import Path

from msagent.cli.builders.message import MessageContentBuilder


_ONE_BY_ONE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z0u8AAAAASUVORK5CYII="
)


def test_build_supports_plain_at_file_reference(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello", encoding="utf-8")

    builder = MessageContentBuilder(tmp_path)

    content, reference_mapping = builder.build("请查看 @notes.txt")

    assert content == f"请查看 {file_path}"
    assert reference_mapping == {"notes.txt": str(file_path)}


def test_build_supports_plain_at_image_reference(tmp_path: Path) -> None:
    image_path = tmp_path / "diagram.png"
    image_path.write_bytes(_ONE_BY_ONE_PNG)

    builder = MessageContentBuilder(tmp_path)

    content, reference_mapping = builder.build("分析 @diagram.png")

    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "分析"}
    assert content[1]["type"] == "image"
    assert content[1]["mime_type"] == "image/png"
    assert reference_mapping == {"diagram.png": str(image_path)}


def test_build_supports_standalone_absolute_file_reference(tmp_path: Path) -> None:
    file_path = tmp_path / "report.txt"
    file_path.write_text("ok", encoding="utf-8")

    builder = MessageContentBuilder(tmp_path)

    content, reference_mapping = builder.build(f"read {file_path}")

    assert content == f"read {file_path}"
    assert reference_mapping == {str(file_path): str(file_path)}

