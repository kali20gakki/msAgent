from __future__ import annotations

import base64
from pathlib import Path

import pytest

from msagent.cli.resolvers.image import ImageResolver


@pytest.mark.asyncio
async def test_get_image_files_falls_back_and_filters_non_images(
    monkeypatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "ok.png").write_bytes(b"png-bytes")
    (tmp_path / "note.txt").write_text("hello", encoding="utf-8")

    calls: list[list[str]] = []

    async def fake_execute_bash_command(cmd, cwd, timeout):
        del cwd, timeout
        calls.append(cmd)
        if len(calls) == 1:
            return 1, "", "git failed"
        return 0, "ok.png\nnote.txt\nmissing.jpg\n", ""

    monkeypatch.setattr(
        "msagent.cli.resolvers.image.execute_bash_command",
        fake_execute_bash_command,
    )

    images = await ImageResolver._get_image_files(tmp_path, limit=5, pattern="ok")

    assert images == ["ok.png"]
    assert len(calls) == 2
    assert "git ls-files" in calls[0][2]
    assert "fd --type f" in calls[1][2]


@pytest.mark.asyncio
async def test_image_resolver_complete_formats_completions(
    monkeypatch,
    tmp_path: Path,
) -> None:
    async def fake_get_image_files(working_dir, limit=None, pattern=""):
        del working_dir, limit, pattern
        return ["figures/chart.png"]

    monkeypatch.setattr(
        ImageResolver,
        "_get_image_files",
        staticmethod(fake_get_image_files),
    )

    resolver = ImageResolver()
    completions = await resolver.complete(
        fragment="cha",
        ctx={"working_dir": tmp_path, "start_position": -3},
        limit=10,
    )

    assert len(completions) == 1
    completion = completions[0]
    assert completion.text == "@:image:figures/chart.png"
    assert completion.display_text == "@:image:figures/chart.png"
    assert completion.style == "class:file-completion"
    assert completion.start_position == -3


def test_image_resolver_resolve_returns_absolute_only_when_file_exists(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "assets" / "photo.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image-data")

    resolver = ImageResolver()

    existing = resolver.resolve("assets/photo.png", ctx={"working_dir": tmp_path})
    missing = resolver.resolve("assets/missing.png", ctx={"working_dir": tmp_path})

    assert Path(existing) == image_path.resolve()
    assert missing == "assets/missing.png"


def test_image_resolver_detects_standalone_absolute_image_path(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "img.jpg"
    image_path.write_bytes(b"data")

    resolver = ImageResolver()

    assert resolver.is_standalone_reference(str(image_path)) is True
    assert resolver.is_standalone_reference("relative/img.jpg") is False


def test_image_resolver_build_content_block_success(tmp_path: Path) -> None:
    image_path = tmp_path / "chart.png"
    image_path.write_bytes(b"binary-image")

    resolver = ImageResolver()
    block = resolver.build_content_block(str(image_path))

    assert block == {
        "type": "image",
        "source_type": "base64",
        "data": base64.b64encode(b"binary-image").decode("utf-8"),
        "mime_type": "image/png",
    }


def test_image_resolver_build_content_block_raises_for_missing_file(
    tmp_path: Path,
) -> None:
    resolver = ImageResolver()

    with pytest.raises(FileNotFoundError, match="Image not found"):
        resolver.build_content_block(str(tmp_path / "none.png"))


def test_image_resolver_build_content_block_raises_for_unsupported_format(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "vector.bmp"
    image_path.write_bytes(b"bmp")

    resolver = ImageResolver()

    with pytest.raises(ValueError, match="Unsupported format"):
        resolver.build_content_block(str(image_path))


def test_image_resolver_build_content_block_raises_when_mime_is_unknown(
    monkeypatch,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "img.png"
    image_path.write_bytes(b"img")

    monkeypatch.setattr("msagent.cli.resolvers.image.get_image_mime_type", lambda *_: None)
    resolver = ImageResolver()

    with pytest.raises(ValueError, match="Cannot determine MIME type"):
        resolver.build_content_block(str(image_path))
