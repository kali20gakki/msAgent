"""Message content builder."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from msagent.cli.resolvers import FileResolver, ImageResolver, RefType
from msagent.utils.image import is_image_file
from msagent.utils.path import resolve_path


@dataclass(frozen=True)
class _ExtractedReference:
    ref_type: RefType
    value: str
    token: str
    start: int


class MessageContentBuilder:
    """Builds message content with multimodal support."""

    _EXPLICIT_REF_PATTERN = re.compile(
        r"@:([\w]+):((?:[^\s?,!.;]|[.,!?;](?!\s|$))+)"
    )
    _MENTION_TRIGGER_GUARDS = frozenset(
        (".", "-", "_", "`", "'", '"', ":", "@", "#", "~")
    )
    _TRAILING_MENTION_PUNCTUATION = ".,!?;:"

    def __init__(self, working_dir: Path):
        """Initialize message content builder."""
        self.working_dir = working_dir
        self.resolvers = {
            RefType.FILE: FileResolver(),
            RefType.IMAGE: ImageResolver(),
        }

    def extract_references(self, text: str) -> dict[RefType, list[str]]:
        """Extract all typed references and standalone paths from text."""
        references: dict[RefType, list[str]] = {}

        for reference in self._extract_reference_specs(text):
            if reference.ref_type not in references:
                references[reference.ref_type] = []
            references[reference.ref_type].append(reference.value)

        for word in text.split():
            if word.startswith("@:"):
                continue
            # Strip common punctuation from end of word
            cleaned_word = word.rstrip(".,!?;:")
            for resolver in self.resolvers.values():
                if resolver.is_standalone_reference(cleaned_word):
                    ref_type = resolver.type
                    if ref_type not in references:
                        references[ref_type] = []
                    references[ref_type].append(cleaned_word)
                    break

        return references

    def _extract_reference_specs(self, text: str) -> list[_ExtractedReference]:
        references: list[_ExtractedReference] = []

        for match in self._EXPLICIT_REF_PATTERN.finditer(text):
            type_str, value = match.groups()
            try:
                ref_type = RefType(type_str)
            except ValueError:
                continue

            references.append(
                _ExtractedReference(
                    ref_type=ref_type,
                    value=value,
                    token=match.group(0),
                    start=match.start(),
                )
            )

        references.extend(self._extract_plain_mentions(text))
        references.sort(key=lambda reference: reference.start)
        return references

    def _extract_plain_mentions(self, text: str) -> list[_ExtractedReference]:
        references: list[_ExtractedReference] = []

        for index, char in enumerate(text):
            if char != "@":
                continue
            if index + 1 >= len(text) or text[index + 1] == ":":
                continue

            if index > 0:
                previous = text[index - 1]
                if previous.isalnum() or previous in self._MENTION_TRIGGER_GUARDS:
                    continue

            end = index + 1
            while end < len(text) and not text[end].isspace():
                end += 1

            value = text[index + 1 : end].rstrip(self._TRAILING_MENTION_PUNCTUATION)
            if not value:
                continue

            ref_type = self._detect_plain_reference_type(value)
            if ref_type is None:
                continue

            references.append(
                _ExtractedReference(
                    ref_type=ref_type,
                    value=value,
                    token=f"@{value}",
                    start=index,
                )
            )

        return references

    def _detect_plain_reference_type(self, value: str) -> RefType | None:
        try:
            resolved = resolve_path(str(self.working_dir), value)
        except Exception:
            return None

        if not resolved.exists():
            return None
        if resolved.is_file() and is_image_file(resolved):
            return RefType.IMAGE
        return RefType.FILE

    def build(
        self, text: str
    ) -> tuple[str | list[str | dict[str, Any]], dict[str, str]]:
        """Build message content with multimodal support.

        Raises:
            FileNotFoundError: If referenced file/image doesn't exist
            ValueError: If referenced file/image is invalid
        """
        references = self._extract_reference_specs(text)

        if not references:
            return text, {}

        ctx = {"working_dir": str(self.working_dir)}
        reference_mapping = {}
        text_content = text
        errors: list[str] = []
        attachment_blocks: list[dict[str, Any]] = []

        for reference in references:
            resolver = self.resolvers[reference.ref_type]
            resolved = resolver.resolve(reference.value, ctx)
            reference_mapping[reference.value] = resolved

            try:
                block = resolver.build_content_block(resolved)
                if block:
                    attachment_blocks.append(block)
                    text_content = text_content.replace(reference.token, "", 1)
                else:
                    text_content = text_content.replace(reference.token, resolved, 1)
            except (FileNotFoundError, ValueError) as e:
                errors.append(str(e))
                text_content = text_content.replace(reference.token, "", 1)

        if errors:
            raise ValueError("\n".join(errors))

        content_blocks: list[str | dict[str, Any]] = []

        if text_content.strip():
            content_blocks.append({"type": "text", "text": text_content.strip()})

        content_blocks.extend(attachment_blocks)

        if len(content_blocks) == 1:
            first_block = content_blocks[0]
            if isinstance(first_block, dict) and first_block.get("type") == "text":
                return first_block["text"], reference_mapping

        return content_blocks, reference_mapping
