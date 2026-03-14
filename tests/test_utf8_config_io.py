import builtins
import json
from pathlib import Path

import aiofiles
import pytest

from msagent.configs.agent import BatchAgentConfig
from msagent.configs.approval import ToolApprovalConfig
from msagent.configs.mcp import MCPConfig

PROMPT_TEXT = 'Follow the prompt with \u201csmart quotes\u201d and \u4e2d\u6587\u3002'
CHINESE_TEXT = "\u4e2d\u6587"


@pytest.mark.asyncio
async def test_batch_agent_config_reads_yaml_and_prompts_with_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agents_dir = tmp_path / "agents"
    prompt_dir = tmp_path / "prompts" / "agents"
    agents_dir.mkdir(parents=True)
    prompt_dir.mkdir(parents=True)

    (prompt_dir / "general.md").write_text(PROMPT_TEXT, encoding="utf-8")
    (agents_dir / "general.yml").write_text(
        """
name: general
default: true
prompt:
  - prompts/agents/general.md
llm:
  provider: openai
  model: gpt-4o-mini
  alias: default
  max_tokens: 0
  temperature: 0
""".strip(),
        encoding="utf-8",
    )

    path_type = type(tmp_path)
    original_read_text = path_type.read_text

    def strict_read_text(self, *args, **kwargs):
        if "encoding" not in kwargs:
            raise AssertionError(f"encoding is required for {self}")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(path_type, "read_text", strict_read_text)

    config = await BatchAgentConfig.from_yaml(dir_path=agents_dir)

    assert config.agents[0].prompt == PROMPT_TEXT


def test_tool_approval_config_reads_and_writes_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.approval.json"
    config_path.write_text(
        json.dumps(
            {
                "always_allow": [
                    {"name": "run_command", "args": {"command": f"echo {CHINESE_TEXT}"}}
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    original_open = builtins.open

    def strict_open(file, mode="r", *args, **kwargs):
        if "b" not in mode and "encoding" not in kwargs:
            raise AssertionError(f"encoding is required for {file}")
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", strict_open)

    config = ToolApprovalConfig.from_json_file(config_path)
    config.save_to_json_file(config_path)

    assert config.always_allow[0].args == {"command": f"echo {CHINESE_TEXT}"}
    saved = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved["always_allow"][0]["args"]["command"] == f"echo {CHINESE_TEXT}"


@pytest.mark.asyncio
async def test_mcp_config_reads_utf8_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.mcp.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "msprof-mcp": {
                        "command": "uvx",
                        "args": ["server"],
                        "env": {"LABEL": CHINESE_TEXT},
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    original_aiofiles_open = aiofiles.open

    def strict_aiofiles_open(*args, **kwargs):
        if "encoding" not in kwargs:
            raise AssertionError("encoding is required for aiofiles.open")
        return original_aiofiles_open(*args, **kwargs)

    monkeypatch.setattr(aiofiles, "open", strict_aiofiles_open)

    config = await MCPConfig.from_json(config_path)

    assert config.servers["msprof-mcp"].env["LABEL"] == CHINESE_TEXT
