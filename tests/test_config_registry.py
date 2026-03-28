from pathlib import Path
import json
from importlib.resources import files
import yaml

import pytest

from msagent.configs.registry import ConfigRegistry
from msagent.core.constants import CONFIG_APPROVAL_FILE_NAME


def _load_default_msprof_server() -> dict:
    config_path = files("resources")
    for part in ("configs", "default", "config.mcp.json"):
        config_path = config_path.joinpath(part)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config["mcpServers"]["msprof-mcp"]


def _load_default_msagent_config() -> dict:
    config_path = files("resources")
    for part in ("configs", "default", "agents", "msagent.yml"):
        config_path = config_path.joinpath(part)
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def test_default_msprof_server_uses_packaged_executable() -> None:
    default_msprof_server = _load_default_msprof_server()

    assert default_msprof_server["command"] == "msprof-mcp"
    assert default_msprof_server["args"] == []
    assert default_msprof_server["repair_timeout"] == 30
    assert default_msprof_server["invoke_timeout"] == 60.0


@pytest.mark.asyncio
async def test_config_registry_bootstraps_default_layout(tmp_path: Path) -> None:
    registry = ConfigRegistry(tmp_path)

    await registry.ensure_config_dir()

    config_dir = tmp_path / ".msagent"
    assert config_dir.exists()
    assert (config_dir / "README.md").exists()
    assert (config_dir / "agents").is_dir()
    assert (config_dir / "llms").is_dir()
    assert (config_dir / "sandboxes").is_dir()
    assert (config_dir / "subagents").is_dir()
    assert (config_dir / "config.llms.yml").exists()
    assert (config_dir / "config.mcp.json").exists()
    assert (config_dir / CONFIG_APPROVAL_FILE_NAME.name).exists()

    mcp_config = json.loads((config_dir / "config.mcp.json").read_text())
    assert "msprof-mcp" in mcp_config["mcpServers"]
    msprof_server = mcp_config["mcpServers"]["msprof-mcp"]
    default_msprof_server = _load_default_msprof_server()
    assert msprof_server["args"] == default_msprof_server["args"]
    assert msprof_server["stateful"] == default_msprof_server["stateful"]

    approval_config = json.loads(
        (config_dir / CONFIG_APPROVAL_FILE_NAME.name).read_text(encoding="utf-8")
    )
    assert approval_config["always_allow"] == []
    assert approval_config["always_deny"] == []
    assert any(
        rule == {"name": "run_command", "args": {"command": "sudo\\s+.*"}}
        for rule in approval_config["always_ask"]
    )

    msagent = yaml.safe_load((config_dir / "agents" / "msagent.yml").read_text())
    default_msagent = _load_default_msagent_config()
    assert msagent["tools"]["patterns"] == default_msagent["tools"]["patterns"]


@pytest.mark.asyncio
async def test_config_registry_preserves_existing_mcp_server_config(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / ".msagent"
    config_dir.mkdir(parents=True)
    existing_mcp = {
        "mcpServers": {
            "msprof-mcp": {
                "command": "uvx",
                "args": [
                    "--isolated",
                    "--refresh",
                    "--from",
                    "git+https://gitcode.com/kali20gakki1/msprof_mcp.git",
                    "msprof-mcp",
                ],
                "transport": "stdio",
                "env": {},
                "include": [],
                "exclude": [],
                "enabled": False,
                "stateful": False,
            }
        }
    }
    (config_dir / "config.mcp.json").write_text(
        json.dumps(existing_mcp, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    registry = ConfigRegistry(tmp_path)

    await registry.ensure_config_dir()

    mcp_config = json.loads((config_dir / "config.mcp.json").read_text(encoding="utf-8"))
    msprof_server = mcp_config["mcpServers"]["msprof-mcp"]
    assert msprof_server["args"] == existing_mcp["mcpServers"]["msprof-mcp"]["args"]
    assert msprof_server["stateful"] is False
    assert msprof_server["enabled"] is False


@pytest.mark.asyncio
async def test_config_registry_adds_missing_default_mcp_server(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / ".msagent"
    config_dir.mkdir(parents=True)
    (config_dir / "config.mcp.json").write_text(
        json.dumps({"mcpServers": {}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    registry = ConfigRegistry(tmp_path)

    await registry.ensure_config_dir()

    mcp_config = json.loads((config_dir / "config.mcp.json").read_text(encoding="utf-8"))
    msprof_server = mcp_config["mcpServers"]["msprof-mcp"]
    default_msprof_server = _load_default_msprof_server()
    assert msprof_server == default_msprof_server


@pytest.mark.asyncio
async def test_config_registry_copies_missing_approval_template_into_existing_config_dir(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / ".msagent"
    config_dir.mkdir(parents=True)
    (config_dir / "config.mcp.json").write_text(
        json.dumps({"mcpServers": {}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    registry = ConfigRegistry(tmp_path)

    await registry.ensure_config_dir()

    approval_path = config_dir / CONFIG_APPROVAL_FILE_NAME.name
    assert approval_path.exists()
    approval_config = json.loads(approval_path.read_text(encoding="utf-8"))
    assert approval_config["always_allow"] == []
    assert len(approval_config["always_ask"]) >= 1
