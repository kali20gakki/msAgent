"""Configuration management for msagent."""

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class LLMConfig(BaseModel):
    """Configuration for LLM providers."""
    
    provider: Literal["openai", "anthropic", "gemini", "custom"] = "openai"
    api_key: str = ""
    base_url: str = ""
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 4096
    
    def is_configured(self) -> bool:
        """Check if the configuration is valid."""
        return bool(self.api_key)


class MCPConfig(BaseModel):
    """Configuration for MCP server."""
    
    name: str
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True


class DeepAgentsConfig(BaseModel):
    """Configuration for deepagents features."""

    skills: list[str] = Field(default_factory=list)
    memory: list[str] = Field(default_factory=list)


def get_default_mcp_servers() -> list[MCPConfig]:
    """Get default MCP servers."""
    return [
        MCPConfig(
            name="msprof-mcp",
            command="msprof-mcp",
            args=[],
            enabled=True
        )
    ]


class AppConfig(BaseSettings):
    """Application configuration."""
    
    # LLM Configuration
    llm: LLMConfig = Field(default_factory=LLMConfig)
    
    # MCP Servers Configuration
    mcp_servers: list[MCPConfig] = Field(default_factory=get_default_mcp_servers)

    # deepagents Configuration
    deepagents: DeepAgentsConfig = Field(default_factory=DeepAgentsConfig)
    
    # UI Configuration
    theme: Literal["dark", "light"] = "dark"
    
    class Config:
        env_prefix = "MSAGENT_"
        env_nested_delimiter = "__"


class ConfigManager:
    """Manages configuration loading and saving."""
    
    CONFIG_DIR = Path.home() / ".config" / "msagent"
    CONFIG_FILE = CONFIG_DIR / "config.json"
    
    def __init__(self):
        self._config: AppConfig | None = None
        self._config_path: Path | None = None
    
    def _ensure_config_dir(self) -> None:
        """Ensure configuration directory exists."""
        self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    def load_config(self) -> AppConfig:
        """Load configuration from file or create default."""
        if self._config is not None:
            return self._config
            
        # Check for local config.json first
        local_config = Path.cwd() / "config.json"
        if local_config.exists():
            try:
                with open(local_config, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._config = AppConfig(**data)
                self._config_path = local_config
                # Ensure global config dir exists anyway for saving global preferences if needed
                self._ensure_config_dir()
                return self._config
            except Exception:
                pass
        
        self._ensure_config_dir()
        
        # Try to load from file
        if self.CONFIG_FILE.exists():
            try:
                with open(self.CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._config = AppConfig(**data)
                self._config_path = self.CONFIG_FILE
                return self._config
            except Exception:
                pass
        
        # Try to load from environment variables
        self._config = AppConfig()
        self._config_path = self.CONFIG_FILE
        
        # Override with environment variables if present
        if os.getenv("OPENAI_API_KEY"):
            self._config.llm.provider = "openai"
            self._config.llm.api_key = os.getenv("OPENAI_API_KEY", "")
            self._config.llm.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        elif os.getenv("ANTHROPIC_API_KEY"):
            self._config.llm.provider = "anthropic"
            self._config.llm.api_key = os.getenv("ANTHROPIC_API_KEY", "")
            self._config.llm.model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        elif os.getenv("GEMINI_API_KEY"):
            self._config.llm.provider = "gemini"
            self._config.llm.api_key = os.getenv("GEMINI_API_KEY", "")
            self._config.llm.model = os.getenv("GEMINI_MODEL", "gemini-pro")
        
        # Custom API configuration
        if os.getenv("CUSTOM_API_KEY"):
            self._config.llm.provider = "custom"
            self._config.llm.api_key = os.getenv("CUSTOM_API_KEY", "")
            self._config.llm.base_url = os.getenv("CUSTOM_BASE_URL", "")
            self._config.llm.model = os.getenv("CUSTOM_MODEL", "")
        
        return self._config
    
    def save_config(self, config: AppConfig) -> None:
        """Save configuration to file."""
        self._config = config
        target_path = self._config_path or self.CONFIG_FILE
        if target_path == self.CONFIG_FILE:
            self._ensure_config_dir()
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(config.model_dump(), f, indent=2, ensure_ascii=False)
    
    def get_config(self) -> AppConfig:
        """Get current configuration."""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def update_llm_config(self, llm_config: LLMConfig) -> None:
        """Update LLM configuration."""
        config = self.get_config()
        config.llm = llm_config
        self.save_config(config)
    
    def add_mcp_server(self, mcp_config: MCPConfig) -> None:
        """Add an MCP server configuration."""
        config = self.get_config()
        # Remove existing server with same name
        config.mcp_servers = [s for s in config.mcp_servers if s.name != mcp_config.name]
        config.mcp_servers.append(mcp_config)
        self.save_config(config)
    
    def remove_mcp_server(self, name: str) -> bool:
        """Remove an MCP server configuration."""
        config = self.get_config()
        original_len = len(config.mcp_servers)
        config.mcp_servers = [s for s in config.mcp_servers if s.name != name]
        if len(config.mcp_servers) < original_len:
            self.save_config(config)
            return True
        return False
    
    def get_mcp_servers(self) -> list[MCPConfig]:
        """Get all enabled MCP server configurations."""
        config = self.get_config()
        return [s for s in config.mcp_servers if s.enabled]


# Global config manager instance
config_manager = ConfigManager()
