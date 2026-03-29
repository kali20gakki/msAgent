from __future__ import annotations

from msagent.configs.agent import AgentConfig, RetryPolicyConfig


def test_retry_policy_accepts_deepagents_aligned_shape() -> None:
    retry = RetryPolicyConfig.model_validate(
        {
            "enabled": True,
            "model": {
                "enabled": True,
                "max_retries": 6,
                "timeout": 150.0,
            },
            "tool": {
                "enabled": True,
                "max_retries": 3,
                "tools": ["run_tool"],
                "retry_on": ["TimeoutError"],
                "on_failure": "error",
                "backoff_factor": 1.5,
                "initial_delay": 2.0,
                "max_delay": 20.0,
                "jitter": False,
            },
        }
    )

    assert retry.model.max_retries == 6
    assert retry.model.timeout == 150.0
    assert retry.tool.max_retries == 3
    assert retry.tool.tools == ["run_tool"]
    assert retry.tool.retry_on == ["TimeoutError"]
    assert retry.tool.on_failure == "error"


def test_retry_policy_migrates_legacy_flat_fields() -> None:
    retry = RetryPolicyConfig.model_validate(
        {
            "enabled": True,
            "llm_max_retries": 8,
            "llm_base_delay": 3.0,
            "llm_max_delay": 40.0,
            "enable_circuit_breaker": True,
            "circuit_breaker_threshold": 9,
            "circuit_breaker_recovery": 30.0,
        }
    )

    assert retry.model.max_retries == 8
    assert retry.tool.max_retries == 8
    assert retry.tool.initial_delay == 3.0
    assert retry.tool.max_delay == 40.0


def test_agent_config_migrate_to_2_4_rewrites_retry_shape() -> None:
    migrated = AgentConfig.migrate(
        {
            "version": "2.3.0",
            "name": "msagent",
            "prompt": "x",
            "llm": {"provider": "openai", "model": "gpt-5.4"},
            "retry": {
                "enabled": True,
                "llm_max_retries": 4,
                "llm_base_delay": 2.0,
                "llm_max_delay": 25.0,
            },
        },
        from_version="2.3.0",
    )

    assert migrated["retry"]["enabled"] is True
    assert migrated["retry"]["model"]["max_retries"] == 4
    assert migrated["retry"]["tool"]["max_retries"] == 4
    assert migrated["retry"]["tool"]["initial_delay"] == 2.0
    assert migrated["retry"]["tool"]["max_delay"] == 25.0
