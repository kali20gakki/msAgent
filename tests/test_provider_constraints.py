import pytest

from msagent.configs import LLMConfig, LLMProvider
from msagent.configs.llm import BatchLLMConfig


def test_supported_providers_are_constrained() -> None:
    assert sorted(provider.value for provider in LLMProvider) == [
        "anthropic",
        "google",
        "openai",
    ]


def test_gemini_alias_normalizes_to_google_provider() -> None:
    config = LLMConfig.model_validate(
        {
            "provider": "gemini",
            "model": "gemini-2.5-pro",
            "alias": "default",
            "max_tokens": 0,
            "temperature": 0.0,
            "streaming": True,
        }
    )
    assert config.provider == LLMProvider.GOOGLE


@pytest.mark.asyncio
async def test_batch_llm_config_filters_unsupported_providers(tmp_path) -> None:
    (tmp_path / "openai.yml").write_text(
        "\n".join(
            [
                "provider: openai",
                "model: gpt-5.4",
                "alias: default",
                "max_tokens: 0",
                "temperature: 0.0",
                "streaming: true",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "gemini.yml").write_text(
        "\n".join(
            [
                "provider: gemini",
                "model: gemini-2.5-pro",
                "alias: gemini-main",
                "max_tokens: 0",
                "temperature: 0.0",
                "streaming: true",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "bedrock.yml").write_text(
        "\n".join(
            [
                "provider: bedrock",
                "model: claude-sonnet",
                "alias: legacy-bedrock",
                "max_tokens: 0",
                "temperature: 0.0",
                "streaming: true",
            ]
        ),
        encoding="utf-8",
    )

    batch = await BatchLLMConfig.from_yaml(dir_path=tmp_path)

    assert sorted(batch.llm_names) == ["default", "gemini-main"]
    gemini = batch.get_llm_config("gemini-main")
    assert gemini is not None
    assert gemini.provider == LLMProvider.GOOGLE
