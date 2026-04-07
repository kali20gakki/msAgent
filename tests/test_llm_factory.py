from __future__ import annotations

from types import SimpleNamespace

from msagent.configs import LLMConfig
from msagent.llms.factory import LLMFactory


def test_llm_factory_maps_gemini_to_google_provider(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_init_chat_model(model_name: str, **kwargs):
        captured["model_name"] = model_name
        captured["kwargs"] = kwargs
        return SimpleNamespace(model_name=model_name, kwargs=kwargs)

    monkeypatch.setattr("msagent.llms.factory.init_chat_model", fake_init_chat_model)

    config = LLMConfig.model_validate(
        {
            "provider": "gemini",
            "model": "gemini-2.5-pro",
            "alias": "default",
            "max_tokens": 0,
            "temperature": 0.2,
            "streaming": True,
            "request_timeout_seconds": 66,
        }
    )

    model = LLMFactory().create(config)

    assert model.model_name == "google_genai:gemini-2.5-pro"
    assert captured["kwargs"]["timeout"] == 66


def test_llm_factory_injects_openai_timeout_and_stream_usage(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_init_chat_model(model_name: str, **kwargs):
        captured["model_name"] = model_name
        captured["kwargs"] = kwargs
        return SimpleNamespace(model_name=model_name, kwargs=kwargs)

    monkeypatch.setattr("msagent.llms.factory.init_chat_model", fake_init_chat_model)

    config = LLMConfig(
        provider="openai",
        model="gpt-5.4",
        alias="default",
        max_tokens=2048,
        temperature=0.1,
        streaming=True,
        request_timeout_seconds=120,
    )

    LLMFactory().create(config)

    assert captured["model_name"] == "openai:gpt-5.4"
    assert captured["kwargs"]["timeout"] == 120
    assert captured["kwargs"]["stream_usage"] is True
    assert captured["kwargs"]["use_responses_api"] is True
    assert captured["kwargs"]["max_tokens"] == 2048


def test_llm_factory_disables_responses_api_for_openai_compatible_endpoint(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_init_chat_model(model_name: str, **kwargs):
        captured["model_name"] = model_name
        captured["kwargs"] = kwargs
        return SimpleNamespace(model_name=model_name, kwargs=kwargs)

    monkeypatch.setattr("msagent.llms.factory.init_chat_model", fake_init_chat_model)

    config = LLMConfig(
        provider="openai",
        model="deepseek-chat",
        alias="default",
        base_url="https://api.deepseek.com/v1",
        max_tokens=0,
        temperature=0.1,
        streaming=True,
        request_timeout_seconds=120,
    )

    LLMFactory().create(config)

    assert captured["model_name"] == "openai:deepseek-chat"
    assert captured["kwargs"]["base_url"] == "https://api.deepseek.com/v1"
    assert captured["kwargs"]["use_responses_api"] is False
    assert captured["kwargs"]["stream_usage"] is True


def test_llm_factory_normalizes_deepseek_base_url_without_v1(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_init_chat_model(model_name: str, **kwargs):
        captured["model_name"] = model_name
        captured["kwargs"] = kwargs
        return SimpleNamespace(model_name=model_name, kwargs=kwargs)

    monkeypatch.setattr("msagent.llms.factory.init_chat_model", fake_init_chat_model)

    config = LLMConfig(
        provider="openai",
        model="deepseek-chat",
        alias="default",
        base_url="https://api.deepseek.com",
        max_tokens=0,
        temperature=0.1,
        streaming=True,
        request_timeout_seconds=120,
    )

    LLMFactory().create(config)

    assert captured["kwargs"]["base_url"] == "https://api.deepseek.com/v1"
    assert captured["kwargs"]["use_responses_api"] is False


def test_llm_factory_applies_retry_override_kwargs(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_init_chat_model(model_name: str, **kwargs):
        captured["model_name"] = model_name
        captured["kwargs"] = kwargs
        return SimpleNamespace(model_name=model_name, kwargs=kwargs)

    monkeypatch.setattr("msagent.llms.factory.init_chat_model", fake_init_chat_model)

    config = LLMConfig(
        provider="openai",
        model="gpt-5.4",
        alias="default",
        max_tokens=0,
        temperature=0.1,
        streaming=True,
        request_timeout_seconds=120,
    )

    LLMFactory().create(config, max_retries=7, timeout_seconds=45)

    assert captured["model_name"] == "openai:gpt-5.4"
    assert captured["kwargs"]["max_retries"] == 7
    assert captured["kwargs"]["timeout"] == 45


def test_llm_factory_disables_trust_env_for_custom_openai_endpoint_by_default(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_init_chat_model(model_name: str, **kwargs):
        captured["model_name"] = model_name
        captured["kwargs"] = kwargs
        return SimpleNamespace(model_name=model_name, kwargs=kwargs)

    monkeypatch.setattr("msagent.llms.factory.init_chat_model", fake_init_chat_model)

    config = LLMConfig(
        provider="openai",
        model="gpt-5.4",
        alias="default",
        base_url="https://gmn.chuangzuoli.com/v1",
        max_tokens=0,
        temperature=0.1,
        streaming=True,
        request_timeout_seconds=120,
    )

    LLMFactory().create(config)

    kwargs = captured["kwargs"]
    http_client = kwargs["http_client"]
    http_async_client = kwargs["http_async_client"]
    assert http_client._trust_env is False
    assert http_async_client._trust_env is False
    http_client.close()
    import asyncio

    asyncio.run(http_async_client.aclose())


def test_llm_factory_respects_explicit_trust_env_for_custom_endpoint(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_init_chat_model(model_name: str, **kwargs):
        captured["model_name"] = model_name
        captured["kwargs"] = kwargs
        return SimpleNamespace(model_name=model_name, kwargs=kwargs)

    monkeypatch.setattr("msagent.llms.factory.init_chat_model", fake_init_chat_model)

    config = LLMConfig(
        provider="openai",
        model="gpt-5.4",
        alias="default",
        base_url="https://gmn.chuangzuoli.com/v1",
        trust_env=True,
        max_tokens=0,
        temperature=0.1,
        streaming=True,
        request_timeout_seconds=120,
    )

    LLMFactory().create(config)

    kwargs = captured["kwargs"]
    assert "http_client" not in kwargs
    assert "http_async_client" not in kwargs


def test_llm_factory_forwards_extra_params(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_init_chat_model(model_name: str, **kwargs):
        captured["model_name"] = model_name
        captured["kwargs"] = kwargs
        return SimpleNamespace(model_name=model_name, kwargs=kwargs)

    monkeypatch.setattr("msagent.llms.factory.init_chat_model", fake_init_chat_model)

    config = LLMConfig(
        provider="openai",
        model="gpt-5.4",
        alias="default",
        max_tokens=0,
        temperature=0.1,
        streaming=True,
        request_timeout_seconds=120,
        params={"openai_proxy": "http://127.0.0.1:7890", "foo": "bar"},
    )

    LLMFactory().create(config)

    kwargs = captured["kwargs"]
    assert kwargs["openai_proxy"] == "http://127.0.0.1:7890"
    assert kwargs["foo"] == "bar"


def test_llm_factory_uses_default_env_api_key_for_anthropic(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_init_chat_model(model_name: str, **kwargs):
        captured["model_name"] = model_name
        captured["kwargs"] = kwargs
        return SimpleNamespace(model_name=model_name, kwargs=kwargs)

    monkeypatch.setattr("msagent.llms.factory.init_chat_model", fake_init_chat_model)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")

    config = LLMConfig(
        provider="anthropic",
        model="claude-sonnet-4-5",
        alias="default",
        base_url="https://anthropic-proxy.example.com",
        max_tokens=0,
        temperature=0.1,
        streaming=True,
        request_timeout_seconds=120,
    )

    LLMFactory().create(config)

    kwargs = captured["kwargs"]
    assert kwargs["api_key"] == "anthropic-key"
    assert kwargs["anthropic_api_key"] == "anthropic-key"
    assert kwargs["base_url"] == "https://anthropic-proxy.example.com"
    assert kwargs["anthropic_api_url"] == "https://anthropic-proxy.example.com"


def test_llm_factory_uses_default_env_api_key_for_gemini(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_init_chat_model(model_name: str, **kwargs):
        captured["model_name"] = model_name
        captured["kwargs"] = kwargs
        return SimpleNamespace(model_name=model_name, kwargs=kwargs)

    monkeypatch.setattr("msagent.llms.factory.init_chat_model", fake_init_chat_model)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")

    config = LLMConfig(
        provider="gemini",
        model="gemini-2.5-pro",
        alias="default",
        base_url="https://generativelanguage.googleapis.com",
        max_tokens=0,
        temperature=0.1,
        streaming=True,
        request_timeout_seconds=120,
    )

    LLMFactory().create(config)

    kwargs = captured["kwargs"]
    assert captured["model_name"] == "google_genai:gemini-2.5-pro"
    assert kwargs["api_key"] == "google-key"
    assert kwargs["google_api_key"] == "google-key"
    assert kwargs["base_url"] == "https://generativelanguage.googleapis.com"
