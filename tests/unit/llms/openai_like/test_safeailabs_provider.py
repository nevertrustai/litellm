import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache


def test_safeailabs_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("SAFE_AI_KEY", "sail_test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="safeailabs/deepseek/deepseek-v4.1-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "deepseek/deepseek-v4.1-flash"
    assert provider == "safeailabs"
    assert api_key == "sail_test-key"
    assert api_base == "https://us-region.api.safeailabs.ai/v1"


def test_safeailabs_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("SAFE_AI_KEY", "sail_env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model="safeailabs/deepseek/deepseek-v4.1-flash",
        custom_llm_provider=None,
        api_base="https://gbl.api.safeailabs.ai/v1",
        api_key="sail_explicit-key",
    )

    assert provider == "safeailabs"
    assert api_key == "sail_explicit-key"
    assert api_base == "https://gbl.api.safeailabs.ai/v1"


def test_safeailabs_base_url_env_selects_another_region(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("SAFE_AI_KEY", "sail_test-key")
    monkeypatch.setenv("SAFE_AI_BASE_URL", "https://gbl.api.safeailabs.ai/v1")

    _, provider, _, api_base = get_llm_provider(
        model="safeailabs/deepseek/deepseek-v4.1-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "safeailabs"
    assert api_base == "https://gbl.api.safeailabs.ai/v1"


@pytest.mark.usefixtures("local_model_cost_map")
def test_safeailabs_model_info_matches_the_models_endpoint():
    info = litellm.get_model_info("safeailabs/deepseek/deepseek-v4.1-flash")

    assert info["litellm_provider"] == "safeailabs"
    assert info["max_input_tokens"] == 1_048_576
    assert info["max_output_tokens"] == 65_536
    assert info["input_cost_per_token"] == 3e-07
    assert info["output_cost_per_token"] == 1e-06
    assert info["cache_read_input_token_cost"] == 3e-08
    assert info["supports_function_calling"] is True
    assert info["supports_reasoning"] is True


def test_safeailabs_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    safeailabs = next(provider for provider in providers if provider["litellm_provider"] == "safeailabs")

    assert safeailabs["provider"] == "SAFEAILABS"
    assert safeailabs["provider_display_name"] == "Safe AI Labs"
    assert safeailabs["default_model_placeholder"] == "safeailabs/deepseek/deepseek-v4.1-flash"
    assert {field["key"]: field["required"] for field in safeailabs["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_safeailabs_supported_endpoints():
    matrix_path = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    providers = json.loads(matrix_path.read_text())["providers"]

    assert providers["safeailabs"]["endpoints"] == {
        "chat_completions": True,
        "messages": True,
        "responses": True,
        "embeddings": False,
        "image_generations": False,
        "audio_transcriptions": False,
        "audio_speech": False,
        "moderations": False,
        "batches": False,
        "rerank": False,
        "a2a": False,
        "interactions": False,
    }


def test_safeailabs_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://us-region.api.safeailabs.ai/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_safeailabs",
                "object": "chat.completion",
                "created": 1_789_550_000,
                "model": "deepseek/deepseek-v4.1-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from Safe AI Labs"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.completion(
            model="safeailabs/deepseek/deepseek-v4.1-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="sail_test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://us-region.api.safeailabs.ai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer sail_test-key"
    assert body["model"] == "deepseek/deepseek-v4.1-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from Safe AI Labs"


def test_safeailabs_responses_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://us-region.api.safeailabs.ai/v1/responses").respond(
            200,
            json={
                "id": "resp_safeailabs",
                "object": "response",
                "created_at": 1_789_550_000,
                "model": "deepseek/deepseek-v4.1-flash",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_safeailabs",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from Safe AI Labs", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model="safeailabs/deepseek/deepseek-v4.1-flash",
            input="Say hello",
            api_key="sail_test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://us-region.api.safeailabs.ai/v1/responses"
    assert request.headers["authorization"] == "Bearer sail_test-key"
    assert body["model"] == "deepseek/deepseek-v4.1-flash"
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from Safe AI Labs"


@pytest.mark.asyncio
async def test_safeailabs_anthropic_messages_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://us-region.api.safeailabs.ai/v1/messages").respond(
            200,
            json={
                "id": "msg_safeailabs",
                "type": "message",
                "role": "assistant",
                "model": "deepseek/deepseek-v4.1-flash",
                "content": [{"type": "text", "text": "Hello from Safe AI Labs"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model="safeailabs/deepseek/deepseek-v4.1-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="sail_test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://us-region.api.safeailabs.ai/v1/messages"
    assert request.headers["authorization"] == "Bearer sail_test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == "deepseek/deepseek-v4.1-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response["content"][0]["text"] == "Hello from Safe AI Labs"
