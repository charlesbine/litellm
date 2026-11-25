import base64
from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm.llms.novita.common_utils import NovitaError, novita_media_api_base
from litellm.llms.novita.image_generation import (
    NovitaImageGenerationConfig,
    get_novita_image_generation_config,
)
from litellm.types.utils import ImageResponse, LlmProviders
from litellm.utils import ProviderConfigManager


def _mock_response(payload: object, url: str, status_code: int = 200, method: str = "POST") -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=payload,
        request=httpx.Request(method, url, headers={"Authorization": "Bearer nv-key"}),
    )


def test_provider_config_manager_returns_novita_image_generation_config() -> None:
    config: Final = ProviderConfigManager.get_provider_image_generation_config(
        model="qwen-image-txt2img",
        provider=LlmProviders.NOVITA,
    )
    assert isinstance(config, NovitaImageGenerationConfig)


def test_novita_media_api_base_strips_openai_suffix() -> None:
    assert novita_media_api_base("https://api.novita.ai/v3/openai") == "https://api.novita.ai/v3"
    assert novita_media_api_base("https://api.novita.ai/v3/openai/") == "https://api.novita.ai/v3"
    assert novita_media_api_base("https://api.novita.ai/v3") == "https://api.novita.ai/v3"


@pytest.mark.parametrize(
    ("model", "expected_url"),
    [
        ("seedream-4.5", "https://api.novita.ai/v3/seedream-4.5"),
        ("seedream-5.0-lite", "https://api.novita.ai/v3/seedream-5.0-lite"),
        ("qwen-image-txt2img", "https://api.novita.ai/v3/async/qwen-image-txt2img"),
        ("async/qwen-image-txt2img", "https://api.novita.ai/v3/async/qwen-image-txt2img"),
    ],
)
def test_get_complete_url_routes_sync_and_async_endpoints(model: str, expected_url: str) -> None:
    config: Final = get_novita_image_generation_config(model)
    assert (
        config.get_complete_url(
            api_base="https://api.novita.ai/v3/openai",
            api_key=None,
            model=model,
            optional_params={},
            litellm_params={},
        )
        == expected_url
    )


def test_map_openai_params_formats_size_and_count_per_endpoint_family() -> None:
    config: Final = NovitaImageGenerationConfig()

    async_mapped: Final = config.map_openai_params(
        non_default_params={"size": "1024x768", "n": 2, "output_format": "PNG"},
        optional_params={},
        model="qwen-image-txt2img",
        drop_params=False,
    )
    assert async_mapped == {"size": "1024*768", "image_num": 2, "output_format": "png"}

    sync_mapped: Final = config.map_openai_params(
        non_default_params={"size": "2048x2048", "n": 2},
        optional_params={},
        model="seedream-5.0-lite",
        drop_params=False,
    )
    assert sync_mapped == {
        "size": "2048x2048",
        "sequential_image_generation": "auto",
        "sequential_image_generation_options": {"max_images": 2},
    }


def test_map_openai_params_rejects_quality_unless_drop_params() -> None:
    config: Final = NovitaImageGenerationConfig()
    with pytest.raises(ValueError, match="quality"):
        config.map_openai_params(
            non_default_params={"quality": "hd"},
            optional_params={},
            model="qwen-image-txt2img",
            drop_params=False,
        )
    dropped: Final = config.map_openai_params(
        non_default_params={"quality": "hd", "size": "512x512"},
        optional_params={},
        model="qwen-image-txt2img",
        drop_params=True,
    )
    assert dropped == {"size": "512*512"}


def test_async_image_generation_e2e_polls_task_result_and_passes_extras() -> None:
    post_calls: Final[list[dict[str, object]]] = []
    get_calls: Final[list[str]] = []
    poll_count: Final[list[int]] = [0]

    def fake_post(self: object, *args: object, **kwargs: object) -> httpx.Response:
        post_calls.append(
            {
                "url": kwargs.get("url"),
                "headers": kwargs.get("headers"),
                "json": kwargs.get("json"),
            }
        )
        return _mock_response({"task_id": "nv-task-123"}, url="https://api.novita.ai/v3/async/qwen-image-txt2img")

    def fake_get(self: object, url: str, *args: object, **kwargs: object) -> httpx.Response:
        get_calls.append(url)
        poll_count[0] += 1
        if poll_count[0] == 1:
            return _mock_response(
                {"task": {"task_id": "nv-task-123", "status": "TASK_STATUS_PROCESSING", "progress_percent": 50}},
                url=url,
                method="GET",
            )
        return _mock_response(
            {
                "task": {"task_id": "nv-task-123", "status": "TASK_STATUS_SUCCEED", "progress_percent": 100},
                "images": [{"image_url": "https://cdn.novita.ai/img1.png", "image_type": "png"}],
            },
            url=url,
            method="GET",
        )

    with (
        patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post", new=fake_post),
        patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.get", new=fake_get),
        patch("litellm.llms.novita.image_generation.transformation.time.sleep", return_value=None),
    ):
        response: Final = litellm.image_generation(
            model="novita/qwen-image-txt2img",
            prompt="a red apple on a wooden table",
            size="1024x1024",
            n=1,
            api_key="nv-secret",
            seed=42,
        )

    assert len(post_calls) == 1
    assert post_calls[0]["url"] == "https://api.novita.ai/v3/async/qwen-image-txt2img"
    headers: Final = post_calls[0]["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == "Bearer nv-secret"
    assert headers["X-Novita-Source"] == "litellm"
    assert post_calls[0]["json"] == {
        "prompt": "a red apple on a wooden table",
        "image_num": 1,
        "size": "1024*1024",
        "seed": 42,
    }
    assert get_calls == [
        "https://api.novita.ai/v3/async/task-result?task_id=nv-task-123",
        "https://api.novita.ai/v3/async/task-result?task_id=nv-task-123",
    ]
    assert isinstance(response, ImageResponse)
    assert response.data is not None
    assert response.data[0].url == "https://cdn.novita.ai/img1.png"


def test_sync_seedream_image_generation_e2e_supports_b64_json() -> None:
    fake_png: Final[bytes] = b"\x89PNG\r\nfake"
    post_calls: Final[list[dict[str, object]]] = []

    def fake_post(self: object, *args: object, **kwargs: object) -> httpx.Response:
        post_calls.append({"url": kwargs.get("url"), "json": kwargs.get("json")})
        return _mock_response(
            {"images": ["https://cdn.novita.ai/seedream.png"]},
            url="https://api.novita.ai/v3/seedream-5.0-lite",
        )

    def fake_get(self: object, url: str, *args: object, **kwargs: object) -> httpx.Response:
        return httpx.Response(status_code=200, content=fake_png, request=httpx.Request("GET", url))

    with (
        patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post", new=fake_post),
        patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.get", new=fake_get),
    ):
        response: Final = litellm.image_generation(
            model="novita/seedream-5.0-lite",
            prompt="minimalist ceramic mug",
            size="2048x2048",
            response_format="b64_json",
            api_key="nv-secret",
            watermark=False,
        )

    assert post_calls == [
        {
            "url": "https://api.novita.ai/v3/seedream-5.0-lite",
            "json": {
                "prompt": "minimalist ceramic mug",
                "size": "2048x2048",
                "watermark": False,
            },
        }
    ]
    assert isinstance(response, ImageResponse)
    assert response.data is not None
    assert response.data[0].b64_json == base64.b64encode(fake_png).decode("utf-8")


def test_async_task_failure_raises_novita_error() -> None:
    fake_client: Final = MagicMock()
    fake_client.get.return_value = _mock_response(
        {"task": {"task_id": "bad-task", "status": "TASK_STATUS_FAILED", "reason": "NSFW blocked"}},
        url="https://api.novita.ai/v3/async/task-result?task_id=bad-task",
        method="GET",
    )
    config: Final = NovitaImageGenerationConfig(sync_client_factory=lambda _: fake_client, sleep_fn=lambda _: None)
    initial: Final = _mock_response(
        {"task_id": "bad-task"},
        url="https://api.novita.ai/v3/async/qwen-image-txt2img",
    )
    with pytest.raises(NovitaError, match="NSFW blocked"):
        config.transform_image_generation_response(
            model="qwen-image-txt2img",
            raw_response=initial,
            model_response=ImageResponse(),
            logging_obj=MagicMock(),
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )
