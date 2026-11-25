import base64
import io
from typing import Final
from unittest.mock import patch

import httpx
import pytest

import litellm
from litellm.llms.novita.image_edit import NovitaImageEditConfig, get_novita_image_edit_config
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import ImageResponse, LlmProviders
from litellm.utils import ProviderConfigManager

_PNG_BYTES: Final[bytes] = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)
_PNG_DATA_URL: Final[str] = f"data:image/png;base64,{base64.b64encode(_PNG_BYTES).decode('utf-8')}"


def _mock_response(payload: object, url: str, status_code: int = 200, method: str = "POST") -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=payload,
        request=httpx.Request(method, url, headers={"Authorization": "Bearer nv-edit-key"}),
    )


def test_provider_config_manager_returns_novita_image_edit_config() -> None:
    config: Final = ProviderConfigManager.get_provider_image_edit_config(
        model="qwen-image-edit",
        provider=LlmProviders.NOVITA,
    )
    assert isinstance(config, NovitaImageEditConfig)
    assert config.use_multipart_form_data() is False


def test_edit_request_sends_images_list_for_seedream_and_image_for_qwen() -> None:
    config: Final = get_novita_image_edit_config("seedream-4.5")
    seedream_body, files = config.transform_image_edit_request(
        model="seedream-4.5",
        prompt="turn it into watercolor",
        image=io.BytesIO(_PNG_BYTES),
        image_edit_optional_request_params={"size": "2048x2048"},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert files == ()
    assert seedream_body == {
        "prompt": "turn it into watercolor",
        "images": [_PNG_DATA_URL],
        "size": "2048x2048",
    }

    qwen_body, _ = config.transform_image_edit_request(
        model="qwen-image-edit",
        prompt="add sunglasses",
        image=io.BytesIO(_PNG_BYTES),
        image_edit_optional_request_params={"mask": io.BytesIO(_PNG_BYTES), "seed": 7},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert qwen_body == {
        "prompt": "add sunglasses",
        "image": _PNG_DATA_URL,
        "mask_image": _PNG_DATA_URL,
        "seed": 7,
    }


def test_edit_request_requires_at_least_one_image() -> None:
    config: Final = NovitaImageEditConfig()
    with pytest.raises(ValueError, match="at least one input image"):
        config.transform_image_edit_request(
            model="qwen-image-edit",
            prompt="edit",
            image=None,
            image_edit_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )


def test_image_edit_e2e_async_polling_sends_exact_url_and_json_body() -> None:
    post_calls: Final[list[dict[str, object]]] = []
    get_calls: Final[list[str]] = []

    def fake_post(self: object, *args: object, **kwargs: object) -> httpx.Response:
        post_calls.append(
            {
                "url": kwargs.get("url"),
                "headers": kwargs.get("headers"),
                "json": kwargs.get("json"),
            }
        )
        return _mock_response(
            {"task_id": "nv-edit-456"},
            url="https://api.novita.ai/v3/async/qwen-image-edit",
        )

    def fake_get(self: object, url: str, *args: object, **kwargs: object) -> httpx.Response:
        get_calls.append(url)
        return _mock_response(
            {
                "task": {"task_id": "nv-edit-456", "status": "TASK_STATUS_SUCCEED"},
                "images": [{"image_url": "https://cdn.novita.ai/edited.png", "image_type": "png"}],
            },
            url=url,
            method="GET",
        )

    with (
        patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post", new=fake_post),
        patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.get", new=fake_get),
    ):
        response: Final = litellm.image_edit(
            model="novita/qwen-image-edit",
            image=io.BytesIO(_PNG_BYTES),
            prompt="make the background sunset orange",
            output_format="png",
            api_key="nv-edit-key",
            seed=12,
        )

    assert len(post_calls) == 1
    assert post_calls[0]["url"] == "https://api.novita.ai/v3/async/qwen-image-edit"
    headers: Final = post_calls[0]["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == "Bearer nv-edit-key"
    assert headers["X-Novita-Source"] == "litellm"
    assert post_calls[0]["json"] == {
        "prompt": "make the background sunset orange",
        "image": _PNG_DATA_URL,
        "output_format": "png",
        "seed": 12,
    }
    assert get_calls == ["https://api.novita.ai/v3/async/task-result?task_id=nv-edit-456"]
    assert isinstance(response, ImageResponse)
    assert response.data is not None
    assert response.data[0].url == "https://cdn.novita.ai/edited.png"
