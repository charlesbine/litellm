import base64
import io
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.openrouter.common_utils import OpenRouterException
from litellm.llms.openrouter.image_edit.transformation import OpenRouterImageEditConfig
from litellm.types.router import GenericLiteLLMParams

MODEL = "bytedance-seed/seedream-4.5"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
PNG_URL = "data:image/png;base64," + base64.b64encode(PNG).decode()


def _response(payload: object) -> httpx.Response:
    return httpx.Response(200, json=payload, request=httpx.Request("POST", "https://openrouter.ai/api/v1/images"))


def _request(image: object, prompt: str | None = "make it red", params: dict | None = None) -> dict:
    body, files = OpenRouterImageEditConfig().transform_image_edit_request(
        model=MODEL,
        prompt=prompt,
        image=image,
        image_edit_optional_request_params=params or {},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert files == []
    return body


def test_get_supported_openai_params():
    assert OpenRouterImageEditConfig().get_supported_openai_params(MODEL) == [
        "size",
        "quality",
        "n",
        "response_format",
        "background",
    ]


def test_use_multipart_form_data_returns_false():
    assert OpenRouterImageEditConfig().use_multipart_form_data() is False


def test_map_openai_params_maps_size_and_quality_and_drops_unsupported():
    mapped = OpenRouterImageEditConfig().map_openai_params(
        image_edit_optional_params={
            "size": "1024x1536",
            "quality": "hd",
            "n": 2,
            "background": "transparent",
            "input_fidelity": "high",
        },
        model=MODEL,
        drop_params=False,
    )
    assert mapped == {"aspect_ratio": "2:3", "quality": "high", "n": 2, "background": "transparent"}


@pytest.mark.parametrize(
    ("api_base", "expected"),
    [
        (None, "https://openrouter.ai/api/v1/images"),
        ("https://proxy.example.com/v1", "https://proxy.example.com/v1/images"),
    ],
)
def test_get_complete_url_targets_images_api(api_base: str | None, expected: str):
    with patch("litellm.llms.openrouter.image_generation.transformation.get_secret_str", return_value=None):
        assert (
            OpenRouterImageEditConfig().get_complete_url(model=MODEL, api_base=api_base, litellm_params={}) == expected
        )


def test_validate_environment_sets_bearer_token():
    headers = OpenRouterImageEditConfig().validate_environment(headers={}, model=MODEL, api_key="sk-or-test")
    assert headers["Authorization"] == "Bearer sk-or-test"


@patch("litellm.llms.openrouter.image_edit.transformation.get_secret_str", return_value=None)
def test_validate_environment_missing_api_key_raises(_mock_get_secret):
    with patch.object(litellm, "api_key", None), pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterImageEditConfig().validate_environment(headers={}, model=MODEL)


def test_transform_request_sends_images_as_input_references():
    assert _request(PNG) == {
        "model": MODEL,
        "prompt": "make it red",
        "input_references": [{"type": "image_url", "image_url": {"url": PNG_URL}}],
    }


def test_transform_request_reads_file_objects_without_moving_them():
    image = io.BytesIO(PNG)
    image.seek(4)
    body = _request([image, None, PNG])
    assert body["input_references"] == [
        {"type": "image_url", "image_url": {"url": PNG_URL}},
        {"type": "image_url", "image_url": {"url": PNG_URL}},
    ]
    assert image.tell() == 4


def test_transform_request_merges_params_and_extra_body():
    body = _request(
        PNG,
        prompt=None,
        params={
            "aspect_ratio": "1:1",
            "seed": 7,
            "extra_body": {"provider": {"only": ["seed"]}, "input_references": ["https://evil.example/x.png"]},
        },
    )
    assert body == {
        "provider": {"only": ["seed"]},
        "aspect_ratio": "1:1",
        "seed": 7,
        "model": MODEL,
        "prompt": "",
        "input_references": [{"type": "image_url", "image_url": {"url": PNG_URL}}],
    }


def test_read_image_bytes_rejects_unsupported_type():
    with pytest.raises(ValueError, match="Unsupported image type"):
        OpenRouterImageEditConfig()._read_image_bytes("not-bytes")


def test_transform_response_maps_images_usage_and_cost():
    response = OpenRouterImageEditConfig().transform_image_edit_response(
        model=MODEL,
        raw_response=_response(
            {
                "data": [{"b64_json": "aGVsbG8="}],
                "usage": {
                    "prompt_tokens": 300,
                    "prompt_tokens_details": {"image_tokens": 258},
                    "completion_tokens": 1290,
                    "total_tokens": 1590,
                    "cost": "0.04",
                },
            }
        ),
        logging_obj=MagicMock(),
    )
    assert [image.b64_json for image in response.data] == ["aGVsbG8="]
    assert response.usage is not None
    assert response.usage.input_tokens_details.image_tokens == 258
    assert response.usage.input_tokens_details.text_tokens == 42
    assert response._hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"] == 0.04


def test_transform_response_malformed_payload_names_edit_operation():
    with pytest.raises(OpenRouterException, match="Error transforming OpenRouter image edit response"):
        OpenRouterImageEditConfig().transform_image_edit_response(
            model=MODEL, raw_response=_response({"choices": []}), logging_obj=MagicMock()
        )


def test_get_error_class():
    error = OpenRouterImageEditConfig().get_error_class("boom", 429, {})
    assert isinstance(error, OpenRouterException)
    assert error.status_code == 429


def test_image_edit_posts_to_images_api_with_provider_extras():
    """End to end through litellm.image_edit with the HTTP call mocked."""
    client = HTTPHandler()
    with patch.object(client, "post", return_value=_response({"data": [{"b64_json": "aGVsbG8="}]})) as post:
        response = litellm.image_edit(
            model=f"openrouter/{MODEL}",
            image=io.BytesIO(PNG),
            prompt="make it red",
            size="1024x1024",
            api_key="sk-or-test",
            client=client,
            provider={"zdr": True},
            seed=7,
        )

    assert post.call_args.kwargs["url"] == "https://openrouter.ai/api/v1/images"
    assert post.call_args.kwargs["json"] == {
        "aspect_ratio": "1:1",
        "provider": {"zdr": True},
        "seed": 7,
        "model": MODEL,
        "prompt": "make it red",
        "input_references": [{"type": "image_url", "image_url": {"url": PNG_URL}}],
    }
    assert response.data[0].b64_json == "aGVsbG8="
