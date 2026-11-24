from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.openrouter.common_utils import OpenRouterException
from litellm.llms.openrouter.image_generation.transformation import (
    OpenRouterImageGenerationConfig,
)
from litellm.types.utils import ImageResponse

MODEL = "black-forest-labs/flux-3-image"


def _response(payload: object, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=payload,
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/images"),
    )


def _transform(payload: object) -> ImageResponse:
    return OpenRouterImageGenerationConfig().transform_image_generation_response(
        model=MODEL,
        raw_response=_response(payload),
        model_response=ImageResponse(),
        logging_obj=MagicMock(),
        request_data={},
        optional_params={},
        litellm_params={},
        encoding=None,
    )


def _map(params: dict[str, object], drop_params: bool = False) -> dict:
    return OpenRouterImageGenerationConfig().map_openai_params(
        non_default_params=params, optional_params={}, model=MODEL, drop_params=drop_params
    )


def test_get_supported_openai_params():
    assert OpenRouterImageGenerationConfig().get_supported_openai_params(MODEL) == [
        "n",
        "quality",
        "size",
        "response_format",
    ]


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ("1024x1024", {"aspect_ratio": "1:1"}),
        ("512x512", {"aspect_ratio": "1:1"}),
        ("1536x1024", {"aspect_ratio": "3:2"}),
        ("1792x1024", {"aspect_ratio": "16:9"}),
        ("1024x1536", {"aspect_ratio": "2:3"}),
        ("1024x1792", {"aspect_ratio": "9:16"}),
        ("auto", {"aspect_ratio": "auto"}),
        # Tiers and other explicit pixel sizes are native OpenRouter `size` values.
        ("2K", {"size": "2K"}),
        ("2048x2048", {"size": "2048x2048"}),
    ],
)
def test_map_openai_params_size(size: str, expected: dict[str, str]):
    assert _map({"size": size}) == expected


@pytest.mark.parametrize(
    ("quality", "expected"),
    [
        ("auto", {"quality": "auto"}),
        ("low", {"quality": "low"}),
        ("medium", {"quality": "medium"}),
        ("high", {"quality": "high"}),
        ("standard", {"quality": "medium"}),
        ("hd", {"quality": "high"}),
        ("ultra", {}),
    ],
)
def test_map_openai_params_quality(quality: str, expected: dict[str, str]):
    assert _map({"quality": quality}) == expected


def test_map_openai_params_passes_n_and_response_format():
    assert _map({"n": 2, "response_format": "url"}) == {"n": 2, "response_format": "url"}


def test_map_openai_params_unsupported_param_respects_drop_params():
    assert _map({"style": "vivid"}, drop_params=False) == {"style": "vivid"}
    assert _map({"style": "vivid"}, drop_params=True) == {}


@pytest.mark.parametrize(
    ("api_base", "expected"),
    [
        (None, "https://openrouter.ai/api/v1/images"),
        ("https://openrouter.ai/api/v1", "https://openrouter.ai/api/v1/images"),
        ("https://proxy.example.com/v1/", "https://proxy.example.com/v1/images"),
        ("https://proxy.example.com/v1/images", "https://proxy.example.com/v1/images"),
    ],
)
def test_get_complete_url_targets_images_api(api_base: str | None, expected: str):
    with patch("litellm.llms.openrouter.image_generation.transformation.get_secret_str", return_value=None):
        url = OpenRouterImageGenerationConfig().get_complete_url(
            api_base=api_base, api_key=None, model=MODEL, optional_params={}, litellm_params={}
        )
    assert url == expected


def test_validate_environment_sets_bearer_token():
    headers = OpenRouterImageGenerationConfig().validate_environment(
        headers={}, model=MODEL, messages=[], optional_params={}, litellm_params={}, api_key="sk-or-test"
    )
    assert headers["Authorization"] == "Bearer sk-or-test"


def test_transform_request_builds_images_body():
    body = OpenRouterImageGenerationConfig().transform_image_generation_request(
        model=MODEL,
        prompt="a teapot",
        optional_params={"aspect_ratio": "16:9", "quality": "high", "n": 1},
        litellm_params={},
        headers={},
    )
    assert body == {"aspect_ratio": "16:9", "quality": "high", "n": 1, "model": MODEL, "prompt": "a teapot"}
    assert "messages" not in body and "modalities" not in body


def test_transform_request_flattens_extra_body_without_overriding_owned_fields():
    body = OpenRouterImageGenerationConfig().transform_image_generation_request(
        model=MODEL,
        prompt="a teapot",
        optional_params={
            "aspect_ratio": "1:1",
            "extra_body": {
                "provider": {"only": ["bfl"]},
                "seed": 7,
                "aspect_ratio": "4:3",
                "model": "other/model",
                "prompt": "other prompt",
            },
        },
        litellm_params={},
        headers={},
    )
    assert body == {
        "provider": {"only": ["bfl"]},
        "seed": 7,
        "aspect_ratio": "1:1",
        "model": MODEL,
        "prompt": "a teapot",
    }


def test_transform_response_maps_b64_and_url_images():
    response = _transform(
        {
            "created": 1748372400,
            "data": [
                {"b64_json": "aGVsbG8=", "media_type": "image/png"},
                {"url": "https://storage.example.com/a.png", "media_type": "image/png"},
                {"media_type": "image/png"},
            ],
        }
    )
    assert response.created == 1748372400
    assert [(image.b64_json, image.url) for image in response.data] == [
        ("aGVsbG8=", None),
        (None, "https://storage.example.com/a.png"),
    ]


def test_transform_response_sets_usage_and_provider_cost():
    response = _transform(
        {
            "data": [{"b64_json": "aGVsbG8="}],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 4175,
                "total_tokens": 4187,
                "cost": 0.048,
                "cost_details": {"upstream_inference_cost": 0.048},
            },
        }
    )
    assert response.usage is not None
    assert response.usage.input_tokens == 12
    assert response.usage.input_tokens_details.text_tokens == 12
    assert response.usage.input_tokens_details.image_tokens == 0
    assert response.usage.output_tokens == 4175
    assert response.usage.total_tokens == 4187
    assert response._hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"] == 0.048
    assert response._hidden_params["response_cost_details"] == {"upstream_inference_cost": 0.048}
    assert response._hidden_params["model"] == MODEL


def test_transform_response_without_usage_leaves_cost_unset():
    response = _transform({"data": [{"b64_json": "aGVsbG8="}]})
    assert "additional_headers" not in response._hidden_params


def test_transform_response_json_error():
    raw = httpx.Response(200, content=b"not json", request=httpx.Request("POST", "https://openrouter.ai"))
    with pytest.raises(OpenRouterException) as exc_info:
        OpenRouterImageGenerationConfig().transform_image_generation_response(
            model=MODEL,
            raw_response=raw,
            model_response=ImageResponse(),
            logging_obj=MagicMock(),
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )
    assert "Error parsing OpenRouter image generation response" in str(exc_info.value)


@pytest.mark.parametrize(
    "payload",
    [
        ["not", "an", "object"],
        {},
        {"data": 7},
        {"data": ["not an object"]},
        {"data": [{"b64_json": 7}]},
        {"data": [{"b64_json": "aGVsbG8="}], "usage": "not an object"},
    ],
)
def test_transform_response_wraps_malformed_payloads_without_echoing_them(payload: object):
    with pytest.raises(OpenRouterException) as exc_info:
        _transform(payload)
    message = str(exc_info.value)
    assert message.startswith("Error transforming OpenRouter image generation response: ")
    assert "input_value" not in message
    assert "aGVsbG8=" not in message
    assert exc_info.value.status_code == 500


def test_get_error_class():
    error = OpenRouterImageGenerationConfig().get_error_class("boom", 400, {})
    assert isinstance(error, OpenRouterException)
    assert error.status_code == 400


def test_image_generation_posts_to_images_api_with_provider_extras():
    """End to end through litellm.image_generation with the HTTP call mocked."""
    client = HTTPHandler()
    with patch.object(client, "post", return_value=_response({"data": [{"b64_json": "aGVsbG8="}]})) as post:
        response = litellm.image_generation(
            model=f"openrouter/{MODEL}",
            prompt="a teapot",
            size="1536x1024",
            quality="hd",
            api_key="sk-or-test",
            client=client,
            provider={"only": ["bfl"]},
            seed=7,
        )

    assert post.call_args.kwargs["url"] == "https://openrouter.ai/api/v1/images"
    assert post.call_args.kwargs["json"] == {
        "provider": {"only": ["bfl"]},
        "seed": 7,
        "aspect_ratio": "3:2",
        "quality": "high",
        "model": MODEL,
        "prompt": "a teapot",
    }
    assert response.data[0].b64_json == "aGVsbG8="
