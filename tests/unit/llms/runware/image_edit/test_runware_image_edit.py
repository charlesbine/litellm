import base64
import io
import uuid
from typing import Final
from unittest.mock import patch

import httpx
import pytest

import litellm
from litellm.llms.runware.image_edit import RunwareImageEditConfig, get_runware_image_edit_config
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import ImageResponse, LlmProviders
from litellm.utils import ProviderConfigManager

_PNG_BYTES: Final[bytes] = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)
_PNG_DATA_URL: Final[str] = f"data:image/png;base64,{base64.b64encode(_PNG_BYTES).decode('utf-8')}"


def _mock_response(payload: object, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=payload,
        request=httpx.Request("POST", "https://api.runware.ai/v1"),
    )


def test_provider_config_manager_returns_runware_image_edit_config() -> None:
    config: Final = ProviderConfigManager.get_provider_image_edit_config(
        model="bfl:3@1",
        provider=LlmProviders.RUNWARE,
    )
    assert isinstance(config, RunwareImageEditConfig)
    assert config.use_multipart_form_data() is False


def test_edit_request_uses_seed_and_mask_when_mask_provided() -> None:
    config: Final = get_runware_image_edit_config("runware:101@1")
    fixed_uuid: Final = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    with patch("litellm.llms.runware.image_edit.transformation.uuid.uuid4", return_value=uuid.UUID(fixed_uuid)):
        body, files = config.transform_image_edit_request(
            model="runware:101@1",
            prompt="replace background with blue sky",
            image=io.BytesIO(_PNG_BYTES),
            image_edit_optional_request_params={"mask": io.BytesIO(_PNG_BYTES), "width": 512, "height": 512},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

    assert files == ()
    assert body == [
        {
            "taskType": "imageInference",
            "taskUUID": fixed_uuid,
            "model": "runware:101@1",
            "positivePrompt": "replace background with blue sky",
            "includeCost": True,
            "width": 512,
            "height": 512,
            "inputs": {
                "seedImage": _PNG_DATA_URL,
                "maskImage": _PNG_DATA_URL,
            },
        }
    ]


def test_edit_request_uses_seed_image_when_strength_provided() -> None:
    config: Final = RunwareImageEditConfig()
    fixed_uuid: Final = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    with patch("litellm.llms.runware.image_edit.transformation.uuid.uuid4", return_value=uuid.UUID(fixed_uuid)):
        body, _ = config.transform_image_edit_request(
            model="runware:101@1",
            prompt="watercolor painting",
            image=_PNG_BYTES,
            image_edit_optional_request_params={"strength": 0.8},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )

    assert body[0]["strength"] == 0.8
    assert body[0]["inputs"] == {"seedImage": _PNG_DATA_URL}


def test_edit_request_requires_at_least_one_image() -> None:
    config: Final = RunwareImageEditConfig()
    with pytest.raises(ValueError, match="at least one input image"):
        config.transform_image_edit_request(
            model="bfl:3@1",
            prompt="edit",
            image=None,
            image_edit_optional_request_params={},
            litellm_params=GenericLiteLLMParams(),
            headers={},
        )


def test_image_edit_e2e_sends_exact_runware_json_array_and_records_cost() -> None:
    captured: Final[dict[str, object]] = {}
    fixed_uuid: Final = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"

    def fake_post(self: object, *args: object, **kwargs: object) -> httpx.Response:
        captured["url"] = kwargs.get("url")
        captured["headers"] = kwargs.get("headers")
        captured["json"] = kwargs.get("json")
        return _mock_response(
            {
                "data": [
                    {
                        "taskType": "imageInference",
                        "taskUUID": fixed_uuid,
                        "imageURL": "https://im.runware.ai/edited.png",
                        "cost": 0.04,
                    }
                ]
            }
        )

    with (
        patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post", new=fake_post),
        patch("litellm.llms.runware.image_edit.transformation.uuid.uuid4", return_value=uuid.UUID(fixed_uuid)),
    ):
        response: Final = litellm.image_edit(
            model="runware/bfl:3@1",
            image=io.BytesIO(_PNG_BYTES),
            prompt="make the cube blue",
            size="1024x1024",
            n=1,
            api_key="rw-edit-key",
            seed=7,
        )

    assert captured["url"] == "https://api.runware.ai/v1"
    headers: Final = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == "Bearer rw-edit-key"
    assert captured["json"] == [
        {
            "taskType": "imageInference",
            "taskUUID": fixed_uuid,
            "model": "bfl:3@1",
            "positivePrompt": "make the cube blue",
            "includeCost": True,
            "numberResults": 1,
            "width": 1024,
            "height": 1024,
            "seed": 7,
            "inputs": {
                "referenceImages": [_PNG_DATA_URL],
            },
        }
    ]
    assert isinstance(response, ImageResponse)
    assert response.data is not None
    assert response.data[0].url == "https://im.runware.ai/edited.png"
    assert response.hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"] == pytest.approx(0.04)
