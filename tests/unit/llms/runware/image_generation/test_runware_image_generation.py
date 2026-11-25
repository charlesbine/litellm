import uuid
from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm.cost_calculator import response_cost_calculator
from litellm.llms.runware.common_utils import RunwareError
from litellm.llms.runware.image_generation import (
    RunwareImageGenerationConfig,
    get_runware_image_generation_config,
)
from litellm.types.utils import ImageResponse, LlmProviders
from litellm.utils import ProviderConfigManager


def _mock_response(payload: object, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=payload,
        request=httpx.Request("POST", "https://api.runware.ai/v1"),
    )


def test_provider_config_manager_returns_runware_image_generation_config() -> None:
    config: Final = ProviderConfigManager.get_provider_image_generation_config(
        model="runware:100@1",
        provider=LlmProviders.RUNWARE,
    )
    assert isinstance(config, RunwareImageGenerationConfig)


def test_validate_environment_sets_bearer_auth_and_rejects_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUNWARE_API_KEY", raising=False)
    config: Final = get_runware_image_generation_config("runware:100@1")

    with pytest.raises(ValueError, match="RUNWARE_API_KEY"):
        config.validate_environment(
            headers={},
            model="runware:100@1",
            messages=[],
            optional_params={},
            litellm_params={},
        )

    headers: Final = config.validate_environment(
        headers={"X-Custom": "1"},
        model="runware:100@1",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="rw-test-key",
    )
    assert headers["Authorization"] == "Bearer rw-test-key"
    assert headers["Content-Type"] == "application/json"
    assert headers["X-Custom"] == "1"


@pytest.mark.parametrize(
    ("non_default", "expected"),
    [
        ({"size": "512x768", "n": 2}, {"width": 512, "height": 768, "numberResults": 2}),
        ({"size": "auto", "quality": "auto"}, {}),
        (
            {"quality": "low", "response_format": "b64_json", "output_format": "png"},
            {"outputQuality": 60, "outputType": "base64Data", "outputFormat": "PNG"},
        ),
        (
            {"quality": "hd", "response_format": "url", "output_format": "jpeg"},
            {"outputQuality": 95, "outputType": "URL", "outputFormat": "JPG"},
        ),
    ],
)
def test_map_openai_params_translates_standard_fields(
    non_default: dict[str, object],
    expected: dict[str, object],
) -> None:
    config: Final = RunwareImageGenerationConfig()
    mapped: Final = config.map_openai_params(
        non_default_params=non_default,
        optional_params={},
        model="runware:100@1",
        drop_params=False,
    )
    assert mapped == expected


def test_map_openai_params_rejects_unsupported_unless_drop_params() -> None:
    config: Final = RunwareImageGenerationConfig()
    with pytest.raises(ValueError, match="style"):
        config.map_openai_params(
            non_default_params={"style": "vivid"},
            optional_params={},
            model="runware:100@1",
            drop_params=False,
        )

    dropped: Final = config.map_openai_params(
        non_default_params={"style": "vivid", "n": 1},
        optional_params={},
        model="runware:100@1",
        drop_params=True,
    )
    assert dropped == {"numberResults": 1}


def test_transform_response_extracts_url_b64_datauri_and_total_cost() -> None:
    config: Final = RunwareImageGenerationConfig()
    raw: Final = _mock_response(
        {
            "data": [
                {"taskType": "imageInference", "imageURL": "https://im.runware.ai/a.jpg", "cost": 0.0006},
                {"taskType": "imageInference", "imageBase64Data": "QUJD", "cost": 0.0006},
                {"taskType": "imageInference", "imageDataURI": "data:image/png;base64,REVG", "cost": 0.0008},
            ]
        }
    )

    result: Final = config.transform_image_generation_response(
        model="runware:100@1",
        raw_response=raw,
        model_response=ImageResponse(),
        logging_obj=MagicMock(),
        request_data=[],
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    assert result.data is not None
    assert len(result.data) == 3
    assert result.data[0].url == "https://im.runware.ai/a.jpg"
    assert result.data[0].b64_json is None
    assert result.data[1].b64_json == "QUJD"
    assert result.data[2].b64_json == "REVG"
    additional_headers: Final = result.hidden_params["additional_headers"]
    assert isinstance(additional_headers, dict)
    assert additional_headers["llm_provider-x-litellm-response-cost"] == pytest.approx(0.002)

    billed_cost: Final = response_cost_calculator(
        response_object=result,
        model="runware/runware:100@1",
        custom_llm_provider="runware",
        call_type="image_generation",
        optional_params={},
    )
    assert billed_cost == pytest.approx(0.002)


def test_transform_response_raises_runware_error_on_errors_payload() -> None:
    config: Final = RunwareImageGenerationConfig()
    raw: Final = _mock_response(
        {
            "errors": [
                {
                    "code": "invalidModel",
                    "message": "Model not found",
                    "taskUUID": "11111111-1111-4111-8111-111111111111",
                }
            ]
        },
        status_code=200,
    )
    with pytest.raises(RunwareError, match="invalidModel: Model not found"):
        config.transform_image_generation_response(
            model="runware:bad@1",
            raw_response=raw,
            model_response=ImageResponse(),
            logging_obj=MagicMock(),
            request_data=[],
            optional_params={},
            litellm_params={},
            encoding=None,
        )


def test_image_generation_e2e_sends_exact_runware_array_payload() -> None:
    captured: Final[dict[str, object]] = {}
    fixed_uuid: Final = "12345678-1234-4123-8123-123456789abc"

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
                        "imageURL": "https://im.runware.ai/out.webp",
                        "cost": 0.0006,
                    }
                ]
            }
        )

    with (
        patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post", new=fake_post),
        patch("litellm.llms.runware.image_generation.transformation.uuid.uuid4", return_value=uuid.UUID(fixed_uuid)),
    ):
        response: Final = litellm.image_generation(
            model="runware/runware:100@1",
            prompt="a red cube on white background",
            size="512x512",
            n=1,
            quality="high",
            response_format="url",
            api_key="rw-secret",
            steps=4,
            CFGScale=3.5,
            extra_body={"seed": 42},
        )

    assert captured["url"] == "https://api.runware.ai/v1"
    headers: Final = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == "Bearer rw-secret"
    assert captured["json"] == [
        {
            "taskType": "imageInference",
            "taskUUID": fixed_uuid,
            "model": "runware:100@1",
            "positivePrompt": "a red cube on white background",
            "includeCost": True,
            "numberResults": 1,
            "width": 512,
            "height": 512,
            "outputQuality": 95,
            "outputType": "URL",
            "steps": 4,
            "CFGScale": 3.5,
            "seed": 42,
        }
    ]
    assert isinstance(response, ImageResponse)
    assert response.data is not None
    assert response.data[0].url == "https://im.runware.ai/out.webp"
    assert response.hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"] == pytest.approx(0.0006)
