import uuid
from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm.llms.runware.common_utils import RunwareError
from litellm.llms.runware.videos.transformation import RunwareVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import decode_video_id_with_provider, encode_video_id_with_provider
from litellm.utils import ProviderConfigManager


def _mock_response(payload: object, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=payload,
        request=httpx.Request("POST", "https://api.runware.ai/v1"),
    )


def test_provider_config_manager_returns_runware_video_config() -> None:
    config: Final = ProviderConfigManager.get_provider_video_config(
        model="bytedance:2@2",
        provider=LlmProviders.RUNWARE,
    )
    assert isinstance(config, RunwareVideoConfig)


def test_map_openai_params_translates_seconds_size_and_input_reference() -> None:
    config: Final = RunwareVideoConfig()
    mapped: Final = config.map_openai_params(
        video_create_optional_params={
            "seconds": "5",
            "size": "1280x720",
            "input_reference": "https://example.com/start.png",
            "seed": 99,
        },
        model="bytedance:2@2",
        drop_params=False,
    )
    assert mapped == {
        "duration": 5,
        "width": 1280,
        "height": 720,
        "seed": 99,
        "inputs": {
            "frameImages": [{"image": "https://example.com/start.png", "frame": "first"}],
        },
    }


def test_map_openai_params_maps_resolution_tier_and_frame_images() -> None:
    config: Final = RunwareVideoConfig()
    mapped: Final = config.map_openai_params(
        video_create_optional_params={
            "size": "720p",
            "frame_images": [
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}, "frame_type": "first_frame"},
                {"type": "image_url", "image_url": {"url": "https://example.com/b.png"}, "frame_type": "last_frame"},
            ],
        },
        model="bytedance:2@2",
        drop_params=False,
    )
    assert mapped == {
        "resolution": "720p",
        "inputs": {
            "frameImages": [
                {"image": "https://example.com/a.png", "frame": "first"},
                {"image": "https://example.com/b.png", "frame": "last"},
            ]
        },
    }


def test_video_status_response_maps_processing_completed_and_error() -> None:
    config: Final = RunwareVideoConfig()
    task_uuid: Final = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"
    encoded_id: Final = encode_video_id_with_provider(task_uuid, "runware", "bytedance:2@2")
    logging_obj: Final = MagicMock()
    logging_obj.litellm_params = {"video_id": encoded_id}

    processing: Final = config.transform_video_status_retrieve_response(
        raw_response=_mock_response(
            {"data": [{"taskType": "videoInference", "taskUUID": task_uuid, "status": "processing"}]}
        ),
        logging_obj=logging_obj,
        custom_llm_provider="runware",
    )
    assert processing.status == "in_progress"
    assert decode_video_id_with_provider(processing.id)["video_id"] == task_uuid

    completed: Final = config.transform_video_status_retrieve_response(
        raw_response=_mock_response(
            {
                "data": [
                    {
                        "taskType": "videoInference",
                        "taskUUID": task_uuid,
                        "status": "success",
                        "videoURL": "https://vm.runware.ai/v.mp4",
                        "cost": 0.18,
                    }
                ]
            }
        ),
        logging_obj=logging_obj,
        custom_llm_provider="runware",
    )
    assert completed.status == "completed"
    assert completed.usage == {"provider_reported_cost_usd": 0.18}
    assert completed.hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"] == pytest.approx(0.18)

    failed: Final = config.transform_video_status_retrieve_response(
        raw_response=_mock_response(
            {"errors": [{"code": "timeoutProvider", "message": "Provider timed out", "taskUUID": task_uuid}]}
        ),
        logging_obj=logging_obj,
        custom_llm_provider="runware",
    )
    assert failed.status == "failed"
    assert failed.error == {"code": "timeoutProvider", "message": "timeoutProvider: Provider timed out"}


def test_video_content_raises_when_still_processing() -> None:
    config: Final = RunwareVideoConfig()
    with pytest.raises(RunwareError, match="not complete yet"):
        config.transform_video_content_response(
            raw_response=_mock_response({"data": [{"taskType": "videoInference", "status": "processing"}]}),
            logging_obj=MagicMock(),
        )


def test_unsupported_video_operations_raise_not_implemented() -> None:
    config: Final = RunwareVideoConfig()
    params: Final = GenericLiteLLMParams()
    with pytest.raises(NotImplementedError):
        config.transform_video_list_request("https://api.runware.ai/v1", params, {})
    with pytest.raises(NotImplementedError):
        config.transform_video_delete_request("vid", "https://api.runware.ai/v1", params, {})
    with pytest.raises(NotImplementedError):
        config.transform_video_remix_request("vid", "remix", "https://api.runware.ai/v1", params, {})


def test_video_lifecycle_e2e_create_poll_and_download() -> None:
    fixed_uuid: Final = "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"
    post_calls: Final[list[dict[str, object]]] = []
    get_calls: Final[list[str]] = []
    mp4_bytes: Final[bytes] = b"\x00\x00\x00\x18ftypmp42fakevideo"

    def fake_post(self: object, *args: object, **kwargs: object) -> httpx.Response:
        post_calls.append(
            {
                "url": kwargs.get("url"),
                "headers": kwargs.get("headers"),
                "json": kwargs.get("json"),
            }
        )
        body: Final = kwargs.get("json")
        assert isinstance(body, list)
        if body[0]["taskType"] == "videoInference":
            return _mock_response({"data": [{"taskType": "videoInference", "taskUUID": fixed_uuid}]})
        return _mock_response(
            {
                "data": [
                    {
                        "taskType": "videoInference",
                        "taskUUID": fixed_uuid,
                        "status": "success",
                        "videoURL": "https://vm.runware.ai/clip.mp4",
                        "cost": 0.14,
                    }
                ]
            }
        )

    def fake_get(self: object, url: str, *args: object, **kwargs: object) -> httpx.Response:
        get_calls.append(url)
        return httpx.Response(status_code=200, content=mp4_bytes, request=httpx.Request("GET", url))

    with (
        patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post", new=fake_post),
        patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.get", new=fake_get),
        patch("litellm.llms.runware.videos.transformation.uuid.uuid4", return_value=uuid.UUID(fixed_uuid)),
    ):
        created: Final = litellm.video_generation(
            model="runware/bytedance:2@2",
            prompt="waves on a sunny beach",
            seconds="5",
            size="1280x720",
            api_key="rw-video-key",
            extra_body={"fps": 24},
        )
        assert isinstance(created, VideoObject)
        assert created.status == "queued"
        assert created.seconds == "5"
        assert created.size == "1280x720"

        polled: Final = litellm.video_status(video_id=created.id, api_key="rw-video-key")
        assert isinstance(polled, VideoObject)
        assert polled.status == "completed"
        assert polled.usage == {"provider_reported_cost_usd": 0.14}

        content: Final = litellm.video_content(video_id=created.id, api_key="rw-video-key")
        assert content == mp4_bytes

    assert len(post_calls) == 3
    assert post_calls[0]["url"] == "https://api.runware.ai/v1"
    assert post_calls[0]["json"] == [
        {
            "taskType": "videoInference",
            "taskUUID": fixed_uuid,
            "model": "bytedance:2@2",
            "positivePrompt": "waves on a sunny beach",
            "deliveryMethod": "async",
            "includeCost": True,
            "duration": 5,
            "width": 1280,
            "height": 720,
            "fps": 24,
        }
    ]
    assert post_calls[1]["json"] == [{"taskType": "getResponse", "taskUUID": fixed_uuid}]
    assert post_calls[2]["json"] == [{"taskType": "getResponse", "taskUUID": fixed_uuid}]
    assert get_calls == ["https://vm.runware.ai/clip.mp4"]
