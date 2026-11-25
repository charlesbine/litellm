from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm.llms.novita.common_utils import NovitaError
from litellm.llms.novita.videos.transformation import NovitaVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import decode_video_id_with_provider, encode_video_id_with_provider
from litellm.utils import ProviderConfigManager


def _mock_response(payload: object, url: str, status_code: int = 200, method: str = "POST") -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=payload,
        request=httpx.Request(method, url),
    )


def test_provider_config_manager_returns_novita_video_config() -> None:
    config: Final = ProviderConfigManager.get_provider_video_config(
        model="wan2.6-t2v",
        provider=LlmProviders.NOVITA,
    )
    assert isinstance(config, NovitaVideoConfig)


def test_map_openai_params_translates_seconds_size_and_input_reference() -> None:
    config: Final = NovitaVideoConfig()
    mapped: Final = config.map_openai_params(
        video_create_optional_params={
            "seconds": "5",
            "size": "1280x720",
            "input_reference": "https://example.com/frame.png",
            "seed": 7,
        },
        model="wan2.6-i2v",
        drop_params=False,
    )
    assert mapped == {
        "image": "https://example.com/frame.png",
        "duration": 5,
        "size": "1280*720",
        "seed": 7,
    }


def test_map_openai_params_maps_resolution_tier() -> None:
    config: Final = NovitaVideoConfig()
    mapped: Final = config.map_openai_params(
        video_create_optional_params={"size": "720p", "seconds": "5"},
        model="kling-v2.5-turbo-t2v",
        drop_params=False,
    )
    assert mapped == {"duration": 5, "resolution": "720p"}


def test_video_status_response_maps_novita_statuses_and_progress() -> None:
    config: Final = NovitaVideoConfig()
    encoded_id: Final = encode_video_id_with_provider("nv-vid-1", "novita", "wan2.6-t2v")
    logging_obj: Final = MagicMock()
    logging_obj.litellm_params = {"video_id": encoded_id}

    processing: Final = config.transform_video_status_retrieve_response(
        raw_response=_mock_response(
            {"task": {"task_id": "nv-vid-1", "status": "TASK_STATUS_PROCESSING", "progress_percent": 40}},
            url="https://api.novita.ai/v3/async/task-result?task_id=nv-vid-1",
            method="GET",
        ),
        logging_obj=logging_obj,
        custom_llm_provider="novita",
    )
    assert processing.status == "in_progress"
    assert processing.progress == 40
    assert decode_video_id_with_provider(processing.id)["video_id"] == "nv-vid-1"

    failed: Final = config.transform_video_status_retrieve_response(
        raw_response=_mock_response(
            {"task": {"task_id": "nv-vid-1", "status": "TASK_STATUS_FAILED", "reason": "Invalid resolution"}},
            url="https://api.novita.ai/v3/async/task-result?task_id=nv-vid-1",
            method="GET",
        ),
        logging_obj=logging_obj,
        custom_llm_provider="novita",
    )
    assert failed.status == "failed"
    assert failed.error == {"code": "novita_error", "message": "Invalid resolution"}


def test_video_content_raises_when_still_processing() -> None:
    config: Final = NovitaVideoConfig()
    with pytest.raises(NovitaError, match="not complete yet"):
        config.transform_video_content_response(
            raw_response=_mock_response(
                {"task": {"task_id": "nv-vid-1", "status": "TASK_STATUS_PROCESSING"}},
                url="https://api.novita.ai/v3/async/task-result?task_id=nv-vid-1",
                method="GET",
            ),
            logging_obj=MagicMock(),
        )


def test_unsupported_video_operations_raise_not_implemented() -> None:
    config: Final = NovitaVideoConfig()
    params: Final = GenericLiteLLMParams()
    with pytest.raises(NotImplementedError):
        config.transform_video_list_request("https://api.novita.ai/v3", params, {})
    with pytest.raises(NotImplementedError):
        config.transform_video_delete_request("vid", "https://api.novita.ai/v3", params, {})
    with pytest.raises(NotImplementedError):
        config.transform_video_remix_request("vid", "remix", "https://api.novita.ai/v3", params, {})


def test_video_lifecycle_e2e_create_poll_and_download() -> None:
    post_calls: Final[list[dict[str, object]]] = []
    get_calls: Final[list[str]] = []
    mp4_bytes: Final[bytes] = b"\x00\x00\x00\x18ftypmp42novita"

    def fake_post(self: object, *args: object, **kwargs: object) -> httpx.Response:
        post_calls.append(
            {
                "url": kwargs.get("url"),
                "headers": kwargs.get("headers"),
                "json": kwargs.get("json"),
            }
        )
        return _mock_response(
            {"task_id": "nv-video-999"},
            url="https://api.novita.ai/v3/async/wan2.6-t2v",
        )

    def fake_get(self: object, url: str, *args: object, **kwargs: object) -> httpx.Response:
        get_calls.append(url)
        if "task-result" in url:
            return _mock_response(
                {
                    "task": {"task_id": "nv-video-999", "status": "TASK_STATUS_SUCCEED", "progress_percent": 100},
                    "videos": [{"video_url": "https://cdn.novita.ai/clip.mp4"}],
                },
                url=url,
                method="GET",
            )
        return httpx.Response(status_code=200, content=mp4_bytes, request=httpx.Request("GET", url))

    with (
        patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.post", new=fake_post),
        patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.get", new=fake_get),
    ):
        created: Final = litellm.video_generation(
            model="novita/wan2.6-t2v",
            prompt="calm ocean ripples at sunrise",
            seconds="5",
            size="1280x720",
            api_key="nv-video-key",
            extra_body={"watermark": False},
        )
        assert isinstance(created, VideoObject)
        assert created.status == "queued"
        assert created.seconds == "5"
        assert created.size == "1280x720"

        polled: Final = litellm.video_status(video_id=created.id, api_key="nv-video-key")
        assert isinstance(polled, VideoObject)
        assert polled.status == "completed"
        assert polled.progress == 100

        content: Final = litellm.video_content(video_id=created.id, api_key="nv-video-key")
        assert content == mp4_bytes

    assert len(post_calls) == 1
    assert post_calls[0]["url"] == "https://api.novita.ai/v3/async/wan2.6-t2v"
    headers: Final = post_calls[0]["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == "Bearer nv-video-key"
    assert headers["X-Novita-Source"] == "litellm"
    assert post_calls[0]["json"] == {
        "prompt": "calm ocean ripples at sunrise",
        "duration": 5,
        "size": "1280*720",
        "watermark": False,
    }
    assert get_calls == [
        "https://api.novita.ai/v3/async/task-result?task_id=nv-video-999",
        "https://api.novita.ai/v3/async/task-result?task_id=nv-video-999",
        "https://cdn.novita.ai/clip.mp4",
    ]
