import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Final, TypeAlias

import httpx
from httpx._types import FileContent, RequestFiles
from pydantic import ConfigDict, TypeAdapter, ValidationError

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.videos.async_task_utils import (
    input_reference_url,
    polled_video_models,
    unsupported_video_fields,
)
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # shared HTTP factory lacks typed params
    get_httpx_client,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # shared HTTP factory is private
)
from litellm.llms.runware.common_utils import (
    RunwareError,
    finite_non_negative_float,
    format_runware_errors,
    parse_runware_envelope,
    parse_runware_size,
    runware_api_base,
    runware_auth_headers,
    set_runware_response_cost,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.types.videos.main import (
    CharacterObject,
    VideoCreateOptionalRequestParams,
    VideoFrameImage,
    VideoInputReference,
    VideoObject,
)
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_video_id_with_provider,
)

_RUNWARE_PROVIDER: Final[str] = LlmProviders.RUNWARE.value
_STATUS_MAP: Final[Mapping[str, str]] = MappingProxyType(
    {
        "processing": "in_progress",
        "queued": "queued",
        "pending": "queued",
        "success": "completed",
        "completed": "completed",
        "error": "failed",
        "failed": "failed",
    }
)
_RESOLUTION_TIERS: Final[frozenset[str]] = frozenset({"480p", "540p", "720p", "1080p", "2k", "4k"})
_FRAME_POSITION_MAP: Final[Mapping[str, str]] = MappingProxyType(
    {
        "first_frame": "first",
        "last_frame": "last",
    }
)
_SUPPORTED_NATIVE_OPENROUTER_FIELDS: Final[frozenset[str]] = frozenset(
    {"duration", "frame_images", "input_references", "seed"}
)
_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))
_OBJECT_SEQUENCE: Final = TypeAdapter(tuple[object, ...], config=ConfigDict(hide_input_in_errors=True))
_FRAME_IMAGES_ADAPTER: Final = TypeAdapter(tuple[VideoFrameImage, ...])
_INPUT_REFERENCES_ADAPTER: Final = TypeAdapter(tuple[VideoInputReference, ...])

_SupportedParams: TypeAlias = list[str]
_VideoParams: TypeAlias = dict[str, object]
_VideoHeaders: TypeAlias = dict[str, str]
_VideoFiles: TypeAlias = list[object]


def _get_runware_async_httpx_client() -> AsyncHTTPHandler:
    return get_async_httpx_client(llm_provider=LlmProviders.RUNWARE)


def _parse_duration(value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("Runware seconds must be a numeric value")
    try:
        numeric: Final = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Runware seconds must be a numeric value") from exc
    if numeric <= 0:
        raise ValueError("Runware seconds must be a positive number")
    return int(numeric) if numeric.is_integer() else numeric


def _size_params(size: object) -> Mapping[str, object]:
    if not isinstance(size, str):
        raise ValueError("Runware video size must be '<width>x<height>' or a resolution like '720p'")
    normalized: Final = size.strip().lower()
    if normalized == "auto":
        return {}
    if normalized in _RESOLUTION_TIERS:
        return {"resolution": normalized}
    dimensions: Final = parse_runware_size(size)
    if dimensions is None:
        return {}
    width, height = dimensions
    return {"width": width, "height": height}


def _first_task(request_data: object) -> Mapping[str, object]:
    if isinstance(request_data, Mapping):
        return _JSON_OBJECT.validate_python(request_data)
    if isinstance(request_data, Sequence) and not isinstance(request_data, (str, bytes)):
        items: Final = _OBJECT_SEQUENCE.validate_python(request_data)
        if items and isinstance(items[0], Mapping):
            return _JSON_OBJECT.validate_python(items[0])
    return {}


def _request_size_label(task: Mapping[str, object]) -> str | None:
    width: Final = task.get("width")
    height: Final = task.get("height")
    if (
        isinstance(width, int)
        and not isinstance(width, bool)
        and isinstance(height, int)
        and not isinstance(height, bool)
    ):
        return f"{width}x{height}"
    resolution: Final = task.get("resolution")
    return resolution if isinstance(resolution, str) else None


def _response_str(item: Mapping[str, object], key: str) -> str | None:
    value: Final = item.get(key)
    return value if isinstance(value, str) and value else None


def _task_uuid_from_logging_obj(logging_obj: object) -> tuple[str | None, str | None, str | None]:
    raw_params: Final = getattr(logging_obj, "litellm_params", None)
    params_map: Final[Mapping[str, object]] = (
        _JSON_OBJECT.validate_python(raw_params) if isinstance(raw_params, Mapping) else {}
    )
    video_id: Final = params_map.get("video_id")
    if not isinstance(video_id, str) or not video_id:
        return None, None, None
    decoded: Final = decode_video_id_with_provider(video_id)
    model_id, served_model = polled_video_models(logging_obj)
    return decoded.get("video_id") or None, model_id, served_model


def _error_payload(
    errors: tuple[Mapping[str, object], ...],
) -> dict[str, object]:  # mutable-ok: VideoObject.error is a dict
    first_code: Final = _response_str(errors[0], "code") if errors else None
    return {
        "code": first_code or "runware_error",
        "message": format_runware_errors(errors),
    }


def _extract_video_url(
    data_items: tuple[Mapping[str, object], ...], error_items: tuple[Mapping[str, object], ...]
) -> str:
    if error_items and not data_items:
        raise ValueError(format_runware_errors(error_items))
    for item in data_items:
        video_url = _response_str(item, "videoURL")
        if video_url is not None:
            return video_url
    if data_items:
        status: Final = _response_str(data_items[0], "status")
        if status and status.lower() == "processing":
            raise ValueError(
                "Video generation is not complete yet. Please check status with video_status() before downloading."
            )
    if error_items:
        raise ValueError(format_runware_errors(error_items))
    raise ValueError("Runware video result did not include a videoURL")


class RunwareVideoConfig(BaseVideoConfig):
    def __init__(
        self,
        sync_client_factory: Callable[[], HTTPHandler] = get_httpx_client,
        async_client_factory: Callable[[], AsyncHTTPHandler] = _get_runware_async_httpx_client,
    ) -> None:
        super().__init__()
        self._sync_client_factory: Final = sync_client_factory
        self._async_client_factory: Final = async_client_factory

    def get_supported_openai_params(self, model: str) -> _SupportedParams:
        supported_params: Final[_SupportedParams] = [  # mutable-ok: base class contract returns a list
            "model",
            "prompt",
            "input_reference",
            "seconds",
            "size",
            "user",
            "extra_headers",
        ]
        return supported_params

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> _VideoParams:
        unsupported: Final = unsupported_video_fields(
            video_create_optional_params,
            model,
            _RUNWARE_PROVIDER,
            drop_params,
            _SUPPORTED_NATIVE_OPENROUTER_FIELDS,
        )
        supported_params: Final = frozenset(self.get_supported_openai_params(model))
        mapped_inputs: Final = self._build_video_inputs(video_create_optional_params, model, drop_params)
        duration_params: Final[Mapping[str, object]] = (
            {"duration": _parse_duration(video_create_optional_params["seconds"])}
            if "seconds" in video_create_optional_params and video_create_optional_params["seconds"] is not None
            else {}
        )
        size_params: Final[Mapping[str, object]] = (
            _size_params(video_create_optional_params["size"])
            if "size" in video_create_optional_params and video_create_optional_params["size"] is not None
            else {}
        )
        passthrough_excluded: Final = (
            supported_params | unsupported | frozenset({"frame_images", "input_references", "inputs"})
        )
        mapped_params: Final[_VideoParams] = {  # mutable-ok: base class contract returns a dict
            **duration_params,
            **size_params,
            **{
                key: value
                for key, value in video_create_optional_params.items()
                if key not in passthrough_excluded and value is not None
            },
            **({"inputs": dict(mapped_inputs)} if mapped_inputs else {}),
        }
        return mapped_params

    @staticmethod
    def _build_video_inputs(
        params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Mapping[str, object]:
        raw_inputs: Final = params.get("inputs")
        caller_inputs: Final[Mapping[str, object]] = (
            _JSON_OBJECT.validate_python(raw_inputs) if isinstance(raw_inputs, Mapping) else {}
        )
        frame_images_param: Final = params.get("frame_images")
        input_references_param: Final = params.get("input_references")
        input_reference: Final = params.get("input_reference")

        frame_images_field: Final[Mapping[str, object]] = (
            RunwareVideoConfig._map_frame_images(frame_images_param, model, drop_params)
            if frame_images_param is not None
            else (
                {"frameImages": [{"image": input_reference_url(input_reference), "frame": "first"}]}
                if input_reference is not None
                else {}
            )
        )
        ref_images_field: Final[Mapping[str, object]] = (
            RunwareVideoConfig._map_input_references(input_references_param, model, drop_params)
            if input_references_param is not None
            else {}
        )
        return {
            **frame_images_field,
            **ref_images_field,
            **caller_inputs,
        }

    @staticmethod
    def _map_frame_images(value: object, model: str, drop_params: bool) -> Mapping[str, object]:
        try:
            parsed: Final = _FRAME_IMAGES_ADAPTER.validate_python(value)
        except ValidationError as exc:
            if drop_params:
                return {}
            raise ValueError(f"Invalid frame_images for Runware model {model}: {exc}") from exc
        mapped: Final = [
            {
                "image": item["image_url"]["url"],
                "frame": _FRAME_POSITION_MAP[item["frame_type"]],
            }
            for item in parsed
        ]
        return {"frameImages": mapped} if mapped else {}

    @staticmethod
    def _map_input_references(value: object, model: str, drop_params: bool) -> Mapping[str, object]:
        try:
            parsed: Final = _INPUT_REFERENCES_ADAPTER.validate_python(value)
        except ValidationError as exc:
            if drop_params:
                return {}
            raise ValueError(f"Invalid input_references for Runware model {model}: {exc}") from exc
        urls: Final = [item["image_url"]["url"] for item in parsed]
        return {"referenceImages": urls} if urls else {}

    def validate_environment(
        self,
        headers: _VideoHeaders,
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> _VideoHeaders:
        resolved_key: Final = api_key or (litellm_params.api_key if litellm_params is not None else None)
        return runware_auth_headers(headers=headers, api_key=resolved_key)

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: _VideoParams,
    ) -> str:
        return runware_api_base(api_base)

    def transform_video_create_request(  # pyright: ignore[reportIncompatibleMethodOverride]  # Runware HTTP API requires a top-level JSON task array
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: _VideoParams,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
    ) -> tuple[list[_VideoParams], RequestFiles, str]:
        task: Final[_VideoParams] = {  # mutable-ok: JSON task payload for httpx
            "taskType": "videoInference",
            "taskUUID": str(uuid.uuid4()),
            "model": model,
            "positivePrompt": prompt,
            "deliveryMethod": "async",
            "includeCost": True,
            **{key: value for key, value in video_create_optional_request_params.items() if key != "model"},
        }
        return [task], [], runware_api_base(api_base)

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: object,
        custom_llm_provider: str | None = None,
        request_data: object = None,
    ) -> VideoObject:
        data_items, error_items = parse_runware_envelope(raw_response)
        if error_items and not data_items:
            raise RunwareError(
                status_code=max(raw_response.status_code, 400),
                message=format_runware_errors(error_items),
                headers=raw_response.headers,
            )
        first_item: Final[Mapping[str, object]] = data_items[0] if data_items else {}
        sent_task: Final = _first_task(request_data)
        task_uuid: Final = _response_str(first_item, "taskUUID") or _response_str(sent_task, "taskUUID") or ""
        provider: Final = custom_llm_provider or _RUNWARE_PROVIDER
        raw_status: Final = (_response_str(first_item, "status") or "queued").lower()
        status: Final = _STATUS_MAP.get(raw_status, "queued")
        duration: Final = finite_non_negative_float(sent_task.get("duration"))
        seconds: Final = (
            (str(int(duration)) if duration.is_integer() else str(duration)) if duration is not None else None
        )
        size: Final = _request_size_label(sent_task)
        cost: Final = finite_non_negative_float(first_item.get("cost"))
        resolution: Final = _response_str(sent_task, "resolution")
        usage: Final[_VideoParams] = {  # mutable-ok: VideoObject.usage expects a mapping
            key: value
            for key, value in (
                ("duration_seconds", duration),
                ("video_resolution", resolution),
                ("provider_reported_cost_usd", cost),
            )
            if value is not None
        }
        video_object: Final = VideoObject(
            id=encode_video_id_with_provider(task_uuid, provider, model),
            object="video",
            status=status,
            created_at=int(time.time()),
            model=model,
            seconds=seconds,
            size=size,
        )
        if usage:
            video_object.usage = usage
        set_runware_response_cost(video_object, cost)
        return video_object

    @staticmethod
    def _decode_task_uuid(video_id: str) -> str:
        decoded: Final = decode_video_id_with_provider(video_id)
        task_uuid: Final = decoded.get("video_id", video_id)
        if not task_uuid:
            raise ValueError("Runware video_id must contain a taskUUID")
        return task_uuid

    def transform_video_status_retrieve_request(  # pyright: ignore[reportIncompatibleMethodOverride]  # Runware getResponse requires a top-level JSON task array
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
    ) -> tuple[str, list[_VideoParams]]:
        task_uuid: Final = self._decode_task_uuid(video_id)
        return (
            runware_api_base(api_base),
            [{"taskType": "getResponse", "taskUUID": task_uuid}],
        )

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
        custom_llm_provider: str | None = None,
        client: HTTPHandler | None = None,
    ) -> VideoObject:
        data_items, error_items = parse_runware_envelope(raw_response)
        logged_uuid, model_id, served_model = _task_uuid_from_logging_obj(logging_obj)
        provider: Final = custom_llm_provider or _RUNWARE_PROVIDER

        if error_items and not data_items:
            error_uuid: Final = _response_str(error_items[0], "taskUUID") or logged_uuid or ""
            return VideoObject(
                id=encode_video_id_with_provider(error_uuid, provider, model_id, served_model=served_model),
                object="video",
                status="failed",
                created_at=0,
                model=model_id,
                error=_error_payload(error_items),
            )

        first_item: Final[Mapping[str, object]] = data_items[0] if data_items else {}
        task_uuid: Final = _response_str(first_item, "taskUUID") or logged_uuid or ""
        raw_status: Final = (_response_str(first_item, "status") or "processing").lower()
        status: Final = _STATUS_MAP.get(raw_status, "in_progress")
        error: Final = _error_payload(error_items) if error_items and status == "failed" else None
        cost: Final = finite_non_negative_float(first_item.get("cost"))
        video_object: Final = VideoObject(
            id=encode_video_id_with_provider(task_uuid, provider, model_id, served_model=served_model),
            object="video",
            status="failed" if error is not None else status,
            created_at=0,
            model=model_id,
            error=error,
        )
        if cost is not None:
            video_object.usage = {"provider_reported_cost_usd": cost}
            set_runware_response_cost(video_object, cost)
        return video_object

    def transform_video_content_request(  # pyright: ignore[reportIncompatibleMethodOverride]  # Runware getResponse requires a top-level JSON task array
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
        variant: str | None = None,
    ) -> tuple[str, list[_VideoParams]]:
        task_uuid: Final = self._decode_task_uuid(video_id)
        return (
            runware_api_base(api_base),
            [{"taskType": "getResponse", "taskUUID": task_uuid}],
        )

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
        client: HTTPHandler | None = None,
    ) -> bytes:
        data_items, error_items = parse_runware_envelope(raw_response)
        try:
            video_url: Final = _extract_video_url(data_items, error_items)
        except ValueError as exc:
            raise RunwareError(
                status_code=max(raw_response.status_code, 400),
                message=str(exc),
                headers=raw_response.headers,
            ) from exc
        httpx_client: Final = client if client is not None else self._sync_client_factory()
        video_response: Final[httpx.Response] = httpx_client.get(  # pyright: ignore[reportUnknownMemberType]  # HTTP handler stubs are untyped
            video_url
        )
        video_response.raise_for_status()
        return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
        client: AsyncHTTPHandler | None = None,
    ) -> bytes:
        data_items, error_items = parse_runware_envelope(raw_response)
        try:
            video_url: Final = _extract_video_url(data_items, error_items)
        except ValueError as exc:
            raise RunwareError(
                status_code=max(raw_response.status_code, 400),
                message=str(exc),
                headers=raw_response.headers,
            ) from exc
        async_httpx_client: Final = client if client is not None else self._async_client_factory()
        video_response: Final[httpx.Response] = await async_httpx_client.get(  # pyright: ignore[reportUnknownMemberType]  # HTTP handler stubs are untyped
            video_url
        )
        video_response.raise_for_status()
        return video_response.content

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
        extra_body: Mapping[str, object] | None = None,
    ) -> tuple[str, _VideoParams]:
        raise NotImplementedError("video remix is not supported for runware")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError("video remix is not supported for runware")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
        after: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        extra_query: Mapping[str, object] | None = None,
    ) -> tuple[str, _VideoParams]:
        raise NotImplementedError("video listing is not supported for runware")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
        custom_llm_provider: str | None = None,
    ) -> _VideoParams:
        raise NotImplementedError("video listing is not supported for runware")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
    ) -> tuple[str, _VideoParams]:
        raise NotImplementedError("video delete is not supported for runware")

    def transform_video_delete_response(self, raw_response: httpx.Response, logging_obj: object) -> VideoObject:
        raise NotImplementedError("video delete is not supported for runware")

    def transform_video_create_character_request(
        self,
        name: str,
        video: object,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
    ) -> tuple[str, _VideoFiles]:
        raise NotImplementedError("video character creation is not supported for runware")

    def transform_video_create_character_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
    ) -> CharacterObject:
        raise NotImplementedError("video character creation is not supported for runware")

    def transform_video_get_character_request(
        self,
        character_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
    ) -> tuple[str, _VideoParams]:
        raise NotImplementedError("video character retrieval is not supported for runware")

    def transform_video_get_character_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
    ) -> CharacterObject:
        raise NotImplementedError("video character retrieval is not supported for runware")

    def transform_video_edit_request(
        self,
        prompt: str,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
        video_file: FileContent | None = None,
        extra_body: Mapping[str, object] | None = None,
        prefetched_source_data: Mapping[str, object] | None = None,
    ) -> tuple[str, Mapping[str, object], RequestFiles | None]:
        raise NotImplementedError("video edit is not supported for runware")

    def transform_video_edit_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
        custom_llm_provider: str | None = None,
        request_data: Mapping[str, object] | None = None,
    ) -> VideoObject:
        raise NotImplementedError("video edit is not supported for runware")

    def transform_video_extension_request(
        self,
        prompt: str,
        video_id: str,
        seconds: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
        extra_body: Mapping[str, object] | None = None,
    ) -> tuple[str, _VideoParams]:
        raise NotImplementedError("video extension is not supported for runware")

    def transform_video_extension_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError("video extension is not supported for runware")

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: _VideoHeaders | httpx.Headers,
    ) -> BaseLLMException:
        return RunwareError(status_code=status_code, message=error_message, headers=headers)
