import time
from collections.abc import Callable, Mapping
from typing import Final, TypeAlias

import httpx
from httpx._types import FileContent, RequestFiles
from pydantic import ConfigDict, TypeAdapter

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
from litellm.llms.novita.common_utils import (
    NovitaError,
    extract_novita_task_id,
    extract_novita_task_status,
    map_novita_status_to_openai,
    novita_auth_headers,
    novita_media_api_base,
    novita_media_endpoint_url,
    novita_task_result_url,
    parse_novita_json,
    parse_novita_size,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.types.videos.main import (
    CharacterObject,
    VideoCreateOptionalRequestParams,
    VideoObject,
)
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_video_id_with_provider,
)

_NOVITA_PROVIDER: Final[str] = LlmProviders.NOVITA.value
_RESOLUTION_TIERS: Final[frozenset[str]] = frozenset({"480p", "540p", "720p", "1080p", "2k", "4k"})
_SUPPORTED_NATIVE_OPENROUTER_FIELDS: Final[frozenset[str]] = frozenset({"duration", "aspect_ratio", "seed"})
_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))
_OBJECT_LIST: Final = TypeAdapter(tuple[object, ...], config=ConfigDict(hide_input_in_errors=True))

_SupportedParams: TypeAlias = list[str]
_VideoParams: TypeAlias = dict[str, object]
_VideoHeaders: TypeAlias = dict[str, str]
_VideoStringParams: TypeAlias = dict[str, str]
_VideoFiles: TypeAlias = list[object]


def _get_novita_async_httpx_client() -> AsyncHTTPHandler:
    return get_async_httpx_client(llm_provider=LlmProviders.NOVITA)


def _parse_duration(value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("Novita seconds must be a numeric value")
    try:
        numeric: Final = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Novita seconds must be a numeric value") from exc
    if numeric <= 0:
        raise ValueError("Novita seconds must be a positive number")
    return int(numeric) if numeric.is_integer() else numeric


def _size_params(size: object) -> Mapping[str, object]:
    if not isinstance(size, str):
        raise ValueError(
            "Novita video size must be '<width>x<height>', '<width>*<height>', or a resolution like '720p'"
        )
    normalized: Final = size.strip().lower()
    if normalized == "auto":
        return {}
    if normalized in _RESOLUTION_TIERS:
        return {"resolution": normalized}
    dimensions: Final = parse_novita_size(size)
    if dimensions is None:
        return {}
    width, height = dimensions
    return {"size": f"{width}*{height}"}


def _numeric_duration(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        numeric: Final = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric > 0 else None


def _task_id_from_logging_obj(logging_obj: object) -> tuple[str | None, str | None, str | None]:
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


def _extract_video_url(payload: Mapping[str, object]) -> str:
    raw_status, reason, _ = extract_novita_task_status(payload)
    mapped_status: Final = map_novita_status_to_openai(raw_status) if raw_status else None
    if mapped_status == "failed":
        raise ValueError(reason or "Novita video generation failed")
    raw_videos: Final = payload.get("videos")
    if isinstance(raw_videos, list):
        entries: Final = _OBJECT_LIST.validate_python(raw_videos)
        for entry in entries:
            if isinstance(entry, str) and entry:
                return entry
            if isinstance(entry, Mapping):
                entry_map = _JSON_OBJECT.validate_python(entry)
                url = entry_map.get("video_url") or entry_map.get("url")
                if isinstance(url, str) and url:
                    return url
    direct_url: Final = payload.get("video_url")
    if isinstance(direct_url, str) and direct_url:
        return direct_url
    if mapped_status in ("queued", "in_progress"):
        raise ValueError(
            "Video generation is not complete yet. Please check status with video_status() before downloading."
        )
    raise ValueError("Novita video result did not include a video_url")


class NovitaVideoConfig(BaseVideoConfig):
    def __init__(
        self,
        sync_client_factory: Callable[[], HTTPHandler] = get_httpx_client,
        async_client_factory: Callable[[], AsyncHTTPHandler] = _get_novita_async_httpx_client,
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
            _NOVITA_PROVIDER,
            drop_params,
            _SUPPORTED_NATIVE_OPENROUTER_FIELDS,
        )
        supported_params: Final = frozenset(self.get_supported_openai_params(model))
        input_reference: Final = video_create_optional_params.get("input_reference")
        image_params: Final[Mapping[str, object]] = (
            {"image": input_reference_url(input_reference)} if input_reference is not None else {}
        )
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
        excluded: Final = supported_params | unsupported
        mapped_params: Final[_VideoParams] = {  # mutable-ok: base class contract returns a dict
            **image_params,
            **duration_params,
            **size_params,
            **{
                key: value
                for key, value in video_create_optional_params.items()
                if key not in excluded and value is not None
            },
        }
        return mapped_params

    def validate_environment(
        self,
        headers: _VideoHeaders,
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> _VideoHeaders:
        resolved_key: Final = api_key or (litellm_params.api_key if litellm_params is not None else None)
        return novita_auth_headers(headers=headers, api_key=resolved_key)

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: _VideoParams,
    ) -> str:
        return novita_media_api_base(api_base)

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: _VideoParams,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
    ) -> tuple[_VideoParams, RequestFiles, str]:
        target_url: Final = novita_media_endpoint_url(api_base, model, always_async=True)
        raw_input: Final = video_create_optional_request_params.get("input")
        if isinstance(raw_input, Mapping):
            input_map: Final = _JSON_OBJECT.validate_python(raw_input)
            merged_input: Final = {"prompt": prompt, **input_map}
            request_data: _VideoParams = {  # mutable-ok: base class contract returns a dict
                **{
                    key: value
                    for key, value in video_create_optional_request_params.items()
                    if key not in ("model", "input")
                },
                "input": merged_input,
            }
            return request_data, [], target_url

        flat_request: Final[_VideoParams] = {  # mutable-ok: base class contract returns a dict
            "prompt": prompt,
            **{key: value for key, value in video_create_optional_request_params.items() if key != "model"},
        }
        return flat_request, [], target_url

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: object,
        custom_llm_provider: str | None = None,
        request_data: Mapping[str, object] | None = None,
    ) -> VideoObject:
        payload: Final = parse_novita_json(raw_response)
        task_id: Final = extract_novita_task_id(payload)
        if task_id is None:
            raise NovitaError(
                status_code=max(raw_response.status_code, 500),
                message="Novita video generation response did not include a task_id",
                headers=raw_response.headers,
            )
        provider: Final = custom_llm_provider or _NOVITA_PROVIDER
        request_params: Final = request_data or {}
        raw_nested_params: Final = request_params.get("parameters")
        nested_params: Final[Mapping[str, object]] = (
            _JSON_OBJECT.validate_python(raw_nested_params) if isinstance(raw_nested_params, Mapping) else {}
        )
        duration: Final = _numeric_duration(request_params.get("duration") or nested_params.get("duration"))
        seconds: Final = (
            (str(int(duration)) if duration.is_integer() else str(duration)) if duration is not None else None
        )
        raw_size: Final = request_params.get("size") or nested_params.get("size")
        raw_resolution: Final = request_params.get("resolution") or nested_params.get("resolution")
        size: Final = (
            raw_size.replace("*", "x")
            if isinstance(raw_size, str)
            else raw_resolution
            if isinstance(raw_resolution, str)
            else None
        )
        resolution: Final = raw_resolution if isinstance(raw_resolution, str) else None
        usage: Final[_VideoParams] = {  # mutable-ok: VideoObject.usage expects a mapping
            key: value
            for key, value in (
                ("duration_seconds", duration),
                ("video_resolution", resolution),
            )
            if value is not None
        }
        video_object: Final = VideoObject(
            id=encode_video_id_with_provider(task_id, provider, model),
            object="video",
            status="queued",
            created_at=int(time.time()),
            model=model,
            seconds=seconds,
            size=size,
        )
        if usage:
            video_object.usage = usage
        return video_object

    @staticmethod
    def _decode_task_id(video_id: str) -> str:
        decoded: Final = decode_video_id_with_provider(video_id)
        task_id: Final = decoded.get("video_id", video_id)
        if not task_id:
            raise ValueError("Novita video_id must contain a task_id")
        return task_id

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
    ) -> tuple[str, _VideoParams]:
        task_id: Final = self._decode_task_id(video_id)
        return novita_task_result_url(api_base, task_id), {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
        custom_llm_provider: str | None = None,
        client: HTTPHandler | None = None,
    ) -> VideoObject:
        payload: Final = parse_novita_json(raw_response)
        raw_task: Final = payload.get("task")
        task_map: Final[Mapping[str, object]] = (
            _JSON_OBJECT.validate_python(raw_task) if isinstance(raw_task, Mapping) else {}
        )
        logged_task_id, model_id, served_model = _task_id_from_logging_obj(logging_obj)
        raw_task_id: Final = task_map.get("task_id")
        task_id: Final = raw_task_id if isinstance(raw_task_id, str) and raw_task_id else (logged_task_id or "")
        raw_status, reason, progress = extract_novita_task_status(payload)
        status: Final = map_novita_status_to_openai(raw_status) if raw_status else "in_progress"
        provider: Final = custom_llm_provider or _NOVITA_PROVIDER
        error: Final[dict[str, object] | None] = (  # mutable-ok: VideoObject.error is a dict
            {"code": "novita_error", "message": reason or "Novita video generation failed"}
            if status == "failed"
            else None
        )
        return VideoObject(
            id=encode_video_id_with_provider(task_id, provider, model_id, served_model=served_model),
            object="video",
            status=status,
            created_at=0,
            model=model_id,
            progress=progress,
            error=error,
        )

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
        variant: str | None = None,
    ) -> tuple[str, _VideoStringParams]:
        task_id: Final = self._decode_task_id(video_id)
        return novita_task_result_url(api_base, task_id), {}

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
        client: HTTPHandler | None = None,
    ) -> bytes:
        payload: Final = parse_novita_json(raw_response)
        try:
            video_url: Final = _extract_video_url(payload)
        except ValueError as exc:
            raise NovitaError(
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
        payload: Final = parse_novita_json(raw_response)
        try:
            video_url: Final = _extract_video_url(payload)
        except ValueError as exc:
            raise NovitaError(
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
        raise NotImplementedError("video remix is not supported for novita")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError("video remix is not supported for novita")

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
        raise NotImplementedError("video listing is not supported for novita")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
        custom_llm_provider: str | None = None,
    ) -> _VideoParams:
        raise NotImplementedError("video listing is not supported for novita")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
    ) -> tuple[str, _VideoParams]:
        raise NotImplementedError("video delete is not supported for novita")

    def transform_video_delete_response(self, raw_response: httpx.Response, logging_obj: object) -> VideoObject:
        raise NotImplementedError("video delete is not supported for novita")

    def transform_video_create_character_request(
        self,
        name: str,
        video: object,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
    ) -> tuple[str, _VideoFiles]:
        raise NotImplementedError("video character creation is not supported for novita")

    def transform_video_create_character_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
    ) -> CharacterObject:
        raise NotImplementedError("video character creation is not supported for novita")

    def transform_video_get_character_request(
        self,
        character_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
    ) -> tuple[str, _VideoParams]:
        raise NotImplementedError("video character retrieval is not supported for novita")

    def transform_video_get_character_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
    ) -> CharacterObject:
        raise NotImplementedError("video character retrieval is not supported for novita")

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
        raise NotImplementedError("video edit is not supported for novita")

    def transform_video_edit_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
        custom_llm_provider: str | None = None,
        request_data: Mapping[str, object] | None = None,
    ) -> VideoObject:
        raise NotImplementedError("video edit is not supported for novita")

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
        raise NotImplementedError("video extension is not supported for novita")

    def transform_video_extension_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError("video extension is not supported for novita")

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: _VideoHeaders | httpx.Headers,
    ) -> BaseLLMException:
        return NovitaError(status_code=status_code, message=error_message, headers=headers)
