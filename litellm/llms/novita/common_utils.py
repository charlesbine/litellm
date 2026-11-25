import asyncio
import base64
import time
from collections.abc import Awaitable, Callable, Mapping
from types import MappingProxyType
from typing import Final
from urllib.parse import quote

import httpx
from pydantic import ConfigDict, TypeAdapter, ValidationError

from litellm.litellm_core_utils.url_utils import encode_url_path_segments, join_api_base
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # shared HTTP factory lacks typed params
    get_httpx_client,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # shared HTTP factory is private
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import ImageObject, LlmProviders

NOVITA_DEFAULT_MEDIA_API_BASE: Final[str] = "https://api.novita.ai/v3"
NOVITA_DEFAULT_POLL_INTERVAL_SECONDS: Final[float] = 2.0
NOVITA_DEFAULT_MAX_POLL_TIME_SECONDS: Final[float] = 300.0

_SYNC_IMAGE_MODEL_PREFIXES: Final[tuple[str, ...]] = (
    "seedream-4",
    "seedream-5",
    "glm-image",
    "hunyuan-image-3",
)
_SUCCESS_STATUSES: Final[frozenset[str]] = frozenset({"TASK_STATUS_SUCCEED", "SUCCESS", "COMPLETED"})
_FAILURE_STATUSES: Final[frozenset[str]] = frozenset({"TASK_STATUS_FAILED", "FAILED"})
_TASK_STATUS_TO_OPENAI: Final[Mapping[str, str]] = MappingProxyType(
    {
        "TASK_STATUS_QUEUED": "queued",
        "TASK_STATUS_PROCESSING": "in_progress",
        "TASK_STATUS_SUCCEED": "completed",
        "TASK_STATUS_FAILED": "failed",
        "QUEUED": "queued",
        "PENDING": "queued",
        "PROCESSING": "in_progress",
        "SUCCESS": "completed",
        "COMPLETED": "completed",
        "FAILED": "failed",
    }
)

_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))
_OBJECT_LIST: Final = TypeAdapter(tuple[object, ...], config=ConfigDict(hide_input_in_errors=True))


class NovitaError(BaseLLMException):
    pass


def novita_httpx_client(ssl_verify: object = None) -> HTTPHandler:
    return get_httpx_client(params={"ssl_verify": ssl_verify})


def novita_async_httpx_client(ssl_verify: object = None) -> AsyncHTTPHandler:
    return get_async_httpx_client(llm_provider=LlmProviders.NOVITA, params={"ssl_verify": ssl_verify})


def novita_media_api_base(api_base: str | None) -> str:
    raw: Final = (api_base or get_secret_str("NOVITA_API_BASE") or NOVITA_DEFAULT_MEDIA_API_BASE).rstrip("/")
    if "/async/" in raw:
        return raw.split("/async/", 1)[0]
    if raw.endswith("/async"):
        return raw[: -len("/async")]
    if raw.endswith("/openai"):
        return raw[: -len("/openai")]
    return raw


def novita_auth_headers(
    headers: Mapping[str, object],
    api_key: str | None = None,
) -> dict[str, str]:  # mutable-ok: base config validate_environment returns a dict
    resolved_key: Final = api_key or get_secret_str("NOVITA_API_KEY")
    if not resolved_key:
        raise ValueError("Missing Novita AI API Key - set NOVITA_API_KEY in environment variables or pass api_key")
    existing: Final = {str(key): str(value) for key, value in headers.items() if value is not None}
    has_source: Final = any(name.lower() == "x-novita-source" for name in existing)
    return {
        **existing,
        "Authorization": f"Bearer {resolved_key}",
        "Content-Type": "application/json",
        **({} if has_source else {"X-Novita-Source": "litellm"}),
    }


def is_novita_sync_image_model(model: str) -> bool:
    normalized: Final = model.strip("/").lower()
    if normalized.startswith("async/"):
        return False
    return normalized.startswith(_SYNC_IMAGE_MODEL_PREFIXES)


def novita_media_endpoint_url(api_base: str | None, model: str, *, always_async: bool = False) -> str:
    base: Final = novita_media_api_base(api_base)
    encoded: Final = encode_url_path_segments(model.strip("/"), field_name="model")
    if encoded.startswith("async/"):
        return join_api_base(base, encoded)
    if not always_async and is_novita_sync_image_model(encoded):
        return join_api_base(base, encoded)
    return join_api_base(base, f"async/{encoded}")


def novita_task_result_url(api_base: str | None, task_id: str) -> str:
    base: Final = novita_media_api_base(api_base)
    encoded_id: Final = quote(task_id, safe="")
    return f"{base}/async/task-result?task_id={encoded_id}"


def parse_novita_size(size: object) -> tuple[int, int] | None:
    if not isinstance(size, str):
        raise ValueError("Novita size must be formatted as '<width>x<height>', '<width>*<height>', or 'auto'")
    normalized: Final = size.strip().lower()
    if normalized == "auto":
        return None
    separator: Final = "*" if "*" in normalized else "x"
    if normalized.count(separator) != 1:
        raise ValueError(f"Invalid Novita size '{size}'. Expected '<width>x<height>' or '<width>*<height>'.")
    width_text, height_text = normalized.split(separator)
    if not (width_text.isdigit() and height_text.isdigit()):
        raise ValueError(f"Invalid Novita size '{size}'. Width and height must be positive integers.")
    width: Final = int(width_text)
    height: Final = int(height_text)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid Novita size '{size}'. Width and height must be positive integers.")
    return width, height


def parse_novita_json(raw_response: httpx.Response) -> Mapping[str, object]:
    try:
        return _JSON_OBJECT.validate_python(raw_response.json())
    except (ValueError, ValidationError) as exc:
        raise NovitaError(
            status_code=raw_response.status_code,
            message=f"Error parsing Novita response: {exc}",
            headers=raw_response.headers,
        ) from exc


def extract_novita_task_id(payload: Mapping[str, object]) -> str | None:
    top_level: Final = payload.get("task_id")
    if isinstance(top_level, str) and top_level:
        return top_level
    nested: Final = payload.get("data")
    if isinstance(nested, Mapping):
        nested_map: Final = _JSON_OBJECT.validate_python(nested)
        nested_id: Final = nested_map.get("task_id")
        if isinstance(nested_id, str) and nested_id:
            return nested_id
    return None


def _image_url_from_entry(entry: object) -> str | None:
    if isinstance(entry, str) and entry:
        return entry
    if isinstance(entry, Mapping):
        entry_map: Final = _JSON_OBJECT.validate_python(entry)
        url: Final = entry_map.get("image_url") or entry_map.get("url")
        if isinstance(url, str) and url:
            return url
    return None


def extract_novita_image_urls(payload: Mapping[str, object]) -> tuple[str, ...]:
    raw_images: Final = payload.get("images")
    if not isinstance(raw_images, list):
        return ()
    entries: Final = _OBJECT_LIST.validate_python(raw_images)
    return tuple(url for entry in entries if (url := _image_url_from_entry(entry)) is not None)


def extract_novita_task_status(payload: Mapping[str, object]) -> tuple[str, str, int | None]:
    raw_task: Final = payload.get("task")
    task_map: Final[Mapping[str, object]] = (
        _JSON_OBJECT.validate_python(raw_task) if isinstance(raw_task, Mapping) else {}
    )
    raw_status: Final = task_map.get("status") or payload.get("status")
    status_str: Final = raw_status if isinstance(raw_status, str) else ""
    raw_reason: Final = task_map.get("reason") or payload.get("reason") or payload.get("message")
    reason: Final = raw_reason if isinstance(raw_reason, str) else ""
    raw_progress: Final = task_map.get("progress_percent")
    progress: Final = (
        raw_progress
        if isinstance(raw_progress, int) and not isinstance(raw_progress, bool) and 0 <= raw_progress <= 100
        else None
    )
    return status_str, reason, progress


def map_novita_status_to_openai(raw_status: str) -> str:
    return _TASK_STATUS_TO_OPENAI.get(raw_status.upper(), "in_progress")


def _request_or_none(raw_response: httpx.Response) -> httpx.Request | None:
    try:
        return raw_response.request
    except RuntimeError:
        return None


def _polling_headers_from_response(
    raw_response: httpx.Response, api_key: str | None
) -> dict[str, str]:  # mutable-ok: httpx headers
    if api_key:
        return novita_auth_headers({}, api_key=api_key)
    request: Final = _request_or_none(raw_response)
    request_auth: Final = request.headers.get("authorization") if request is not None else None
    if request_auth:
        return {"Authorization": request_auth, "X-Novita-Source": "litellm"}
    return novita_auth_headers({}, api_key=None)


def _api_base_from_response(raw_response: httpx.Response, api_base: str | None) -> str:
    if api_base:
        return novita_media_api_base(api_base)
    request: Final = _request_or_none(raw_response)
    if request is not None:
        request_url: Final = str(request.url)
        marker: Final = "/async/"
        if marker in request_url:
            return request_url.split(marker, 1)[0]
    return novita_media_api_base(None)


def poll_novita_task_sync(
    task_id: str,
    raw_response: httpx.Response,
    client: HTTPHandler,
    *,
    api_base: str | None = None,
    api_key: str | None = None,
    poll_interval: float = NOVITA_DEFAULT_POLL_INTERVAL_SECONDS,
    max_wait: float = NOVITA_DEFAULT_MAX_POLL_TIME_SECONDS,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Mapping[str, object]:
    url: Final = novita_task_result_url(_api_base_from_response(raw_response, api_base), task_id)
    headers: Final = _polling_headers_from_response(raw_response, api_key)
    deadline: Final = time.monotonic() + max_wait

    while True:
        poll_response = client.get(  # pyright: ignore[reportUnknownMemberType]  # HTTP handler stubs are untyped
            url=url, headers=headers
        )
        if poll_response.status_code >= 400:
            raise NovitaError(
                status_code=poll_response.status_code,
                message=f"Novita task polling failed: {poll_response.text}",
                headers=poll_response.headers,
            )
        payload = parse_novita_json(poll_response)
        status, reason, _ = extract_novita_task_status(payload)
        normalized = status.upper()
        if normalized in _SUCCESS_STATUSES:
            return payload
        if normalized in _FAILURE_STATUSES:
            raise NovitaError(
                status_code=400,
                message=reason or f"Novita task {task_id} failed with status {status}",
                headers=poll_response.headers,
            )
        if time.monotonic() >= deadline:
            raise NovitaError(
                status_code=408,
                message=f"Novita task {task_id} timed out after {max_wait} seconds (last status: {status})",
                headers=poll_response.headers,
            )
        sleep_fn(poll_interval)


async def poll_novita_task_async(
    task_id: str,
    raw_response: httpx.Response,
    client: AsyncHTTPHandler,
    *,
    api_base: str | None = None,
    api_key: str | None = None,
    poll_interval: float = NOVITA_DEFAULT_POLL_INTERVAL_SECONDS,
    max_wait: float = NOVITA_DEFAULT_MAX_POLL_TIME_SECONDS,
    sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> Mapping[str, object]:
    url: Final = novita_task_result_url(_api_base_from_response(raw_response, api_base), task_id)
    headers: Final = _polling_headers_from_response(raw_response, api_key)
    deadline: Final = time.monotonic() + max_wait

    while True:
        poll_response = await client.get(  # pyright: ignore[reportUnknownMemberType]  # HTTP handler stubs are untyped
            url=url, headers=headers
        )
        if poll_response.status_code >= 400:
            raise NovitaError(
                status_code=poll_response.status_code,
                message=f"Novita task polling failed: {poll_response.text}",
                headers=poll_response.headers,
            )
        payload = parse_novita_json(poll_response)
        status, reason, _ = extract_novita_task_status(payload)
        normalized = status.upper()
        if normalized in _SUCCESS_STATUSES:
            return payload
        if normalized in _FAILURE_STATUSES:
            raise NovitaError(
                status_code=400,
                message=reason or f"Novita task {task_id} failed with status {status}",
                headers=poll_response.headers,
            )
        if time.monotonic() >= deadline:
            raise NovitaError(
                status_code=408,
                message=f"Novita task {task_id} timed out after {max_wait} seconds (last status: {status})",
                headers=poll_response.headers,
            )
        await sleep_fn(poll_interval)


def novita_image_objects_sync(
    urls: tuple[str, ...],
    response_format: object,
    client: HTTPHandler,
) -> tuple[ImageObject, ...]:
    if response_format != "b64_json":
        return tuple(ImageObject(url=url, b64_json=None, revised_prompt=None) for url in urls)
    return tuple(_download_b64_image_sync(client, url) for url in urls)


def _download_b64_image_sync(client: HTTPHandler, url: str) -> ImageObject:
    response: Final = client.get(  # pyright: ignore[reportUnknownMemberType]  # HTTP handler stubs are untyped
        url=url
    )
    response.raise_for_status()
    encoded: Final = base64.b64encode(response.content).decode("utf-8")
    return ImageObject(url=None, b64_json=encoded, revised_prompt=None)


async def novita_image_objects_async(
    urls: tuple[str, ...],
    response_format: object,
    client: AsyncHTTPHandler,
) -> tuple[ImageObject, ...]:
    if response_format != "b64_json":
        return tuple(ImageObject(url=url, b64_json=None, revised_prompt=None) for url in urls)
    return tuple(await asyncio.gather(*(_download_b64_image_async(client, url) for url in urls)))


async def _download_b64_image_async(client: AsyncHTTPHandler, url: str) -> ImageObject:
    response: Final = await client.get(  # pyright: ignore[reportUnknownMemberType]  # HTTP handler stubs are untyped
        url=url
    )
    response.raise_for_status()
    encoded: Final = base64.b64encode(response.content).decode("utf-8")
    return ImageObject(url=None, b64_json=encoded, revised_prompt=None)
