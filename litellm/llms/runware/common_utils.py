import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import httpx
from pydantic import ConfigDict, TypeAdapter, ValidationError

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import ImageObject, ImageResponse
from litellm.types.videos.main import VideoObject

RUNWARE_DEFAULT_API_BASE: Final[str] = "https://api.runware.ai/v1"
_RESPONSE_COST_HEADER: Final[str] = "llm_provider-x-litellm-response-cost"

_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))
_JSON_OBJECTS: Final = TypeAdapter(tuple[Mapping[str, object], ...], config=ConfigDict(hide_input_in_errors=True))
_STRING_HEADERS: Final = TypeAdapter(dict[str, object], config=ConfigDict(hide_input_in_errors=True))

_QUALITY_TO_OUTPUT_QUALITY: Final[Mapping[str, int]] = MappingProxyType(
    {
        "low": 60,
        "standard": 85,
        "medium": 85,
        "high": 95,
        "hd": 95,
    }
)
_RESPONSE_FORMAT_TO_OUTPUT_TYPE: Final[Mapping[str, str]] = MappingProxyType(
    {
        "url": "URL",
        "b64_json": "base64Data",
    }
)
_OUTPUT_FORMAT_MAP: Final[Mapping[str, str]] = MappingProxyType(
    {
        "jpg": "JPG",
        "jpeg": "JPG",
        "png": "PNG",
        "webp": "WEBP",
    }
)


class RunwareError(BaseLLMException):
    pass


def runware_api_base(api_base: str | None) -> str:
    resolved: Final = api_base or get_secret_str("RUNWARE_API_BASE") or RUNWARE_DEFAULT_API_BASE
    return resolved.rstrip("/")


def runware_auth_headers(
    headers: Mapping[str, object],
    api_key: str | None = None,
) -> dict[str, str]:  # mutable-ok: base config validate_environment returns a dict
    resolved_key: Final = api_key or get_secret_str("RUNWARE_API_KEY")
    if not resolved_key:
        raise ValueError("RUNWARE_API_KEY is not set")
    existing: Final = {str(key): str(value) for key, value in headers.items() if value is not None}
    return {
        **existing,
        "Authorization": f"Bearer {resolved_key}",
        "Content-Type": "application/json",
    }


def parse_runware_size(size: object) -> tuple[int, int] | None:
    if not isinstance(size, str):
        raise ValueError("Runware size must be formatted as '<width>x<height>' or 'auto'")
    normalized: Final = size.strip().lower()
    if normalized == "auto":
        return None
    if normalized.count("x") != 1:
        raise ValueError(f"Invalid Runware size '{size}'. Expected '<width>x<height>' or 'auto'.")
    width_text, height_text = normalized.split("x")
    if not (width_text.isdigit() and height_text.isdigit()):
        raise ValueError(f"Invalid Runware size '{size}'. Width and height must be positive integers.")
    width: Final = int(width_text)
    height: Final = int(height_text)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid Runware size '{size}'. Width and height must be positive integers.")
    return width, height


def map_runware_quality(quality: object) -> int | None:
    if isinstance(quality, int) and not isinstance(quality, bool):
        if 20 <= quality <= 99:
            return quality
        raise ValueError(f"Runware outputQuality must be between 20 and 99, got {quality}")
    if not isinstance(quality, str):
        raise ValueError("Runware quality must be a string or an integer between 20 and 99")
    normalized: Final = quality.strip().lower()
    if normalized == "auto":
        return None
    mapped: Final = _QUALITY_TO_OUTPUT_QUALITY.get(normalized)
    if mapped is None:
        raise ValueError(f"Unsupported Runware quality '{quality}'")
    return mapped


def map_runware_response_format(response_format: object) -> str | None:
    if not isinstance(response_format, str):
        return None
    normalized: Final = response_format.strip().lower()
    mapped: Final = _RESPONSE_FORMAT_TO_OUTPUT_TYPE.get(normalized)
    if mapped is None:
        raise ValueError(f"Unsupported Runware response_format '{response_format}'. Expected 'url' or 'b64_json'.")
    return mapped


def map_runware_output_format(output_format: object) -> str | None:
    if not isinstance(output_format, str):
        return None
    normalized: Final = output_format.strip().lower()
    mapped: Final = _OUTPUT_FORMAT_MAP.get(normalized)
    if mapped is None:
        raise ValueError(f"Unsupported Runware output_format '{output_format}'. Expected 'jpg', 'png', or 'webp'.")
    return mapped


def finite_non_negative_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric: Final = float(value)
    return numeric if math.isfinite(numeric) and numeric >= 0 else None


def parse_runware_envelope(
    raw_response: httpx.Response,
) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    try:
        payload: Final = _JSON_OBJECT.validate_python(raw_response.json())
    except (ValueError, ValidationError) as exc:
        raise RunwareError(
            status_code=raw_response.status_code,
            message=f"Error parsing Runware response: {exc}",
            headers=raw_response.headers,
        ) from exc

    raw_data: Final = payload.get("data")
    raw_errors: Final = payload.get("errors")
    data_items: Final = _validate_object_list(raw_data)
    error_items: Final = _validate_object_list(raw_errors)
    return data_items, error_items


def _validate_object_list(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, list):
        return ()
    try:
        return _JSON_OBJECTS.validate_python(value)
    except ValidationError:
        return ()


def format_runware_errors(errors: tuple[Mapping[str, object], ...]) -> str:
    messages: Final = tuple(_error_item_message(item) for item in errors)
    non_empty: Final = tuple(msg for msg in messages if msg)
    return "; ".join(non_empty) if non_empty else "Runware returned an unknown error"


def _error_item_message(item: Mapping[str, object]) -> str:
    code: Final = item.get("code")
    message: Final = item.get("message")
    if isinstance(code, str) and code and isinstance(message, str) and message:
        return f"{code}: {message}"
    if isinstance(message, str) and message:
        return message
    if isinstance(code, str) and code:
        return code
    return ""


def _data_uri_base64(uri: str) -> str | None:
    parts: Final = uri.split(",", 1)
    return parts[1] if len(parts) == 2 and parts[1] else None


def _item_to_image_object(item: Mapping[str, object]) -> ImageObject | None:
    b64_data: Final = item.get("imageBase64Data")
    if isinstance(b64_data, str) and b64_data:
        return ImageObject(url=None, b64_json=b64_data, revised_prompt=None)
    data_uri: Final = item.get("imageDataURI")
    if isinstance(data_uri, str) and data_uri:
        return ImageObject(url=None, b64_json=_data_uri_base64(data_uri), revised_prompt=None)
    image_url: Final = item.get("imageURL")
    if isinstance(image_url, str) and image_url:
        return ImageObject(url=image_url, b64_json=None, revised_prompt=None)
    return None


def extract_runware_images(
    data_items: tuple[Mapping[str, object], ...],
) -> tuple[tuple[ImageObject, ...], float | None]:
    images: Final = tuple(obj for item in data_items if (obj := _item_to_image_object(item)) is not None)
    costs: Final = tuple(
        cost for item in data_items if (cost := finite_non_negative_float(item.get("cost"))) is not None
    )
    total_cost: Final = sum(costs) if costs else None
    return images, total_cost


def set_runware_response_cost(response: ImageResponse | VideoObject, cost: float | None) -> None:
    if cost is None or not math.isfinite(cost) or cost < 0:
        return
    hidden_params: Final = response.hidden_params
    existing_headers: Final = hidden_params.get("additional_headers")
    base_headers: Final = (
        _STRING_HEADERS.validate_python(existing_headers) if isinstance(existing_headers, Mapping) else {}
    )
    hidden_params["additional_headers"] = {
        **base_headers,
        _RESPONSE_COST_HEADER: float(cost),
    }
