import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Final

import httpx
from httpx._types import RequestFiles
from pydantic import ConfigDict, TypeAdapter

from litellm.images.utils import image_to_data_url
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.novita.common_utils import (
    NOVITA_DEFAULT_MAX_POLL_TIME_SECONDS,
    NOVITA_DEFAULT_POLL_INTERVAL_SECONDS,
    NovitaError,
    extract_novita_image_urls,
    extract_novita_task_id,
    is_novita_sync_image_model,
    novita_async_httpx_client,
    novita_auth_headers,
    novita_httpx_client,
    novita_image_objects_async,
    novita_image_objects_sync,
    novita_media_endpoint_url,
    parse_novita_json,
    parse_novita_size,
    poll_novita_task_async,
    poll_novita_task_sync,
)
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes, ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_OBJECT_LIST: Final = TypeAdapter(tuple[object, ...], config=ConfigDict(hide_input_in_errors=True))
_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))
_SUPPORTED_OPENAI_PARAMS: Final[tuple[str, ...]] = (
    "mask",
    "n",
    "size",
    "response_format",
    "output_format",
)


def _first_item(value: object) -> object:
    if not isinstance(value, list):
        return value
    items: Final = _OBJECT_LIST.validate_python(value)
    return items[0] if items else None


def _logged_edit_options(logging_obj: object) -> tuple[object, str | None, str | None, object]:
    raw_params: Final = getattr(logging_obj, "litellm_params", None)
    params_map: Final[Mapping[str, object]] = (
        _JSON_OBJECT.validate_python(raw_params) if isinstance(raw_params, Mapping) else {}
    )
    response_format: Final = params_map.get("response_format")
    raw_api_base: Final = params_map.get("api_base")
    raw_api_key: Final = params_map.get("api_key")
    ssl_verify: Final = params_map.get("ssl_verify")
    return (
        response_format,
        raw_api_base if isinstance(raw_api_base, str) else None,
        raw_api_key if isinstance(raw_api_key, str) else None,
        ssl_verify,
    )


class NovitaImageEditConfig(BaseImageEditConfig):
    def __init__(
        self,
        sync_client_factory: Callable[[object], HTTPHandler] = novita_httpx_client,
        async_client_factory: Callable[[object], AsyncHTTPHandler] = novita_async_httpx_client,
        poll_interval: float = NOVITA_DEFAULT_POLL_INTERVAL_SECONDS,
        max_wait: float = NOVITA_DEFAULT_MAX_POLL_TIME_SECONDS,
        sleep_fn: Callable[[float], None] = time.sleep,
        async_sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        super().__init__()
        self._sync_client_factory: Final = sync_client_factory
        self._async_client_factory: Final = async_client_factory
        self._poll_interval: Final = poll_interval
        self._max_wait: Final = max_wait
        self._sleep_fn: Final = sleep_fn
        self._async_sleep_fn: Final = async_sleep_fn

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: base class contract returns a list
        return list(_SUPPORTED_OPENAI_PARAMS)

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: base class contract returns a dict
        sync_model: Final = is_novita_sync_image_model(model)
        return {
            **self._map_count(image_edit_optional_params.get("n"), sync_model=sync_model),
            **self._map_size(image_edit_optional_params.get("size"), sync_model=sync_model),
            **self._map_response_format(image_edit_optional_params.get("response_format")),
            **self._map_output_format(image_edit_optional_params.get("output_format")),
            **(
                {"mask": image_edit_optional_params["mask"]}
                if "mask" in image_edit_optional_params and image_edit_optional_params["mask"] is not None
                else {}
            ),
        }

    @staticmethod
    def _map_count(count: object, *, sync_model: bool) -> Mapping[str, object]:
        if count is None:
            return {}
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError(f"Novita n must be a positive integer, got {count!r}")
        if sync_model:
            if count == 1:
                return {}
            return {
                "sequential_image_generation": "auto",
                "sequential_image_generation_options": {"max_images": count},
            }
        return {"image_num": count}

    @staticmethod
    def _map_size(size: object, *, sync_model: bool) -> Mapping[str, object]:
        if size is None:
            return {}
        dimensions: Final = parse_novita_size(size)
        if dimensions is None:
            return {}
        width, height = dimensions
        formatted: Final = f"{width}x{height}" if sync_model else f"{width}*{height}"
        return {"size": formatted}

    @staticmethod
    def _map_response_format(response_format: object) -> Mapping[str, object]:
        if response_format is None:
            return {}
        if not isinstance(response_format, str) or response_format not in ("url", "b64_json"):
            raise ValueError(f"Unsupported Novita response_format '{response_format}'. Expected 'url' or 'b64_json'.")
        return {"response_format": response_format}

    @staticmethod
    def _map_output_format(output_format: object) -> Mapping[str, object]:
        if output_format is None:
            return {}
        if not isinstance(output_format, str):
            raise ValueError(f"Novita output_format must be a string, got {output_format!r}")
        return {"output_format": output_format.strip().lower()}

    def validate_environment(
        self,
        headers: Mapping[str, object],
        model: str,
        api_key: str | None = None,
        litellm_params: Mapping[str, object] | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: base class contract returns a dict
        return novita_auth_headers(headers=headers, api_key=api_key)

    def use_multipart_form_data(self) -> bool:
        return False

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: Mapping[str, object],
    ) -> str:
        return novita_media_endpoint_url(api_base, model, always_async=False)

    def transform_image_edit_request(
        self,
        model: str,
        prompt: str | None,
        image: FileTypes | Sequence[FileTypes | None] | None,
        image_edit_optional_request_params: Mapping[str, object],
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, object],
    ) -> tuple[dict[str, object], RequestFiles]:  # mutable-ok: base class contract returns a dict
        candidates: Final = image if isinstance(image, list) else (image,)
        image_urls: Final = tuple(image_to_data_url(item) for item in candidates if item is not None)
        if not image_urls:
            raise ValueError("Novita image edit requires at least one input image")

        mask: Final = _first_item(image_edit_optional_request_params.get("mask"))
        mask_field: Final[Mapping[str, str]] = {"mask_image": image_to_data_url(mask)} if mask is not None else {}
        provider_params: Final = {
            key: value
            for key, value in image_edit_optional_request_params.items()
            if key not in ("mask", "response_format")
        }
        sync_model: Final = is_novita_sync_image_model(model)
        first_image: Final = image_urls[0]
        image_field: Final[Mapping[str, object]] = (
            {"images": list(image_urls)} if sync_model or len(image_urls) > 1 else {"image": first_image}
        )
        request_body: Final[dict[str, object]] = {  # mutable-ok: base class contract returns a dict
            **({"prompt": prompt} if prompt is not None else {}),
            **image_field,
            **mask_field,
            **provider_params,
        }
        return request_body, ()

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> ImageResponse:
        initial: Final = parse_novita_json(raw_response)
        response_format, api_base, api_key, ssl_verify = _logged_edit_options(logging_obj)
        client: Final = self._sync_client_factory(ssl_verify)
        final_payload: Final = self._resolve_payload_sync(initial, raw_response, client, api_base, api_key)
        urls: Final = extract_novita_image_urls(final_payload)
        if not urls:
            raise NovitaError(
                status_code=max(raw_response.status_code, 500),
                message="Novita image edit completed without returning any images",
                headers=raw_response.headers,
            )
        model_response: Final = ImageResponse()
        model_response.data = list(novita_image_objects_sync(urls, response_format, client))
        return model_response

    async def async_transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> ImageResponse:
        initial: Final = parse_novita_json(raw_response)
        response_format, api_base, api_key, ssl_verify = _logged_edit_options(logging_obj)
        client: Final = self._async_client_factory(ssl_verify)
        final_payload: Final = await self._resolve_payload_async(initial, raw_response, client, api_base, api_key)
        urls: Final = extract_novita_image_urls(final_payload)
        if not urls:
            raise NovitaError(
                status_code=max(raw_response.status_code, 500),
                message="Novita image edit completed without returning any images",
                headers=raw_response.headers,
            )
        model_response: Final = ImageResponse()
        model_response.data = list(await novita_image_objects_async(urls, response_format, client))
        return model_response

    def _resolve_payload_sync(
        self,
        initial: Mapping[str, object],
        raw_response: httpx.Response,
        client: HTTPHandler,
        api_base: str | None,
        api_key: str | None,
    ) -> Mapping[str, object]:
        if extract_novita_image_urls(initial):
            return initial
        task_id: Final = extract_novita_task_id(initial)
        if task_id is None:
            return initial
        return poll_novita_task_sync(
            task_id=task_id,
            raw_response=raw_response,
            client=client,
            api_base=api_base,
            api_key=api_key,
            poll_interval=self._poll_interval,
            max_wait=self._max_wait,
            sleep_fn=self._sleep_fn,
        )

    async def _resolve_payload_async(
        self,
        initial: Mapping[str, object],
        raw_response: httpx.Response,
        client: AsyncHTTPHandler,
        api_base: str | None,
        api_key: str | None,
    ) -> Mapping[str, object]:
        if extract_novita_image_urls(initial):
            return initial
        task_id: Final = extract_novita_task_id(initial)
        if task_id is None:
            return initial
        return await poll_novita_task_async(
            task_id=task_id,
            raw_response=raw_response,
            client=client,
            api_base=api_base,
            api_key=api_key,
            poll_interval=self._poll_interval,
            max_wait=self._max_wait,
            sleep_fn=self._async_sleep_fn,
        )

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, object] | httpx.Headers,
    ) -> BaseLLMException:
        return NovitaError(status_code=status_code, message=error_message, headers=headers)
