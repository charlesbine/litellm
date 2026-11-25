import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Final

import httpx
from pydantic import ConfigDict, TypeAdapter

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_generation.transformation import BaseImageGenerationConfig
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
from litellm.types.llms.openai import AllMessageValues, OpenAIImageGenerationOptionalParams
from litellm.types.utils import ImageResponse
from litellm.utils import EXTRA_BODY_ROUTING_KEYS

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer

_EXTRA_BODY: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))
_SUPPORTED_OPENAI_PARAMS: Final[tuple[OpenAIImageGenerationOptionalParams, ...]] = (
    "n",
    "size",
    "response_format",
    "output_format",
)


class NovitaImageGenerationConfig(BaseImageGenerationConfig):
    def __init__(
        self,
        sync_client_factory: Callable[[object], HTTPHandler] = novita_httpx_client,
        async_client_factory: Callable[[object], AsyncHTTPHandler] = novita_async_httpx_client,
        poll_interval: float = NOVITA_DEFAULT_POLL_INTERVAL_SECONDS,
        max_wait: float = NOVITA_DEFAULT_MAX_POLL_TIME_SECONDS,
        sleep_fn: Callable[[float], None] = time.sleep,
        async_sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._sync_client_factory: Final = sync_client_factory
        self._async_client_factory: Final = async_client_factory
        self._poll_interval: Final = poll_interval
        self._max_wait: Final = max_wait
        self._sleep_fn: Final = sleep_fn
        self._async_sleep_fn: Final = async_sleep_fn

    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIImageGenerationOptionalParams]:  # mutable-ok: base class contract returns a list
        return list(_SUPPORTED_OPENAI_PARAMS)

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: base class contract returns a dict
        supported: Final = frozenset(_SUPPORTED_OPENAI_PARAMS)
        unsupported: Final = tuple(key for key in non_default_params if key not in supported)
        if unsupported and not drop_params:
            raise ValueError(
                f"Parameters {', '.join(unsupported)} are not supported for model {model}. "
                f"Supported parameters are {sorted(supported)}. Set drop_params=True to drop unsupported parameters."
            )

        sync_model: Final = is_novita_sync_image_model(model)
        return {
            **optional_params,
            **self._map_count(non_default_params.get("n"), sync_model=sync_model),
            **self._map_size(non_default_params.get("size"), sync_model=sync_model),
            **self._map_response_format(non_default_params.get("response_format")),
            **self._map_output_format(non_default_params.get("output_format")),
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

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        return novita_media_endpoint_url(api_base, model, always_async=False)

    def validate_environment(
        self,
        headers: Mapping[str, object],
        model: str,
        messages: list[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: base class contract returns a dict
        return novita_auth_headers(headers=headers, api_key=api_key)

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, object],
    ) -> dict[str, object]:  # mutable-ok: base class contract returns a dict
        extra_body: Final = optional_params.get("extra_body")
        extra_fields: Final[Mapping[str, object]] = (
            _EXTRA_BODY.validate_python(extra_body) if isinstance(extra_body, Mapping) else {}
        )
        body_params: Final = {
            **{key: value for key, value in optional_params.items() if key not in ("extra_body", "response_format")},
            **{key: value for key, value in extra_fields.items() if key not in EXTRA_BODY_ROUTING_KEYS},
        }
        return {
            "prompt": prompt,
            **body_params,
        }

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: "LiteLLMLoggingObj",
        request_data: Mapping[str, object],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        initial: Final = parse_novita_json(raw_response)
        client: Final = self._sync_client_factory(litellm_params.get("ssl_verify"))
        final_payload: Final = self._resolve_payload_sync(initial, raw_response, client, litellm_params, api_key)
        urls: Final = extract_novita_image_urls(final_payload)
        if not urls:
            raise NovitaError(
                status_code=max(raw_response.status_code, 500),
                message="Novita image generation completed without returning any images",
                headers=raw_response.headers,
            )
        images: Final = novita_image_objects_sync(urls, optional_params.get("response_format"), client)
        existing: Final = list(model_response.data or ())
        model_response.data = [*existing, *images]
        return model_response

    async def async_transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: "LiteLLMLoggingObj",
        request_data: Mapping[str, object],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        initial: Final = parse_novita_json(raw_response)
        client: Final = self._async_client_factory(litellm_params.get("ssl_verify"))
        final_payload: Final = await self._resolve_payload_async(initial, raw_response, client, litellm_params, api_key)
        urls: Final = extract_novita_image_urls(final_payload)
        if not urls:
            raise NovitaError(
                status_code=max(raw_response.status_code, 500),
                message="Novita image generation completed without returning any images",
                headers=raw_response.headers,
            )
        images: Final = await novita_image_objects_async(urls, optional_params.get("response_format"), client)
        existing: Final = list(model_response.data or ())
        model_response.data = [*existing, *images]
        return model_response

    def _resolve_payload_sync(
        self,
        initial: Mapping[str, object],
        raw_response: httpx.Response,
        client: HTTPHandler,
        litellm_params: Mapping[str, object],
        api_key: str | None,
    ) -> Mapping[str, object]:
        if extract_novita_image_urls(initial):
            return initial
        task_id: Final = extract_novita_task_id(initial)
        if task_id is None:
            return initial
        raw_api_base: Final = litellm_params.get("api_base")
        raw_api_key: Final = api_key or litellm_params.get("api_key")
        return poll_novita_task_sync(
            task_id=task_id,
            raw_response=raw_response,
            client=client,
            api_base=raw_api_base if isinstance(raw_api_base, str) else None,
            api_key=raw_api_key if isinstance(raw_api_key, str) else None,
            poll_interval=self._poll_interval,
            max_wait=self._max_wait,
            sleep_fn=self._sleep_fn,
        )

    async def _resolve_payload_async(
        self,
        initial: Mapping[str, object],
        raw_response: httpx.Response,
        client: AsyncHTTPHandler,
        litellm_params: Mapping[str, object],
        api_key: str | None,
    ) -> Mapping[str, object]:
        if extract_novita_image_urls(initial):
            return initial
        task_id: Final = extract_novita_task_id(initial)
        if task_id is None:
            return initial
        raw_api_base: Final = litellm_params.get("api_base")
        raw_api_key: Final = api_key or litellm_params.get("api_key")
        return await poll_novita_task_async(
            task_id=task_id,
            raw_response=raw_response,
            client=client,
            api_base=raw_api_base if isinstance(raw_api_base, str) else None,
            api_key=raw_api_key if isinstance(raw_api_key, str) else None,
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
