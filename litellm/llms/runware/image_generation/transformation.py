import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

import httpx
from pydantic import ConfigDict, TypeAdapter

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.llms.runware.common_utils import (
    RunwareError,
    extract_runware_images,
    format_runware_errors,
    map_runware_output_format,
    map_runware_quality,
    map_runware_response_format,
    parse_runware_envelope,
    parse_runware_size,
    runware_api_base,
    runware_auth_headers,
    set_runware_response_cost,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import ImageResponse
from litellm.utils import EXTRA_BODY_ROUTING_KEYS

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer

_EXTRA_BODY: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))
_SUPPORTED_OPENAI_PARAMS: Final[tuple[OpenAIImageGenerationOptionalParams, ...]] = (
    "n",
    "size",
    "quality",
    "response_format",
    "output_format",
)


class RunwareImageGenerationConfig(BaseImageGenerationConfig):
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

        mapped: Final = {
            **optional_params,
            **self._map_count(non_default_params.get("n")),
            **self._map_size(non_default_params.get("size")),
            **self._map_quality(non_default_params.get("quality")),
            **self._map_response_format(non_default_params.get("response_format")),
            **self._map_output_format(non_default_params.get("output_format")),
        }
        return mapped

    @staticmethod
    def _map_count(count: object) -> Mapping[str, object]:
        if count is None:
            return {}
        if isinstance(count, int) and not isinstance(count, bool) and count > 0:
            return {"numberResults": count}
        raise ValueError(f"Runware n must be a positive integer, got {count!r}")

    @staticmethod
    def _map_size(size: object) -> Mapping[str, object]:
        if size is None:
            return {}
        dimensions: Final = parse_runware_size(size)
        if dimensions is None:
            return {}
        width, height = dimensions
        return {"width": width, "height": height}

    @staticmethod
    def _map_quality(quality: object) -> Mapping[str, object]:
        if quality is None:
            return {}
        output_quality: Final = map_runware_quality(quality)
        return {"outputQuality": output_quality} if output_quality is not None else {}

    @staticmethod
    def _map_response_format(response_format: object) -> Mapping[str, object]:
        if response_format is None:
            return {}
        output_type: Final = map_runware_response_format(response_format)
        return {"outputType": output_type} if output_type is not None else {}

    @staticmethod
    def _map_output_format(output_format: object) -> Mapping[str, object]:
        if output_format is None:
            return {}
        mapped_format: Final = map_runware_output_format(output_format)
        return {"outputFormat": mapped_format} if mapped_format is not None else {}

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        return runware_api_base(api_base)

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
        return runware_auth_headers(headers=headers, api_key=api_key)

    def transform_image_generation_request(  # pyright: ignore[reportIncompatibleMethodOverride]  # Runware HTTP API requires a top-level JSON task array
        self,
        model: str,
        prompt: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, object],
    ) -> list[dict[str, object]]:  # mutable-ok: httpx serializes list JSON arrays
        extra_body: Final = optional_params.get("extra_body")
        extra_fields: Final[Mapping[str, object]] = (
            _EXTRA_BODY.validate_python(extra_body) if isinstance(extra_body, Mapping) else {}
        )
        merged_params: Final = {
            **{key: value for key, value in optional_params.items() if key != "extra_body"},
            **{key: value for key, value in extra_fields.items() if key not in EXTRA_BODY_ROUTING_KEYS},
        }
        task: Final[dict[str, object]] = {  # mutable-ok: JSON task payload for httpx
            "taskType": "imageInference",
            "taskUUID": str(uuid.uuid4()),
            "model": model,
            "positivePrompt": prompt,
            "includeCost": True,
            **merged_params,
        }
        return [task]

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: "LiteLLMLoggingObj",
        request_data: object,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        data_items, error_items = parse_runware_envelope(raw_response)
        if error_items and not data_items:
            raise RunwareError(
                status_code=max(raw_response.status_code, 400),
                message=format_runware_errors(error_items),
                headers=raw_response.headers,
            )
        images, total_cost = extract_runware_images(data_items)
        if not images and error_items:
            raise RunwareError(
                status_code=max(raw_response.status_code, 400),
                message=format_runware_errors(error_items),
                headers=raw_response.headers,
            )
        existing_data: Final = list(model_response.data or ())
        model_response.data = [*existing_data, *images]
        set_runware_response_cost(model_response, total_cost)
        return model_response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, object] | httpx.Headers,
    ) -> BaseLLMException:
        return RunwareError(status_code=status_code, message=error_message, headers=headers)
