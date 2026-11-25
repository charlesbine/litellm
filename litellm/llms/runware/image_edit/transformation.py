import uuid
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

import httpx
from httpx._types import RequestFiles
from pydantic import ConfigDict, TypeAdapter

from litellm.images.utils import image_to_data_url
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
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
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes, ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))
_OBJECT_LIST: Final = TypeAdapter(tuple[object, ...], config=ConfigDict(hide_input_in_errors=True))
_SUPPORTED_OPENAI_PARAMS: Final[tuple[str, ...]] = (
    "mask",
    "n",
    "size",
    "quality",
    "response_format",
    "output_format",
)


def _first_item(value: object) -> object:
    if not isinstance(value, list):
        return value
    items: Final = _OBJECT_LIST.validate_python(value)
    return items[0] if items else None


def _build_edit_inputs(
    image_urls: tuple[str, ...],
    mask_url: str | None,
    provider_params: Mapping[str, object],
) -> Mapping[str, object]:
    raw_inputs: Final = provider_params.get("inputs")
    caller_inputs: Final[Mapping[str, object]] = (
        _JSON_OBJECT.validate_python(raw_inputs) if isinstance(raw_inputs, Mapping) else {}
    )
    if not image_urls:
        return caller_inputs
    first_image: Final = image_urls[0]
    if mask_url is not None:
        extra_refs: Final = list(image_urls[1:]) if len(image_urls) > 1 else None
        return {
            "seedImage": first_image,
            "maskImage": mask_url,
            **({"referenceImages": extra_refs} if extra_refs else {}),
            **caller_inputs,
        }
    if len(image_urls) == 1 and "strength" in provider_params:
        return {
            "seedImage": first_image,
            **caller_inputs,
        }
    return {
        "referenceImages": list(image_urls),
        **caller_inputs,
    }


class RunwareImageEditConfig(BaseImageEditConfig):
    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: base class contract returns a list
        return list(_SUPPORTED_OPENAI_PARAMS)

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: base class contract returns a dict
        return {
            **self._map_count(image_edit_optional_params.get("n")),
            **self._map_size(image_edit_optional_params.get("size")),
            **self._map_quality(image_edit_optional_params.get("quality")),
            **self._map_response_format(image_edit_optional_params.get("response_format")),
            **self._map_output_format(image_edit_optional_params.get("output_format")),
            **(
                {"mask": image_edit_optional_params["mask"]}
                if "mask" in image_edit_optional_params and image_edit_optional_params["mask"] is not None
                else {}
            ),
        }

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

    def validate_environment(
        self,
        headers: Mapping[str, object],
        model: str,
        api_key: str | None = None,
        litellm_params: Mapping[str, object] | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: base class contract returns a dict
        return runware_auth_headers(headers=headers, api_key=api_key)

    def use_multipart_form_data(self) -> bool:
        return False

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: Mapping[str, object],
    ) -> str:
        return runware_api_base(api_base)

    def transform_image_edit_request(  # pyright: ignore[reportIncompatibleMethodOverride]  # Runware HTTP API requires a top-level JSON task array
        self,
        model: str,
        prompt: str | None,
        image: FileTypes | Sequence[FileTypes | None] | None,
        image_edit_optional_request_params: Mapping[str, object],
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, object],
    ) -> tuple[list[dict[str, object]], RequestFiles]:  # mutable-ok: httpx serializes list JSON arrays
        candidates: Final = image if isinstance(image, list) else (image,)
        image_urls: Final = tuple(image_to_data_url(item) for item in candidates if item is not None)
        if not image_urls:
            raise ValueError("Runware image edit requires at least one input image")

        mask: Final = _first_item(image_edit_optional_request_params.get("mask"))
        mask_url: Final = image_to_data_url(mask) if mask is not None else None
        provider_params: Final = {
            key: value for key, value in image_edit_optional_request_params.items() if key != "mask"
        }
        inputs: Final = _build_edit_inputs(
            image_urls=image_urls,
            mask_url=mask_url,
            provider_params=provider_params,
        )
        task: Final[dict[str, object]] = {  # mutable-ok: JSON task payload for httpx
            "taskType": "imageInference",
            "taskUUID": str(uuid.uuid4()),
            "model": model,
            **({"positivePrompt": prompt} if prompt is not None else {}),
            "includeCost": True,
            **{key: value for key, value in provider_params.items() if key != "inputs"},
            "inputs": dict(inputs),
        }
        return [task], ()

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
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
        model_response: Final = ImageResponse()
        model_response.data = list(images)
        set_runware_response_cost(model_response, total_cost)
        return model_response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, object] | httpx.Headers,
    ) -> BaseLLMException:
        return RunwareError(status_code=status_code, message=error_message, headers=headers)
