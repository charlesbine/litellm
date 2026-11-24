"""
OpenRouter Image Generation Support

OpenRouter serves every image model (GPT Image, Gemini, Seedream, FLUX, ...) through its
dedicated Image API, ``POST /api/v1/images``. Image-only models such as FLUX or Seedream are
not available through chat completions at all.
Docs: https://openrouter.ai/docs/guides/overview/multimodal/image-generation

Request format:
{
    "model": "bytedance-seed/seedream-4.5",
    "prompt": "a red panda astronaut",
    "aspect_ratio": "16:9",
    "quality": "high",
    "n": 1
}

Response format:
{
    "created": 1748372400,
    "data": [{"b64_json": "<base64>", "media_type": "image/png"}],
    "usage": {"prompt_tokens": 0, "completion_tokens": 4175, "total_tokens": 4175, "cost": 0.04}
}
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, cast

import httpx
from pydantic import ConfigDict, RootModel, TypeAdapter, with_config
from typing_extensions import NotRequired, ReadOnly, TypedDict

import litellm
from litellm.litellm_core_utils.url_utils import ensure_api_route
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.llms.openrouter.common_utils import OpenRouterException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import (
    ImageObject,
    ImageResponse,
    ImageUsage,
    ImageUsageInputTokensDetails,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer
else:
    LiteLLMLoggingObj = Any

OPENROUTER_API_BASE: Final = "https://openrouter.ai/api/v1"
OPENROUTER_IMAGES_ROUTE: Final = "images"

_DECODED_JSON: Final = TypeAdapter(object)

# OpenAI pixel sizes -> OpenRouter normalized aspect ratios. Explicit pixel sizes are
# provider-dependent on OpenRouter (xAI rejects them), while aspect_ratio works everywhere.
_SIZE_TO_ASPECT_RATIO: Final[Mapping[str, str]] = {
    "256x256": "1:1",
    "512x512": "1:1",
    "1024x1024": "1:1",
    "1536x1024": "3:2",
    "1792x1024": "16:9",
    "1024x1536": "2:3",
    "1024x1792": "9:16",
    "auto": "auto",
}

# OpenAI quality values -> OpenRouter quality values (auto | low | medium | high).
_QUALITY: Final[Mapping[str, str]] = {
    "auto": "auto",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "standard": "medium",
    "hd": "high",
}

# Request fields LiteLLM owns; caller-supplied extras never override them.
_RESERVED_FIELDS: Final = frozenset({"model", "prompt", "input_references", "extra_body"})


def map_openai_image_params(params: Mapping[str, object]) -> dict[str, object]:
    """Map OpenAI image params (size, quality, n, ...) to OpenRouter Image API fields."""
    mapped: Final[dict[str, object]] = {}
    for key, value in params.items():
        if value is None:
            continue
        if key == "size" and isinstance(value, str):
            aspect_ratio = _SIZE_TO_ASPECT_RATIO.get(value)
            if aspect_ratio is not None:
                mapped["aspect_ratio"] = aspect_ratio
            else:
                # Resolution tiers ("2K") and other pixel sizes are accepted natively.
                mapped["size"] = value
        elif key == "quality" and isinstance(value, str):
            quality = _QUALITY.get(value)
            if quality is not None:
                mapped["quality"] = quality
        else:
            mapped[key] = value
    return mapped


def build_images_request_body(
    model: str,
    prompt: str,
    optional_params: Mapping[str, object],
    input_references: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Build the ``POST /images`` body, flattening OpenRouter-specific extras from ``extra_body``."""
    body: Final[dict[str, object]] = {}
    extra_body: Final = optional_params.get("extra_body")
    if isinstance(extra_body, Mapping):
        body.update(
            {
                key: value
                for key, value in cast(Mapping[str, object], extra_body).items()  # cast-ok: JSON object keys
                if key not in _RESERVED_FIELDS
            }
        )
    body.update({key: value for key, value in optional_params.items() if key not in _RESERVED_FIELDS})
    body["model"] = model
    body["prompt"] = prompt
    if input_references:
        body["input_references"] = list(input_references)
    return body


def get_images_url(api_base: str | None) -> str:
    base: Final = api_base or get_secret_str("OPENROUTER_API_BASE") or OPENROUTER_API_BASE
    return ensure_api_route(base, OPENROUTER_IMAGES_ROUTE)


@with_config(ConfigDict(extra="allow", strict=True, hide_input_in_errors=True))
class _ImageData(TypedDict):
    b64_json: ReadOnly[NotRequired[str | None]]
    url: ReadOnly[NotRequired[str | None]]
    revised_prompt: ReadOnly[NotRequired[str | None]]


@with_config(ConfigDict(extra="allow", strict=True, hide_input_in_errors=True))
class _TokensDetails(TypedDict):
    image_tokens: ReadOnly[NotRequired[int | float | None]]


@with_config(ConfigDict(extra="allow", strict=True, hide_input_in_errors=True))
class _Usage(TypedDict):
    prompt_tokens: ReadOnly[NotRequired[int | float | None]]
    completion_tokens: ReadOnly[NotRequired[int | float | None]]
    total_tokens: ReadOnly[NotRequired[int | float | None]]
    prompt_tokens_details: ReadOnly[NotRequired[_TokensDetails | None]]
    cost: ReadOnly[NotRequired[int | float | str | None]]
    cost_details: ReadOnly[NotRequired[Mapping[str, object] | None]]


@with_config(ConfigDict(extra="allow", strict=True, hide_input_in_errors=True))
class _ImagesResponse(TypedDict):
    created: ReadOnly[NotRequired[int | None]]
    data: ReadOnly[Sequence[_ImageData]]
    usage: ReadOnly[NotRequired[_Usage | None]]


class _ImagesResponseRoot(RootModel[_ImagesResponse]):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)


def transform_images_response(
    model: str,
    raw_response: httpx.Response,
    model_response: ImageResponse,
    operation: str,
) -> ImageResponse:
    """Map an OpenRouter ``/images`` response onto LiteLLM's ``ImageResponse``."""
    try:
        response_json: Final = _DECODED_JSON.validate_python(raw_response.json())
    except Exception as e:
        raise OpenRouterException(
            message=f"Error parsing OpenRouter {operation} response: {e}",
            status_code=raw_response.status_code,
            headers=raw_response.headers,
        )

    try:
        payload: Final = _ImagesResponseRoot.model_validate(response_json).root
    except Exception:
        # Never echo the payload: it can carry base64 image bytes.
        raise OpenRouterException(
            message=f"Error transforming OpenRouter {operation} response: unexpected response shape",
            status_code=500,
            headers={},
        )

    created: Final = payload.get("created")
    if created is not None:
        model_response.created = created
    model_response.data = [
        ImageObject(
            b64_json=image.get("b64_json"),
            url=image.get("url"),
            revised_prompt=image.get("revised_prompt"),
        )
        for image in payload["data"]
        if image.get("b64_json") or image.get("url")
    ]

    usage: Final = payload.get("usage")
    if usage:
        prompt_tokens: Final = int(usage.get("prompt_tokens") or 0)
        prompt_tokens_details: Final = usage.get("prompt_tokens_details") or {}
        input_image_tokens: Final = int(prompt_tokens_details.get("image_tokens") or 0)
        model_response.usage = ImageUsage(
            input_tokens=prompt_tokens,
            input_tokens_details=ImageUsageInputTokensDetails(
                image_tokens=input_image_tokens,
                text_tokens=max(0, prompt_tokens - input_image_tokens),
            ),
            output_tokens=int(usage.get("completion_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
        )

        cost: Final = usage.get("cost")
        if cost is not None:
            additional_headers: Final = cast(  # cast-ok: provider headers are stored as a mutable mapping
                dict[str, object], model_response.hidden_params.setdefault("additional_headers", {})
            )
            additional_headers["llm_provider-x-litellm-response-cost"] = float(cost)

        cost_details: Final = usage.get("cost_details")
        if cost_details:
            response_cost_details: Final = cast(  # cast-ok: provider cost details are stored as a mutable mapping
                dict[str, object], model_response.hidden_params.setdefault("response_cost_details", {})
            )
            response_cost_details.update(cost_details)

    model_response.hidden_params["model"] = model
    return model_response


class OpenRouterImageGenerationConfig(BaseImageGenerationConfig):
    """Image generation through OpenRouter's Image API (``POST /api/v1/images``)."""

    def get_supported_openai_params(self, model: str) -> list[OpenAIImageGenerationOptionalParams]:
        return ["n", "quality", "size", "response_format"]

    def map_openai_params(
        self,
        non_default_params: dict[str, object],
        optional_params: dict[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:
        supported_params: Final = self.get_supported_openai_params(model)
        requested: Final[dict[str, object]] = {
            key: value for key, value in non_default_params.items() if key in supported_params or not drop_params
        }
        optional_params.update(map_openai_image_params(requested))
        return optional_params

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        return get_images_url(api_base)

    def validate_environment(
        self,
        headers: dict[str, object],
        model: str,
        messages: list[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, object]:
        api_key = api_key or litellm.api_key or get_secret_str("OPENROUTER_API_KEY")
        headers.update(
            {
                "Authorization": f"Bearer {api_key}",
            }
        )
        return headers

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, object],
    ) -> dict[str, object]:
        return build_images_request_body(model=model, prompt=prompt, optional_params=optional_params)

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: Mapping[str, object],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        return transform_images_response(
            model=model,
            raw_response=raw_response,
            model_response=model_response,
            operation="image generation",
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: dict[str, object] | httpx.Headers
    ) -> BaseLLMException:
        return OpenRouterException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )
