"""
OpenRouter Image Edit Support

OpenRouter edits images through the same Image API as generation, ``POST /api/v1/images``:
the source image(s) are sent as ``input_references`` (base64 data URLs) next to the prompt.
Docs: https://openrouter.ai/docs/guides/overview/multimodal/image-generation

Request format:
{
    "model": "bytedance-seed/seedream-4.5",
    "prompt": "Make the teapot red",
    "input_references": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]
}

The response has the same shape as image generation.
"""

import base64
from collections.abc import Mapping, Sequence
from io import BufferedReader, BytesIO
from typing import TYPE_CHECKING, Any, Final, cast

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.images.utils import ImageEditRequestUtils
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.llms.openrouter.common_utils import OpenRouterException
from litellm.llms.openrouter.image_generation.transformation import (
    build_images_request_body,
    get_images_url,
    map_openai_image_params,
    transform_images_response,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes, ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class OpenRouterImageEditConfig(BaseImageEditConfig):
    """Image editing through OpenRouter's Image API (``POST /api/v1/images``)."""

    def get_supported_openai_params(self, model: str) -> list[str]:
        return ["size", "quality", "n", "response_format", "background"]

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:
        supported_params: Final = self.get_supported_openai_params(model)
        return map_openai_image_params(
            {key: value for key, value in image_edit_optional_params.items() if key in supported_params}
        )

    def validate_environment(
        self,
        headers: dict[str, object],
        model: str,
        api_key: str | None = None,
        litellm_params: dict[str, object] | None = None,
        api_base: str | None = None,
    ) -> dict[str, object]:
        api_key = api_key or litellm.api_key or get_secret_str("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        headers.update(
            {
                "Authorization": f"Bearer {api_key}",
            }
        )
        return headers

    def use_multipart_form_data(self) -> bool:
        """OpenRouter uses JSON requests, not multipart/form-data."""
        return False

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict[str, object],
    ) -> str:
        return get_images_url(api_base)

    def transform_image_edit_request(
        self,
        model: str,
        prompt: str | None,
        image: FileTypes | Sequence[FileTypes | None] | None,
        image_edit_optional_request_params: Mapping[str, object],
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, object],
    ) -> tuple[dict[str, object], RequestFiles]:
        images: Final = image if isinstance(image, list) else [image]
        input_references: Final = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{ImageEditRequestUtils.get_image_content_type(img)};base64,"
                    f"{base64.b64encode(self._read_image_bytes(img)).decode('utf-8')}"
                },
            }
            for img in images
            if img is not None
        ]
        request_body: Final = build_images_request_body(
            model=model,
            prompt=prompt or "",
            optional_params=image_edit_optional_request_params,
            input_references=input_references,
        )
        empty_files: Final = cast(RequestFiles, [])
        return request_body, empty_files

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ImageResponse:
        return transform_images_response(
            model=model,
            raw_response=raw_response,
            model_response=ImageResponse(),
            operation="image edit",
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: dict[str, object] | httpx.Headers
    ) -> BaseLLMException:
        return OpenRouterException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )

    def _read_image_bytes(self, image: object) -> bytes:
        """Read raw bytes from various image input types."""
        if isinstance(image, bytes):
            return image
        if isinstance(image, (BytesIO, BufferedReader)):
            current_pos = image.tell()
            image.seek(0)
            data = image.read()
            image.seek(current_pos)
            return data
        raise ValueError("Unsupported image type for OpenRouter image edit.")
