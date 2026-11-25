from .transformation import NovitaImageGenerationConfig


def get_novita_image_generation_config(model: str) -> NovitaImageGenerationConfig:
    return NovitaImageGenerationConfig()


__all__ = [
    "NovitaImageGenerationConfig",
    "get_novita_image_generation_config",
]
