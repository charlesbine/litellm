from .transformation import RunwareImageGenerationConfig


def get_runware_image_generation_config(model: str) -> RunwareImageGenerationConfig:
    return RunwareImageGenerationConfig()


__all__ = [
    "RunwareImageGenerationConfig",
    "get_runware_image_generation_config",
]
