from .transformation import NovitaImageEditConfig


def get_novita_image_edit_config(model: str) -> NovitaImageEditConfig:
    return NovitaImageEditConfig()


__all__ = [
    "NovitaImageEditConfig",
    "get_novita_image_edit_config",
]
