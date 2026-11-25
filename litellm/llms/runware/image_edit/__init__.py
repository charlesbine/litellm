from .transformation import RunwareImageEditConfig


def get_runware_image_edit_config(model: str) -> RunwareImageEditConfig:
    return RunwareImageEditConfig()


__all__ = [
    "RunwareImageEditConfig",
    "get_runware_image_edit_config",
]
