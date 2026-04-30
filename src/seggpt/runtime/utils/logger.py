"""Thin logging wrappers used by Layer 1.

The upstream ``generative_services.utils.logger`` module bundled
custom logger backends, config loaders, and message routers (~400 LOC).
Layer 1 only needs four level-specific print helpers, so this module
delegates to the standard ``logging`` package and skips the custom
plumbing.
"""
import logging
from typing import Any

_LOGGER = logging.getLogger("seggpt.runtime")


def logd_print(message: Any) -> None:
    _LOGGER.log(logging.DEBUG, message)


def logi_print(message: Any) -> None:
    _LOGGER.log(logging.INFO, message)


def logw_print(message: Any) -> None:
    _LOGGER.log(logging.WARNING, message)


def loge_print(message: Any) -> None:
    _LOGGER.log(logging.ERROR, message)
