"""Type aliases and descriptors used across Layer 1.

Direct port of ``generative_services.utils.types`` minus the inactive
TYPE_CHECKING branch. ``class_property`` is the only non-standard
descriptor; ``AbstractService.default_config`` and several services use
it to expose a yacs ``CfgNode`` from a classmethod without instantiation.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, ParamSpec, TypeVar, Union

T = TypeVar("T")
P = ParamSpec("P")
ConfigLike = Dict[str, Union[str, int, float, bool, None, Dict[str, "ConfigLike"], List["ConfigLike"]]]
PathLike = Union[str, bytes, os.PathLike, Path]
ListLikeInOut = Union[T, List["ListLikeInOut"]]


class class_property(property):  # noqa: N801 - public descriptor name kept stable
    """Descriptor to be used as decorator for ``@classmethod``.

    Equivalent to ``property`` but resolves on the class (not the
    instance), so ``Cls.foo`` works without instantiation.
    """

    def __get__(self, obj: object, objtype: type | None = None):
        return self.fget.__get__(None, objtype)()  # type: ignore[union-attr]
