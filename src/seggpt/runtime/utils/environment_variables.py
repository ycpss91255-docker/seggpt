"""Environment variable wrappers used by Layer 1.

Slimmed port of ``generative_services.utils.environment_variables``.
Only the descriptors actually referenced by the SegGPT runtime
(``USE_CUDA`` and ``GSI_HOME``) are kept; the upstream module also
defined logger / transformers env vars that are not needed here.

Implementation pattern (typed env var with default, lazy resolve via
``.get()``) is borrowed from MLflow's ``environment_variables``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Type

from seggpt.runtime.utils.types import PathLike, class_property


class EnvironmentVariable:
    """Represents a single environment variable with a typed default."""

    __envs: Dict[str, "EnvironmentVariable"] = {}

    def __init__(self, name: str, type_: Type, default: Any, *, force: bool = False):
        if name in self.__envs:
            raise ValueError(f"{name} is already defined")
        self.__envs[name] = self
        self.name = name
        self.type = type_
        self.default = default
        if force and name not in os.environ:
            self.set(default)

    @property
    def is_defined(self) -> bool:
        return self.name in os.environ

    def get_raw(self) -> str | None:
        return os.getenv(self.name)

    def set(self, value: Any) -> None:
        os.environ[self.name] = str(value)

    def get(self) -> Any:
        val = self.get_raw()
        if val is not None:
            try:
                return self.type(val)
            except Exception as exc:
                raise ValueError(
                    f"Failed to convert {val!r} to {self.type} for {self.name}: {exc}"
                ) from exc
        return self.default

    def __str__(self) -> str:
        return f"{self.name} (default: {self.default}, type: {self.type.__name__})"

    def __repr__(self) -> str:
        return repr(self.name)

    @class_property
    @classmethod
    def variables(cls) -> Dict[str, "EnvironmentVariable"]:
        return cls.__envs


class BooleanEnvironmentVariable(EnvironmentVariable):
    """Env var that parses common truthy / falsy strings into ``bool``."""

    TRUE_VALUES = ("true", "1", "on", "yes")
    FALSE_VALUES = ("false", "0", "off", "no")

    def __init__(self, name: str, default: bool, **kwargs: Any):
        if not (default is True or default is False or default is None):
            raise ValueError(f"{name} default value must be one of [True, False, None]")
        super().__init__(name, bool, default, **kwargs)

    def set(self, value: bool) -> None:
        super().set(int(value))

    def get(self) -> bool:
        if not self.is_defined:
            return self.default
        val = os.getenv(self.name)
        lowercased = val.lower()
        if lowercased not in self.TRUE_VALUES + self.FALSE_VALUES:
            raise ValueError(
                f"{self.name} value must be one of "
                f"['true', 'false', '1', '0'] (case-insensitive), but got {val}"
            )
        return lowercased in self.TRUE_VALUES


class PathEnvironmentVariable(EnvironmentVariable):
    """Env var resolved as a ``pathlib.Path``."""

    def __init__(self, name: str, default: PathLike, *, is_dir: bool = True, **kwargs: Any):
        super().__init__(name, str, default, **kwargs)
        self._is_dir = is_dir

    def get(self) -> Path:
        path = Path(super().get())
        if self._is_dir:
            path.mkdir(parents=True, exist_ok=True)
            return path.absolute()
        return path


USE_CUDA = BooleanEnvironmentVariable("USE_CUDA", True)
GSI_HOME = PathEnvironmentVariable("GSI_HOME", "", is_dir=True)
