"""Unit tests for ``seggpt.runtime.utils.types``."""
from __future__ import annotations

import os
from pathlib import Path

from seggpt.runtime.utils.types import (
    ConfigLike,
    ListLikeInOut,
    PathLike,
    class_property,
)


class TestPathLike:
    def test_accepts_str(self) -> None:
        path: PathLike = "/tmp/example"
        assert isinstance(path, str)

    def test_accepts_pathlib_Path(self) -> None:
        path: PathLike = Path("/tmp/example")
        assert isinstance(path, Path)

    def test_accepts_bytes(self) -> None:
        path: PathLike = b"/tmp/example"
        assert isinstance(path, bytes)

    def test_accepts_os_PathLike(self) -> None:
        path: PathLike = Path("/tmp/example")
        assert isinstance(path, os.PathLike)


class TestClassProperty:
    def test_resolves_on_class_without_instance(self) -> None:
        class Cfg:
            @class_property
            @classmethod
            def name(cls) -> str:
                return cls.__name__

        assert Cfg.name == "Cfg"

    def test_resolves_through_subclass(self) -> None:
        class Base:
            @class_property
            @classmethod
            def label(cls) -> str:
                return cls.__name__

        class Child(Base):
            pass

        assert Base.label == "Base"
        assert Child.label == "Child"

    def test_does_not_require_instantiation(self) -> None:
        instantiated = []

        class Eager:
            def __init__(self) -> None:
                instantiated.append(self)

            @class_property
            @classmethod
            def value(cls) -> int:
                return 42

        assert Eager.value == 42
        assert instantiated == []


class TestConfigLikeAlias:
    def test_nested_dict_passes_isinstance_dict_check(self) -> None:
        cfg: ConfigLike = {"name": "seggpt", "params": {"depth": 24, "heads": 16}}
        assert isinstance(cfg, dict)
        assert cfg["params"]["heads"] == 16


class TestListLikeInOutAlias:
    def test_scalar_value(self) -> None:
        value: ListLikeInOut[int] = 7
        assert value == 7

    def test_nested_list(self) -> None:
        nested: ListLikeInOut[int] = [1, [2, [3, 4]], 5]
        assert nested[1][1][0] == 3
