"""Defer ``import`` of heavy modules until first attribute access.

Slimmed port of ``generative_services.utils.lazy_import``: only
``LazyModuleImporter`` is kept. The upstream module also defined a
fancier ``LazySelectedImporter`` and a ``try_import()`` deferred-error
context manager; neither is used by the SegGPT runtime, so they are
out of scope for this port.

Why lazy-import here at all: the SegGPT runtime imports torch /
detectron2 / timm / fvcore / fairscale at the top of its module file.
Pulling all of them eagerly would slow down ``import seggpt`` by ~3-5
seconds even for callers that never touch the model (e.g. unit tests
of pure utilities below). ``LazyModuleImporter`` defers the actual
``importlib.import_module`` until the first attribute lookup.
"""
from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any


class LazyModuleImporter(ModuleType):
    """Module wrapper that imports the underlying module on first access."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._name = name
        self._module: ModuleType | None = None

    def _load(self) -> ModuleType:
        if self._module is None:
            self._module = importlib.import_module(self._name)
            self.__dict__.update(self._module.__dict__)
        return self._module

    def __getattr__(self, item: str) -> Any:
        return getattr(self._load(), item)
