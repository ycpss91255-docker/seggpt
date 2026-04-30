"""Unit tests for ``seggpt.runtime.utils.lazy_import``."""
from __future__ import annotations

import sys
import types
from unittest.mock import patch

from seggpt.runtime.utils.lazy_import import LazyModuleImporter


class TestLazyModuleImporter:
    def test_import_deferred_until_attribute_access(self) -> None:
        # Wrap a fake module that increments a counter every time it is
        # imported; verify the increment happens only on first attribute
        # access, not at construction time.
        load_count = {"n": 0}
        fake_module = types.ModuleType("fake_lazy_target")
        fake_module.value = 99

        def _fake_import_module(name: str) -> types.ModuleType:
            load_count["n"] += 1
            return fake_module

        importer = LazyModuleImporter("fake_lazy_target")
        assert load_count["n"] == 0  # constructor must not trigger import

        with patch("importlib.import_module", _fake_import_module):
            assert importer.value == 99
        assert load_count["n"] == 1

    def test_subsequent_access_does_not_reimport(self) -> None:
        load_count = {"n": 0}
        fake_module = types.ModuleType("fake_lazy_target_2")
        fake_module.alpha = "a"
        fake_module.beta = "b"

        def _fake_import_module(name: str) -> types.ModuleType:
            load_count["n"] += 1
            return fake_module

        importer = LazyModuleImporter("fake_lazy_target_2")
        with patch("importlib.import_module", _fake_import_module):
            _ = importer.alpha
            _ = importer.beta
            _ = importer.alpha
        assert load_count["n"] == 1

    def test_attributes_resolve_correctly_against_real_module(self) -> None:
        # Use the stdlib `os.path` as a real lazy target; confirm a
        # known attribute round-trips.
        importer = LazyModuleImporter("os.path")
        assert importer.sep in {"/", "\\"}

    def test_initialised_with_modulename_only(self) -> None:
        importer = LazyModuleImporter("json")
        assert importer.__class__.__name__ == "LazyModuleImporter"
        # Underlying not loaded yet
        assert importer._module is None  # noqa: SLF001 - private read for test
