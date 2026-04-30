"""Unit tests for ``seggpt.runtime.utils.environment_variables``."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Each test gets a fresh module so the module-level singletons (USE_CUDA,
# GSI_HOME) and the `EnvironmentVariable.__envs` registry are isolated.
@pytest.fixture
def env_module(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("USE_CUDA", raising=False)
    monkeypatch.delenv("GSI_HOME", raising=False)
    sys.modules.pop("seggpt.runtime.utils.environment_variables", None)
    import importlib

    return importlib.import_module("seggpt.runtime.utils.environment_variables")


class TestEnvironmentVariable:
    def test_default_returned_when_unset(self, env_module, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MY_VAR", raising=False)
        var = env_module.EnvironmentVariable("MY_VAR", int, 42)
        assert var.get() == 42

    def test_environment_value_overrides_default(self, env_module, monkeypatch: pytest.MonkeyPatch) -> None:
        var = env_module.EnvironmentVariable("MY_VAR_2", int, 1)
        monkeypatch.setenv("MY_VAR_2", "99")
        assert var.get() == 99

    def test_double_registration_raises(self, env_module) -> None:
        env_module.EnvironmentVariable("DUP_VAR", str, "")
        with pytest.raises(ValueError, match="already defined"):
            env_module.EnvironmentVariable("DUP_VAR", str, "")

    def test_set_writes_to_environ(self, env_module, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("WRITE_VAR", raising=False)
        var = env_module.EnvironmentVariable("WRITE_VAR", str, "")
        var.set("hello")
        assert os.environ["WRITE_VAR"] == "hello"

    def test_unparseable_value_raises(self, env_module, monkeypatch: pytest.MonkeyPatch) -> None:
        var = env_module.EnvironmentVariable("BAD_INT", int, 0)
        monkeypatch.setenv("BAD_INT", "not-an-int")
        with pytest.raises(ValueError, match="Failed to convert"):
            var.get()


class TestBooleanEnvironmentVariable:
    @pytest.mark.parametrize("raw", ["true", "1", "on", "yes", "TRUE", "Yes"])
    def test_truthy_strings_resolve_to_true(
        self, env_module, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        var = env_module.BooleanEnvironmentVariable(f"BOOL_T_{raw}", False)
        monkeypatch.setenv(f"BOOL_T_{raw}", raw)
        assert var.get() is True

    @pytest.mark.parametrize("raw", ["false", "0", "off", "no", "FALSE", "No"])
    def test_falsy_strings_resolve_to_false(
        self, env_module, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        var = env_module.BooleanEnvironmentVariable(f"BOOL_F_{raw}", True)
        monkeypatch.setenv(f"BOOL_F_{raw}", raw)
        assert var.get() is False

    def test_invalid_string_raises(self, env_module, monkeypatch: pytest.MonkeyPatch) -> None:
        var = env_module.BooleanEnvironmentVariable("BOOL_BAD", True)
        monkeypatch.setenv("BOOL_BAD", "maybe")
        with pytest.raises(ValueError, match="must be one of"):
            var.get()

    def test_default_returned_when_unset(self, env_module, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BOOL_DEFAULT", raising=False)
        var = env_module.BooleanEnvironmentVariable("BOOL_DEFAULT", True)
        assert var.get() is True

    def test_non_bool_default_rejected(self, env_module) -> None:
        with pytest.raises(ValueError, match="must be one of"):
            env_module.BooleanEnvironmentVariable("BOOL_INT_DEFAULT", 1)  # type: ignore[arg-type]

    def test_set_writes_int_form(self, env_module, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BOOL_WRITE", raising=False)
        var = env_module.BooleanEnvironmentVariable("BOOL_WRITE", False)
        var.set(True)
        assert os.environ["BOOL_WRITE"] == "1"
        assert var.get() is True


class TestPathEnvironmentVariable:
    def test_creates_dir_when_is_dir_true(
        self, env_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        target = tmp_path / "newdir"
        monkeypatch.setenv("PATH_DIR_VAR", str(target))
        var = env_module.PathEnvironmentVariable("PATH_DIR_VAR", "", is_dir=True)
        result = var.get()
        assert result.is_dir()
        assert result == target.resolve().absolute()

    def test_does_not_create_when_is_dir_false(
        self, env_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        target = tmp_path / "logging.yaml"
        monkeypatch.setenv("PATH_FILE_VAR", str(target))
        var = env_module.PathEnvironmentVariable("PATH_FILE_VAR", "", is_dir=False)
        result = var.get()
        assert not target.exists()
        assert isinstance(result, Path)


class TestRegisteredSingletons:
    def test_use_cuda_default_true(self, env_module) -> None:
        # Module-level USE_CUDA singleton; default is True per spec.
        assert env_module.USE_CUDA.default is True

    def test_gsi_home_is_path_dir(self, env_module) -> None:
        assert isinstance(env_module.GSI_HOME, env_module.PathEnvironmentVariable)
