"""Unit tests for ``seggpt.runtime.utils.tools``.

Some tests need ``yacs`` and ``yaml`` installed. The container always
has them (Dockerfile base stage); host-side runs without them get
auto-skipped via ``pytest.importorskip``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("yacs")
pytest.importorskip("yaml")

from seggpt.runtime.utils import tools  # noqa: E402 - import after skip
from seggpt.runtime.utils.environment_variables import GSI_HOME  # noqa: E402


@pytest.fixture
def fake_gsi_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("GSI_HOME", str(tmp_path))
    return tmp_path


class TestCheckPath:
    def test_returns_path_when_absolute_exists(self, tmp_path: Path) -> None:
        f = tmp_path / "existing.txt"
        f.write_text("hello")
        assert tools.check_path(f) == f

    def test_falls_back_to_gsi_home_for_relative_paths(self, fake_gsi_home: Path) -> None:
        relative = Path("config/model.yaml")
        target = fake_gsi_home / relative
        target.parent.mkdir(parents=True)
        target.write_text("x: 1")
        assert tools.check_path(relative) == target

    def test_raises_when_neither_path_exists(self, fake_gsi_home: Path) -> None:
        with pytest.raises(FileNotFoundError):
            tools.check_path("does-not-exist.yaml")


class TestPathWithHome:
    def test_absolute_path_returned_verbatim(self, tmp_path: Path) -> None:
        assert tools.path_with_home(tmp_path) == tmp_path

    def test_relative_path_prefixed_with_gsi_home(self, fake_gsi_home: Path) -> None:
        assert tools.path_with_home("subdir/file.txt") == fake_gsi_home / "subdir/file.txt"


class TestLoadYaml:
    def test_returns_yacs_cfgnode_by_default(self, tmp_path: Path) -> None:
        from yacs.config import CfgNode

        f = tmp_path / "cfg.yaml"
        f.write_text("name: seggpt\ndepth: 24\n")
        cfg = tools.load_yaml(f)
        assert isinstance(cfg, CfgNode)
        assert cfg.name == "seggpt"
        assert cfg.depth == 24

    def test_returns_plain_dict_when_use_yacs_false(self, tmp_path: Path) -> None:
        f = tmp_path / "plain.yaml"
        f.write_text("a: 1\nb: 2\n")
        cfg = tools.load_yaml(f, use_yacs=False)
        assert isinstance(cfg, dict)
        assert cfg == {"a": 1, "b": 2}

    def test_uses_safe_load(self, tmp_path: Path) -> None:
        # yaml.unsafe_load would interpret !!python/object as instantiation;
        # safe_load raises a ConstructorError. This locks down the choice.
        import yaml

        f = tmp_path / "evil.yaml"
        f.write_text("foo: !!python/object/apply:os.system ['echo pwned']\n")
        with pytest.raises(yaml.YAMLError):
            tools.load_yaml(f, use_yacs=False)
