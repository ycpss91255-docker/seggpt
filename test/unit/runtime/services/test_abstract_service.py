"""Unit tests for ``seggpt.runtime.services.abstract_service``.

The factory + ABC machinery touches ``yacs`` (for ``default_config``)
and ``yaml`` (for ``PathService.__new__``). Auto-skip on hosts where
those aren't installed; in the docker image both are present.
"""
from __future__ import annotations

import sys
from typing import Any

import pytest

pytest.importorskip("yacs")
pytest.importorskip("yaml")

from seggpt.runtime.services.abstract_service import (  # noqa: E402
    AbstractService,
    PathService,
    ServiceFactory,
    _AbstractService,
)


@pytest.fixture(autouse=True)
def _isolate_factory():
    """Snapshot ``ServiceFactory.__service_map`` so each test starts clean.

    ``__init_subclass__`` registers concrete services into the
    process-global factory. Without this fixture, repeated test runs in
    the same session would hit ``Keyword X has been registered`` errors.
    """
    name_mangled = "_ServiceFactory__service_map"
    snapshot = dict(getattr(ServiceFactory, name_mangled))
    yield
    setattr(ServiceFactory, name_mangled, snapshot)


def _build_concrete_service(name: str, *, keywords: Any = None):
    """Return a concrete AbstractService subclass with stubbed methods."""

    cls = type(
        name,
        (AbstractService,),
        {
            "target": lambda self, **kw: self,
            "prompt": lambda self, **kw: {"out": True},
            "output_keys": property(lambda self: {"out"}),
        },
        # __init_subclass__ kwargs go through the metaclass-style hook
        keywords=keywords,
    )
    return cls


class TestServiceFactoryRegistration:
    def test_class_name_auto_registered(self) -> None:
        cls = _build_concrete_service("AutoRegisteredA")
        assert "AutoRegisteredA" in ServiceFactory.keywords
        assert ServiceFactory.service_class("AutoRegisteredA") is cls

    def test_extra_keywords_registered(self) -> None:
        cls = _build_concrete_service("AutoRegisteredB", keywords={"alias-b", "another"})
        assert {"AutoRegisteredB", "alias-b", "another"} <= ServiceFactory.keywords
        assert ServiceFactory.service_class("alias-b") is cls

    def test_string_keyword_normalised_to_set(self) -> None:
        cls = _build_concrete_service("StringKw", keywords="just-one")
        assert ServiceFactory.service_class("just-one") is cls

    def test_list_keyword_normalised(self) -> None:
        cls = _build_concrete_service("ListKw", keywords=["x", "y"])
        assert ServiceFactory.service_class("x") is cls
        assert ServiceFactory.service_class("y") is cls

    def test_double_registration_raises(self) -> None:
        _build_concrete_service("DupServiceClass", keywords={"shared-keyword"})
        with pytest.raises(ValueError, match="has been registered"):
            _build_concrete_service("DupServiceClass2", keywords={"shared-keyword"})

    def test_unsupported_keyword_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported keywords type"):

            class _Bad(AbstractService, keywords=42):  # type: ignore[arg-type]
                target = lambda self, **kw: self  # noqa: E731
                prompt = lambda self, **kw: {}  # noqa: E731

                @property
                def output_keys(self):
                    return set()

    def test_lookup_unknown_keyword_raises(self) -> None:
        with pytest.raises(ValueError, match="Service .* is not supported"):
            ServiceFactory.service_class("not-a-real-service")


class TestServiceFactoryProtocol:
    def test_singleton(self) -> None:
        assert ServiceFactory() is ServiceFactory()

    def test_iter_yields_keywords(self) -> None:
        _build_concrete_service("IterCheck", keywords={"alpha"})
        kws = set(iter(ServiceFactory()))
        assert {"IterCheck", "alpha"} <= kws

    def test_contains_check(self) -> None:
        _build_concrete_service("ContainsCheck")
        assert "ContainsCheck" in ServiceFactory()

    def test_initial_constructs_via_keyword(self) -> None:
        _build_concrete_service("InstanceCheck")
        instance = ServiceFactory.initial("InstanceCheck")
        assert isinstance(instance, _AbstractService)


class TestDefaultConfig:
    def test_introspects_init_parameters(self) -> None:
        from yacs.config import CfgNode

        class CfgInitService(AbstractService):
            def __init__(self, model_path: str = "weights.pth", batch: int = 4, **kw):
                super().__init__(**kw)
                self.model_path = model_path
                self.batch = batch

            def target(self, **kw):
                return self

            def prompt(self, **kw):
                return {}

            @property
            def output_keys(self):
                return {"x"}

        cfg = CfgInitService.default_config
        assert isinstance(cfg, CfgNode)
        assert cfg.keyword == "CfgInitService"
        assert cfg.model_path == "weights.pth"
        assert cfg.batch == 4

    def test_kwargs_excluded_from_config(self) -> None:
        class KwargsExclService(AbstractService):
            def __init__(self, alpha: int = 1, **kwargs):
                super().__init__(**kwargs)

            def target(self, **kw):
                return self

            def prompt(self, **kw):
                return {}

            @property
            def output_keys(self):
                return set()

        cfg = KwargsExclService.default_config
        assert "kwargs" not in cfg
        assert cfg.alpha == 1


class TestSignatureKeys:
    def test_prompt_target_reset_keys_extracted(self) -> None:
        class SigTestService(AbstractService):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            def target(self, image, **kw):
                return self

            def prompt(self, refs, masks, mode="instance", **kw):
                return {}

            def reset(self, **kw):
                return self

            @property
            def output_keys(self):
                return set()

        svc = SigTestService()
        assert {"image", "kw"} <= svc.target_keys
        assert {"refs", "masks", "mode", "kw"} <= svc.prompt_keys
        assert "kw" in svc.reset_keys


class TestPathServiceFactory:
    def test_constructs_via_yaml(self, tmp_path) -> None:
        # Register a concrete service the path-config will resolve to.
        class FromYamlService(AbstractService):
            def __init__(self, factor: int = 7, **kwargs):
                super().__init__(**kwargs)
                self.factor = factor

            def target(self, **kw):
                return self

            def prompt(self, **kw):
                return {}

            @property
            def output_keys(self):
                return set()

        cfg = tmp_path / "svc.yaml"
        cfg.write_text("keyword: FromYamlService\nfactor: 11\n")

        # PathService.__new__ reads cfg, then forwards to ServiceFactory.initial(**cfg)
        instance = PathService(cfg)
        assert isinstance(instance, FromYamlService)
        assert instance.factor == 11
