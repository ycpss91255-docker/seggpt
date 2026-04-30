"""Service base class + factory registry.

Direct port of ``generative_services.services.abstract_service`` with
the only real change being the import paths: every
``generative_services.utils.*`` becomes ``seggpt.runtime.utils.*`` and
sibling-service imports drop the upstream ``generative_services``
prefix.

The factory pattern (``ServiceFactory.register``,
``AbstractService.__init_subclass__``) is preserved verbatim because
SegGPT's ``__init_subclass__`` registers itself with the factory using
case-conversion keywords; pulling the factory out would force a
matching change in ``seggpt_service.py`` and risk silently breaking
keyword lookup.

``PathService`` is also kept even though Layer 1 doesn't construct
services from a config file directly (Layer 2 wraps that), so removing
it would diverge from upstream without a forcing function.
"""
from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

from yacs.config import CfgNode as CN

from seggpt.runtime.services.utils import contains_var_keyword, get_var_keyword
from seggpt.runtime.utils.logger import logw_print
from seggpt.runtime.utils.naming import to_snake_case
from seggpt.runtime.utils.tools import load_yaml
from seggpt.runtime.utils.types import class_property


class _AbstractService(ABC):
    """Abstract Service class."""

    def __init__(self, **kwargs: Any):
        if kwargs:
            logw_print(f"Unused arguments in initialization: {kwargs}")
        self.name = ""

    def reset(self, **kwargs: Any) -> _AbstractService:
        if kwargs:
            logw_print(f"Unused arguments in reset: {kwargs}")
        return self

    @abstractmethod
    def target(self, **kwargs: Any) -> _AbstractService:
        if kwargs:
            logw_print(f"Unused arguments in target: {kwargs}")
        return self

    @abstractmethod
    def prompt(self, **kwargs: Any) -> Dict[str, Any]:
        if kwargs:
            logw_print(f"Unused arguments in prompt: {kwargs}")
        return {}

    @property
    def prompt_signature(self) -> inspect.Signature:
        return inspect.signature(self.prompt)

    @property
    def target_signature(self) -> inspect.Signature:
        return inspect.signature(self.target)

    @property
    def reset_signature(self) -> inspect.Signature:
        return inspect.signature(self.reset)

    @property
    @abstractmethod
    def output_keys(self) -> Set[str]:
        return set()

    @property
    def prompt_keys(self) -> Set[str]:
        return set(self.prompt_signature.parameters.keys())

    @property
    def target_keys(self) -> Set[str]:
        return set(self.target_signature.parameters.keys())

    @property
    def reset_keys(self) -> Set[str]:
        return set(self.reset_signature.parameters.keys())

    @class_property
    @classmethod
    def default_config(cls) -> CN:
        """Reflect ``cls.__init__``'s parameters into a yacs ``CfgNode``."""

        def _check_value(value):
            if value is inspect.Parameter.empty:
                return None
            if isinstance(value, (int, float, str, bool, type(None), list, dict)):
                return value
            if isinstance(value, (tuple, set)):
                return list(value)
            return None

        init_sign = inspect.signature(cls.__init__)
        init_param = dict(init_sign.parameters)
        del init_param["self"]
        if contains_var_keyword(init_sign):
            del init_param[get_var_keyword(init_sign)]
        cfg = CN({k: _check_value(p) for k, p in init_param.items()})
        cfg.keyword = cls.__name__
        return cfg


class ServiceFactory:
    """Service Factory (singleton + classmethod registry)."""

    __instance: Optional["ServiceFactory"] = None
    __service_map: Dict[str, Type[_AbstractService]] = {}

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __contains__(self, keyword: str) -> bool:
        return keyword in self.__service_map

    def __iter__(self):
        return iter(self.__service_map)

    @class_property
    @classmethod
    def services(cls) -> Set[str]:
        return set(c.__name__ for c in cls.__service_map.values())

    @class_property
    @classmethod
    def keywords(cls) -> Set[str]:
        return set(cls.__service_map.keys())

    @classmethod
    def register(
        cls,
        class_type: Type[_AbstractService],
        keywords: Optional[Set[str]] = None,
    ) -> None:
        keywords = keywords or set()
        keywords.update({class_type.__name__})
        for key in keywords:
            if key in cls.__service_map:
                raise ValueError(f"Keyword {key} has been registered.")
            cls.__service_map[key] = class_type

    @classmethod
    def service_class(cls, keyword: str) -> Type[_AbstractService]:
        if keyword in cls.__service_map:
            return cls.__service_map[keyword]
        raise ValueError(f"Service {keyword} is not supported.")

    @classmethod
    def initial(cls, keyword: str, **kwargs: Any) -> _AbstractService:
        return cls.service_class(keyword)(**kwargs)


class AbstractService(_AbstractService, ABC):
    """Abstract Service base — auto-registers subclasses with ``ServiceFactory``."""

    def __init_subclass__(
        cls,
        keywords: Optional[Union[Set[str], List[str], Tuple[str, ...], str]] = None,
        **kwargs: Any,
    ) -> None:
        if keywords is None:
            keywords = set()
        elif isinstance(keywords, str):
            keywords = {keywords}
        elif isinstance(keywords, (list, tuple)):
            keywords = set(keywords)
        elif isinstance(keywords, set):
            pass
        else:
            raise ValueError(f"Unsupported keywords type: {type(keywords)}")
        _AbstractService.__init_subclass__()
        ABC.__init_subclass__()
        ServiceFactory.register(cls, keywords)

    def __init__(self, **kwargs: Any):
        _AbstractService.__init__(self, **kwargs)
        ABC.__init__(self)


class PathService(AbstractService, ABC, keywords={"Path", "path", to_snake_case("PathService")}):
    """Construct a service from a YAML config path via ``ServiceFactory``."""

    def __new__(cls, path):
        cfg = load_yaml(path)
        return ServiceFactory.initial(**cfg)
