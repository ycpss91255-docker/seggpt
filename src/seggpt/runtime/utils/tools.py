"""Filesystem + YAML helpers used by Layer 1.

Slimmed port of ``generative_services.utils.tools``. Only the helpers
referenced by the SegGPT runtime survive: ``check_path`` /
``path_with_home`` resolve relative paths against ``GSI_HOME``, and
``load_yaml`` reads a config into a yacs ``CfgNode``.

Image I/O helpers, RLE writers, and the dictionary-conversion
machinery (~250 LOC of upstream ``tools``) are out of Layer 1's scope.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import yaml
from yacs.config import CfgNode as CN

from seggpt.runtime.utils.environment_variables import GSI_HOME
from seggpt.runtime.utils.types import PathLike


def check_path(path: PathLike) -> Path:
    """Resolve ``path`` to an existing file/dir or raise.

    If ``path`` is relative and missing, fall back to
    ``GSI_HOME / path`` before raising.
    """
    if not isinstance(path, Path):
        path = Path(path)
    if path.exists():
        return path
    if not path.is_absolute():
        new_path = GSI_HOME.get() / path
        if new_path.exists():
            return new_path
    raise FileNotFoundError(f"File not found: {path}")


def path_with_home(path: PathLike) -> Path:
    """Return ``path`` as-is if absolute, else prefix ``GSI_HOME``."""
    if not isinstance(path, Path):
        path = Path(path)
    if path.is_absolute():
        return path
    return GSI_HOME.get() / path


def load_yaml(yaml_file: PathLike, use_yacs: bool = True) -> Union[CN, Dict]:
    """Load a YAML file via ``yaml.safe_load``, optionally as a yacs ``CfgNode``."""
    with open(check_path(yaml_file), "r", encoding="ascii") as fin:
        raw = yaml.safe_load(fin.read())
    if use_yacs:
        return CN(raw)
    return raw
