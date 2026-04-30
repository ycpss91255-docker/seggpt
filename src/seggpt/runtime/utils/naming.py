"""Case conversion helpers used by service registration keywords.

The upstream ``generative_services.utils.naming`` module bundled a
random-noun name generator (~1000 LOC) along with these two case
converters; only the converters are referenced by the SegGPT runtime,
so the rest is dropped.
"""
import re

_CAMEL_TO_SNAKE_PATTERNS = (
    (re.compile(r"(.)([A-Z][a-z]+)"), r"\1_\2"),
    (re.compile(r"__([A-Z])"), r"_\1"),
    (re.compile(r"([a-z0-9])([A-Z])"), r"\1_\2"),
)


def to_snake_case(name: str) -> str:
    """Convert a ``CamelCase`` or ``mixedCase`` name to ``snake_case``."""
    for pattern, replacement in _CAMEL_TO_SNAKE_PATTERNS:
        name = pattern.sub(replacement, name)
    return name.lower()


def to_camel_case(name: str) -> str:
    """Convert a ``snake_case`` name to ``CamelCase``."""
    return "".join(word.title() for word in to_snake_case(name).split("_"))
