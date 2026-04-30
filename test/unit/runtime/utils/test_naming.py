"""Unit tests for ``seggpt.runtime.utils.naming``."""
from __future__ import annotations

import pytest

from seggpt.runtime.utils.naming import to_camel_case, to_snake_case


class TestToSnakeCase:
    @pytest.mark.parametrize(
        "src,expected",
        [
            ("SegGPT", "seg_gpt"),
            ("SegGPTService", "seg_gpt_service"),
            ("SegGPTDaemonService", "seg_gpt_daemon_service"),
            ("AbstractService", "abstract_service"),
            ("XMLHttpRequest", "xml_http_request"),
            ("simple", "simple"),
            ("alreadysnake", "alreadysnake"),
            ("already_snake_case", "already_snake_case"),
            ("camelCase", "camel_case"),
        ],
    )
    def test_round_trip_examples(self, src: str, expected: str) -> None:
        assert to_snake_case(src) == expected

    def test_empty_string(self) -> None:
        assert to_snake_case("") == ""


class TestToCamelCase:
    @pytest.mark.parametrize(
        "src,expected",
        [
            ("seg_gpt", "SegGpt"),
            ("seg_gpt_service", "SegGptService"),
            ("abstract_service", "AbstractService"),
            ("camelCase", "CamelCase"),
            ("simple", "Simple"),
            ("multiple__underscores", "MultipleUnderscores"),
        ],
    )
    def test_round_trip_examples(self, src: str, expected: str) -> None:
        assert to_camel_case(src) == expected

    def test_empty_string(self) -> None:
        assert to_camel_case("") == ""

    def test_idempotent_on_already_camel(self) -> None:
        # CamelCase input goes through to_snake_case first, then re-camels;
        # acronyms don't perfectly round-trip but the final form is stable.
        once = to_camel_case("SegGptService")
        twice = to_camel_case(once)
        assert once == twice
