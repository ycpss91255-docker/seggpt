"""Unit tests for ``seggpt.runtime.utils.logger``."""
from __future__ import annotations

import logging

import pytest

from seggpt.runtime.utils.logger import (
    _LOGGER,
    logd_print,
    loge_print,
    logi_print,
    logw_print,
)


@pytest.fixture
def captured_logs(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.DEBUG, logger="seggpt.runtime")
    return caplog


class TestLevelWrappers:
    def test_logd_print_emits_at_debug(self, captured_logs: pytest.LogCaptureFixture) -> None:
        logd_print("debug-msg")
        record = captured_logs.records[-1]
        assert record.levelno == logging.DEBUG
        assert record.message == "debug-msg"

    def test_logi_print_emits_at_info(self, captured_logs: pytest.LogCaptureFixture) -> None:
        logi_print("info-msg")
        record = captured_logs.records[-1]
        assert record.levelno == logging.INFO
        assert record.message == "info-msg"

    def test_logw_print_emits_at_warning(self, captured_logs: pytest.LogCaptureFixture) -> None:
        logw_print("warn-msg")
        record = captured_logs.records[-1]
        assert record.levelno == logging.WARNING
        assert record.message == "warn-msg"

    def test_loge_print_emits_at_error(self, captured_logs: pytest.LogCaptureFixture) -> None:
        loge_print("err-msg")
        record = captured_logs.records[-1]
        assert record.levelno == logging.ERROR
        assert record.message == "err-msg"

    def test_logger_namespaced_under_seggpt_runtime(self) -> None:
        # Every record routes through one stable logger so external code
        # can opt-in or filter via the standard logging hierarchy.
        assert _LOGGER.name == "seggpt.runtime"


class TestNonStringPayloads:
    def test_dict_message_round_trips(self, captured_logs: pytest.LogCaptureFixture) -> None:
        logi_print({"event": "loaded", "size_mb": 350})
        record = captured_logs.records[-1]
        assert record.msg == {"event": "loaded", "size_mb": 350}

    def test_list_message_round_trips(self, captured_logs: pytest.LogCaptureFixture) -> None:
        logw_print([1, 2, 3])
        record = captured_logs.records[-1]
        assert record.msg == [1, 2, 3]
