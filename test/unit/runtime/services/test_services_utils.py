"""Unit tests for ``seggpt.runtime.services.utils``.

The module imports ``import_torch`` (lazy) and ``USE_CUDA`` (env-var
descriptor) at module-load time. ``torch_use_cuda()`` resolves the
torch import on first call, which on a host without torch raises
ImportError — gate that test with ``pytest.importorskip("torch")``.
The variadic-keyword helpers don't touch torch, so they always run.
"""
from __future__ import annotations

import inspect

import pytest

from seggpt.runtime.services.utils import contains_var_keyword, get_var_keyword


class TestContainsVarKeyword:
    def test_true_when_kwargs_present(self) -> None:
        def fn(a, b, **kwargs):
            pass

        assert contains_var_keyword(inspect.signature(fn)) is True

    def test_false_when_only_positional(self) -> None:
        def fn(a, b, c):
            pass

        assert contains_var_keyword(inspect.signature(fn)) is False

    def test_false_when_only_var_positional(self) -> None:
        def fn(*args):
            pass

        assert contains_var_keyword(inspect.signature(fn)) is False

    def test_true_alongside_args(self) -> None:
        def fn(a, *args, **kwargs):
            pass

        assert contains_var_keyword(inspect.signature(fn)) is True


class TestGetVarKeyword:
    def test_returns_kwargs_name(self) -> None:
        def fn(a, **kwargs):
            pass

        assert get_var_keyword(inspect.signature(fn)) == "kwargs"

    def test_returns_custom_name(self) -> None:
        def fn(**opts):
            pass

        assert get_var_keyword(inspect.signature(fn)) == "opts"

    def test_empty_when_no_var_keyword(self) -> None:
        def fn(a, b):
            pass

        assert get_var_keyword(inspect.signature(fn)) == ""


class TestTorchUseCuda:
    """Torch-resolution test runs only when torch is importable."""

    def test_returns_cpu_when_use_cuda_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("torch")

        from seggpt.runtime.services.utils import torch_use_cuda

        monkeypatch.setenv("USE_CUDA", "0")
        assert torch_use_cuda() == "cpu"

    def test_returns_cpu_when_no_cuda_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("torch")

        from seggpt.runtime.services import utils as svc_utils

        monkeypatch.setenv("USE_CUDA", "1")
        # Force cuda.is_available() to False to simulate CPU-only host.
        monkeypatch.setattr(svc_utils.torch.cuda, "is_available", lambda: False)
        assert svc_utils.torch_use_cuda() == "cpu"

    def test_returns_cuda_when_both_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("torch")

        from seggpt.runtime.services import utils as svc_utils

        monkeypatch.setenv("USE_CUDA", "1")
        monkeypatch.setattr(svc_utils.torch.cuda, "is_available", lambda: True)
        assert svc_utils.torch_use_cuda() == "cuda"
