"""Unit tests for the SEGGPT_PRECISION switch (``configure_precision`` /
``apply_precision_from_env`` in ``seggpt.runtime.services.utils``, #14).

The helper only flips ``torch.backends`` flags, so it is testable on any
host with torch — no GPU, no weights.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from seggpt.runtime.services.utils import apply_precision_from_env, configure_precision  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_backend_flags():
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = matmul, cudnn


class TestConfigurePrecision:
    def test_tf32_enables_tensor_core_matmul_and_cudnn(self) -> None:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        assert configure_precision("tf32") == "tf32"
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True

    def test_fp32_turns_matmul_tf32_off(self) -> None:
        torch.backends.cuda.matmul.allow_tf32 = True
        assert configure_precision("fp32") == "fp32"
        assert torch.backends.cuda.matmul.allow_tf32 is False

    def test_mode_is_case_insensitive_and_normalised(self) -> None:
        assert configure_precision("TF32") == "tf32"
        assert torch.backends.cuda.matmul.allow_tf32 is True

    def test_unknown_mode_raises_listing_choices(self) -> None:
        with pytest.raises(ValueError, match=r"SEGGPT_PRECISION.*'int4'.*fp32.*tf32"):
            configure_precision("int4")

    def test_fp16_is_not_accepted_yet(self) -> None:
        # reserved for #12 (autocast); must fail loudly instead of silently running fp32
        with pytest.raises(ValueError, match="fp16"):
            configure_precision("fp16")


class TestApplyPrecisionFromEnv:
    def test_default_is_fp32(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SEGGPT_PRECISION", raising=False)
        torch.backends.cuda.matmul.allow_tf32 = True
        assert apply_precision_from_env() == "fp32"
        assert torch.backends.cuda.matmul.allow_tf32 is False

    def test_env_tf32_is_applied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SEGGPT_PRECISION", "tf32")
        torch.backends.cuda.matmul.allow_tf32 = False
        assert apply_precision_from_env() == "tf32"
        assert torch.backends.cuda.matmul.allow_tf32 is True
