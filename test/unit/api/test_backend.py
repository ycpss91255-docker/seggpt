"""Unit tests for ``seggpt.api.SegGPTBackend``.

The wrapper is dep-light (numpy + a Layer 1 service handle), but
importing ``seggpt.api`` chains into ``seggpt_service`` which imports
``yacs`` at module-load time. Auto-skip on hosts without yacs;
inside the docker image yacs is installed and the full suite runs.

End-to-end mIoU lives in the integration suite — see
``test/integration/runtime/test_seggpt_backend_e2e.py``.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("yacs")


# --- Layer 1 stub used by every test --------------------------------------


class _StubService:
    """Minimal SegGPTService stand-in that records call order + returns canned output."""

    def __init__(self, *, device: str = "cpu", canned: Any = None) -> None:
        self.calls: list[str] = []
        self._device = device
        self._canned = canned or {
            "mask": np.zeros((1, 4, 4), dtype=np.uint8),
            "class_id": np.zeros((1,), dtype=np.int64),
        }

    def reset(self) -> "_StubService":
        self.calls.append("reset")
        return self

    def target(self, image: np.ndarray) -> "_StubService":  # noqa: ARG002
        self.calls.append("target")
        return self

    def prompt(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append("prompt")
        return self._canned


@pytest.fixture
def stub_service():
    return _StubService()


@pytest.fixture
def backend_with(stub_service, monkeypatch):
    """Build a SegGPTBackend whose underlying Layer 1 service is the stub."""
    from seggpt.api import backend as backend_mod

    monkeypatch.setattr(
        backend_mod,
        "SegGPTService",
        lambda **kwargs: stub_service,
    )
    return backend_mod.SegGPTBackend(model_path="m.pth", config_path="c.yaml")


# --- Tests -----------------------------------------------------------------


class TestInferContract:
    def test_returns_expected_keys(self, backend_with) -> None:
        result = backend_with.infer(
            target=np.zeros((4, 4, 3), dtype=np.uint8),
            refs=[np.zeros((4, 4, 3), dtype=np.uint8)],
            masks=[np.zeros((1, 4, 4), dtype=np.uint8)],
        )
        assert set(result.keys()) == {"mask", "class_id", "inference_latency_ms", "gpu_mem_mb"}

    def test_calls_reset_target_prompt_in_order(self, backend_with, stub_service) -> None:
        backend_with.infer(
            target=np.zeros((4, 4, 3), dtype=np.uint8),
            refs=[np.zeros((4, 4, 3), dtype=np.uint8)],
            masks=[np.zeros((1, 4, 4), dtype=np.uint8)],
        )
        assert stub_service.calls == ["reset", "target", "prompt"]

    def test_consecutive_calls_each_reset(self, backend_with, stub_service) -> None:
        for _ in range(3):
            backend_with.infer(
                target=np.zeros((4, 4, 3), dtype=np.uint8),
                refs=[np.zeros((4, 4, 3), dtype=np.uint8)],
                masks=[np.zeros((1, 4, 4), dtype=np.uint8)],
            )
        assert stub_service.calls == ["reset", "target", "prompt"] * 3

    def test_mode_passes_through_to_layer1(self, backend_with, stub_service) -> None:
        captured: dict[str, Any] = {}

        def _record_prompt(**kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            stub_service.calls.append("prompt")
            return stub_service._canned

        stub_service.prompt = _record_prompt  # type: ignore[method-assign]
        backend_with.infer(
            target=np.zeros((4, 4, 3), dtype=np.uint8),
            refs=[np.zeros((4, 4, 3), dtype=np.uint8)],
            masks=[np.zeros((1, 4, 4), dtype=np.uint8)],
            mode="semantic",
        )
        assert captured["segmentation_mode"] == "semantic"

    def test_class_id_passes_through_to_layer1(self, backend_with, stub_service) -> None:
        captured: dict[str, Any] = {}

        def _record_prompt(**kwargs: Any) -> dict[str, Any]:
            captured.update(kwargs)
            stub_service.calls.append("prompt")
            return stub_service._canned

        stub_service.prompt = _record_prompt  # type: ignore[method-assign]
        cid = [np.array([0]), np.array([0])]
        backend_with.infer(
            target=np.zeros((4, 4, 3), dtype=np.uint8),
            refs=[np.zeros((4, 4, 3), dtype=np.uint8)] * 2,
            masks=[np.zeros((1, 4, 4), dtype=np.uint8)] * 2,
            class_id=cid,
        )
        assert captured["class_id"] is cid


class TestValidation:
    def test_rejects_refs_masks_length_mismatch(self, backend_with) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            backend_with.infer(
                target=np.zeros((4, 4, 3), dtype=np.uint8),
                refs=[np.zeros((4, 4, 3), dtype=np.uint8)] * 2,
                masks=[np.zeros((1, 4, 4), dtype=np.uint8)],  # length 1
            )


class TestTelemetry:
    def test_latency_is_positive(self, backend_with) -> None:
        result = backend_with.infer(
            target=np.zeros((4, 4, 3), dtype=np.uint8),
            refs=[np.zeros((4, 4, 3), dtype=np.uint8)],
            masks=[np.zeros((1, 4, 4), dtype=np.uint8)],
        )
        assert result["inference_latency_ms"] > 0.0

    def test_gpu_mem_zero_on_cpu_device(self, backend_with) -> None:
        result = backend_with.infer(
            target=np.zeros((4, 4, 3), dtype=np.uint8),
            refs=[np.zeros((4, 4, 3), dtype=np.uint8)],
            masks=[np.zeros((1, 4, 4), dtype=np.uint8)],
        )
        # stub service.device == 'cpu' → wrapper skips CUDA peak read
        assert result["gpu_mem_mb"] == 0.0

    def test_cuda_path_calls_synchronize_and_peak_read(self, monkeypatch) -> None:
        """CUDA branch synchronises around the prompt and reads peak memory."""
        from seggpt.api import backend as backend_mod

        stub = _StubService(device="cuda:0")
        monkeypatch.setattr(backend_mod, "SegGPTService", lambda **kw: stub)

        fake_cuda = MagicMock()
        fake_cuda.max_memory_allocated.return_value = 256 * 1024 * 1024  # 256 MB
        fake_torch = MagicMock()
        fake_torch.cuda = fake_cuda
        monkeypatch.setattr(backend_mod, "torch", fake_torch)

        backend = backend_mod.SegGPTBackend(model_path="m.pth", config_path="c.yaml")
        result = backend.infer(
            target=np.zeros((4, 4, 3), dtype=np.uint8),
            refs=[np.zeros((4, 4, 3), dtype=np.uint8)],
            masks=[np.zeros((1, 4, 4), dtype=np.uint8)],
        )

        assert fake_cuda.reset_peak_memory_stats.call_count == 1
        # synchronize before + after → 2 calls
        assert fake_cuda.synchronize.call_count == 2
        assert result["gpu_mem_mb"] == pytest.approx(256.0, rel=1e-3)


class TestServiceExposure:
    def test_service_property_returns_underlying_layer1(self, backend_with, stub_service) -> None:
        assert backend_with.service is stub_service
