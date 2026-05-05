"""Stateless ``SegGPTBackend.infer()`` wrapper around Layer 1.

Each ``infer(...)`` call resets the underlying ``SegGPTService``,
sets the target, runs prompt, and returns raw output plus telemetry.
Model weights are loaded once at construction time and cached on the
instance — multiple ``infer()`` calls reuse the same model.

The output dict is **raw**: callers (CoreSAM in production, the
Phase 0 benchmark harness during validation) decide whether the mask
is good enough to surface, what bounding box to derive, what status
code to set. This split keeps the backend reusable beyond CoreSAM
(see CLAUDE.md SegGPT Backend Repo job-boundary section).

Output schema::

    {
        "mask":               np.ndarray (C, H, W) bool,   # raw prediction
        "class_id":           np.ndarray (C,)    int,      # class indices
        "inference_latency_ms": float,                     # prompt() wall time
        "gpu_mem_mb":          float,                      # peak CUDA alloc
    }
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from seggpt.runtime.services.import_modules import import_torch as torch
from seggpt.runtime.services.seggpt_service import SegGPTService
from seggpt.runtime.utils.types import PathLike


class SegGPTBackend:
    """Stateless visual-prompt segmentation backend.

    Args:
        model_path: Path to ``seggpt_vit_large.pth`` (or another
            compatible checkpoint).
        config_path: Path to the model architecture YAML
            (img_size / depth / heads / etc).
        warmup: Whether to run a single dummy forward pass at construction
            time to JIT-compile CUDA kernels and lock in stable latency
            measurements for subsequent ``infer()`` calls. Defaults to
            ``False`` so cold-start latency is also measurable; flip on
            for benchmark harnesses that average across runs.
    """

    def __init__(
        self,
        model_path: PathLike,
        config_path: PathLike,
        *,
        warmup: bool = False,
    ) -> None:
        self._service = SegGPTService(
            checkpoint_path=model_path,
            config_path=config_path,
        )
        if warmup:
            self._warmup()

    @property
    def service(self) -> SegGPTService:
        """Underlying Layer 1 service. Exposed for advanced introspection."""
        return self._service

    def infer(
        self,
        target: np.ndarray,
        refs: Sequence[np.ndarray],
        masks: Sequence[np.ndarray],
        *,
        mode: str = "instance",
        class_id: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Run a single one-shot inference.

        Args:
            target: Target RGB image, shape ``(H, W, 3)``.
            refs: Reference RGB images, each shape ``(H, W, 3)``.
            masks: Reference masks, each shape ``(C, H, W)`` or ``(H, W)``.
                Must match ``len(refs)``.
            mode: ``"instance"`` (default) or ``"semantic"``. ``"panoptic"``
                is intentionally not exposed (upstream marks it as
                not-yet-supported).
            class_id: Optional list of class-id arrays (one per ref);
                if omitted, every reference contributes equally.

        Returns:
            Dict with keys ``mask`` / ``class_id`` /
            ``inference_latency_ms`` / ``gpu_mem_mb``. The first two
            come straight from Layer 1; the latter two are telemetry
            this wrapper measures.
        """
        if len(refs) != len(masks):
            raise ValueError(
                f"refs ({len(refs)}) and masks ({len(masks)}) length mismatch"
            )

        # Stateless contract: each call starts from a clean Layer 1 state.
        self._service.reset()
        self._service.target(target)

        # Wall-time + peak-GPU-mem around the actual forward pass. Reset
        # the CUDA peak counter first so the reading reflects only this
        # infer() call, not history.
        on_cuda = self._device_is_cuda()
        if on_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        result = self._service.prompt(
            images=list(refs),
            masks=list(masks),
            class_id=class_id,
            segmentation_mode=mode,
        )
        if on_cuda:
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        gpu_mem_mb = 0.0
        if on_cuda:
            gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

        return {
            "mask": result["mask"],
            "class_id": result["class_id"],
            "inference_latency_ms": latency_ms,
            "gpu_mem_mb": gpu_mem_mb,
        }

    def _device_is_cuda(self) -> bool:
        device = getattr(self._service, "_device", "cpu")
        return str(device).startswith("cuda")

    def _warmup(self) -> None:
        """One-shot dummy forward to JIT CUDA kernels.

        Uses a 64x64 black target + single 64x64 black ref/mask so the
        warm-up cost is negligible vs. a real forward but exercises the
        full kernel-compilation path.
        """
        dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
        dummy_mask = np.zeros((1, 64, 64), dtype=np.uint8)
        self.infer(dummy_img, [dummy_img], [dummy_mask])
