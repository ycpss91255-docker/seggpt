"""End-to-end Phase 0 integration test.

Loads the in-repo hmbb fixtures, runs ``SegGPTBackend.infer()``, and
asserts the predicted mask matches the reference output above the
0.9 mIoU threshold (matches the upstream test_seggpt_service.py
contract that this port must preserve).

Auto-skips on:
  * hosts without ``torch`` / ``cv2``;
  * environments where the SegGPT checkpoint is not on disk;
  * environments with no CUDA device (CPU forward of ViT-Large is
    ~minutes per call — too slow for CI's 30-min budget).

Inside the docker image with the checkpoint mounted at
``/home/<user>/work/model/seggpt_vit_large.pth`` and a GPU visible,
this test runs the same shape of check the legacy
``tests/services/test_seggpt_service.py`` did.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")
pytest.importorskip("yacs")

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = Path(os.environ.get("SEGGPT_MODEL_PATH", REPO_ROOT / "model" / "seggpt_vit_large.pth"))
CONFIG_PATH = Path(
    os.environ.get("SEGGPT_CONFIG_PATH", REPO_ROOT / "model" / "seggpt_vit_large.yaml")
)
HMBB = REPO_ROOT / "test" / "assets" / "hmbb"
EXPECTED = REPO_ROOT / "test" / "assets" / "expected" / "output_hmbb_3.png"

requires_weights = pytest.mark.skipif(
    not MODEL_PATH.exists() or not CONFIG_PATH.exists(),
    reason=f"model weights or config missing ({MODEL_PATH} / {CONFIG_PATH})",
)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device not available; ViT-Large CPU forward is too slow",
)


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_mask(path: Path) -> np.ndarray:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def _miou(pred: np.ndarray, true: np.ndarray, eps: float = 1e-8) -> float:
    pred_b = pred.astype(bool)
    true_b = true.astype(bool)
    intersection = np.logical_and(pred_b, true_b).sum()
    union = np.logical_or(pred_b, true_b).sum()
    return float(intersection) / float(union + eps)


@pytest.fixture(scope="module")
def backend():
    from seggpt.api import SegGPTBackend

    return SegGPTBackend(model_path=MODEL_PATH, config_path=CONFIG_PATH)


@requires_weights
@requires_cuda
def test_hmbb_infer_returns_expected_keys(backend) -> None:
    target = _read_rgb(HMBB / "hmbb_3.jpg")
    refs = [_read_rgb(HMBB / "hmbb_1.jpg"), _read_rgb(HMBB / "hmbb_2.jpg")]
    masks = [_read_mask(HMBB / "hmbb_1_target.png"), _read_mask(HMBB / "hmbb_2_target.png")]

    result = backend.infer(target, refs, masks)

    assert set(result.keys()) == {"mask", "class_id", "inference_latency_ms", "gpu_mem_mb"}
    assert isinstance(result["mask"], np.ndarray)
    assert isinstance(result["class_id"], np.ndarray)
    assert result["inference_latency_ms"] > 0.0
    assert result["gpu_mem_mb"] >= 0.0


@requires_weights
@requires_cuda
def test_hmbb_infer_miou_above_0p9(backend) -> None:
    """Locks down the ported pipeline against upstream's 0.9 mIoU bar."""
    target = _read_rgb(HMBB / "hmbb_3.jpg")
    refs = [_read_rgb(HMBB / "hmbb_1.jpg"), _read_rgb(HMBB / "hmbb_2.jpg")]
    masks = [_read_mask(HMBB / "hmbb_1_target.png"), _read_mask(HMBB / "hmbb_2_target.png")]
    expected = _read_mask(EXPECTED)

    result = backend.infer(target, refs, masks)
    pred = result["mask"][0]
    if pred.shape != expected.shape:
        pred = cv2.resize(
            pred.astype(np.uint8),
            (expected.shape[1], expected.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    assert _miou(pred, expected) > 0.9


@requires_weights
@requires_cuda
def test_infer_is_stateless_across_calls(backend) -> None:
    """Two consecutive ``infer()`` calls with the same inputs return the same mask."""
    target = _read_rgb(HMBB / "hmbb_3.jpg")
    refs = [_read_rgb(HMBB / "hmbb_1.jpg"), _read_rgb(HMBB / "hmbb_2.jpg")]
    masks = [_read_mask(HMBB / "hmbb_1_target.png"), _read_mask(HMBB / "hmbb_2_target.png")]

    a = backend.infer(target, refs, masks)
    b = backend.infer(target, refs, masks)

    assert np.array_equal(a["mask"], b["mask"])


@requires_weights
@requires_cuda
def test_infer_rejects_mismatched_refs_and_masks(backend) -> None:
    target = _read_rgb(HMBB / "hmbb_3.jpg")
    refs = [_read_rgb(HMBB / "hmbb_1.jpg"), _read_rgb(HMBB / "hmbb_2.jpg")]
    masks = [_read_mask(HMBB / "hmbb_1_target.png")]  # length-1, refs is length-2

    with pytest.raises(ValueError, match="length mismatch"):
        backend.infer(target, refs, masks)
