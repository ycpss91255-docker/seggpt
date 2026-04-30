"""Unit tests for ``scripts/phase0_driver.py`` mask helpers.

The driver itself is an end-to-end harness — its real coverage lives in
running it against the test set inside the docker image. These tests pin
down the small pure-numpy helpers that wrap mIoU and resize-to-GT, since
those are the bits where bugs would silently distort every reported
score. SegGPT-side imports (``seggpt.api``) are deferred inside ``main``
and never touched here, so this suite runs on any host with numpy + cv2.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

_REPO = Path(__file__).resolve().parents[3]
_DRIVER = _REPO / "scripts" / "phase0_driver.py"


def _load_driver_module():
    """Import the driver as a module without going through ``scripts`` package."""
    spec = importlib.util.spec_from_file_location("phase0_driver", _DRIVER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    return _load_driver_module()


class TestMiou:
    def test_identical_masks_return_one(self, driver):
        a = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        assert driver._miou(a, a) == pytest.approx(1.0)

    def test_disjoint_masks_return_zero(self, driver):
        a = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        b = np.array([[0, 1], [0, 0]], dtype=np.uint8)
        assert driver._miou(a, b) == pytest.approx(0.0)

    def test_half_overlap_returns_one_third(self, driver):
        # 1 pixel intersection, 3 pixel union -> 1/3
        a = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        b = np.array([[1, 0], [1, 0]], dtype=np.uint8)
        assert driver._miou(a, b) == pytest.approx(1.0 / 3.0)

    def test_treats_nonzero_as_foreground(self, driver):
        # 0/255 PNG-style mask vs 0/1 mask should still match.
        a = np.array([[255, 255], [0, 0]], dtype=np.uint8)
        b = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        assert driver._miou(a, b) == pytest.approx(1.0)

    def test_empty_masks_return_zero_via_eps(self, driver):
        # Both empty: intersection 0, union 0; eps prevents div-by-zero.
        zeros = np.zeros((4, 4), dtype=np.uint8)
        assert driver._miou(zeros, zeros) == pytest.approx(0.0)


class TestResizeTo:
    def test_noop_when_shapes_match(self, driver):
        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        out = driver._resize_to(mask, mask.shape)
        assert out is mask  # identity, no copy

    def test_resizes_to_target_shape_with_nearest(self, driver):
        # 2x2 -> 4x4 nearest-neighbor: each pixel doubles in both axes.
        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        out = driver._resize_to(mask, (4, 4))
        assert out.shape == (4, 4)
        # Top-left and bottom-right quadrants are 1, others 0.
        assert out[0:2, 0:2].all()
        assert not out[0:2, 2:4].any()
        assert not out[2:4, 0:2].any()
        assert out[2:4, 2:4].all()

    def test_preserves_binary_values_no_interpolation(self, driver):
        # Resize must NOT introduce interpolated values like 128.
        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        out = driver._resize_to(mask, (8, 8))
        assert set(np.unique(out).tolist()).issubset({0, 1})


class TestReadMask:
    """Lock the 3 mask-on-disk shapes _read_mask must accept.

    Regression: GIMP-exported RGBA masks (R=G=B=0, alpha=mask) silently
    decoded to all-zero under the old ``cv2.imread(IMREAD_GRAYSCALE)``
    path because that codec drops alpha. Result was 96/96 empty SegGPT
    predictions in a real Phase 0 run. Recover alpha when present.
    """

    def _write_and_read(self, driver, tmp_path, name, arr):
        path = tmp_path / name
        cv2.imwrite(str(path), arr)
        return driver._read_mask(path)

    def test_rgba_alpha_encoded_mask_recovers_alpha(self, driver, tmp_path):
        # GIMP-style RGBA: R=G=B=0, alpha is the actual mask
        h, w = 8, 8
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[2:5, 3:6, 3] = 255  # alpha-only foreground patch
        out = self._write_and_read(driver, tmp_path, "rgba.png", rgba)
        assert out.ndim == 2
        assert out.shape == (h, w)
        assert out[2:5, 3:6].all()
        assert out.astype(bool).sum() == 9

    def test_bgr_three_channel_mask_grayscale_converted(self, driver, tmp_path):
        # Hmbb-style RGB mask (no alpha): BGR -> grayscale conversion path
        h, w = 8, 8
        bgr = np.zeros((h, w, 3), dtype=np.uint8)
        bgr[1:3, 1:3, :] = 255  # white square
        out = self._write_and_read(driver, tmp_path, "bgr.png", bgr)
        assert out.ndim == 2
        assert out.shape == (h, w)
        assert out[1:3, 1:3].all()

    def test_single_channel_grayscale_passthrough(self, driver, tmp_path):
        # Already grayscale: return as-is, no conversion
        h, w = 8, 8
        gray = np.zeros((h, w), dtype=np.uint8)
        gray[4:6, 4:6] = 200
        out = self._write_and_read(driver, tmp_path, "gray.png", gray)
        assert out.ndim == 2
        assert out.shape == (h, w)
        assert (out[4:6, 4:6] == 200).all()


class TestNSubsetsAreSubsetsOfPool:
    """Sanity: every preset N must be drawable from prompt_01..08."""

    def test_indices_within_pool_bounds(self, driver):
        for n, indices in driver._N_SUBSETS.items():
            assert len(indices) == n, f"N={n} subset size mismatch"
            assert all(1 <= i <= 8 for i in indices), (
                f"N={n} indices outside iron_beam pool 1..8: {indices}"
            )

    def test_indices_unique_per_subset(self, driver):
        for n, indices in driver._N_SUBSETS.items():
            assert len(set(indices)) == n, f"N={n} has duplicate indices: {indices}"
