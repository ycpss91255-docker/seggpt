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
        # Already grayscale, fg=bright minority: pass through, no invert
        h, w = 8, 8
        gray = np.zeros((h, w), dtype=np.uint8)
        gray[4:6, 4:6] = 200
        out = self._write_and_read(driver, tmp_path, "gray.png", gray)
        assert out.ndim == 2
        assert out.shape == (h, w)
        assert (out[4:6, 4:6] == 200).all()

    def test_white_bg_black_fg_three_channel_auto_inverts(self, driver, tmp_path):
        # User's other authoring style: white background, black foreground
        # (no alpha). Heuristic flips it so SegGPT still gets fg=255.
        h, w = 8, 8
        bgr = np.full((h, w, 3), 255, dtype=np.uint8)  # white bg
        bgr[3:5, 3:5, :] = 0  # black fg patch
        out = self._write_and_read(driver, tmp_path, "white_bg_bgr.png", bgr)
        assert out.ndim == 2
        assert out.shape == (h, w)
        assert out[3:5, 3:5].all(), "black-on-white fg must be inverted to 255"
        # background (was 255) should now be 0
        assert not out[0:3, :].any()

    def test_white_bg_black_fg_grayscale_auto_inverts(self, driver, tmp_path):
        # Same scenario in single-channel: bright bg + dark fg -> invert
        h, w = 8, 8
        gray = np.full((h, w), 255, dtype=np.uint8)
        gray[2:4, 2:4] = 0
        out = self._write_and_read(driver, tmp_path, "white_bg_gray.png", gray)
        assert out[2:4, 2:4].all()
        assert not out[0:2, :].any()

    def test_rgba_all_opaque_falls_back_to_rgb(self, driver, tmp_path):
        # Edge: RGBA exists but alpha is constant (all-255). Shouldn't pick
        # alpha (it's all 255 = "everything is foreground"); fall back to
        # the BGR channels and apply the invert heuristic on those.
        h, w = 8, 8
        rgba = np.full((h, w, 4), 255, dtype=np.uint8)  # white bg + opaque
        rgba[3:5, 3:5, :3] = 0  # black fg in BGR
        out = self._write_and_read(driver, tmp_path, "rgba_opaque.png", rgba)
        assert out[3:5, 3:5].all()
        assert not out[0:3, :].any()


class TestApplyOverlay:
    """Lock the overlay blend semantics: mask-only tint, alpha-correct."""

    def test_only_mask_pixels_change(self, driver):
        target = np.full((4, 4, 3), 100, dtype=np.uint8)
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        out = driver._apply_overlay(target, mask, (0, 0, 0), 0.5)
        # Outside the mask: unchanged
        assert (out[0, :, :] == 100).all()
        assert (out[3, :, :] == 100).all()
        # Inside the mask: blended toward black at alpha=0.5 -> 50
        assert (out[1:3, 1:3, :] == 50).all()

    def test_alpha_one_paints_pure_tint(self, driver):
        target = np.full((4, 4, 3), 200, dtype=np.uint8)
        mask = np.ones((4, 4), dtype=bool)
        out = driver._apply_overlay(target, mask, (10, 20, 30), 1.0)
        assert (out[..., 0] == 10).all()
        assert (out[..., 1] == 20).all()
        assert (out[..., 2] == 30).all()

    def test_alpha_zero_leaves_target_unchanged(self, driver):
        target = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        mask = np.ones((8, 8), dtype=bool)
        out = driver._apply_overlay(target, mask, (0, 0, 0), 0.0)
        assert np.array_equal(out, target)

    def test_resizes_mismatched_mask_via_nearest(self, driver):
        # 4x4 target, 2x2 mask: nearest-neighbour upsample fills 2x2 blocks
        target = np.full((4, 4, 3), 200, dtype=np.uint8)
        mask = np.array([[True, False], [False, True]], dtype=bool)
        out = driver._apply_overlay(target, mask, (0, 0, 0), 1.0)
        # Top-left 2x2 quadrant tinted, bottom-right 2x2 tinted, others not
        assert (out[0:2, 0:2, :] == 0).all()
        assert (out[0:2, 2:4, :] == 200).all()
        assert (out[2:4, 0:2, :] == 200).all()
        assert (out[2:4, 2:4, :] == 0).all()


class TestParseBgr:
    def test_three_ints_parses_to_tuple(self, driver):
        assert driver._parse_bgr("0,0,0") == (0, 0, 0)
        assert driver._parse_bgr(" 10, 20, 30 ") == (10, 20, 30)
        assert driver._parse_bgr("255,255,255") == (255, 255, 255)

    def test_wrong_count_rejected(self, driver):
        import argparse
        with pytest.raises(argparse.ArgumentTypeError):
            driver._parse_bgr("0,0")
        with pytest.raises(argparse.ArgumentTypeError):
            driver._parse_bgr("0,0,0,0")

    def test_out_of_range_rejected(self, driver):
        import argparse
        with pytest.raises(argparse.ArgumentTypeError):
            driver._parse_bgr("256,0,0")
        with pytest.raises(argparse.ArgumentTypeError):
            driver._parse_bgr("-1,0,0")

    def test_non_int_rejected(self, driver):
        import argparse
        with pytest.raises(argparse.ArgumentTypeError):
            driver._parse_bgr("0,0,red")


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


class TestPassStats:
    """Lock the pass/fail definition: pass = mIoU >= threshold (matches the
    failure-copy condition `miou < threshold` in main()), so pass_count +
    fail_count always equals total. Empty input returns 0/0/0/0.0."""

    def test_empty_returns_zero_rates(self, driver):
        out = driver._pass_stats([], 0.5)
        assert out == {"total": 0, "pass_count": 0, "fail_count": 0, "pass_rate": 0.0}

    def test_all_pass(self, driver):
        out = driver._pass_stats([0.6, 0.7, 0.8], 0.5)
        assert out["pass_count"] == 3
        assert out["fail_count"] == 0
        assert out["total"] == 3
        assert out["pass_rate"] == pytest.approx(1.0)

    def test_all_fail(self, driver):
        out = driver._pass_stats([0.1, 0.2, 0.4], 0.5)
        assert out["pass_count"] == 0
        assert out["fail_count"] == 3
        assert out["pass_rate"] == pytest.approx(0.0)

    def test_threshold_is_inclusive(self, driver):
        # mIoU == threshold counts as a pass (>=), mirroring the inverse
        # `< threshold` failure-copy condition in main().
        out = driver._pass_stats([0.5, 0.5, 0.4999], 0.5)
        assert out["pass_count"] == 2
        assert out["fail_count"] == 1

    def test_mixed_rate_rounds_correctly(self, driver):
        # 4/10 pass -> 0.4
        mious = [0.6, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
        out = driver._pass_stats(mious, 0.5)
        assert out["pass_count"] == 4
        assert out["total"] == 10
        assert out["pass_rate"] == pytest.approx(0.4)

    def test_pass_plus_fail_equals_total(self, driver):
        # Invariant: pass_count + fail_count == total for any input.
        for vals in ([0.0], [1.0], [0.3, 0.5, 0.7], [0.5] * 5):
            out = driver._pass_stats(vals, 0.5)
            assert out["pass_count"] + out["fail_count"] == out["total"]


class TestLoadDriverYaml:
    """Lock the YAML override contract: missing -> {}, unknown keys -> SystemExit,
    overlay_color list -> tuple, malformed file rejected."""

    yaml = pytest.importorskip("yaml")

    def _write(self, tmp_path, body: str):
        p = tmp_path / "driver.yaml"
        p.write_text(body)
        return p

    def test_missing_file_returns_empty(self, driver, tmp_path):
        # No file on disk -> caller falls back to hardcoded argparse defaults.
        assert driver._load_driver_yaml(tmp_path / "nope.yaml") == {}

    def test_empty_file_returns_empty(self, driver, tmp_path):
        path = self._write(tmp_path, "")
        assert driver._load_driver_yaml(path) == {}

    def test_full_valid_file_parses(self, driver, tmp_path):
        path = self._write(
            tmp_path,
            "n_values: [1, 4]\n"
            "mode: semantic\n"
            "failure_threshold: 0.6\n"
            "overlay_color: [10, 20, 30]\n"
            "overlay_alpha: 0.3\n",
        )
        out = driver._load_driver_yaml(path)
        assert out == {
            "n_values": [1, 4],
            "mode": "semantic",
            "failure_threshold": 0.6,
            "overlay_color": (10, 20, 30),
            "overlay_alpha": 0.3,
        }

    def test_overlay_color_coerced_to_tuple_of_ints(self, driver, tmp_path):
        # YAML parses [0, 0, 0] as a list; loader coerces so downstream
        # cv2/_apply_overlay calls receive a hashable BGR tuple.
        path = self._write(tmp_path, "overlay_color: [0, 0, 0]\n")
        out = driver._load_driver_yaml(path)
        assert out["overlay_color"] == (0, 0, 0)
        assert isinstance(out["overlay_color"], tuple)
        assert all(isinstance(c, int) for c in out["overlay_color"])

    def test_unknown_key_raises_systemexit(self, driver, tmp_path):
        # Typo guard: silent ignore would leave the user thinking their
        # YAML edit applied. Surface the bad key in the error message.
        path = self._write(tmp_path, "n_value: [1, 4]\n")  # missing 's'
        with pytest.raises(SystemExit) as exc:
            driver._load_driver_yaml(path)
        assert "n_value" in str(exc.value)

    def test_overlay_color_wrong_length_rejected(self, driver, tmp_path):
        path = self._write(tmp_path, "overlay_color: [0, 0]\n")
        with pytest.raises(SystemExit):
            driver._load_driver_yaml(path)

    def test_overlay_color_non_list_rejected(self, driver, tmp_path):
        path = self._write(tmp_path, "overlay_color: black\n")
        with pytest.raises(SystemExit):
            driver._load_driver_yaml(path)

    def test_top_level_must_be_mapping(self, driver, tmp_path):
        # YAML scalar / list at top level is a misconfiguration.
        path = self._write(tmp_path, "- 1\n- 2\n")
        with pytest.raises(SystemExit):
            driver._load_driver_yaml(path)

    def test_no_gt_yaml_override(self, driver, tmp_path):
        # `no_gt: true` is the default in the shipped YAML (annotated GT
        # targets are not yet available); loader must accept the bool.
        path = self._write(tmp_path, "no_gt: true\n")
        out = driver._load_driver_yaml(path)
        assert out == {"no_gt": True}


class TestRepoYamlIsSelfConsistent:
    """The shipped config/phase0_driver.yaml must round-trip through the loader."""

    def test_repo_default_yaml_loads(self, driver):
        path = _REPO / "config" / "phase0_driver.yaml"
        assert path.exists(), "config/phase0_driver.yaml missing — driver default points here"
        out = driver._load_driver_yaml(path)
        # Sanity: every key in the shipped file is whitelisted.
        assert set(out).issubset(driver._DRIVER_CONFIG_KEYS)
        # Sanity: shipped n_values must all be drawable from the subset table.
        for n in out.get("n_values", []):
            assert n in driver._N_SUBSETS, f"shipped n_values has {n} not in _N_SUBSETS"
