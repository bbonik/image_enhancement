"""Unit tests for color space transforms and color operators."""

import numpy as np

from image_enhancement import (
    change_color_saturation,
    correct_colors,
    linear_to_srgb,
    srgb_to_linear,
    transfer_graytone_to_color,
)


class TestSrgbLinearTransforms:

    def test_srgb_to_linear_range_and_dtype(self, color_uint8):
        linear = srgb_to_linear(color_uint8)
        assert linear.shape == color_uint8.shape
        assert linear.dtype == np.float64
        assert linear.min() >= 0.0
        assert linear.max() <= 1.0

    def test_linear_to_srgb_returns_uint8(self, color_uint8):
        # uint8 [0,255] output is the documented, intentional convention
        srgb = linear_to_srgb(srgb_to_linear(color_uint8))
        assert srgb.dtype == np.uint8
        assert srgb.shape == color_uint8.shape

    def test_round_trip_within_one_level_full_range(self, color_uint8):
        # linear_to_srgb truncates to uint8, so allow off-by-one; this
        # must hold across the whole range, including deep shadows
        # (historical bug: sRGB levels <= ~10/255 collapsed to 0)
        round_trip = linear_to_srgb(srgb_to_linear(color_uint8))
        diff = np.abs(round_trip.astype(int) - color_uint8.astype(int))
        assert diff.max() <= 1

    def test_all_256_levels_round_trip(self):
        # every uint8 level must survive the round trip within one level
        levels = np.arange(256, dtype=np.uint8).reshape(1, 256, 1).repeat(3, axis=2)
        round_trip = linear_to_srgb(srgb_to_linear(levels))
        diff = np.abs(round_trip.astype(int) - levels.astype(int))
        assert diff.max() <= 1

    def test_linear_to_srgb_matches_easyrgb_reference(self):
        # ground truth: the EasyRGB piecewise formula, evaluated per pixel
        # https://www.easyrgb.com/en/math.php
        def easyrgb(v):
            if v > 0.0031308:
                return 1.055 * (v ** (1 / 2.4)) - 0.055
            return 12.92 * v

        linear = np.linspace(0, 1, 4097)
        linear = np.append(linear, [0.0031308, np.nextafter(0.0031308, 1)])
        expected = np.clip([easyrgb(v) for v in linear], 0, 1)
        expected = (np.asarray(expected) * 255).astype(np.uint8)

        image = np.tile(linear.reshape(1, -1, 1), (1, 1, 3))
        result = linear_to_srgb(image)
        assert np.array_equal(result[0, :, 0], expected)

    def test_uint8_and_float_inputs_equivalent(self, color_uint8):
        as_float = color_uint8.astype(np.float64) / 255.0
        out_uint8 = srgb_to_linear(color_uint8)
        out_float = srgb_to_linear(as_float)
        assert np.allclose(out_uint8, out_float)

    def test_linearization_darkens(self, color_uint8):
        # removing gamma correction lowers mid-tone values
        as_float = color_uint8.astype(np.float64) / 255.0
        linear = srgb_to_linear(color_uint8)
        assert linear.mean() < as_float.mean()

    def test_monotonic(self):
        ramp = np.linspace(0, 1, 256).reshape(1, -1, 1).repeat(3, axis=2)
        linear = srgb_to_linear(ramp)
        assert np.all(np.diff(linear[0, :, 0]) >= 0)


class TestChangeColorSaturation:

    def test_degree_one_is_identity_without_mask(self, color_uint8):
        out = change_color_saturation(color_uint8, None, sat_degree=1.0)
        expected = color_uint8.astype(np.float64) / 255.0
        assert np.allclose(out, expected, atol=1e-12)

    def test_degree_zero_gives_grayscale(self, color_uint8):
        out = change_color_saturation(color_uint8, None, sat_degree=0.0)
        assert np.allclose(out[:, :, 0], out[:, :, 1])
        assert np.allclose(out[:, :, 1], out[:, :, 2])

    def test_higher_degree_increases_saturation(self, color_uint8):
        base = change_color_saturation(color_uint8, None, sat_degree=1.0)
        boosted = change_color_saturation(color_uint8, None, sat_degree=2.0)

        def mean_sat(img):
            return (img.max(axis=2) - img.min(axis=2)).mean()

        assert mean_sat(boosted) > mean_sat(base)

    def test_output_range_and_shape(self, color_uint8):
        out = change_color_saturation(color_uint8, None, sat_degree=3.0)
        assert out.shape == color_uint8.shape
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_with_photometric_mask(self, color_uint8, gray_float):
        out = change_color_saturation(
            color_uint8, image_ph_mask=gray_float, sat_degree=1.5)
        assert out.shape == color_uint8.shape
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestCorrectColors:

    def test_output_range_and_shape(self, color_uint8):
        out = correct_colors(color_uint8)
        assert out.shape == color_uint8.shape
        assert out.dtype == np.float64
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_reduces_color_cast(self, color_uint8):
        # add a strong warm cast, then check channel means get closer
        cast = color_uint8.astype(np.float64)
        cast[:, :, 0] = np.clip(cast[:, :, 0] * 1.5, 0, 255)
        cast[:, :, 2] = cast[:, :, 2] * 0.5
        cast = cast.astype(np.uint8)

        corrected = correct_colors(cast)

        def channel_mean_spread(img):
            means = [img[:, :, c].mean() for c in range(3)]
            return max(means) - min(means)

        spread_before = channel_mean_spread(cast.astype(np.float64) / 255.0)
        spread_after = channel_mean_spread(corrected)
        assert spread_after < spread_before

    def test_constant_channel_does_not_crash(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        image[:, :, 0] = 128                       # constant channel
        image[:, :, 1] = np.tile(np.arange(16, dtype=np.uint8) * 16, (16, 1))
        image[:, :, 2] = 200                       # constant channel
        out = correct_colors(image)
        assert np.isfinite(out).all()


class TestTransferGraytoneToColor:

    def test_output_is_uint8_display_ready(self, color_uint8, gray_float):
        # uint8 [0,255] output is the documented, intentional convention
        out = transfer_graytone_to_color(color_uint8, gray_float)
        assert out.dtype == np.uint8
        assert out.shape == color_uint8.shape

    def test_brighter_guide_brightens_output(self, color_uint8):
        from skimage.color import rgb2gray
        gray = rgb2gray(color_uint8)
        brighter = np.clip(gray * 1.5, 0, 1)
        darker = np.clip(gray * 0.5, 0, 1)
        out_bright = transfer_graytone_to_color(color_uint8, brighter)
        out_dark = transfer_graytone_to_color(color_uint8, darker)
        assert out_bright.mean() > out_dark.mean()
