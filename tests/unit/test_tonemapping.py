"""Unit tests for the photometric mask and tone mapping operators."""

import numpy as np

from image_enhancement import (
    adjust_brightness,
    apply_local_contrast_enhancement,
    apply_spatial_tonemapping,
    get_photometric_mask,
)


class TestGetPhotometricMask:

    def test_color_input_grayscale_output(self, color_uint8):
        mask = get_photometric_mask(color_uint8)
        assert mask.ndim == 2
        assert mask.shape == color_uint8.shape[:2]
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_color_input_color_output(self, color_uint8):
        mask = get_photometric_mask(color_uint8, grayscale_out=False)
        assert mask.shape == color_uint8.shape

    def test_grayscale_input(self, gray_float):
        mask = get_photometric_mask(gray_float)
        assert mask.shape == gray_float.shape
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_smooths_the_image(self, gray_float):
        # the mask is an edge-aware blur: neighbor differences must shrink
        mask = get_photometric_mask(gray_float)
        input_roughness = np.abs(np.diff(gray_float, axis=0)).mean()
        mask_roughness = np.abs(np.diff(mask, axis=0)).mean()
        assert mask_roughness < input_roughness

    def test_does_not_mutate_input(self, gray_float):
        original = gray_float.copy()
        get_photometric_mask(gray_float)
        assert np.array_equal(gray_float, original)


class TestApplyLocalContrastEnhancement:

    def test_degree_zero_flattens_details_to_mask(self, gray_float):
        mask = get_photometric_mask(gray_float)
        out = apply_local_contrast_enhancement(gray_float, mask, degree=0.0)
        assert np.allclose(out, np.clip(mask, 0, 1), atol=1e-12)

    def test_higher_degree_increases_detail_energy(self, gray_float):
        mask = get_photometric_mask(gray_float)
        low = apply_local_contrast_enhancement(gray_float, mask, degree=0.5)
        high = apply_local_contrast_enhancement(gray_float, mask, degree=2.0)
        detail_low = np.abs(low - mask).mean()
        detail_high = np.abs(high - mask).mean()
        assert detail_high > detail_low

    def test_output_range_and_shape(self, gray_float):
        mask = get_photometric_mask(gray_float)
        out = apply_local_contrast_enhancement(gray_float, mask, degree=5.0)
        assert out.shape == gray_float.shape
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestApplySpatialTonemapping:

    def test_output_range_and_shape(self, gray_float):
        mask = get_photometric_mask(gray_float)
        out = apply_spatial_tonemapping(gray_float, mask, verbose=False)
        assert out.shape == gray_float.shape
        assert out.min() >= -1e-9
        assert out.max() <= 1.0 + 1e-9

    def test_brightens_dark_regions(self, gray_float):
        dark = gray_float * 0.25   # uniformly dark image
        mask = get_photometric_mask(dark)
        out = apply_spatial_tonemapping(
            dark, mask, areas_dark=1.0, areas_bright=0.0,
            preserve_tones=False, verbose=False)
        assert out.mean() > dark.mean()

    def test_darkens_bright_regions(self, gray_float):
        bright = 0.75 + gray_float * 0.25  # uniformly bright image
        mask = get_photometric_mask(bright)
        out = apply_spatial_tonemapping(
            bright, mask, areas_dark=0.0, areas_bright=1.0,
            preserve_tones=False, verbose=False)
        assert out.mean() < bright.mean()

    def test_zero_enhancement_is_gentle(self, gray_float):
        mask = get_photometric_mask(gray_float)
        out = apply_spatial_tonemapping(
            gray_float, mask, areas_dark=0.0, areas_bright=0.0,
            preserve_tones=True, verbose=False)
        # with no enhancement requested and tone preservation on, the
        # image should stay close to the input
        assert np.abs(out - gray_float).mean() < 0.1


class TestAdjustBrightness:

    def test_degree_zero_is_identity(self, gray_float):
        out = adjust_brightness(gray_float, degree=0)
        assert np.array_equal(out, gray_float)

    def test_positive_degree_brightens(self, gray_float):
        out = adjust_brightness(gray_float, degree=0.5)
        assert out.mean() > gray_float.mean()

    def test_negative_degree_darkens(self, gray_float):
        out = adjust_brightness(gray_float, degree=-0.5)
        assert out.mean() < gray_float.mean()

    def test_stronger_degree_is_stronger(self, gray_float):
        mild = adjust_brightness(gray_float, degree=0.2)
        strong = adjust_brightness(gray_float, degree=0.9)
        assert strong.mean() > mild.mean()

    def test_output_range(self, gray_float):
        for degree in (-1, -0.5, 0.5, 1):
            out = adjust_brightness(gray_float, degree=degree)
            assert out.min() >= 0.0
            assert out.max() <= 1.0 + 1e-9

    def test_does_not_mutate_input(self, gray_float):
        original = gray_float.copy()
        adjust_brightness(gray_float, degree=0.5)
        assert np.array_equal(gray_float, original)
