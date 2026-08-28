"""Unit tests for the full enhancement pipeline and exposure fusion."""

import subprocess
import sys
import textwrap

import numpy as np
from skimage.color import rgb2gray

import image_enhancement as ie
from image_enhancement import blend_exposures, enhance_image


class TestEnhanceImage:

    def test_defaults_output_contract(self, color_uint8):
        # float [0,1] output is the documented, intentional convention
        out = enhance_image(color_uint8, {})
        assert out.shape == color_uint8.shape
        assert out.dtype == np.float64
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_does_not_mutate_parameters(self, color_uint8):
        params = {"local_contrast": 1.5}
        enhance_image(color_uint8, params)
        assert params == {"local_contrast": 1.5}

    def test_does_not_mutate_image(self, color_uint8):
        original = color_uint8.copy()
        enhance_image(color_uint8, {})
        assert np.array_equal(color_uint8, original)

    def test_out_of_range_parameters_are_clamped(self, color_uint8):
        params = {
            "local_contrast": -5,   # clamped to 0
            "mid_tones": 7,         # clamped to 1
            "areas_dark": -1,       # clamped to 0
            "brightness": 100,      # clamped to 1
        }
        out = enhance_image(color_uint8, params)
        assert np.isfinite(out).all()
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_dark_enhancement_brightens_shadows(self, color_uint8):
        dark = (color_uint8 * 0.3).astype(np.uint8)
        no_boost = enhance_image(dark, {
            "areas_dark": 0.0, "areas_bright": 0.0, "brightness": 0.0,
            "local_contrast": 1.0, "saturation_degree": 1.0})
        boost = enhance_image(dark, {
            "areas_dark": 1.0, "areas_bright": 0.0, "brightness": 0.0,
            "local_contrast": 1.0, "saturation_degree": 1.0})
        assert boost.mean() > no_boost.mean()

    def test_color_correction_stage_runs(self, color_uint8):
        out = enhance_image(color_uint8, {"color_correction": True})
        assert np.isfinite(out).all()


class TestBlendExposures:

    def test_color_output_contract(self, exposure_stack):
        out = blend_exposures(exposure_stack)
        assert out.shape == exposure_stack[0].shape
        assert out.dtype == np.float64
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_grayscale_exposures(self, exposure_stack):
        gray_stack = [rgb2gray(e) for e in exposure_stack]
        out = blend_exposures(gray_stack)
        assert out.ndim == 2
        assert out.shape == gray_stack[0].shape

    def test_blend_between_extremes(self, exposure_stack):
        out = blend_exposures(exposure_stack)
        darkest = exposure_stack[0].astype(np.float64).mean() / 255.0
        brightest = exposure_stack[2].astype(np.float64).mean() / 255.0
        assert darkest < out.mean() < brightest

    def test_exposure_order_does_not_matter(self, exposure_stack):
        # exposures are sorted by mean luminance internally
        out_sorted = blend_exposures(list(exposure_stack))
        out_shuffled = blend_exposures(
            [exposure_stack[2], exposure_stack[0], exposure_stack[1]])
        assert np.allclose(out_sorted, out_shuffled)

    def test_misspelled_alias_is_same_function(self):
        assert ie.blend_expoures is ie.blend_exposures


class TestHeadlessUse:

    def test_import_and_run_without_matplotlib(self):
        # simulate an environment where matplotlib is not installed
        code = textwrap.dedent("""
            import sys
            sys.modules['matplotlib'] = None
            sys.modules['matplotlib.pyplot'] = None
            import numpy as np
            import image_enhancement as ie
            img = (np.random.RandomState(0).rand(16, 16, 3) * 255).astype(np.uint8)
            out = ie.enhance_image(img, {})
            assert out.shape == img.shape
            try:
                ie.adjust_brightness(np.zeros((4, 4)), degree=0.5, verbose=True)
            except ImportError:
                pass  # expected: verbose visualization needs matplotlib
            else:
                raise AssertionError('expected ImportError for verbose=True')
        """)
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
