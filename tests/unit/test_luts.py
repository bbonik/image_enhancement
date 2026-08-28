"""Unit tests for the scalar mapping and LUT-generating functions."""

import numpy as np

from image_enhancement import get_membership_luts, get_sigmoid_lut, map_value


class TestMapValue:

    def test_identity(self):
        assert map_value(0.3) == 0.3

    def test_linear_range_mapping(self):
        assert map_value(0.5, range_in=(0, 1), range_out=(10, 20)) == 15
        assert map_value(1.0, range_in=(0, 2), range_out=(0, 10)) == 5

    def test_invert(self):
        assert map_value(0.2, invert=True) == 0.8
        assert map_value(0.0, invert=True) == 1.0

    def test_clamps_to_input_range(self):
        assert map_value(5.0, range_in=(0, 1)) == 1.0
        assert map_value(-3.0, range_in=(0, 1)) == 0.0

    def test_convex_lowers_midrange_preserves_endpoints(self):
        assert map_value(0.0, non_lin_convex=0.1) == 0.0
        # endpoints preserved up to floating point error
        assert np.isclose(map_value(1.0, non_lin_convex=0.1), 1.0)
        assert map_value(0.5, non_lin_convex=0.1) < 0.5

    def test_concave_raises_midrange_preserves_endpoints(self):
        assert map_value(0.0, non_lin_concave=0.1) == 0.0
        assert np.isclose(map_value(1.0, non_lin_concave=0.1), 1.0)
        assert map_value(0.5, non_lin_concave=0.1) > 0.5


class TestSigmoidLut:

    def test_shape_and_range(self):
        lut = get_sigmoid_lut(resolution=256, threshold=0.2, non_linearirty=0.2)
        assert lut.shape == (256,)
        assert lut.min() >= 0.0
        assert lut.max() <= 1.0

    def test_endpoints(self):
        lut = get_sigmoid_lut(resolution=256, threshold=0.2, non_linearirty=0.2)
        assert lut[0] == 0.0
        assert np.isclose(lut[-1], 1.0)

    def test_monotonic_non_decreasing(self):
        for threshold in (0.04, 0.2, 0.5, 0.9):
            lut = get_sigmoid_lut(resolution=256, threshold=threshold,
                                  non_linearirty=0.12)
            assert np.all(np.diff(lut) >= 0), f"not monotonic at thr={threshold}"

    def test_inflection_near_half(self):
        # at the threshold index the sigmoid should cross ~0.5
        lut = get_sigmoid_lut(resolution=256, threshold=0.3, non_linearirty=0.2)
        assert abs(lut[int(0.3 * 255)] - 0.5) < 0.01

    def test_custom_resolution(self):
        lut = get_sigmoid_lut(resolution=64)
        assert lut.shape == (64,)


class TestMembershipLuts:

    def test_shapes_and_ranges(self):
        lower, mid, upper = get_membership_luts(
            resolution=256, lower_threshold=0.35, upper_threshold=0.65)
        for lut in (lower, mid, upper):
            assert lut.shape == (256,)
            assert lut.min() >= 0.0
            assert lut.max() <= 1.0

    def test_trapezoid_structure(self):
        lower, mid, upper = get_membership_luts(
            resolution=256, lower_threshold=0.35, upper_threshold=0.65)
        # lower: ramps from 0 up to 1 and stays 1
        assert lower[0] == 0.0
        assert lower[-1] == 1.0
        # upper: starts at 1 and ramps down to 0
        assert upper[0] == 1.0
        assert np.isclose(upper[-1], 0.0)
        # mid: plateau of 1 between the two thresholds
        plateau = mid[int(0.4 * 255):int(0.6 * 255)]
        assert np.all(plateau == 1.0)

    def test_full_coverage(self):
        # at least one membership function is active for every input level
        lower, mid, upper = get_membership_luts(resolution=256)
        combined = np.maximum(np.maximum(lower, mid), upper)
        assert np.all(combined > 0.0)
