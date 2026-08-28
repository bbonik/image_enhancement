"""
Pytest wrapper around the golden-reference regression harness.

Compares the current code's outputs against the captured references at
pixel level. The default tolerance is 0 (exact match), which is the right
setting when running in the same environment that captured the references.

Across platforms or dependency versions, floating point results can differ
legitimately: libm/BLAS last-ulp differences flip isolated pixels across
uint8 truncation boundaries (a whole 1/255 step) and LUT index boundaries.
CI therefore sets REGRESSION_CROSS_PLATFORM=1, which enables a per-pixel
tolerance plus a small allowance of outlier pixels and a hard cap:

    REGRESSION_CROSS_PLATFORM=1 pytest -m regression

A real algorithm change moves most pixels by amounts far beyond both the
outlier allowance and the hard cap, so the check stays meaningful.
"""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "regression"))

from regression_harness import REF_DIR, compare  # noqa: E402


@pytest.mark.regression
def test_outputs_match_golden_references():
    if not os.path.exists(os.path.join(REF_DIR, "references.npz")):
        pytest.skip("no golden references captured (run regression_harness.py capture)")

    if os.environ.get("REGRESSION_CROSS_PLATFORM") == "1":
        # cross-platform mode: tolerate isolated last-ulp boundary flips
        # (1 uint8 level = ~4e-3 normalized; LUT flips reach ~1e-2), on at
        # most 0.5% of pixels, with no pixel differing by more than 0.05
        exit_code = compare(
            tolerance=5e-3, max_outlier_fraction=0.005, hard_cap=0.05)
    else:
        # same-environment mode: bit-exact
        tolerance = float(os.environ.get("REGRESSION_TOLERANCE", "0"))
        exit_code = compare(tolerance)

    assert exit_code == 0, (
        "outputs diverged from golden references; "
        "see stdout for the per-output diff report"
    )
