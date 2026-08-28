"""
Pytest wrapper around the golden-reference regression harness.

Compares the current code's outputs against the captured references at
pixel level. The default tolerance is 0 (exact match), which is the right
setting when running in the same environment that captured the references.

Across platforms or dependency versions, floating point results can differ
legitimately (e.g. BLAS-backed rgb2gray), so CI may set a small tolerance:

    REGRESSION_TOLERANCE=1e-3 pytest -m regression

A real algorithm change produces differences orders of magnitude larger
than that, across a large fraction of pixels, so the check stays meaningful.
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

    tolerance = float(os.environ.get("REGRESSION_TOLERANCE", "0"))
    exit_code = compare(tolerance)
    assert exit_code == 0, (
        f"outputs diverged from golden references (tolerance={tolerance}); "
        "see stdout for the per-output diff report"
    )
