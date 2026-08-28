"""
Robust recursive envelope kernels for get_photometric_mask().

The envelope filter is five sequential directional passes (up-down,
left-right, down-up, right-left, up-down) where each row/column update
depends on the previously updated one, so the scan direction cannot be
vectorized away. Two implementations are provided:

- NumPy (always available): the original implementation, vectorized
  across the axis perpendicular to the scan direction.
- numba (optional, installed via the [fast] extra): compiled loops that
  mirror the NumPy arithmetic operation-for-operation. Because the
  expression order, float64 types and LUT index truncation are identical,
  and numba does not enable fastmath (so no FMA contraction or
  reassociation), the results are bit-exact, which is enforced by the
  golden-reference regression suite.

Dispatch: the numba kernel is used when numba is importable, unless the
environment variable IMAGE_ENHANCEMENT_NO_NUMBA=1 is set.
"""

import os

import numpy as np


def _envelope_numpy(mask, lut_a, lut_b):
    """Original NumPy implementation. Operates in-place on a 3D float64
    array (H x W x C) and returns it."""
    lut_a_max = len(lut_a) - 1
    lut_b_max = len(lut_b) - 1

    # up -> down
    for i in range(1, mask.shape[0]-1):
        d = np.abs(mask[i-1, :, :] - mask[i+1, :, :])  # diff
        d = lut_a[(d * lut_a_max).astype(int)]
        mask[i, :, :] = ((mask[i, :, :] * d) +
                         (mask[i-1, :, :] * (1-d)))

    # left -> right
    for j in range(1, mask.shape[1]-1):
        d = np.abs(mask[:, j-1, :] - mask[:, j+1, :])  # diff
        d = lut_a[(d * lut_a_max).astype(int)]
        mask[:, j, :] = ((mask[:, j, :] * d) +
                         (mask[:, j-1, :] * (1-d)))

    # down -> up
    for i in range(mask.shape[0]-2, 1, -1):
        d = np.abs(mask[i-1, :, :] - mask[i+1, :, :])  # diff
        d = lut_a[(d * lut_a_max).astype(int)]
        mask[i, :, :] = ((mask[i, :, :] * d) +
                         (mask[i+1, :, :] * (1-d)))

    # right -> left
    for j in range(mask.shape[1]-2, 1, -1):
        d = np.abs(mask[:, j-1, :] - mask[:, j+1, :])  # diff
        d = lut_b[(d * lut_b_max).astype(int)]
        mask[:, j, :] = ((mask[:, j, :] * d) +
                         (mask[:, j+1, :] * (1-d)))

    # up -> down
    for i in range(1, mask.shape[0]-1):
        d = np.abs(mask[i-1, :, :] - mask[i+1, :, :])  # diff
        d = lut_b[(d * lut_b_max).astype(int)]
        mask[i, :, :] = ((mask[i, :, :] * d) +
                         (mask[i-1, :, :] * (1-d)))

    return mask


_NUMBA_KERNEL = None
_NUMBA_CHECKED = False


def _get_numba_kernel():
    """Build (once) and return the numba kernel, or None if numba is not
    installed."""
    global _NUMBA_KERNEL, _NUMBA_CHECKED
    if _NUMBA_CHECKED:
        return _NUMBA_KERNEL
    _NUMBA_CHECKED = True

    try:
        import numba
    except ImportError:
        return None

    @numba.njit(parallel=True, cache=True)
    def _envelope_numba(mask, lut_a, lut_b):  # pragma: no cover
        height, width, channels = mask.shape
        lut_a_max = lut_a.shape[0] - 1
        lut_b_max = lut_b.shape[0] - 1

        # Vertical passes are parallel over column blocks: each thread
        # owns a contiguous span of columns and walks rows in scan order,
        # giving cache-friendly row-segment access. Blocking cannot change
        # results: vertical passes have no dependency across columns.
        block = 128
        n_blocks = (width + block - 1) // block

        # up -> down (columns independent -> parallel over column blocks)
        for b in numba.prange(n_blocks):
            j0 = b * block
            j1 = min(j0 + block, width)
            for i in range(1, height - 1):
                for j in range(j0, j1):
                    for c in range(channels):
                        d = abs(mask[i-1, j, c] - mask[i+1, j, c])
                        s = lut_a[int(d * lut_a_max)]
                        mask[i, j, c] = ((mask[i, j, c] * s) +
                                         (mask[i-1, j, c] * (1-s)))

        # left -> right (rows independent -> parallel over i)
        for i in numba.prange(height):
            for c in range(channels):
                for j in range(1, width - 1):
                    d = abs(mask[i, j-1, c] - mask[i, j+1, c])
                    s = lut_a[int(d * lut_a_max)]
                    mask[i, j, c] = ((mask[i, j, c] * s) +
                                     (mask[i, j-1, c] * (1-s)))

        # down -> up
        for b in numba.prange(n_blocks):
            j0 = b * block
            j1 = min(j0 + block, width)
            for i in range(height - 2, 1, -1):
                for j in range(j0, j1):
                    for c in range(channels):
                        d = abs(mask[i-1, j, c] - mask[i+1, j, c])
                        s = lut_a[int(d * lut_a_max)]
                        mask[i, j, c] = ((mask[i, j, c] * s) +
                                         (mask[i+1, j, c] * (1-s)))

        # right -> left
        for i in numba.prange(height):
            for c in range(channels):
                for j in range(width - 2, 1, -1):
                    d = abs(mask[i, j-1, c] - mask[i, j+1, c])
                    s = lut_b[int(d * lut_b_max)]
                    mask[i, j, c] = ((mask[i, j, c] * s) +
                                     (mask[i, j+1, c] * (1-s)))

        # up -> down
        for b in numba.prange(n_blocks):
            j0 = b * block
            j1 = min(j0 + block, width)
            for i in range(1, height - 1):
                for j in range(j0, j1):
                    for c in range(channels):
                        d = abs(mask[i-1, j, c] - mask[i+1, j, c])
                        s = lut_b[int(d * lut_b_max)]
                        mask[i, j, c] = ((mask[i, j, c] * s) +
                                         (mask[i-1, j, c] * (1-s)))

        return mask

    _NUMBA_KERNEL = _envelope_numba
    return _NUMBA_KERNEL


def apply_recursive_envelope(mask, lut_a, lut_b):
    """Apply the 5-pass robust recursive envelope in-place on a 3D
    float64 array and return it. Uses the numba kernel when available
    (identical, bit-exact arithmetic), else the NumPy implementation."""
    if os.environ.get("IMAGE_ENHANCEMENT_NO_NUMBA") != "1":
        kernel = _get_numba_kernel()
        if kernel is not None:
            return kernel(np.ascontiguousarray(mask), lut_a, lut_b)
    return _envelope_numpy(mask, lut_a, lut_b)
