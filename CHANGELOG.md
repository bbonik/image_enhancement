# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.1.0] - 2026-08-28

Performance release. **All outputs remain bit-for-bit identical** to the
reference implementation, enforced by the golden-reference regression
suite on both the fast and fallback paths.

### Added
- Optional numba-compiled kernel for the photometric mask's recursive
  envelope (`pip install ".[fast]"`): 250 ms → 40 ms on an 8 MP image
  (6.3x). Mirrors the NumPy arithmetic operation-for-operation (no
  fastmath), so results are bit-exact. `IMAGE_ENHANCEMENT_NO_NUMBA=1`
  forces the NumPy path.
- Bit-exact uint8 LUT fast path for `srgb_to_linear` (302 ms → 29 ms,
  10x); float inputs keep the original general path.
- CI regression job variant with numba installed, verifying the compiled
  path against the golden references.

### Changed
- End-to-end `enhance_image` on a 2448x3264 photo: 1.65 s → 0.69 s (2.4x)
- Removed six provably redundant defensive array copies at call sites
  whose callees never mutate their inputs
- The recursive envelope now lives in `src/image_enhancement/_envelope.py`
  (NumPy implementation extracted verbatim)
- README: performance section; CUDA port explicitly documented as legacy

## [1.0.0] - 2026-08-28

Modernization release. **All algorithm outputs are bit-for-bit identical to
the original code** — verified by a pixel-exact regression harness capturing
the behavior of every public function before any change was made.

### Added
- Installable package (`pip install .`) with `src/` layout and
  `pyproject.toml`; optional extras: `[plot]`, `[cuda]`, `[dev]`
- Public API re-exported from the package root (existing imports unchanged)
- Type hints on all public function signatures (Python 3.9+)
- Unit test suite: 62 pytest tests covering every public function
- Pixel-exact golden-reference regression harness
  (`tests/regression/regression_harness.py`) with a pytest wrapper
- GitHub Actions CI: lint, unit tests on Python 3.9–3.13, regression check
- Documented data conventions (intentional per-function dtype/range mix)

### Fixed
- `plt.tight_layout(True)` TypeError on matplotlib ≥ 3.6
- `imageio.imread` deprecation (moved to `imageio.v2`)
- Grayscale exposure blending and grayscale medical-image example crashed
  under modern scikit-image (`rgb2gray` no longer accepts 2D arrays)
- `enhance_image` no longer mutates the caller's parameters dict
- `correct_colors` no longer crashes on constant color channels and
  `verbose` is now optional
- Docstring/typo fixes; corrected `enhance_image` docstring (returns
  float [0,1], not uint8)

### Changed
- `blend_expoures` renamed to `blend_exposures`; the original misspelled
  name is kept as a permanent alias
- matplotlib is now an optional dependency, imported lazily only for
  `verbose=True` visualizations; the library runs fully headless
- Repository layout: `src/image_enhancement/`, `examples/`, `cuda/`
  (CUDA code itself untouched)

### Known behaviors (documented, intentionally preserved)
- `linear_to_srgb` collapses deep shadows (sRGB ≤ ~10/255) to 0 on
  round-trip; characterized in the test suite, shared by the CUDA port,
  and locked in by the golden references
