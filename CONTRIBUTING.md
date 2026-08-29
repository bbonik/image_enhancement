# Contributing

Thanks for your interest in improving image_enhancement. This guide covers
the development setup, the test strategy, and the project's one golden rule.

## The golden rule: outputs are protected

This library implements published research; its pixel outputs are the
product. Every change is checked against **golden references**: stored
outputs of every public function on fixed test images, compared at pixel
level. Refactors and optimizations must be *bit-exact*; intentional
behavior changes require deliberately re-capturing the references and
explaining the impact in the commit message (see the 1.2.0 entry in
CHANGELOG.md for an example of how to do that: quantify which outputs
change, by how much, and where).

## Setup

```bash
git clone https://github.com/bbonik/image_enhancement.git
cd image_enhancement
pip install -e ".[dev]"        # pytest + ruff
pip install -e ".[dev,fast]"   # also test the numba-compiled path
```

## Tests

```bash
pytest tests/unit -q          # 60+ fast unit tests (<2 s), no image files needed
pytest -m regression -q -s    # golden-reference comparison (~2 s)
pytest -q                     # everything
```

- `tests/unit/` — behavior tests on small synthetic images: LUTs, color
  transforms, tone mapping, the full pipeline, headless operation.
- `tests/regression/regression_harness.py` — captures and compares the
  golden references (`references.npz`). Run directly for full control:

```bash
python tests/regression/regression_harness.py compare    # bit-exact check
python tests/regression/regression_harness.py capture    # re-capture (deliberate behavior changes only!)
```

Notes:
- Locally, `compare` requires exact equality. Across platforms, floating
  point last-ulp differences can flip isolated pixels over quantization
  boundaries; CI sets `REGRESSION_CROSS_PLATFORM=1`, which allows a tiny
  per-pixel tolerance, a 0.5% outlier allowance, and a hard cap. Real
  algorithm changes exceed these limits by orders of magnitude.
- If numba is installed, the compiled photometric-mask kernel is used
  automatically. Set `IMAGE_ENHANCEMENT_NO_NUMBA=1` to test the plain
  NumPy path. Both paths must pass the regression comparison.

## Lint

```bash
ruff check src tests examples cuda
```

The policy is correctness-only (pyflakes + syntax errors). The original
code base has an established style — extra blank lines, long docstring
rules, spacing — that we deliberately do not churn; don't "fix" style in
otherwise unrelated changes.

## CI

Every push and pull request runs:
- **lint** — ruff correctness rules
- **unit-tests** — Python 3.9 through 3.13
- **regression** — golden-reference comparison (NumPy path)
- **regression-numba** — the same comparison with the compiled kernel

## Code organization

```
src/image_enhancement/
├── __init__.py      Public API re-exports and __version__
├── core.py          All enhancement functions (the original module)
└── _envelope.py     Photometric-mask filter kernels (NumPy + optional numba)
```

`core.py` keeps the original single-module layout on purpose: functions are
self-contained, heavily documented, and reference the papers they implement.
`_envelope.py` holds the only performance-critical inner loop, with the NumPy
reference implementation and the numba mirror side by side — if you touch one,
touch both, and let the regression suite prove they still agree.

The `cuda/` directory is a legacy GPU port, not covered by tests or CI
(needs an NVIDIA GPU and a manual `nvcc` build). Keep it in sync with
`core.py` on a best-effort basis and flag untested changes in commits.
