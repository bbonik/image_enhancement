# image_enhancement

[![CI](https://github.com/bbonik/image_enhancement/actions/workflows/ci.yml/badge.svg)](https://github.com/bbonik/image_enhancement/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A NumPy-based (and CUDA-accelerated) library of image processing functions for image enhancement, built on published center-surround / Retinex-inspired research:

- **Spatial tone mapping** — brighten shadows and recover highlights, guided by each pixel's local neighborhood
- **Local contrast enhancement** — amplify or attenuate fine details without halo artifacts
- **Color correction (white balance)** — remove color casts using a Gray-World + White-Point approach
- **Color saturation adjustment** — vibrance control that avoids the "color noise" of HSV-based methods
- **Exposure fusion** — blend bracketed exposures into a single well-exposed image

![Example enhancement](images/Figure_1.png "Example enhancement")

## Installation

```bash
pip install .            # from the repository root
pip install -e .         # editable / development install
```

Optional extras:

```bash
pip install ".[fast]"    # numba: ~2.4x faster pipeline, bit-identical results
pip install ".[plot]"    # matplotlib, for verbose=True visualizations
pip install ".[cuda]"    # pycuda + Pillow, for the CUDA pipeline
pip install ".[dev]"     # pytest + ruff, for development
```

The core library only needs `numpy`, `scikit-image` and `imageio`, and runs fully headless (matplotlib is loaded lazily, only if you ask for visualizations).

## Quickstart

Full enhancement pipeline (tone mapping + local contrast + saturation):

```python
import imageio.v2 as imageio
from image_enhancement import enhance_image

image = imageio.imread("images/lisbon.jpg")  # uint8 color image

enhanced = enhance_image(image, parameters={
    "local_contrast": 1.5,       # 1.5x increase in local details
    "areas_dark": 0.7,           # strong improvement in dark areas
    "areas_bright": 0.5,         # moderate improvement in bright areas
    "saturation_degree": 1.2,    # slight color boost
    "brightness": 0.1,           # slight global brightening
    "preserve_tones": True,      # protect already well-exposed midtones
    "color_correction": False,   # optional white balance stage
})

# enhance_image returns float [0,1]; convert for saving
imageio.imwrite("enhanced.jpg", (enhanced * 255).astype("uint8"))
```

Exposure fusion:

```python
import imageio.v2 as imageio
from image_enhancement import blend_exposures

exposures = [imageio.imread(f) for f in
             ("short.jpg", "medium.jpg", "long.jpg")]  # any order

fused = blend_exposures(exposures)   # float [0,1]
```

Individual stages can also be used on their own — see the [examples](examples/) directory for runnable scripts, including grayscale/medical image processing.

## Gallery

<details>
<summary>More before/after examples (click to expand)</summary>

![Example2](images/Figure_2.png "Example2")
![Example3](images/Figure_3.png "Example3")
![Example4](images/Figure_4.png "Example4")
![Example5](images/Figure_5.png "Example5")
![Example6](images/Figure_6.png "Example6")
![Example7](images/Figure_7.png "Example7")

</details>

Want challenging real-life test material? Try [TM-DIED: The Most Difficult Image Enhancement Dataset](https://sites.google.com/site/vonikakis/datasets/tm-died).

## API overview

All functions live in [`src/image_enhancement/core.py`](src/image_enhancement/core.py) and are re-exported at package level. Every function has a detailed docstring, an optional `verbose` flag for visual inspection, and references to the underlying publication.

| Function | Purpose | Input | Output |
|---|---|---|---|
| `enhance_image()` | Full pipeline: tone mapping + local contrast + saturation | uint8 color | float [0,1] color |
| `blend_exposures()` | Fuse bracketed exposures via illumination estimation | list of images | float [0,1] |
| `get_photometric_mask()` | Edge-aware estimate of local brightness (no halos) | uint8 or float | float [0,1] |
| `apply_spatial_tonemapping()` | Per-pixel tone curves guided by local surround | float [0,1] gray | float [0,1] gray |
| `apply_local_contrast_enhancement()` | Amplify/attenuate local details | float [0,1] gray | float [0,1] gray |
| `adjust_brightness()` | Global Naka-Rushton-style tone curve | float [0,1] gray | float [0,1] gray |
| `transfer_graytone_to_color()` | Apply grayscale tone edits to a color image | uint8 color + float gray | uint8 color |
| `change_color_saturation()` | Vibrance adjustment (with dark-region boost) | color image | float [0,1] color |
| `correct_colors()` | Gray-World + White-Point white balance | uint8 color | float [0,1] color |
| `srgb_to_linear()` / `linear_to_srgb()` | sRGB gamma removal / re-application | uint8 or float / float | float [0,1] / uint8 |
| `map_value()`, `get_sigmoid_lut()`, `get_membership_luts()` | Parameter mapping and LUT utilities | scalars | scalar / arrays |

### Data conventions

Input/output ranges and dtypes intentionally differ between functions and are part of the API (see table above). In short: grayscale operators work in float [0,1]; `linear_to_srgb` and `transfer_graytone_to_color` return display-ready uint8 [0,255]; everything else returns float [0,1].

One known, intentionally preserved behavior: `linear_to_srgb` collapses deep shadows (sRGB levels ≤ ~10/255) to 0 on round-trip. This is locked in by the golden regression references and characterized in the test suite; the CUDA port shares the same formula.

## Project structure

```
├── src/image_enhancement       The installable Python package
│   ├── __init__.py             Public API re-exports
│   └── core.py                 All enhancement functions
├── examples                    Runnable example scripts
├── cuda                        CUDA implementation + PyCUDA driver
├── tests
│   ├── unit                    Fast unit tests (pytest)
│   └── regression              Pixel-exact golden-reference harness
├── images                      Sample test images
└── .github/workflows           CI: lint + unit tests + regression
```

## Performance

Two optional, **bit-exact** fast paths are built in (results are guaranteed identical to the reference implementation, enforced by the golden-reference regression suite):

- **uint8 LUT for `srgb_to_linear`** (always on): uint8 images have only 256 possible values, so the piecewise formula is precomputed once — 10x faster than evaluating `pow` per pixel.
- **numba-compiled recursive envelope** (`pip install ".[fast]"`): the photometric mask's five sequential scan passes, compiled and parallelized — 6x faster than the NumPy loops. Auto-enabled when numba is installed; set `IMAGE_ENHANCEMENT_NO_NUMBA=1` to force the NumPy path.

Measured on a 2448x3264 (8 MP) photo (Apple Silicon, Python 3.12):

| | before | after |
|---|---|---|
| `enhance_image` (full pipeline) | 1.65 s | **0.69 s** (2.4x) |
| recursive envelope (per mask) | 250 ms | **40 ms** (6.3x) |
| `srgb_to_linear` (uint8 input) | 302 ms | **29 ms** (10x) |

The remaining time is dominated by `pow` over continuous float arrays (`linear_to_srgb`, graytone conversion), which cannot be table-driven without changing results.

## Development

```bash
pip install -e ".[dev]"

pytest tests/unit -q            # fast unit tests (<2 s)
pytest -m regression -q -s      # compare against golden references
ruff check src tests examples cuda
```

The regression harness guards against unintended behavior changes: it compares the output of every public function against stored references at pixel level (bit-exact locally; `REGRESSION_CROSS_PLATFORM=1` absorbs isolated last-ulp quantization flips on other platforms in CI). If you intentionally change algorithm behavior, re-capture the references:

```bash
python tests/regression/regression_harness.py capture
```

CI runs on every push and pull request: lint, unit tests on Python 3.9–3.13, and the regression comparison.

## CUDA acceleration (legacy)

> **Status:** the CUDA port is kept as-is from the original project and is not covered by the test suite or CI (it requires an NVIDIA GPU and a manual build). For CPU-side speedups use the `[fast]` extra above. A CuPy-based port would remove the manual compilation step and is the likely future direction.

The `cuda/` directory contains a CUDA port of the main pipeline (photometric mask + enhancement kernels). The Python driver loads a compiled `.cubin` and runs the kernels on NumPy arrays. Replace `[arch]` with your GPU architecture (e.g. `-arch=sm_72`):

```bash
nvcc --cubin -arch=[arch] -use_fast_math -O3 cuda/image_enhancement.cu -o cuda/image_enhancement.cubin
```

With `image_enhancement.cubin` in `cuda/`, run `cuda/image_enhancement_cuda.py` to generate an enhanced version of `images/alhambra1.jpg`.

## Citations

If you use this code in your research please cite the following papers:

1. [Vonikakis, V., Andreadis, I., & Gasteratos, A. (2008). Fast centre-surround contrast modification. IET Image processing 2(1), 19-34.](https://www.researchgate.net/publication/3481092_Fast_centre-surround_contrast_modification)
2. [Vonikakis, V., Winkler, S. (2016). A center-surround framework for spatial image processing. Proc. IS&T Human Vision & Electronic Imaging, (Retinex020), San Francisco, CA, Feb. 14-18.](https://www.researchgate.net/publication/312243692_A_center-surround_framework_for_spatial_image_processing)
3. [Vonikakis, V., Arapakis, I. & Andreadis, I. (2011). Combining Gray-World assumption, White-Point correction and power transformation for automatic white balance. International Workshop on Advanced Image Technology (IWAIT), 1569353295.](https://www.researchgate.net/publication/235350557_Combining_Gray-World_assumption_White-Point_correction_and_power_transformation_for_automatic_white_balance)
4. [Vonikakis, V., & Andreadis, I. (2008). Multi-Scale Image Contrast Enhancement. ICARCV 2008. (pp. 385-391).](https://www.researchgate.net/publication/221145067_Multi-Scale_Image_Contrast_Enhancement)
5. [Vonikakis, V., Bouzos, O. & Andreadis, I. (2011). Multi-Exposure Image Fusion Based on Illumination Estimation, SIPA2011 (pp.135-142).](https://www.researchgate.net/publication/235350562_Multi-Exposure_Image_Fusion_Based_on_Illumination_Estimation#share)

## License

[MIT](LICENSE) © Vasileios Vonikakis
