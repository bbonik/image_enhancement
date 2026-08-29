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
pip install "git+https://github.com/bbonik/image_enhancement.git"
```

That's it — the library only needs `numpy`, `scikit-image` and `imageio`, and runs fully headless. For a ~2.4x faster pipeline (identical results), add the `fast` extra:

```bash
pip install "image-enhancement[fast] @ git+https://github.com/bbonik/image_enhancement.git"
```

<details>
<summary>Other install options (clone, extras, development)</summary>

```bash
git clone https://github.com/bbonik/image_enhancement.git
cd image_enhancement
pip install -e .         # editable install
pip install -e ".[fast]" # + numba: ~2.4x faster, bit-identical results
pip install -e ".[plot]" # + matplotlib, for verbose=True visualizations
pip install -e ".[cuda]" # + pycuda + Pillow, for the legacy CUDA pipeline
pip install -e ".[dev]"  # + pytest + ruff, for development
```

</details>

## Quickstart

Enhance a photo with sensible defaults — three lines:

```python
import imageio.v2 as imageio
from image_enhancement import enhance_image

image = imageio.imread("my_photo.jpg")               # uint8 color image
enhanced = enhance_image(image, {})                  # defaults just work
imageio.imwrite("enhanced.jpg", (enhanced * 255).astype("uint8"))
```

Every parameter is optional — override only what you want to change:

```python
enhanced = enhance_image(image, {
    "local_contrast": 1.5,       # 1.5x increase in local details
    "areas_dark": 0.7,           # strong improvement in dark areas
    "areas_bright": 0.5,         # moderate improvement in bright areas
    "saturation_degree": 1.2,    # slight color boost
    "brightness": 0.1,           # slight global brightening
    "preserve_tones": True,      # protect already well-exposed midtones
    "color_correction": False,   # optional white balance stage
})
```

Blend bracketed exposures into one well-exposed image:

```python
from image_enhancement import blend_exposures

exposures = [imageio.imread(f) for f in
             ("short.jpg", "medium.jpg", "long.jpg")]  # any order
fused = blend_exposures(exposures)                     # float [0,1]
```

Tips:
- Pass `verbose=True` to any function to visualize the intermediate steps (needs matplotlib, i.e. the `[plot]` extra).
- Individual stages can be used on their own — see the [examples](examples/) directory for runnable scripts, including grayscale/medical image processing.

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

The sRGB transforms follow the [EasyRGB reference math](https://www.easyrgb.com/en/math.php) exactly; the round trip `linear_to_srgb(srgb_to_linear(x))` is within one uint8 level for all 256 input levels (verified by the test suite).

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

The library is fast out of the box (a 8 MP photo takes well under a second on a laptop). Installing the `[fast]` extra makes the full pipeline about **2.4x faster** — with results guaranteed identical, nothing else to configure.

<details>
<summary>How the speedups work (technical details)</summary>

Two fast paths, both producing results verified identical to the reference implementation by the test suite:

- **uint8 lookup table for `srgb_to_linear`** (always on): a uint8 image has only 256 possible values, so the piecewise formula is precomputed once — 10x faster than evaluating `pow` per pixel.
- **Compiled photometric mask** (with the `[fast]` extra): the edge-aware blur's five scan passes, compiled and parallelized with numba — 6x faster than the NumPy loops. Auto-enabled when numba is installed; set `IMAGE_ENHANCEMENT_NO_NUMBA=1` to force the plain NumPy path (e.g. when debugging).

Measured on a 2448x3264 (8 MP) photo (Apple Silicon, Python 3.12):

| | before | after |
|---|---|---|
| `enhance_image` (full pipeline) | 1.65 s | **0.69 s** (2.4x) |
| photometric mask (per call) | 250 ms | **40 ms** (6.3x) |
| `srgb_to_linear` (uint8 input) | 302 ms | **29 ms** (10x) |

</details>

## Development

```bash
git clone https://github.com/bbonik/image_enhancement.git
cd image_enhancement
pip install -e ".[dev]"
pytest tests/unit -q     # fast unit tests (<2 s)
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide: test layout, the golden-reference regression harness, lint policy, and CI. The short version: any change to algorithm output is caught by a pixel-level comparison against stored references, so refactor with confidence.

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
