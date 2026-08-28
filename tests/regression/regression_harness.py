#!/usr/bin/env python3
"""
Regression harness for the image_enhancement library.

Captures golden reference outputs from the library on fixed test inputs, and
compares later runs against them at pixel level (exact, zero-tolerance by
default).

Usage:
    python regression_harness.py capture   # save reference outputs (.npz)
    python regression_harness.py compare   # re-run and diff against references

The harness must be run with the SAME interpreter/venv for capture and
compare, since floating point results can legitimately differ across
numpy/skimage versions. Any nonzero diff after a code change means the change
altered algorithm behavior.
"""

import os
import sys
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REF_DIR = os.path.join(HERE, "references")
IMAGES_DIR = os.path.join(REPO_ROOT, "images")

# Allow running without installation: fall back to the src/ layout
# (an installed/editable package takes priority if present).
sys.path.append(os.path.join(REPO_ROOT, "src"))

import imageio.v2 as imageio  # noqa: E402


def load_test_images():
    """Load deterministic, downsampled test inputs (slicing = deterministic)."""
    lisbon = imageio.imread(os.path.join(IMAGES_DIR, "lisbon.jpg"))[::4, ::4]
    strawberries = imageio.imread(
        os.path.join(IMAGES_DIR, "strawberries.jpg"))[::2, ::2]
    xray = imageio.imread(os.path.join(IMAGES_DIR, "xray.jpg"))[::2, ::2]
    exposures = [
        imageio.imread(
            os.path.join(IMAGES_DIR, f"exposures_B_{n}.jpg"))[::2, ::2]
        for n in ("short", "medium", "long")
    ]
    return lisbon, strawberries, xray, exposures


def compute_all_outputs():
    """Run every public function on fixed inputs. Returns {name: ndarray}."""
    import image_enhancement as ie
    from skimage.color import rgb2gray
    from skimage import img_as_float

    lisbon, strawberries, xray, exposures = load_test_images()
    out = {}

    # --- scalar / LUT utilities -------------------------------------------
    out["map_value_basic"] = np.array([
        ie.map_value(0.3),
        ie.map_value(0.3, range_in=(0, 2), range_out=(10, 20)),
        ie.map_value(0.3, invert=True),
        ie.map_value(0.3, non_lin_convex=0.1),
        ie.map_value(0.3, non_lin_concave=0.1),
        ie.map_value(5.0, range_in=(0, 1)),   # clamp above
        ie.map_value(-1.0, range_in=(0, 1)),  # clamp below
    ])

    out["sigmoid_lut"] = ie.get_sigmoid_lut(
        resolution=256, threshold=0.2, non_linearirty=0.2)
    lo, mid, up = ie.get_membership_luts(
        resolution=256, lower_threshold=0.35, upper_threshold=0.65)
    out["membership_lut_lower"] = lo
    out["membership_lut_mid"] = mid
    out["membership_lut_upper"] = up

    # --- photometric mask --------------------------------------------------
    ph_mask_color = ie.get_photometric_mask(lisbon.copy(), smoothing=0.2)
    out["ph_mask_from_color"] = ph_mask_color

    xray_gray = img_as_float(rgb2gray(xray) if xray.ndim == 3 else xray)
    ph_mask_gray = ie.get_photometric_mask(xray_gray.copy(), smoothing=0.2)
    out["ph_mask_from_gray"] = ph_mask_gray

    # --- grayscale-domain operators ---------------------------------------
    lisbon_gray = rgb2gray(lisbon)
    out["local_contrast"] = ie.apply_local_contrast_enhancement(
        lisbon_gray.copy(), ph_mask_color, degree=1.5)
    out["local_contrast_attenuate"] = ie.apply_local_contrast_enhancement(
        lisbon_gray.copy(), ph_mask_color, degree=0.5)

    out["spatial_tonemap"] = ie.apply_spatial_tonemapping(
        lisbon_gray.copy(), ph_mask_color,
        mid_tone=0.5, tonal_width=0.5, areas_dark=0.7, areas_bright=0.5,
        preserve_tones=True, verbose=False)
    out["spatial_tonemap_nopreserve"] = ie.apply_spatial_tonemapping(
        lisbon_gray.copy(), ph_mask_color,
        mid_tone=0.5, tonal_width=0.5, areas_dark=0.7, areas_bright=0.5,
        preserve_tones=False, verbose=False)

    out["brightness_up"] = ie.adjust_brightness(lisbon_gray.copy(), degree=0.5)
    out["brightness_down"] = ie.adjust_brightness(lisbon_gray.copy(), degree=-0.5)
    out["brightness_zero"] = ie.adjust_brightness(lisbon_gray.copy(), degree=0)

    # --- color space -------------------------------------------------------
    out["srgb_to_linear"] = ie.srgb_to_linear(lisbon.copy())
    out["linear_to_srgb"] = ie.linear_to_srgb(out["srgb_to_linear"].copy())

    # --- color operators ----------------------------------------------------
    out["graytone_to_color"] = ie.transfer_graytone_to_color(
        lisbon.copy(), out["spatial_tonemap"].copy())

    out["saturation_with_mask"] = ie.change_color_saturation(
        lisbon.copy(), image_ph_mask=ph_mask_color, sat_degree=1.5)
    out["saturation_no_mask"] = ie.change_color_saturation(
        lisbon.copy(), image_ph_mask=None, sat_degree=1.5)

    out["correct_colors"] = ie.correct_colors(strawberries.copy(), False)

    # --- full pipeline -------------------------------------------------------
    params = dict(local_contrast=1.5, mid_tones=0.5, tonal_width=0.5,
                  areas_dark=0.7, areas_bright=0.5, saturation_degree=1.2,
                  brightness=0.1, preserve_tones=True, color_correction=False)
    out["enhance_image"] = ie.enhance_image(lisbon.copy(), dict(params))

    params_cc = dict(params, color_correction=True)
    out["enhance_image_color_correction"] = ie.enhance_image(
        lisbon.copy(), dict(params_cc))

    out["enhance_image_defaults"] = ie.enhance_image(lisbon.copy(), {})

    # --- exposure fusion ------------------------------------------------------
    # blend_expoures is the original (misspelled) name, kept as an alias
    blend = getattr(ie, "blend_exposures", None) or ie.blend_expoures
    if hasattr(ie, "blend_exposures"):
        assert ie.blend_expoures is ie.blend_exposures, \
            "blend_expoures alias must point to blend_exposures"

    out["blend_exposures_color"] = blend(
        [e.copy() for e in exposures],
        threshold_dark=0.35, threshold_bright=0.65)
    # NOTE: grayscale exposure blending is broken in the ORIGINAL code under
    # modern skimage (unconditional rgb2gray() rejects 2D input; older skimage
    # passed 2D through unchanged). Captured only once the fix restores it.
    try:
        out["blend_exposures_gray"] = blend(
            [rgb2gray(e) for e in exposures],
            threshold_dark=0.35, threshold_bright=0.65)
    except ValueError as exc:
        print(f"  SKIP  blend_exposures_gray (pre-existing break: {exc})")

    return out


def capture():
    os.makedirs(REF_DIR, exist_ok=True)
    outputs = compute_all_outputs()
    path = os.path.join(REF_DIR, "references.npz")
    np.savez_compressed(path, **outputs)
    print(f"Captured {len(outputs)} reference outputs -> {path}")
    for name, arr in outputs.items():
        print(f"  {name:35s} shape={arr.shape} dtype={arr.dtype}")


def compare(tolerance=0.0):
    path = os.path.join(REF_DIR, "references.npz")
    if not os.path.exists(path):
        print("ERROR: no references captured yet. Run 'capture' first.")
        return 1

    refs = np.load(path)
    outputs = compute_all_outputs()

    failures = []
    for name in refs.files:
        if name not in outputs:
            failures.append((name, "MISSING from current outputs"))
            continue
        ref, cur = refs[name], outputs[name]
        if ref.shape != cur.shape:
            failures.append((name, f"shape {ref.shape} -> {cur.shape}"))
            continue
        if ref.dtype != cur.dtype:
            failures.append((name, f"dtype {ref.dtype} -> {cur.dtype}"))
            continue
        diff = np.abs(ref.astype(np.float64) - cur.astype(np.float64))
        max_diff = diff.max() if diff.size else 0.0
        if max_diff > tolerance:
            failures.append(
                (name,
                 f"max abs diff {max_diff:.3e} "
                 f"({(diff > tolerance).sum()} / {diff.size} px differ)"))
        else:
            print(f"  PASS  {name:35s} (max diff {max_diff:.1e})")

    extra = set(outputs) - set(refs.files)
    for name in sorted(extra):
        print(f"  NOTE  new output '{name}' not in references (ignored)")

    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for name, msg in failures:
            print(f"  FAIL  {name:35s} {msg}")
        return 1
    print(f"\nAll {len(refs.files)} outputs match pixel-exactly.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["capture", "compare"])
    parser.add_argument("--tolerance", type=float, default=0.0,
                        help="max allowed absolute diff (default 0 = exact)")
    args = parser.parse_args()

    if args.mode == "capture":
        capture()
        sys.exit(0)
    sys.exit(compare(args.tolerance))
