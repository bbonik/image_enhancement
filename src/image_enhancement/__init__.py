"""
image_enhancement: NumPy-based image processing functions for image
enhancement, including spatial tone mapping, local contrast enhancement,
color correction (white balance), color saturation adjustment and
exposure fusion.

Author: Vasileios Vonikakis (bbonik@gmail.com)

The public API is re-exported here, so existing imports keep working:

    from image_enhancement import enhance_image
    import image_enhancement as ie
"""

from image_enhancement.core import (
    map_value,
    get_membership_luts,
    get_sigmoid_lut,
    get_photometric_mask,
    blend_exposures,
    blend_expoures,  # backwards-compatible alias (original misspelling)
    apply_local_contrast_enhancement,
    apply_spatial_tonemapping,
    srgb_to_linear,
    linear_to_srgb,
    transfer_graytone_to_color,
    change_color_saturation,
    correct_colors,
    adjust_brightness,
    enhance_image,
)

__version__ = "1.0.0"

__all__ = [
    "map_value",
    "get_membership_luts",
    "get_sigmoid_lut",
    "get_photometric_mask",
    "blend_exposures",
    "blend_expoures",
    "apply_local_contrast_enhancement",
    "apply_spatial_tonemapping",
    "srgb_to_linear",
    "linear_to_srgb",
    "transfer_graytone_to_color",
    "change_color_saturation",
    "correct_colors",
    "adjust_brightness",
    "enhance_image",
]
