#include <stdint.h>

extern "C"
__global__ void color_to_gray(uint8_t* color, float* gray, uint32_t width, uint32_t height);

extern "C"
__global__ void photometric_mask_ud(float* ph_mask, float* lut, uint32_t width, uint32_t height);

extern "C"
__global__ void photometric_mask_du(float* ph_mask, float* lut, uint32_t width, uint32_t height);

extern "C"
__global__ void photometric_mask_lr(float* ph_mask, float* lut, uint32_t width, uint32_t height);

extern "C"
__global__ void photometric_mask_rl(float* ph_mask, float* lut, uint32_t width, uint32_t height);

extern "C"
__global__ void enhance_image(
    uint8_t* image, float* ph_mask, float threshold_dark_tones, float local_boost, float saturation_degree,
    float mid_tone_mapped, float tonal_width_mapped, float areas_dark_mapped, float areas_bright_mapped, float detail_amp_global, uint32_t width, uint32_t height
);
