import os
import numpy
import pycuda.driver as cuda
import pycuda.autoinit

from math import ceil
from PIL import Image

path = os.path.dirname(__file__)
mod = cuda.module_from_file(os.path.join(path, "image_enhancement.cubin"))

color_to_gray_kernel = mod.get_function("color_to_gray")
photometric_mask_ud_kernel = mod.get_function("photometric_mask_ud")
photometric_mask_du_kernel = mod.get_function("photometric_mask_du")
photometric_mask_lr_kernel = mod.get_function("photometric_mask_lr")
photometric_mask_rl_kernel = mod.get_function("photometric_mask_rl")
enhance_image_kernel = mod.get_function("enhance_image")

LUT_RES = 256
EPSILON = 1 / 256


class ToneMapping:
    def __init__(
        self,
        local_contrast = 1.2,
        mid_tones = 0.5,
        tonal_width = 0.5,
        areas_dark = 0.2,
        areas_bright = 0.2,
        brightness = 0.1,
        saturation_degree = 1.2,
        color_correction = False
    ):
        self.detail_amplification_global = numpy.float32(max(0, local_contrast))
        self.mid_tones = numpy.float32(min(1, max(0, mid_tones)))
        self.tonal_width = numpy.float32(min(1, max(0, tonal_width)))
        self.areas_dark = numpy.float32(min(1, max(0, areas_dark)))
        self.areas_bright = numpy.float32(min(1, max(0, areas_bright)))
        self.brightness = numpy.float32(min(1, max(-1, brightness)))
        self.saturation_degree = numpy.float32(max(0, saturation_degree))
        self.color_correction = numpy.float32(color_correction)

        # apply_spatial_tonemapping (adjust range and non-linear response of parameters)
        self.mid_tone_mapped = numpy.float32(self.map_value(self.mid_tones, range_out=(0,1), invert=False, non_lin_convex=None))
        self.tonal_width_mapped = numpy.float32(self.map_value(self.tonal_width, range_out=(EPSILON,1), invert=False, non_lin_convex=0.1))
        self.areas_dark_mapped = numpy.float32(self.map_value(self.areas_dark, range_out=(0,5), invert=True, non_lin_convex=0.05))
        self.areas_bright_mapped = numpy.float32(self.map_value(self.areas_bright, range_out=(0,5), invert=True, non_lin_convex=0.05))

        # photometric_mask
        self.lut_a = numpy.zeros(LUT_RES, dtype=numpy.float32)
        self.lut_b = numpy.zeros(LUT_RES, dtype=numpy.float32)

        self.get_sigmoid_lut(self.lut_a, threshold=51/255, non_linearirty=30/255)
        self.get_sigmoid_lut(self.lut_b, threshold=10/255, non_linearirty=30/255)

        self.d_lut_a = cuda.mem_alloc(self.lut_a.nbytes)
        self.d_lut_b = cuda.mem_alloc(self.lut_b.nbytes)
        cuda.memcpy_htod(self.d_lut_a, self.lut_a)
        cuda.memcpy_htod(self.d_lut_b, self.lut_b)

        # apply_local_contrast_enhancement / change_color_saturation
        self.local_boost = numpy.float32(0.2)
        self.threshold_dark_tones = numpy.float32(100 / 255)

    def map_value(self, value, range_in=(0,1), range_out=(0,1), invert=False, non_lin_convex=None):
        # truncate value to within input range limits
        value = max(min(value, range_in[1]), range_in[0])

        # map values linearly to [0,1]
        value = (value - range_in[0]) / (range_in[1] - range_in[0])

        # invert values
        if invert is True:
            value = 1 - value

        # apply convex non-linearity
        if non_lin_convex is not None:
            value = (value * non_lin_convex) / (1 + non_lin_convex - value)

        # # apply concave non-linearity
        # if non_lin_concave is not None:
        #     value = ((1 + non_lin_concave) * value) / (non_lin_concave + value)

        # mapping value to the output range in a linear way
        value = value * (range_out[1] - range_out[0]) + range_out[0]

        return value

    def get_sigmoid_lut(self, lut, threshold=0.2, non_linearirty=0.2):
        max_value = LUT_RES - 1  # the maximum attainable value
        thr = threshold * max_value  # threshold in the range [0,resolution-1]
        alpha = non_linearirty * max_value  # controls non-linearity degree
        beta = max_value - thr
        if beta == 0:
            beta = 0.001

        for i in range(LUT_RES):
            i_comp = i - thr  # complement of i

            # upper part of the piece-wise sigmoid function
            if i >= thr:
                lut[i] = (((((alpha + beta) * i_comp) / (alpha + i_comp)) * (1 / (2 * beta))) + 0.5)

            # lower part of the piece-wise sigmoid function
            else:
                lut[i] = (alpha * i) / (alpha - i_comp) * (1 / (2 * thr))

    def photometric_mask(self, d_image, d_ph_mask, width, height):
        tile = 8

        color_to_gray_kernel(
            d_image,
            d_ph_mask,
            numpy.uint32(width),
            numpy.uint32(height),
            grid=(ceil(width / tile), ceil(height / tile), 1),
            block=(tile, tile, 1)
        )

        photometric_mask_ud_kernel(
            d_ph_mask,
            self.d_lut_a,
            numpy.uint32(width),
            numpy.uint32(height),
            grid=(ceil(width / tile), 1, 1),
            block=(tile, 1, 1)
        )

        photometric_mask_lr_kernel(
            d_ph_mask,
            self.d_lut_a,
            numpy.uint32(width),
            numpy.uint32(height),
            grid=(ceil(height / tile), 1, 1),
            block=(tile, 1, 1)
        )

        photometric_mask_du_kernel(
            d_ph_mask,
            self.d_lut_a,
            numpy.uint32(width),
            numpy.uint32(height),
            grid=(ceil(width / tile), 1, 1),
            block=(tile, 1, 1)
        )

        photometric_mask_rl_kernel(
            d_ph_mask,
            self.d_lut_b,
            numpy.uint32(width),
            numpy.uint32(height),
            grid=(ceil(height / tile), 1, 1),
            block=(tile, 1, 1)
        )

        photometric_mask_ud_kernel(
            d_ph_mask,
            self.d_lut_b,
            numpy.uint32(width),
            numpy.uint32(height),
            grid=(ceil(width / tile), 1, 1),
            block=(tile, 1, 1)
        )

    def enhance_image(self, d_image, d_ph_mask, width, height):
        tile = 16

        enhance_image_kernel(
            d_image,
            d_ph_mask,
            self.threshold_dark_tones,
            self.local_boost,
            self.saturation_degree,
            self.mid_tone_mapped,
            self.tonal_width_mapped,
            self.areas_dark_mapped,
            self.areas_bright_mapped,
            self.detail_amplification_global,
            numpy.uint32(width),
            numpy.uint32(height),
            grid=(ceil(width / tile), ceil(height / tile), 1),
            block=(tile, tile, 1)
        )


if __name__ == "__main__":
    tone_mapping = ToneMapping(
        local_contrast = 1.0,
        mid_tones = 0.5,
        tonal_width = 0.5,
        areas_dark = 0.2,
        areas_bright = 0.2,
        brightness = 0.1,
        saturation_degree = 1.2,
        color_correction = False
    )

    image = Image.open(os.path.join(path, "..", "images", "alhambra1.jpg"))
    image = numpy.asarray(image.convert("RGB"))

    height, width, _ = image.shape

    d_image = cuda.mem_alloc(image.nbytes)
    d_ph_mask = cuda.mem_alloc(width * height * numpy.float32().nbytes)

    cuda.memcpy_htod(d_image, image)

    tone_mapping.photometric_mask(d_image, d_ph_mask, width, height)
    tone_mapping.enhance_image(d_image, d_ph_mask, width, height)

    enhanced = numpy.empty_like(image)
    cuda.memcpy_dtoh(enhanced, d_image)

    Image.fromarray(numpy.uint8(enhanced)).save(os.path.join(path, "..", "output.png"))
