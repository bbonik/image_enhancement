#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 11:35:49 2020

Example: enhancement of local details in medical image

@author: Vasileios Vonikakis (bbonik@gmail.com)
"""



import imageio
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.color import rgb2gray
from image_enhancement import get_photometric_mask
from image_enhancement import apply_spatial_tonemapping
from image_enhancement import apply_local_contrast_enhancement


if __name__=="__main__":
    
    # select an image
    filename = "../images/xray.jpg"

    image = imageio.imread(filename)  # load image
    
    # grayscale and float
    image = rgb2gray(image)
    image = img_as_float(image)
    
    # get estimation of the local neighborhood
    image_ph_mask = get_photometric_mask(
        image=image,      
        verbose=False
        )

    # increase the local contrast of the grayscale image
    image_contrast = apply_local_contrast_enhancement(
            image=image, 
            image_ph_mask=image_ph_mask, 
            degree=2,  # x2 local details
            verbose=False
            )
    
    # apply spatial tonemapping on the previous stage
    image_tonemapped = apply_spatial_tonemapping(
            image=image_contrast, 
            image_ph_mask=image_ph_mask, 
            mid_tone=0.5,
            tonal_width=0.5,
            areas_dark=0.0,  # no improvement in dark areas
            areas_bright=0.8,  # strong improvement in bright areas
            preserve_tones=False,
            verbose=False
            )

    # display results
    plt.figure(figsize=(7,3))
    plt.subplot(1,2,1)
    plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    plt.title('Input image')
    plt.axis('off')
    plt.tight_layout()
    
    plt.subplot(1,2,2)
    plt.imshow(image_tonemapped, cmap='gray', vmin=0, vmax=1)
    plt.title('Enhanced image')
    plt.axis('off')
    plt.tight_layout()
    
    plt.show()
    
