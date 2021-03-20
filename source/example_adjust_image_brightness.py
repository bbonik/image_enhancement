#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 11:35:49 2020

Example: image enhancement 
(spatial tone-mapping, local contrast enhancement, color enhancement)

@author: Vasileios Vonikakis (bbonik@gmail.com)
"""



import imageio
import matplotlib.pyplot as plt
from image_enhancement import enhance_image



if __name__=="__main__":
    
    # select an image
    filename = "../images/lisbon.jpg"

    image = imageio.imread(filename)  # load image
    
    # setting up parameters
    parameters = {}
    parameters['local_contrast'] = 1.0  # no increase in details
    parameters['mid_tones'] = 0.5  # middle of range
    parameters['tonal_width'] = 0.5  # middle of range
    parameters['areas_dark'] = 0.0  # no change in dark areas
    parameters['areas_bright'] = 0.0  # no change in bright areas
    parameters['saturation_degree'] = 1.0  # no change in color saturation
    parameters['brightness'] = 0.5  # increase overall brightness by 50%
    parameters['preserve_tones'] = True
    parameters['color_correction'] = True
    image_enhanced = enhance_image(image, parameters, verbose=False)  
    
    # display results
    plt.figure(figsize=(7,3))
    plt.subplot(1,2,1)
    plt.imshow(image, vmin=0, vmax=255)
    plt.title('Input image')
    plt.axis('off')
    plt.tight_layout()
    
    plt.subplot(1,2,2)
    plt.imshow(image_enhanced, vmin=0, vmax=255)
    plt.title('Enhanced image')
    plt.axis('off')
    plt.tight_layout()
    
    plt.show()
    
