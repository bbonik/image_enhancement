#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 11:35:49 2020

Example: enhancement of local details 

@author: Vasileios Vonikakis (bbonik@gmail.com)
"""



import imageio
import matplotlib.pyplot as plt
from image_enhancement import enhance_image



if __name__=="__main__":
    
    # select an image
    filename = "../images/waves.jpg"

    image = imageio.imread(filename)  # load image
    
    # setting up parameters
    parameters = {}
    parameters['local_contrast'] = 4  # 4x increase in details
    parameters['mid_tones'] = 0.5
    parameters['tonal_width'] = 0.5
    parameters['areas_dark'] = 0.0  # no change in dark areas
    parameters['areas_bright'] = 0.0  # no change in bright areas
    parameters['preserve_tones'] = False
    parameters['saturation_degree'] = 2  # 2x increase in color saturation
    parameters['color_correction'] = False
    image_enhanced = enhance_image(image, parameters, verbose=False)  
    
    # display results
    plt.figure(figsize=(7,3))
    plt.subplot(1,2,1)
    plt.imshow(image, vmin=0, vmax=255)
    plt.title('Input image')
    plt.axis('off')
    plt.tight_layout()
    
    plt.subplot(1,2,2)
    plt.imshow(image_enhanced, cmap='gray', vmin=0, vmax=1)
    plt.title('Increased local contrast')
    plt.axis('off')
    plt.tight_layout()
    
    plt.show()
    
