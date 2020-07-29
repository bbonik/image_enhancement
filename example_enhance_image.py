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
from src.image_enhancement import enhance_image



if __name__=="__main__":
    
    # select an image
#    filename = "images/alhambra1.jpg"
#    filename = "images/alhambra2.jpg"
    filename = "images/lisbon.jpg"

    image = imageio.imread(filename)  # load image
    
    # setting up parameters
    parameters = {}
    parameters['local_contrast'] = 1.2  # 1.5x increase in details
    parameters['mid_tones'] = 0.5
    parameters['tonal_width'] = 0.5
    parameters['areas_dark'] = 0.7  # 70% improvement in dark areas
    parameters['areas_bright'] = 0.5  # 50% improvement in bright areas
    parameters['preserve_tones'] = True
    parameters['saturation_degree'] = 1.2  # 1.2x increase in color saturation
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
    
