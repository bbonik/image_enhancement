#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 11:35:49 2020

Example: color correction (white balance)

@author: Vasileios Vonikakis (bbonik@gmail.com)
"""



import imageio
import matplotlib.pyplot as plt
from image_enhancement import correct_colors



if __name__=="__main__":

    
    # select an image
    filename = "../images/strawberries.jpg"
    # filename = "../images/napoleon.jpg"
    # filename = "../images/shark.jpg"
    # filename = "../images/underwater1.jpg"
    # filename = "../images/underwater2.jpg"
    
    image = imageio.imread(filename)  # load image
    
    image_enhanced = correct_colors(image, verbose=False)
    
    # display results
    plt.figure(figsize=(7,3))
    plt.subplot(1,2,1)
    plt.imshow(image, vmin=0, vmax=255)
    plt.title('Input image')
    plt.axis('off')
    plt.tight_layout()
    
    plt.subplot(1,2,2)
    plt.imshow(image_enhanced, vmin=0, vmax=1)
    plt.title('Corrected colors')
    plt.axis('off')
    plt.tight_layout()
    
    plt.show()
    
