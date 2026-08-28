#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 11:35:49 2020

Example: image enhancement 
(spatial tone-mapping, local contrast enhancement, color enhancement)

@author: Vasileios Vonikakis (bbonik@gmail.com)
"""


import glob
import imageio.v2 as imageio
from image_enhancement import blend_exposures



if __name__=="__main__":
    
    # select a collection of image exposures
    
    # exposure_filenames = glob.glob('../images/exposures_A*.jpg')
    exposure_filenames = glob.glob('../images/exposures_B*.jpg')

    # put the exposures in a list
    image_list = []
    for filename in exposure_filenames:
        image = imageio.imread(filename)
        image_list.append(image)
    
    # blend exposures
    exposure_blend = blend_exposures(
        image_list,             
        threshold_dark=0.35,                              
        threshold_bright=0.65, 
        verbose=True
        )
   
    
