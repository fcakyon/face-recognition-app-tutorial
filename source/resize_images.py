# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 18:16:24 2019

@author: FCA
"""

import cv2 # OpenCV for image editing, computer vision and deep learning
from source.utils import get_folder_dir # Custom function for better directory name handling

# Get images directory
images_dir = get_folder_dir("images") 

# Read image
image = cv2.imread(images_dir + "sample4.jpg")
 
print('Original Dimensions : ',image.shape)
 
scale_percent = 50 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) 
print('Resized Dimensions : ',resized.shape) 

# save image
cv2.imwrite(images_dir + "sample4.jpg", resized) 