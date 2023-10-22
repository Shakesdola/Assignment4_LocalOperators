#!/usr/bin/env python
# coding: utf-8

import numpy as np
from PIL import Image

# LoG filter
def LoG_filter(img, kernel_size=5, sigma=3):
    height, weight = img.shape
    
    pad = kernel_size // 2
    outlarge = np.zeros((height + pad *2 , weight + pad *2 ), dtype=float)
    outlarge[pad: pad + height, pad: pad + weight] = img.copy().astype(float)
    tmp = outlarge.copy()

    # LoG Kernel
    K = np.zeros((kernel_size, kernel_size), dtype=float)
    for x in range(-pad, -pad + kernel_size):
        for y in range(-pad, -pad + kernel_size):
            K[y + pad, x + pad] = (
                (x ** 2 + y ** 2 - sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
            )
    K /= (2 * np.pi * (sigma ** 6))
    K /= K.sum()

    # Filtering
    for y in range(height):
        for x in range(weight):
            outlarge[pad + y, pad + x] = np.sum(K * tmp[y: y + kernel_size, x: x + kernel_size])

    out = np.clip(outlarge, 0, 255)
    out = outlarge[pad: pad + height, pad: pad + weight].astype(np.uint8)
    return out

# Load the grayscale image
image = Image.open('Stop.png').convert('L')
image_array = np.array(image)

# Apply LoG filter for edge detection
filtered_image = LoG_filter(image_array, kernel_size=5, sigma=3)

# Adjust contrast and brightness
contrast_factor = 5
brightness_offset = 0.5

adjusted_image = filtered_image * contrast_factor + brightness_offset

# Ensure pixel values are within the valid range
adjusted_image = np.clip(adjusted_image, 0, 255).astype(np.uint8)

# Create a PIL image from the adjusted result
output_image = Image.fromarray(adjusted_image)
output_image.save('LoGfilter.png')
