#!/usr/bin/env python
# coding: utf-8

import numpy as np
from PIL import Image

# Load the grayscale image
image = Image.open('Stop.png').convert('L')
data = np.array(image)

# Define gradient kernels for horizontal and vertical edge detection
#horizontal_kernel = np.array([[-1, 0, 1]])
#vertical_kernel = np.array([[-1], [0], [1]])

# Define the Sobel kernels for horizontal and vertical edge detection
horizontal_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
vertical_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Define the Prewitt kernels for horizontal and vertical edge detection
#horizontal_kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
#vertical_kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

# Get the dimensions of the image
height, width = data.shape

# Initialize arrays to store the horizontal and vertical edges
horizontal_edge = np.zeros_like(data)
vertical_edge = np.zeros_like(data)

# Perform convolution with the horizontal and vertical Sobel kernels
for y in range(1, height - 1):
    for x in range(1, width - 1):
        neighborhood = data[y - 1 : y + 2, x - 1 : x + 2]
        horizontal_edge[y, x] = np.sum(neighborhood * 1/4 * horizontal_kernel)
        vertical_edge[y, x] = np.sum(neighborhood * 1/4 * vertical_kernel)

# Combine horizontal and vertical edges to get gradient magnitude
gradient_magnitude = np.sqrt(horizontal_edge**2 + vertical_edge**2)

# Normalize gradient magnitude to [0, 255]
gradient_magnitude = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)

# Create a PIL image from the filtered result
gradient_image = Image.fromarray(gradient_magnitude, 'L')
gradient_image.save('gradient.png')
