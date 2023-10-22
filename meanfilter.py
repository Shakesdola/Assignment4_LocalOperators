#!/usr/bin/env python
# coding: utf-8

import numpy as np
from PIL import Image

# Load the grayscale image
image = Image.open('Stop.png').convert('L')
width, height = image.size

# Define the kernel sizes (3x3, 5x5, and 9x9)
kernel_sizes = [3, 5, 9]

# Initialize an empty list to store the filtered images
filtered_images = []

# Apply mean filter for each kernel size
for size in kernel_sizes:
    kernel = np.ones((size, size), dtype=np.float32) / (size * size)
    data = np.array(image)
    filtered = np.zeros((height, width), dtype=np.uint8)

    for y in range(size // 2, height - size // 2):
        for x in range(size // 2, width - size // 2):
            neighborhood = data[y - size // 2:y + size // 2 + 1, x - size // 2:x + size // 2 + 1]
            filtered[y, x] = np.sum(neighborhood * kernel)
            
    #Save the filtered images as "3k.png", "5k.png" and "3k.png"
    output = Image.fromarray(filtered)
    output.save(f'{size}k.png')
