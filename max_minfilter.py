#!/usr/bin/env python
# coding: utf-8

from PIL import Image
import numpy as np

def min_filter(input_image, kernel_size):
    width, height = input_image.size
    image_data = np.array(input_image)
    output_data = np.copy(image_data)
    pad = kernel_size // 2

    for x in range(pad, width - pad):
        for y in range(pad, height - pad):
            neighborhood = image_data[y - pad : y + pad + 1, x - pad : x + pad + 1]
            min_value = np.min(neighborhood)
            output_data[y, x] = min_value

    return Image.fromarray(output_data)

def max_filter(input_image, kernel_size):
    width, height = input_image.size
    image_data = np.array(input_image)
    output_data = np.copy(image_data)
    pad = kernel_size // 2

    for x in range(pad, width - pad):
        for y in range(pad, height - pad):
            neighborhood = image_data[y - pad : y + pad + 1, x - pad : x + pad + 1]
            max_value = np.max(neighborhood)
            output_data[y, x] = max_value

    return Image.fromarray(output_data)


image = Image.open('Stop.png').convert('L')

# Apply the minimum and maximum filters
min_filtered = min_filter(image, kernel_size=3)
max_filtered = max_filter(image, kernel_size=5)

# Combine (max - min)
max_min = Image.fromarray(np.uint8(np.array(max_filtered) - np.array(min_filtered)))

# Save the filtered images
min_filtered.save('minfilter.png')
max_filtered.save('maxfilter.png')
max_min.save('max-minfilter.png')

