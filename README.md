# Rayleigh-Sommerfeld Propagation Library

This Python library provides a collection of functions for simulating Rayleigh-Sommerfeld (RS) propagation of light, working with multi-page TIFF images, and leveraging parallelization for efficient computation.

## Features

1. **Rayleigh-Sommerfeld Propagation:**
   - Compute light propagation in free space using Rayleigh-Sommerfeld approximations in both spatial and Fourier domains.

2. **Image Handling:**
   - Display multi-page TIFF files with navigation via sliders and keyboard input.
   - Save processed images as multi-page TIFF files.

3. **Visualization:**
   - Extract and visualize intensity and phase images of propagated light fields.
   - Live updates for interactive visualization.

4. **Parallel Processing:**
   - Utilize multithreading for parallelizing backpropagation computations across multiple propagation distances.

## Installation

Dependencies include:
- `numpy`
- `matplotlib`
- `Pillow`
- `tifffile`
- `concurrent.futures` (Python standard library)

## Functions Overview

### Rayleigh-Sommerfeld Propagation

- `Propagation_RS_zero_padd(tim, lamb, z, n, pp, zero_padd, w)`
  Simulates light propagation using Rayleigh-Sommerfeld approximations.

- `RayleighSommerfeldFunction(L, C, z, n, lambda_, pp, w)`
  Computes the RS kernel in the spatial domain.

- `RayleighSommerfeldFunctionFT(L, C, z, n, lambda_, pp, w)`
  Computes the RS kernel in the Fourier domain.

### Encapsulated Backpropagation

- `backpropag_encapsulated(sqrt_im_norm_holo, lambda_, z_val, n, pp, zero_padd, w)`
  Combines propagation, normalization, and extraction of intensity and phase in one function.

### Visualization

- `backpro_live_plot(figure, bck_I, bck_phi)`
  Displays intensity and phase images side by side with live updates.

### TIFF Image Handling

- `display_tiff(file_path, title="images")`
  Displays multi-page TIFF images with navigation via slider and keyboard controls.

- `save_as_multipage_tiff(images, output_path)`
  Saves a list of grayscale images as a multi-page TIFF file.

### Parallel Processing

- `parallel_backpropagation(z, sqrt_im_norm_holo, lambda_, n, pp, zero_padd, w)`
  Executes backpropagation for multiple propagation distances using multithreading.

## Usage

### Example file: `qumin_bck_propag_opti.py`

```python
# Imports
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread, imwrite
import sys
import os

# local file
sys.path.insert(0, os.path.abspath("./python_qumin"))
from qumin_lib import *

# directory and path
# initial file
main_dir = '../20240711_sample_4/test_cell_z_stack/'
file_name = 'test_cell_z_stack_MMStack_Pos0.ome.tif'

# new files to be created
normalized_name = "z_stack.tiff"
intensity_name = "bckprg_I.tiff"
phase_name = "bckprg_phi.tiff"

# creating their path
file_path = os.path.join(main_dir, file_name)
intensity_path = os.path.join(main_dir, intensity_name)
phase_path = os.path.join(main_dir, phase_name)
normalized_path = os.path.join(main_dir, normalized_name)

# setting parameters and slices
background_ranges = (slice(100, 900), slice(50, 1200))
image_ranges = (slice(706, 706+1024), slice(1703, 1703+1024))
ref_image_num = 13                          # image that is going to be used for backpropagation
z = (28 + np.arange(-5, 5.25, 0.25)) * 1e-6 # z values
lambda_ = 638e-9                            # wavelength (m)
pp = 0.0993442e-6                           # Pixel size in m/pix
n = 1.333                                   # refractive index
zero_padd = 0
w = 0                                       # For different warnings


# reading files
im = np.array(imread(file_path), dtype=float)

# get median
bckgrd = im[:,background_ranges[0],background_ranges[1]]    # get background
med = np.median(bckgrd,axis = (-1,-2))                      # get median
med = med[:, np.newaxis, np.newaxis]

# crop and normalize
im_croped = im[:, image_ranges[0], image_ranges[1]]     # crop images
im_norm = im_croped / med                               # normalise
im_norm_holo = im_norm[ref_image_num]                   # catch reference image
im_norm = (128*im_norm).astype(np.uint8)
save_as_multipage_tiff(im_norm, normalized_path)

# display
display_tiff(normalized_path, title = "Cropped and normalized original images")

# backpropagation
sqrt_im_norm_holo = np.sqrt(im_norm_holo)
phase_ims, intensity_ims = parallel_backpropagation(z, sqrt_im_norm_holo, lambda_, n, pp, zero_padd, w)

# save results
save_as_multipage_tiff(phase_ims, phase_path)
save_as_multipage_tiff(intensity_ims, intensity_path)
display_tiff(phase_path, title = "Intensity backpropagation")
display_tiff(intensity_path, title = "Phase backpropagation")
```

