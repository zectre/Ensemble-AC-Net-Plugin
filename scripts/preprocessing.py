import numpy as np
from osgeo import gdal
from numpy import load, save, concatenate


window_size = 3

# STEP 1: LOAD DATA
def load_data(data):
    img_path = data
    data_img = gdal.Open(img_path).ReadAsArray()
    n_bands, n_row, n_col = data_img.shape
    return data_img, n_bands, n_row, n_col


# STEP 2: EXTRACT PATCHES FOR ALL PIXELS
def extract_patches(n_bands, n_row, n_col, data_img):
    n_patch = (n_row - 2) * (n_col - 2)  # Number of patches
    patches_4D = np.zeros([n_patch, n_bands, 3, 3]) # patches in 4 dimensions (n_patch, n_band, 3, 3)
    patch_index = 0
    for i in range(1, n_row - 1):
        for j in range(1, n_col - 1):
            a1 = i - 1
            a2 = i + 2
            b1 = j - 1
            b2 = j + 2
            for z in range(n_bands):
                patches_4D[patch_index, z, ...] = data_img[z, a1:a2, b1:b2]
            patch_index += 1
    return patches_4D, n_patch


# STEP 3: SAVE INTO NPY
def save_patch(path_save, patches_4D):
    # Save the patches to a .npy file
    np.save(path_save, patches_4D)

    # Return the path of the saved file
    return path_save