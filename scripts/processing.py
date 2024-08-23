# 1. SETUP

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, Activation, BatchNormalization, Dense, Dropout,Flatten, Reshape
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.regularizers import l2
from matplotlib import pyplot as plt   
import h5py 

## 2. LOAD DATA
def load_predictdata(): #source: \\140.116.80.130\home\AC-Net\InputforANN\NEWupdateDataset_May2021\dataforTRAINING_update22May\2Train1test

    TOA_xtest_path = r'C:\Users\Yoga\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\ensembleacnet\TOA.npy'
    TOA_xtest = np.load(TOA_xtest_path)
    
    angles_xtest_path = r'C:\Users\Yoga\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\ensembleacnet\Ang.npy'
    angles_xtest = np.load(angles_xtest_path)

    AOT_xtest_path = r'C:\Users\Yoga\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\ensembleacnet\AOT.npy'
    AOT_xtest = np.load(AOT_xtest_path)
    
    WV_xtest_path = r'C:\Users\Yoga\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins\ensembleacnet\WV.npy'
    WV_xtest = np.load(WV_xtest_path)    

    return TOA_xtest, angles_xtest, AOT_xtest, WV_xtest


''' convert prediction output in numpy to TIF image'''


import numpy as np
from numpy import load, save, concatenate
import pandas as pd
from osgeo import gdal
import rasterio as rio
def reshape_and_save_prediction(test_paths, img_path, output_path):
    # Load the prediction data
    y_prediction = np.load(test_paths)
    pred_B1 = y_prediction[:, 0]
    pred_B2 = y_prediction[:, 1]
    pred_B3 = y_prediction[:, 2]
    pred_B4 = y_prediction[:, 3]
    pred_B5 = y_prediction[:, 4]

    # Reshape the prediction data
    pred_B1_2d = np.reshape(pred_B1, (130, 167), order='C')
    pred_B2_2d = np.reshape(pred_B2, (130, 167), order='C')
    pred_B3_2d = np.reshape(pred_B3, (130, 167), order='C')
    pred_B4_2d = np.reshape(pred_B4, (130, 167), order='C')
    pred_B5_2d = np.reshape(pred_B5, (130, 167), order='C')

    filelist2 = np.concatenate((pred_B1_2d, pred_B2_2d, pred_B3_2d, pred_B4_2d, pred_B5_2d), axis=0)
    filelist_reshape2 = filelist2.reshape(5, pred_B1_2d.shape[0], pred_B1_2d.shape[1]).astype('float32')

    # Save the reshaped data to a .tif file
    with rio.open(img_path) as src:  # choose one image
        out_data = src.read()
        out_meta = src.meta
    out_meta.update(count=len(filelist_reshape2))
    out_meta['dtype'] = "float32"
    out_meta['No Data'] = 0.0
    with rio.open(output_path, "w", **out_meta) as dst:
        dst.write(filelist_reshape2)