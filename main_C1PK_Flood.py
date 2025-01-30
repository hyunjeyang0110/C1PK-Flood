# Global parameters
Selected_percentile = 5  # Test set percentile: 95, 85, ..., 25, 15, 5
Reduction_factor = 0.000375 # Uniform Reduction Factor of 0.000375 as used in case of Hurricane Sandy


import os
import sys
import copy
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras # tensorflow version: 2.17.0
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from sklearn.decomposition import PCA
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten, Dropout, Conv1D, MaxPooling1D
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans
import config  # Importing the main config.py file containing PyFlood parameters
import rasterio
import geopandas as gpd
import importlib
from rasterio.transform import xy, from_origin
from scipy.io import loadmat, savemat
from scipy.interpolate import griddata
from scipy.ndimage import zoom, binary_erosion, binary_dilation
from scipy.spatial.distance import cdist
from scipy.linalg import inv
from shapely.geometry import Point
from skimage import measure, morphology, util
from skimage.measure import label, find_contours
from skimage.transform import resize
from skimage.feature import canny
from bayes_opt import BayesianOptimization
import C1PK_Flood_Modules

importlib.reload(config)

if os.path.isdir(os.path.join('models/TESTSET')):
	print('DIR exist')
else:
	os.mkdir('models/TESTSET')
	print('CREATED: models')


left_boundary = -95.14
right_boundary = -94.34
below_boundary = 29.05
upper_boundary = 29.64

plot_left_boundary = -100.5
plot_right_boundary = -93.9
plot_below_boundary = 25.5
plot_upper_boundary = 30.5

lon_min = -101
lon_max = -94
lon_interval = 3
lat_min = 26
lat_max = 31
lat_interval = 2
fz = 25
plot_projection = 'merc'
map_resolution = 'h'
get_ipython().run_line_magic('matplotlib', 'inline')

max_outlier = 100.0
min_outlier = 0.0
# mean_outlier = 0.5
most_common_threshold =0.2
num_most_common_threshold = 200
count_non_nan_threshold = 100
count_non_nan_storm_threshold = 1000
max_surge_elevation_threshold = 0.5 #meter

random_seed = 111
n_split = 10               # outer fold
n_split_valid = 5          # inner fold
n_cluster = 2

verbose = 0
epochs= 10000
batch_size = 32
size_filter=64
size_kernel=3
activation_function='relu'
size_pool=2
drop_out_rate =0.2
neuron_1stlayer = 128
neuron_2ndlayer = 64
dense_activation_function = 'linear'
kernel_init='normal'


testX, testY = pd.read_pickle("TEST_DATA.pkl")
MaxPSS_percentile = [95, 85, 75, 65, 55, 45, 35, 25, 15, 5]
test_index = MaxPSS_percentile.index(Selected_percentile)
testY.index = MaxPSS_percentile

testX = testX[test_index,:,:][np.newaxis, ...]
x_values = np.array(list(range(-25, 10)))
selected_order = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
print('Selected time-series:', x_values[selected_order])
testX = testX[:,selected_order,:]
print('test storms shape:', testX.shape)

row_ = testY.loc[Selected_percentile]
testY = row_.to_frame().T


model_folder_name = 'models'

def load_all_models(k,N_c):
    all_models = list()
    for j in range(n_split_valid):
        filename = model_folder_name + '/model_' + str(k) + '_' + str(j) + '_' + str(N_c) + '.keras'
        model = keras.models.load_model(filename, compile=False)
        all_models.append(model)
    return all_models
    
station_db_panda, storm_para_stack, MaxElev_ori, cluster_total, n_pcs = pd.read_pickle("TRAINING_DATA.pkl")
x_values = np.array(list(range(-25, 10)))
selected_order = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
print('Selected time-series:', x_values[selected_order])
storm_para_stack = storm_para_stack[selected_order,:,:]
print('input storms shape:', storm_para_stack.shape)

MaxElev_total = MaxElev_ori.T
n_storms = MaxElev_total.shape[1]
total_scenarios = np.arange(0,n_storms)
shuf = np.arange(0,n_storms)
np.random.seed(random_seed)
np.random.shuffle(shuf)
shuf = list(shuf)
storm_split = np.array_split(shuf,10)

MaxElev_01_ori = copy.deepcopy(MaxElev_ori)
MaxElev_01_ori_T = MaxElev_01_ori.T
MaxElev_01_ori_T.index = MaxElev_01_ori_T.index.astype(int)
data = {
    'index': range(1, 437),
    'Central Pressure Deficit': [0] * 436,
    'Heading Direction': [0] * 436,
    'Radius of Maximum Wind': [0] * 436,
    'Reference Latitude': [0] * 436,
    'Reference Longitude': [0] * 436,
    'Translational Speed': [0] * 436
}
df = pd.DataFrame(data)
df.index = MaxElev_ori.index
MaxElev_ori = pd.concat([df, MaxElev_ori], axis=1)

MaxElev = MaxElev_ori.iloc[:,7:].T
MaxElev.index = MaxElev.index.astype(int)
station_index = MaxElev.index.values
n_storms = len(MaxElev.columns)
storm_dummy = MaxElev_ori.iloc[:,:7]
station_lon_lat = station_db_panda.loc[station_index].reset_index(drop=True)
station_db_panda['ID'] = station_db_panda.index
storm_para_stack = np.swapaxes(storm_para_stack,0,2)
storm_para_stack = np.swapaxes(storm_para_stack,1,2)

Total_prediction_data=[]
for k in range(n_split):

    test_storm_index = sorted(storm_split[k])
    train_storm_index =sorted(np.delete(np.asarray(total_scenarios),np.asarray(test_storm_index) ))
    cluster_df = cluster_total.iloc[:,k]
    trainX = storm_para_stack[train_storm_index]

    for N_c in range(n_cluster):
    
        cluster_index = cluster_df[cluster_df == N_c].index.values
        MaxElev_01  =  MaxElev_01_ori_T.loc[MaxElev_01_ori_T.index.isin(cluster_index)].T
        test_MaxElev_01 = MaxElev_01.iloc[test_storm_index,:] 

        dataset_tmp = MaxElev.loc[MaxElev.index.isin(cluster_index)].T
        dataset = storm_dummy.join(dataset_tmp)
        train_dataset = dataset.iloc[train_storm_index,:]
        train_dataset = train_dataset.sort_values(by=['index']).reset_index(drop=True)
        trainy = train_dataset.iloc[:,7:]
        testy = testY[cluster_index]
        
        flatTrainX = trainX.reshape((trainX.shape[0] * trainX.shape[1], trainX.shape[2]))
        flatTestX = testX.reshape((testX.shape[0] * testX.shape[1], testX.shape[2]))
        trans_input = StandardScaler()
        trans_input.fit(flatTrainX)
        flatTrainX = trans_input.transform(flatTrainX)
        flatTestX = trans_input.transform(flatTestX)
        trainX_normalized = flatTrainX.reshape((trainX.shape))
        testX_normalized = flatTestX.reshape((testX.shape))
        
        trans_output = StandardScaler()
        trainy_normalized = trans_output.fit_transform(trainy)
        
        Model_number = k*n_cluster + N_c
        pca_number = n_pcs[Model_number]
        pca = PCA(n_components=pca_number)
        Y_pca_train = pca.fit_transform(trainy_normalized)
        pca_df = pd.DataFrame(Y_pca_train)

        
        members = load_all_models(k,N_c)
        prediction_tmp = None
        for model in members:
            test_predictions = model.predict(testX_normalized)
            inv_pca = pca.inverse_transform(test_predictions)
            test_p = trans_output.inverse_transform(inv_pca)
            if prediction_tmp is None:
                prediction_tmp = test_p
            else:
                prediction_tmp = np.dstack((prediction_tmp, test_p))
        final_estimation = np.average(prediction_tmp,axis=2)

        final_df = pd.DataFrame(final_estimation,columns=test_MaxElev_01.columns)

        if N_c == 0:
            final_df_total = final_df
            testy_total = testy
        else:
            final_df_total= pd.concat([final_df_total,final_df],axis=1)
            testy_total= pd.concat([testy_total,testy],axis=1)

    final_df_total = final_df_total.sort_index(axis=1, ascending=True)
    Total_prediction_data.append(final_df_total.values)
    testy_total = testy_total.sort_index(axis=1, ascending=True)

FINAL_ensemble_prediction = np.mean(np.stack(Total_prediction_data, axis = -1),2)
FINAL_ensemble_prediction = pd.DataFrame(FINAL_ensemble_prediction, columns=testy_total.columns, index = [Selected_percentile])

PyFlood_input = pd.DataFrame()
PyFlood_input['ID'] = testy_total.columns
PyFlood_input = PyFlood_input.merge(station_db_panda[['ID', 'x', 'y']], on='ID', how='left')
PyFlood_input['Predicted'] = FINAL_ensemble_prediction.loc[Selected_percentile].reset_index(drop=True)

PyFlood_input.to_csv('models/TESTSET/PyFlood_INPUT_'+str(Selected_percentile)+'_outof0to445.csv')


dem_data_path = config.dem_data_path
z, lon, lat = C1PK_Flood_Modules.load_dem_data(dem_data_path)
print("DEM data path:", dem_data_path)

import config
wb = config.wb

# Print the uniform water level to verify
wl_s1 = config.wl_s1
print("Uniform water level:", wl_s1)

# Find the coastline using the DEM data and known water body point
coastline, _ = C1PK_Flood_Modules.find_coastline(lon, lat, z, wb)  # Discard the second output (fBW) since it is not needed

# Specify the path to the filtered CSV file
filtered_csv_file = 'models/TESTSET/PyFlood_INPUT_'+str(Selected_percentile)+'_outof0to445.csv'

# Load and print the filtered stations
filtered_stations = C1PK_Flood_Modules.load_filtered_csv(filtered_csv_file)

# Load water levels from multiple monitoring stations from config.py
wl_s2 = np.array(filtered_stations)

# Example with spatially varying water levels from monitoring stations:
SF2, fwl2, _ = C1PK_Flood_Modules.PyFlood_StaticFlooding(lon, lat, z, wl_s2, wb)

distmat_int = C1PK_Flood_Modules.compute_interpolated_distance_matrix(lon, lat, z, coastline)
post_ccfm_uniform_rf_0_000375 = C1PK_Flood_Modules.apply_uniform_rf_to_model(lon, lat, z, wb, SF2['FloodDepth'], distmat_int, Reduction_factor)
C1PK_Flood_Modules.create_raster_from_dem(lon, lat, -post_ccfm_uniform_rf_0_000375+z, 'RESULT_C1PK_Flood_storm' + str(Selected_percentile) +'.tif', 'DEM_Galveston.tif')