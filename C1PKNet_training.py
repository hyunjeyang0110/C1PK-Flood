import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import os
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.decomposition import PCA
import random
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans

#-----------------------------------------------------------------------------#
# Parameters                                                                  #
#-----------------------------------------------------------------------------#

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


import pickle
# station_db_panda: 1,217 point information 
# storm_para_stack: time-series of storm parameters
# MaxElev_ori: peak storm surge values
# cluster_total: cluster number for each outer folds
# n_pcs: number of principal components for each outer folds and clusters
station_db_panda, storm_para_stack, MaxElev_ori, cluster_total, n_pcs = pd.read_pickle("TRAINING_DATA.pkl")

x_values = np.array(list(range(-25, 10)))
selected_order = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
print('Selected time-series:', x_values[selected_order])
storm_para_stack = storm_para_stack[selected_order,:,:]
print('input storms shape:', storm_para_stack.shape)

import copy
MaxElev_01_ori = copy.deepcopy(MaxElev_ori)

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

if os.path.isdir(os.path.join('models')):
	print('DIR exist')
else:
	os.mkdir('models')
	print('CREATED: models')


MaxElev = MaxElev_ori.iloc[:,7:].T
MaxElev.index = MaxElev.index.astype(int)
station_index = MaxElev.index.values
n_storms = len(MaxElev.columns)
storm_dummy = MaxElev_ori.iloc[:,:7]
station_lon_lat = station_db_panda.loc[station_index].reset_index(drop=True)
station_db_panda['ID'] = station_db_panda.index

storm_para_stack = np.swapaxes(storm_para_stack,0,2)
storm_para_stack = np.swapaxes(storm_para_stack,1,2)


total_scenarios = np.arange(0,n_storms)
shuf = np.arange(0,n_storms)
np.random.seed(random_seed)
print('Random seed: ',random_seed)
np.random.shuffle(shuf)
shuf = list(shuf)
storm_split = np.array_split(shuf,10)

pca_number = 0
pca_total = pd.DataFrame()

for k in range(0, n_split):
    print('(1) Fold number: ', k+1)
    print(' - Split the storms into training set and test set')
    test_storm_index = sorted(storm_split[k])                                                                       
    train_storm_index =sorted(np.delete(np.asarray(total_scenarios),np.asarray(test_storm_index) ))                 
    cluster_df = cluster_total.iloc[:,k]
    trainX = storm_para_stack[train_storm_index]
    testX = storm_para_stack[test_storm_index]

    final_df_total = pd.DataFrame()
    test_MaxElev_01_total = pd.DataFrame()
    testy_total =pd.DataFrame()
    
    for N_c in range(0, n_cluster):
        print('(2) Cluster number: ', N_c+1)
        cluster_index = cluster_df[cluster_df == N_c].index.values

        MaxElev_01_ori_T = MaxElev_01_ori.T
        MaxElev_01_ori_T.index = MaxElev_01_ori_T.index.astype(int)
        MaxElev_01  =  MaxElev_01_ori_T.loc[MaxElev_01_ori_T.index.isin(cluster_index)].T
        test_MaxElev_01 = MaxElev_01.iloc[test_storm_index,:]


        print(' - Extract the peak storm surges for cluster ', N_c+1)
        dataset_tmp =MaxElev.loc[MaxElev.index.isin(cluster_index)].T
        dataset = storm_dummy.join(dataset_tmp)

        print(' - Split the peak storm surges into training set and test set')
        train_dataset = dataset.iloc[train_storm_index,:]
        test_dataset = dataset.iloc[test_storm_index,:]
        train_dataset = train_dataset.sort_values(by=['index']).reset_index(drop=True)
        test_dataset = test_dataset.sort_values(by=['index']).reset_index(drop=True)
        trainy = train_dataset.iloc[:,7:]
        testy = test_dataset.iloc[:,7:]

        print(' - Normalization (Standard scaler): Input')
        flatTrainX = trainX.reshape((trainX.shape[0] * trainX.shape[1], trainX.shape[2]))
        flatTestX = testX.reshape((testX.shape[0] * testX.shape[1], testX.shape[2]))
        trans_input = StandardScaler()
        trans_input.fit(flatTrainX)
        flatTrainX = trans_input.transform(flatTrainX )
        flatTestX = trans_input.transform(flatTestX )
        trainX_normalized = flatTrainX.reshape((trainX.shape))
        testX_normalized = flatTestX.reshape((testX.shape))

        print(' - Normalization (Standard scaler): Output')
        trans_output = StandardScaler()
        trainy_normalized = trans_output.fit_transform(trainy)


        print(' - PCA')
        number_of_components = n_pcs[pca_number]
        pca = PCA(n_components=number_of_components)
        pca_number = pca_number+1
        Y_pca_train = pca.fit_transform(trainy_normalized)
        pca_df = pd.DataFrame(Y_pca_train)

        print(' - Split the training storms into training set and validation set')
        shuf_valid = np.arange(0,len(pca_df)).ravel()
        np.random.seed(random_seed)
        np.random.shuffle(shuf_valid)
        shuf_valid = list(shuf_valid)
        storm_split_valid = np.array_split(shuf_valid, n_split_valid)

        for j in range(0,n_split_valid):
            print('(3) Validation set number: ', j+1)
            print(' - Split the training peak surges into training set and validation set')
            valid_scenarios = sorted(storm_split_valid[j])
            validX = trainX_normalized[valid_scenarios]
            validy =pca_df.iloc[valid_scenarios].reset_index(drop=True)
            trainX_final = np.delete(trainX_normalized,valid_scenarios,axis=0)
            trainy_final = pca_df.drop(valid_scenarios).reset_index(drop=True)
            n_timesteps, n_features, n_outputs = trainX_final.shape[1], trainX_final.shape[2], trainy_final.shape[1]

            print(' - CNN')
            model_name = 'model_'+str(k)+ '_'+str(j)+'_'+str(N_c)+'.keras'
            if os.path.isfile(os.path.join('models/'+model_name)):
                print('Already exist: ', model_name)
            else:

                np.random.seed(random_seed)
                tf.random.set_seed(random_seed)
                random.seed(random_seed)
                model = Sequential()
                model.add(Conv1D(filters=size_filter, kernel_size=size_kernel, activation=activation_function, input_shape=(n_timesteps,n_features)))
                model.add(BatchNormalization())
                model.add(MaxPooling1D(pool_size=size_pool))
                model.add(Conv1D(filters=size_filter, kernel_size=size_kernel, activation=activation_function))
                model.add(BatchNormalization())
                model.add(MaxPooling1D(pool_size=size_pool))
                model.add(Flatten())
                model.add(Dense(neuron_1stlayer, activation=activation_function,kernel_initializer=kernel_init))
                model.add(Dropout(drop_out_rate))
                model.add(Dense(neuron_2ndlayer, activation=activation_function,kernel_initializer=kernel_init))
                model.add(Dropout(drop_out_rate))
                model.add(Dense(n_outputs, activation=dense_activation_function))
                model.summary()
                model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                                loss='mse',
                                metrics=['mse','mae'])
                
                esCallback = tf.keras.callbacks.EarlyStopping(monitor='val_mse',
                                           min_delta=0,
                                           patience=100,
                                           verbose=0,
                                           mode='auto')
                mc = ModelCheckpoint('models/'+model_name, monitor='val_mse', mode='min', verbose=1, save_best_only=True)
                model.fit(trainX_final, trainy_final, epochs=epochs, batch_size= len(trainX_final),validation_data = (validX, validy), verbose=verbose, callbacks=[esCallback,mc]) 
                print('>Saved %s' % model_name)

        print(' - CNN Model training is finished.')
        
        def load_all_models(k,j,N_c):
            all_models = list()
            for j in range(n_split_valid):
                filename = 'models/'+'model_'+str(k)+ '_'+str(j)+'_'+str(N_c)+'.keras'
                model = keras.models.load_model(filename)
                all_models.append(model)
                print('>loaded %s' % filename)
            return all_models
        members = load_all_models(k,j,N_c)

        print(' - C1PKNet Model prediction')
        prediction_tmp = None
        for model in members:
            print('  - CNN prediction')
            test_predictions = model.predict(testX_normalized)
            print('  - PCA inversion')
            inv_pca = pca.inverse_transform(test_predictions)
            print('  - Normalization inversion')
            test_p = trans_output.inverse_transform(inv_pca)
            print('  - Stack predictions')
            if prediction_tmp is None:
                prediction_tmp = test_p
            else:
                prediction_tmp = np.dstack((prediction_tmp, test_p))
        print('  - Average the preditions')
        final_estimation = np.average(prediction_tmp,axis=2)

        final_df = pd.DataFrame(final_estimation,columns=test_MaxElev_01.columns)
        print(final_df)
        test_MaxElev_01.columns = range(test_MaxElev_01.shape[1])

        if N_c == 0:
            print('N_c = 0')
            final_df_total = final_df
            test_MaxElev_01_total = test_MaxElev_01
            testy_total = testy
        else:
            print('N_c > 0')
            final_df_total= pd.concat([final_df_total,final_df],axis=1)
            test_MaxElev_01_total= pd.concat([test_MaxElev_01_total,test_MaxElev_01],axis=1)
            testy_total= pd.concat([testy_total,testy],axis=1)
            
    print('-Save data')
    for p in range(0,len(final_df_total.index)):
        
        ANN_predict = final_df_total.iloc[p,:].reset_index(drop=True)
        result=pd.DataFrame()
        result['ID'] = testy_total.columns
        result['Observed'] = testy_total.iloc[p,:].reset_index(drop=True)
        result['Predicted']= ANN_predict
        result = result.merge(station_db_panda[['ID', 'x', 'y','bathymetry']], on='ID', how='left')
        result.to_csv('models/CNN_predict_'+str(test_dataset['index'][p])+'.csv')
