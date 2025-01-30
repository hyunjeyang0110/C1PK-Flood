import rasterio
import pyproj
import numpy as np
import pandas as pd
import rasterio
import pickle
from rasterio.transform import xy
from pyproj import Transformer
from scipy.io import savemat
from scipy.io import loadmat
import matplotlib.pyplot as plt
import config
import C1PK_Flood_Modules
from sklearn.neighbors import KNeighborsRegressor
from rasterio.transform import from_origin

# Libraries version
# rasterio version: 1.4.3
# numpy version: 1.26.4
# pyproj version: 3.7.0
# scipy version: 1.13.1
# matplotlib version: 3.9.2

def save_data(elevation, longitude, latitude, filename=config.dem_file_path):
    """
    Save the data to a .mat file with compression.

    Parameters
    ----------
    elevation : numpy.ndarray
        Elevation data array.
    longitude : numpy.ndarray
        Longitude coordinates array.
    latitude : numpy.ndarray
        Latitude coordinates array.
    filename : str, optional
        Name of the .mat file to save (default is 'DEM.mat').

    Returns
    -------
    None
    """
    data = {
        'z': elevation,
        'lon': longitude,
        'lat': latitude
    }
    savemat(filename, data, do_compression=True)

def save_data_mat(elevation, longitude, latitude, filename=config.dem_file_path):
    data = {
        'z': elevation,
        'lon': longitude,
        'lat': latitude
    }
    # Save without compression
    savemat(filename, data, do_compression=False)




ADCIRC_ori_mesh = pd.read_pickle("ADCIRC_ori_mesh.pkl")

left_boundary_PyFlood = -95.14
below_boundary_PyFlood = 29.05
right_boundary_PyFlood = -94.34
upper_boundary_PyFlood = 29.64

step_size = 0.0001

start_x = left_boundary_PyFlood + step_size/2
start_y = below_boundary_PyFlood + step_size/2

x_values = np.arange(start_x, right_boundary_PyFlood, step_size)
y_values = np.arange(start_y, upper_boundary_PyFlood, step_size)

x_grid, y_grid = np.meshgrid(x_values, y_values)

grid_coordinates = np.c_[x_grid.ravel(), y_grid.ravel()]

num_x_points = len(x_values)
num_y_points = len(y_values)

interpolate_input = ADCIRC_ori_mesh[['x','y']]
interpolate_output = ADCIRC_ori_mesh['bathymetry']

knn = KNeighborsRegressor(n_neighbors = 2)
knn.fit(interpolate_input, interpolate_output)
grid_coordinates = pd.DataFrame(grid_coordinates, columns=interpolate_input.columns)
dem_interpolated = knn.predict(grid_coordinates)

x = np.array(grid_coordinates['x'])
y = np.array(grid_coordinates['y'])
z = np.array(dem_interpolated)

grid_shape = (5900, 8000)

grid_z = z.reshape(grid_shape)
grid_z_flipped = np.flipud(grid_z)

x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()

x_res = (x_max - x_min) / (grid_shape[1] - 1)
y_res = (y_max - y_min) / (grid_shape[0] - 1)

transform = from_origin(x_min, y_max, x_res, y_res)

with rasterio.open(
    'DEM_Galveston.tif',
    'w',
    driver='GTiff',
    height=grid_shape[0],
    width=grid_shape[1],
    count=1,
    dtype=grid_z_flipped.dtype,
    crs='EPSG:4326',
    transform=transform,
) as dst:
    dst.write(grid_z_flipped, 1)

    
elevation, transform, crs = C1PK_Flood_Modules.load_raster_data(
    config.raster_file_path, 
    null_value=config.null_value, 
    high_threshold=config.high_threshold, 
    low_threshold=config.low_threshold
)

print("Original Coordinate Reference System of DEM:", crs)

longitude, latitude = C1PK_Flood_Modules.transform_coordinates(elevation, transform, crs, config.crs_to)

save_data(elevation, 
          longitude, 
          latitude)

data = loadmat('DEM.mat')

elevation = data.get('z')
longitude = data.get('lon')
latitude = data.get('lat')

if elevation is None or longitude is None or latitude is None:
    raise ValueError("The .mat file does not contain 'z', 'lon', or 'lat' variables.")

print(f"Shape of elevation: {elevation.shape}")
print(f"Shape of longitude: {longitude.shape}")
print(f"Shape of latitude: {latitude.shape}")

data = loadmat('DEM.mat')

elevation = data.get('z')
longitude = data.get('lon')
latitude = data.get('lat')

print(f"Shape of elevation (original): {elevation.shape}")

# Plot the DEM from the saved .mat file
C1PK_Flood_Modules.plot_dem_from_mat(config.dem_file_path)