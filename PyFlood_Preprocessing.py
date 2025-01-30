import rasterio
import pyproj
import numpy as np
from rasterio.transform import xy
from pyproj import Transformer
from scipy.io import savemat
from scipy.io import loadmat
import matplotlib.pyplot as plt
import config
import C1PK_Flood_Modules

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