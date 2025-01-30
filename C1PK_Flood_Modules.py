import time
import matplotlib.pyplot as plt
import numpy as np
import config # Importing the main config.py file containing PyFlood parameters
import pandas as pd
import rasterio
import geopandas as gpd
import config
import importlib
from rasterio.transform import xy
from rasterio.transform import from_origin
from scipy.io import loadmat, savemat
from scipy.interpolate import griddata
from scipy.ndimage import zoom
from scipy.spatial.distance import cdist
from scipy.linalg import inv
from scipy.ndimage import binary_erosion, binary_dilation
from shapely.geometry import Point
from skimage import measure, morphology, util
from skimage.measure import label, find_contours
from skimage.transform import resize
from skimage.feature import canny
from sklearn.metrics import mean_squared_error
from math import sqrt
from pyproj import Transformer
from bayes_opt import BayesianOptimization

def load_raster_data(raster_file, null_value, high_threshold, low_threshold):
    """
    Load the raster data from a file and handle null and unrealistic values.

    Parameters
    ----------
    raster_file : str
        Path to the raster file.
    null_value : int or float, optional
        Value to consider as null in the raster data (default is -9999).
    high_threshold : int or float, optional
        Threshold above which values are considered unrealistic and set to np.nan (default is 8849).
    low_threshold : int or float, optional
        Threshold below which values are considered unrealistic and set to np.nan (default is -500).

    Returns
    -------
    elevation : numpy.ndarray
        Elevation data array with null and unrealistic values handled.
    transform : affine.Affine
        Affine transformation for the raster.
    crs : rasterio.crs.CRS
        Coordinate reference system of the raster.

    Notes
    -----
    The high_threshold is set to 8849 meters, which corresponds to the height of Mt. Everest, the highest point on Earth.
    The low_threshold is set to -500 meters, which is a depth representing very deep waters far from the coast.
    Any value beyond these thresholds is considered unrealistic and set to np.nan.
    """
    with rasterio.open(raster_file) as dataset:
        elevation = dataset.read(1)
        transform = dataset.transform
        crs = dataset.crs
    
    print(f"Initial elevation min value: {np.nanmin(elevation)}")
    print(f"Initial elevation max value: {np.nanmax(elevation)}")

    elevation[elevation == null_value] = np.nan
    elevation[elevation > high_threshold] = np.nan
    elevation[elevation < low_threshold] = np.nan

    print(f"Updated elevation min value: {np.nanmin(elevation)}")
    print(f"Updated elevation max value: {np.nanmax(elevation)}")

    return elevation, transform, crs

def transform_coordinates(elevation, transform, crs_from, crs_to):
    """
    Transform grid coordinates from one CRS to another, in this case, to EPSG:4326 (Decimal Degrees)

    Parameters
    ----------
    elevation : numpy.ndarray
        Elevation data array.
    transform : affine.Affine
        Affine transformation for the raster.
    crs_from : str
        Coordinate reference system of the input coordinates.
    crs_to : str, optional
        Coordinate reference system of the output coordinates (default is 'epsg:4326' for WGS 84).

    Returns
    -------
    longitude : numpy.ndarray
        Transformed longitude coordinates.
    latitude : numpy.ndarray
        Transformed latitude coordinates.
    """
    from pyproj import Transformer
    cols, rows = np.meshgrid(np.arange(elevation.shape[1]), np.arange(elevation.shape[0]))
    lon, lat = rasterio.transform.xy(transform, rows, cols, offset='center')
    transformer = Transformer.from_crs(crs_from, crs_to, always_xy=True)
    lon_flat, lat_flat = transformer.transform(np.array(lon).flatten(), np.array(lat).flatten())
    longitude = lon_flat.reshape(elevation.shape)
    latitude = lat_flat.reshape(elevation.shape)

    print("New Coordinate Reference System of DEM in PyFlood:", crs_to)
    return longitude, latitude


def plot_dem_from_mat(file_path):
    """
    Read DEM data from a .mat file and plot it, handling nan values properly.

    Parameters
    ----------
    file_path : str
        Path to the .mat file containing DEM data.

    Returns
    -------
    None
    """
    # Load the .mat file
    data = loadmat(file_path)
    
    # Extract elevation, longitude, and latitude data
    elevation = data.get('z')
    longitude = data.get('lon')
    latitude = data.get('lat')
    
    if elevation is None or longitude is None or latitude is None:
        raise ValueError("The .mat file does not contain 'z', 'lon', or 'lat' variables.")
    
    # Print the shapes of the elevation, latitude, and longitude arrays
    print(f"Shape of elevation: {elevation.shape}")
    print(f"Shape of longitude: {longitude.shape}")
    print(f"Shape of latitude: {latitude.shape}")
    
    # Check for consistency in shapes
    if elevation.shape != longitude.shape or elevation.shape != latitude.shape:
        raise ValueError("The shapes of 'z', 'lon', and 'lat' arrays are not consistent.")
    
    # Display the initial maximum and minimum values of the elevation data
    print(f"Elevation min value: {np.nanmin(elevation)}")
    print(f"Elevation max value: {np.nanmax(elevation)}")
    
    # Plot the DEM
    plt.figure(figsize=(8, 8))
    plt.imshow(elevation, cmap='terrain', extent=(np.nanmin(longitude), np.nanmax(longitude), np.nanmin(latitude), np.nanmax(latitude)))
    plt.colorbar(label='Elevation')
    plt.title('DEM Map')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

def plot_data(lon, lat, data, title):
    """
    Plots geographical data on a 2D map using longitude and latitude as axes.

    This function creates a color-mesh plot of the specified data over a geographical
    area defined by longitude and latitude coordinates. It's suitable for visualizing
    spatial data such as flood depth, elevation, and other relevant metrics in coastal engineering.

    Parameters
    ----------
    lon : 2D array-like
        Longitude coordinates of the data points.
    lat : 2D array-like
        Latitude coordinates of the data points.
    data : 2D array-like
        The data to be plotted. Must match the dimensions of lon and lat.
    title : str
        Title of the plot, indicating the type of data being visualized.

    Returns
    -------
    None
        Displays the plot.

    Examples
    --------
    >>> plot_data(lon, lat, FloodDepth, 'Flood Depth')
    >>> plot_data(lon, lat, FloodDepthAndSea, 'Flood Depth and Sea')
    >>> plot_data(lon, lat, z, 'Elevation (z)')
    """
    # Initialize the plot with specified figure size
    plt.figure(figsize=(10, 8))
    
    # Create a color-mesh plot of the data
    plt.pcolormesh(lon, lat, data, shading='auto')
    
    # Add a color scale to the plot for reference
    plt.colorbar()
    
    # Set the plot title and labels for axes
    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Display the plot
    plt.show()

def bwboundaries_sfm(binary_image, connectivity=8):
    """
    Identifies the boundaries of objects and holes in a binary image.

    This function labels the regions in a binary image representing objects
    and holes based on the specified connectivity. It uses the labeling
    algorithm from the scikit-image library to find connected components.

    Parameters
    ----------
    binary_image : 2D array-like
        A binary image where nonzero (True) pixels belong to objects and
        0 (False) pixels to the background.
    connectivity : int, optional
        The connectivity criterion (4 or 8) used for labeling. 8-connectivity
        means that diagonal pixels are considered neighbors, while 4-connectivity
        means only horizontal and vertical neighbors are considered. Default is 8.

    Returns
    -------
    L_SFM : dict of {'objects': 2D numpy.ndarray, 'holes': 2D numpy.ndarray}
        A dictionary containing labeled images of objects and holes. Each unique
        number in the arrays represents a distinct object or hole, with the
        background labeled as 0.

    Raises
    ------
    ValueError
        If `connectivity` is not 4 or 8.

    Examples
    --------
    >>> binary_image = np.array([[0, 0, 1, 1, 0],
    ...                          [0, 1, 1, 0, 0],
    ...                          [0, 0, 0, 0, 0],
    ...                          [0, 1, 1, 1, 0],
    ...                          [0, 0, 0, 0, 0]], dtype=bool)
    >>> boundaries = bwboundaries_sfm(binary_image, connectivity=8)
    >>> print(boundaries['objects'])
    >>> print(boundaries['holes'])
    """
    
    # Start the timer
    start_time = time.time()
    
    # Ensure the input image is binary; convert if necessary
    if not np.issubdtype(binary_image.dtype, np.bool_):
        binary_image = binary_image != 0
    
    # Map the specified connectivity to scikit-image's convention
    if connectivity == 8:
        skimage_connectivity = 2
    elif connectivity == 4:
        skimage_connectivity = 1
    else:
        # Validate the specified connectivity
        raise ValueError("Connectivity must be 4 or 8.")
   
    # Label objects in the binary image
    labeled_image = measure.label(binary_image, connectivity=skimage_connectivity)
    
    # Invert the image to label holes, adjusting connectivity accordingly
    inverted_image = util.invert(binary_image)
    hole_connectivity = 1 if skimage_connectivity == 2 else 2
    labeled_holes = measure.label(inverted_image, connectivity=hole_connectivity)
    
    # Optionally, if you need contours, you can find them here
    # It is commented because it causes a considerable delay in the runtime
    # contours = [measure.find_contours(labeled_image == i, level=0.5) for i in np.unique(labeled_image)[1:]]
    
    # Return the labeled objects and holes as a dictionary
    L_SFM = {'objects': labeled_image, 'holes': labeled_holes}
    
    # End the timer and calculate the duration
    end_time = time.time()
    duration = end_time - start_time
    print(f"bwboundaries_sfm execution time: {duration} seconds")
    
    return L_SFM

def calc_distance_matrix(land, lon_matrix, lat_matrix):
    """
    Calculates the distances from a given land point to each point in a grid defined by longitude and latitude matrices.

    This function uses the Haversine formula to compute the distance between the land point and each point in the grid,
    considering the Earth's curvature. It returns the minimum distance found and the position of the closest point in the grid.

    Parameters
    ----------
    land : tuple or list
        A pair (longitude, latitude) representing the specific land point from which distances are calculated.
    lon_matrix : 2D numpy.ndarray
        The longitude values of the grid points.
    lat_matrix : 2D numpy.ndarray
        The latitude values of the grid points.

    Returns
    -------
    mindist : float
        The minimum distance (in meters) from the land point to the closest grid point.
    minpos : tuple
        The position (row, column) in the grid of the closest point to the land point.

    Examples
    --------
    >>> land = (-95.3698, 29.7604)  # Longitude and latitude of a point
    >>> lon_matrix = np.array([[-95.3697, -95.3696], [-95.3695, -95.3694]])
    >>> lat_matrix = np.array([[29.7603, 29.7602], [29.7601, 29.7600]])
    >>> mindist, minpos = calc_distance_matrix(land, lon_matrix, lat_matrix)
    >>> print(f"Minimum distance: {mindist} meters, Position: {minpos}")
    """
    # Start the timer
    start_time = time.time()
    
    # Earth radius in kilometers
    Rt = 6371
    
    # Convert latitude to radians for distance calculation
    r_lat = (2 * np.pi * Rt) / 360

    # Calculate the longitude radius for each grid point considering Earth's curvature
    r_lon_matrix = 2 * np.pi * Rt * np.cos(np.radians((lat_matrix + land[1]) / 2)) / 360
    
    # Compute distances using the Haversine formula adapted for matrix operations
    dists_matrix = np.sqrt(((lon_matrix - land[0]) * r_lon_matrix)**2 + ((lat_matrix - land[1]) * r_lat)**2)
    
    # Identify the minimum distance and its position within the matrix
    mindist = np.min(dists_matrix)
    minpos = np.unravel_index(np.argmin(dists_matrix), dists_matrix.shape)

    # End the timer and calculate the duration
    end_time = time.time()
    duration = end_time - start_time
    print(f"calc_distance_matrix execution time: {duration} seconds")
    
    # Convert the minimum distance from kilometers to meters and return it with the position
    return mindist * 1000, minpos

def static_flooding(lon, lat, z, wl, wb):
    """
    Simulates flooding based on digital elevation model (DEM) data, water level, and a reference sea point.

    This function creates a simplified static flooding model by subtracting the water level from the
    DEM data to identify flooded areas. It uses binary image processing to distinguish between water and land,
    identifies connected components as water bodies, and calculates the depth of flooding.

    Parameters
    ----------
    lon : 2D numpy.ndarray
        Longitude coordinates corresponding to each point in the DEM data.
    lat : 2D numpy.ndarray
        Latitude coordinates corresponding to each point in the DEM data.
    z : 2D numpy.ndarray
        Elevation data from the DEM, where each value represents the elevation at the corresponding point.
    wl : float
        The water level used to simulate flooding. All elevations below this value are considered flooded.
    wb : tuple or list
        A pair (longitude, latitude) representing a specific point known to be within the main water body (e.g., the sea).

    Returns
    -------
    SF : dict
        A dictionary with two keys:
        'FloodDepth' : 2D numpy.ndarray
            Indicates the depth of flooding at each point where new land is inundated.
        'FloodDepthAndSea' : 2D numpy.ndarray
            Includes both the depth of flooding over land and the original water body.

    Examples
    --------
    >>> lon = np.array([[...], [...]])
    >>> lat = np.array([[...], [...]])
    >>> z = np.array([[...], [...]])
    >>> wl = 1.5
    >>> wb = (-95.3698, 29.7604)
    >>> SF = static_flooding(lon, lat, z, wl, wb)
    >>> print(SF['FloodDepth'])
    >>> print(SF['FloodDepthAndSea'])
    """
    # Start the timer
    start_time = time.time()
    
    # Adjust the topography based on the water level to identify potential flooding areas
    zF = z - wl

    # Convert to a binary image indicating water (1) and land (0)
    zBW = np.where(zF > 0, np.nan, 1)  # Mark potential water areas
    zBW = np.where(np.isnan(zBW), 0, zBW)  # Ensure land is marked as 0

    # Use the custom function to identify the connected components representing bodies of water and land
    Lc = bwboundaries_sfm(zBW)

    # Determine which of the identified bodies of water is connected to the sea
    mindist, minpos = calc_distance_matrix(wb, lon, lat)
    Lwb1 = Lc['objects'][minpos]
    Lwb2 = Lc['holes'][minpos]

    # Mark the main water body and any connected flooded areas
    Lp = np.zeros_like(z)
    Lp[Lc['objects'] == Lwb1] = 1
    holes = np.zeros_like(z)
    holes[Lc['holes'] == Lwb2] = 1
    L = Lp + holes  # Combine to get the final flooded areas

    # Calculate the depth of flooding in these areas
    fpos = np.where(L == 2)
    FloodDepthAndSea = np.full_like(z, np.nan)
    FloodDepthAndSea[fpos] = zF[fpos]

    # Separate new flooded areas from the existing water body
    FloodDepth = np.full_like(z, np.nan)
    FloodDepth[fpos] = FloodDepthAndSea[fpos]
    FloodDepth[z <= 0] = np.nan  # Exclude existing water bodies

    # Compile the results into a dictionary
    SF = {
        'FloodDepth': FloodDepth,
        'FloodDepthAndSea': FloodDepthAndSea,
    }
    # End the timer and calculate the duration
    end_time = time.time()
    duration = end_time - start_time
    print(f"static_flooding execution time: {duration} seconds")

    return SF

def red_fac_grid_FOL(lon, lat, z, rf, L):
    """
    Calculates a grid of spatially varying reduction factors based on observed reduction factors and a correlation length.

    This function uses a Gaussian function to spatially interpolate the observed reduction factors across a grid defined
    by the longitude, latitude, and elevation data. The correlation length (L) controls the spatial influence of each
    observation.

    Parameters
    ----------
    lon : 2D numpy.ndarray
        Longitude values of the grid points.
    lat : 2D numpy.ndarray
        Latitude values of the grid points.
    z : 2D numpy.ndarray
        Elevation values of the grid points.
    rf : 2D numpy.ndarray
        Observed reduction factors with their corresponding longitude and latitude.
        Each row should be [longitude, latitude, reduction factor].
    L : float
        Correlation length determining the spatial influence of the observed reduction factors.

    Returns
    -------
    rfgrid : 2D numpy.ndarray
        The spatially varying reduction factor grid, with the same shape as the elevation data (z).

    Examples
    --------
    >>> lon = np.array([[...], [...]])
    >>> lat = np.array([[...], [...]])
    >>> z = np.array([[...], [...]])
    >>> rf = np.array([[lon1, lat1, rf1], [lon2, lat2, rf2], ...])
    >>> L = 10.0  # Example correlation length
    >>> rfgrid = red_fac_grid_FOL(lon, lat, z, rf, L)
    >>> print(rfgrid)
    """
    # Constants
    g = 0.01  # gamma, representing observational error
    Rt = 6371  # Earth radius in kilometers
    
    # Prepare the reduction factor data by removing any NaN values
    rf = rf[~np.isnan(rf[:, 2])]
    
    # Calculate background and anomalies in the reduction factors
    Vback = np.mean(rf[:, 2])  # Background value of the reduction factors
    Vo = rf - Vback  # Anomalies from the background

    # Prepare the land matrix using the DEM data
    land = np.column_stack((lon.flatten(), lat.flatten(), z.flatten()))
    
    # Calculate the Euclidean distances between land points and observed reduction factor points
    Va = land[:, :2]  # Coordinates of the land points
    DE = cdist(Va, Vo[:, :2], metric='euclidean')  # Distances matrix

    # Weight matrix for land points vs observed points using Gaussian function
    WD = np.exp(-np.square(DE) / (2 * np.square(L)))
    
    # Calculate the distances and weight matrix among observed points
    TE = cdist(Vo[:, :2], Vo[:, :2], metric='euclidean')
    WT = np.exp(-np.square(TE) / (2 * np.square(L)))

    # Gamma matrix to account for observational error
    gamma = g * np.eye(WT.shape[0])
    TTT = WT + gamma  # Combine observational weights with the gamma matrix
    TTT_inv = inv(TTT)  # Invert the combined matrix

    # Normalize the weights and calculate the interpolated reduction factors
    P = np.dot(WD, TTT_inv)
    P[P < 0] = 0  # Ensure no negative weights
    weightsum = np.sum(P, axis=1)
    weightsum[weightsum < 1] = 1
    Pnorm = P / weightsum[:, np.newaxis]  # Normalize the weights

    # Calculate the interpolated reduction factors and reshape back to the grid
    Vinterp = np.dot(Pnorm, Vo[:, 2]) + Vback
    rfgrid = Vinterp.reshape(z.shape)
    
    return rfgrid

def find_opt_L(SF, rf_lonlat):
    """
    Determines the optimal correlation length (L) for spatially varying reduction factors.

    This function iterates over a range of possible L values, calculating a spatially varying reduction factor grid for each.
    It then evaluates each L based on the difference between the interpolated reduction factors and the observed ones,
    choosing the L that minimizes this difference while considering the standard deviation across the grid as a measure of spatial variability.

    Parameters
    ----------
    SF : dict
        A dictionary containing the 'lat', 'lon', and 'z' keys representing latitude, longitude, and elevation data, respectively.
    rf_lonlat : numpy.ndarray
        An array of observed reduction factors with their corresponding longitude and latitude.
        Each row should be [longitude, latitude, reduction factor].

    Returns
    -------
    L_opt : float
        The optimal correlation length that minimizes the difference between interpolated and observed reduction factors,
        while considering spatial variability.

    Examples
    --------
    >>> SF = {
    ...     'lat': np.array([[...], [...]]),
    ...     'lon': np.array([[...], [...]]),
    ...     'z': np.array([[...], [...]])
    ... }
    >>> rf_lonlat = np.array([[lon1, lat1, rf1], [lon2, lat2, rf2], ...])
    >>> L_opt = find_opt_L(SF, rf_lonlat)
    >>> print(f"Optimal correlation length: {L_opt}")
    """
    # Start the timer
    start_time = time.time()
    
    # Define a range of L values to test
    L_test = np.arange(0.5, 100.5, 0.5)
    D2 = []  # Mean differences between interpolated and observed reduction factors
    STDs = []  # Standard deviations of interpolated reduction factors across the grid

    # Reduce spatial resolution for computational efficiency
    sires = [500, 500]  # Desired size
    sires[0] = min(SF['z'].shape[0], sires[0])
    sires[1] = min(SF['z'].shape[1], sires[1])

    # Resize the elevation, latitude, and longitude data
    lat2 = zoom(SF['lat'], (sires[0]/SF['lat'].shape[0], sires[1]/SF['lat'].shape[1]))
    lon2 = zoom(SF['lon'], (sires[0]/SF['lon'].shape[0], sires[1]/SF['lon'].shape[1]))
    z2 = zoom(SF['z'], (sires[0]/SF['z'].shape[0], sires[1]/SF['z'].shape[1]))

    # Find positions of observed reduction factors in the resized grid
    P = []
    xy = np.column_stack((lon2.ravel(), lat2.ravel()))
    
    for rf in rf_lonlat:
        distances = np.sqrt((xy[:, 0] - rf[0])**2 + (xy[:, 1] - rf[1])**2)
        P.append(np.argmin(distances))

    # Evaluate each L value
    for L in L_test:
        rfgrid = red_fac_grid_FOL(lon2, lat2, z2, rf_lonlat, L)
        
        # Differences between the interpolated grid and the values at the scalar positions
        D = rfgrid.ravel()[P] - rf_lonlat[:, 2]
        D2.append(np.mean(D))
        STDs.append(np.std(rfgrid))

    # Normalize the metrics to [0, 1] range and compute the ratio
    STDs_n = (STDs - np.min(STDs)) / (np.max(STDs) - np.min(STDs)) if np.max(STDs) != np.min(STDs) else np.zeros_like(STDs)
    D2_n = (D2 - np.min(D2)) / (np.max(D2) - np.min(D2)) if np.max(D2) != np.min(D2) else np.zeros_like(D2)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = STDs_n / D2_n
        # Where D2_n is 0, set ratio to 0 to avoid divide by zero issues
        ratio[np.isnan(ratio) | np.isinf(ratio)] = 0
        
    # Determine the optimal L value (Criteria for the optimal value of the interpolation length)
    L_opt_index = np.argmax(ratio)
    L_opt = L_test[L_opt_index]

    # End the timer and calculate the duration
    end_time = time.time()
    duration = end_time - start_time
    print(f"find_opt_L execution time: {duration} seconds")
    
    return L_opt

def spat_var_wl(lon, lat, z, tg):
    """
    Calculates spatially varying flood water levels based on given water level observations (tide gauges) and an elevation model (z).

    Parameters
    ----------
    lon : 2D numpy.ndarray
        Longitude values of the grid points.
    lat : 2D numpy.ndarray
        Latitude values of the grid points.
    z : 2D numpy.ndarray
        Elevation values of the grid points.
    tg : numpy.ndarray
        Array of water level observations with columns [longitude, latitude, water level].

    Returns
    -------
    fwl : 2D numpy.ndarray
        Interpolated flood water levels at the resolution of the input elevation model.
    L : float
        The optimal correlation length used in the interpolation.

    Examples
    --------
    >>> lon = np.array([[...], [...]])
    >>> lat = np.array([[...], [...]])
    >>> z = np.array([[...], [...]])
    >>> tg = np.array([[lon1, lat1, wl1], [lon2, lat2, wl2], ...])
    >>> fwl, L = spat_var_wl(lon, lat, z, tg)
    >>> print(f"Flood water levels:\n{fwl}")
    >>> print(f"Optimal correlation length: {L}")
    """
    # Start the timer
    start_time = time.time()
    
    # Reduce spatial resolution for computational efficiency
    sires = np.round(np.array(z.shape) / 10).astype(int)
    sires = np.clip(sires, a_min=None, a_max=5)  # Ensuring it does not exceed 500x500
    sires[0] = min(z.shape[0], sires[0])
    sires[1] = min(z.shape[1], sires[1])
    
    # Resize lat, lon, and z
    lat2 = resize(lat, sires, anti_aliasing=True)
    lon2 = resize(lon, sires, anti_aliasing=True)
    z2 = resize(z, sires, anti_aliasing=True)

    # Prepare data structure for find_opt_L function
    SF = {'lon': lon2, 'lat': lat2, 'z': z2}

    # Determine the optimal correlation length (L)
    L = 0.5

    # Parameters for interpolation
    g = 0.01  # Observation error coefficient
    Rt = 6371  # Earth radius in km
    r_lat = (2 * np.pi * Rt) / 360  # Convert degrees to radians for latitude calculations

    # Remove NaN values from tg data
    tg = tg[~np.isnan(tg[:, 2])]

    # Calculate distances and construct weight matrices
    land = np.column_stack((lon2.flatten(), lat2.flatten()))
    Vback = np.min(tg[:, 2])  # Background water level
    Vo = np.hstack((tg[:, :2], (tg[:, 2] - Vback).reshape(-1, 1)))  # Observations adjusted for background
    DE = cdist(land, Vo[:, :2], metric='euclidean')  # Distances from land points to observation points
    WD = np.exp(-DE**2 / (2 * L**2))  # Weight matrix for distances

    TE = cdist(Vo[:, :2], Vo[:, :2], metric='euclidean')  # Distances among observation points
    WT = np.exp(-TE**2 / (2 * L**2))  # Weight matrix among observations

    # Gamma matrix for observation errors and normalization
    gamma = g * np.eye(WT.shape[0])
    TTT = WT + gamma
    P = WD.dot(inv(TTT))
    P[P < 0] = 0  # Ensure non-negative weights
    weightsum = np.maximum(P.sum(axis=1), 1)
    Pnorm = P / weightsum[:, None]  # Normalize weights

    # Interpolate values back to grid and upsample to original resolution
    Vinterp = Pnorm.dot(Vo[:, 2]) + Vback
    fwl_low_res = Vinterp.reshape(sires)
    # Upsample fwl to original resolution
    fwl = griddata((lon2.flatten(), lat2.flatten()), fwl_low_res.flatten(), (lon, lat), method='cubic')

    # Replace NaN values with nearest non-NaN values
    fwl = pd.DataFrame(fwl).fillna(method='bfill', axis=0).fillna(method='ffill', axis=0).values
    fwl = pd.DataFrame(fwl).fillna(method='bfill', axis=1).fillna(method='ffill', axis=1).values

    # End the timer and calculate the duration
    end_time = time.time()
    duration = end_time - start_time
    print(f"spat_var_wl execution time: {duration} seconds")
    
    return fwl, L

def PyFlood_StaticFlooding(lon, lat, z, wl, wb):
    """
    Simulates flooding over a given topography using either spatially varying
    or uniform flood water levels.

    Parameters
    ----------
    lon : 2D numpy.ndarray
        Array of longitude coordinates.
    lat : 2D numpy.ndarray
        Array of latitude coordinates.
    z : 2D numpy.ndarray
        Array of elevation data.
    wl : numpy.ndarray or float
        Flood water levels. If spatially varying, wl should be an array with
        columns [longitude, latitude, water level]. If uniform, wl can be a scalar
        value or an array with a single row [lon, lat, water level].
    wb : tuple or list
        Coordinates [longitude, latitude] indicating a point on the sea or main water body.

    Returns
    -------
    SF : dict
        A dictionary containing the results of the flood simulation, including flood depth.
    fwl : numpy.ndarray or float
        The flood water level used in the simulation. It will match the input wl format.
    L : float or None
        The optimal correlation length used for spatially varying water level calculations,
        or None if the flood water level is uniform.

    Examples
    --------
    >>> lon = np.array([[...], [...]])
    >>> lat = np.array([[...], [...]])
    >>> z = np.array([[...], [...]])
    >>> wl = np.array([[lon1, lat1, wl1], [lon2, lat2, wl2], ...])
    >>> wb = (-95.3698, 29.7604)
    >>> SF, fwl, L = PyFlood_StaticFlooding(lon, lat, z, wl, wb)
    >>> print(SF['FloodDepth'])
    >>> print(fwl)
    >>> print(L)
    """
    start_time = time.time()
    
    # Determine if spatially varying flood water levels are being used based on the shape of wl
    if np.array(wl).shape[0] > 1:
        print('Calculating spatially varying water level...')
        # Calculate spatially varying flood water levels
        fwl, L = spat_var_wl(lon, lat, z, wl)
    elif np.array(wl).shape[0] == 1:
        print('Using uniform flood water level...')
        # For uniform flood water level, use the input value directly
        fwl = wl if np.isscalar(wl) else wl[-1]
        L = None  # Correlation length is not applicable for uniform water levels

    print('Applying static method to simulate flooding...')
    # Simulate flooding using the specified flood water levels
    SF = static_flooding(lon, lat, z, fwl, wb)

    # Record and print the execution time
    end_time = time.time()
    print(f"PyFlood_StaticFlooding execution time: {end_time - start_time} seconds")
    
    return SF, fwl, L

def calc_distance_to_coast(lon_flat, lat_flat, coastline):
    """
    Calculate the distance of each point in the flattened lon/lat arrays to the nearest coastline point.
    
    Parameters
    ----------
    lon_flat : ndarray
        Flattened array of longitude values.
    lat_flat : ndarray
        Flattened array of latitude values.
    coastline : ndarray
        Array of coastline points where each row is [longitude, latitude, elevation].
        
    Returns
    -------
    min_dists : ndarray
        Array of distances from each lon/lat point to the nearest coastline point in meters.

    Examples
    --------
    >>> lon_flat = np.array([...])
    >>> lat_flat = np.array([...])
    >>> coastline = np.array([[lon1, lat1, elev1], [lon2, lat2, elev2], ...])
    >>> distances = calc_distance_to_coast(lon_flat, lat_flat, coastline)
    >>> print(distances)
    """
    # Start the timer
    start_time = time.time()
    
    Rt = 6371  # Earth radius in km
    r_lat = (2 * np.pi * Rt) / 360
    
    # Initialize an array to store the minimum distances for each point
    min_dists = np.full(lon_flat.shape, np.inf)
    
    # Compute distances from each grid point to each coastline point and update min_dists
    for coast_point in coastline:
        r_lon = 2 * np.pi * Rt * np.cos(np.radians((lat_flat + coast_point[1]) / 2)) / 360
        dists = np.sqrt(((lon_flat - coast_point[0]) * r_lon) ** 2 + ((lat_flat - coast_point[1]) * r_lat) ** 2)
        min_dists = np.minimum(min_dists, dists)
    
    # End the timer and calculate the duration
    end_time = time.time()
    duration = end_time - start_time
    print(f"calc_distance_to_coast execution time: {duration} seconds")
    
    return min_dists * 1000  # Convert to meters and return



def create_raster_from_dem(lon, lat, z, output_filename, original_tif_path):
    """
    Generates a raster image from DEM data and saves it as a GeoTIFF file,
    aligning with the extent and cell size of the original GeoTIFF.

    Parameters
    ----------
    lon : numpy.ndarray
        A 2D array containing the longitude coordinates that define the regular grid of the DEM.
    lat : numpy.ndarray
        A 2D array containing the latitude coordinates that define the regular grid of the DEM.
    z : numpy.ndarray
        A 2D array containing the elevation data of the DEM.
    output_filename : str
        The path and filename for the output GeoTIFF raster file.
    original_tif_path : str
        Path to the original GeoTIFF file to extract metadata for alignment.

    Returns
    -------
    None

    This function ensures that the new raster matches the extent and cell size
    of the original GeoTIFF by using its metadata.
    """
    
    # Start the timer
    start_time = time.time()

    # Load metadata from the original GeoTIFF file
    with rasterio.open(original_tif_path) as src:
        original_transform = src.transform  # Transformation matrix of the original raster
        original_crs = src.crs  # Coordinate reference system of the original raster
        cell_size_x = src.res[0]  # Cell width in the x direction
        cell_size_y = src.res[1]  # Cell height in the y direction

    # Get dimensions for the new raster from the shape of the elevation data
    height, width = z.shape

    # Set the transform to the original transform if using the same resolution
    transform = original_transform

    # Use rasterio to write the DEM data to a new GeoTIFF file
    with rasterio.open(
        output_filename,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=str(z.dtype),
        crs=original_crs,
        transform=transform,
    ) as dst:
        dst.write(z, 1)  # Write the elevation data to the first band of the raster

    # End the timer and calculate the duration
    end_time = time.time()
    duration = end_time - start_time
    print(f"create_raster_from_dem execution time: {duration} seconds")

def red_fac_appl(lon, lat, z, wb, FloodDepth, rfgrid, prevcalc=None):
    """
    Applies a reduction factor to adjust flood depth values based on distance to the coastline.
    This can refine flood mapping by considering geographical features and elevation.

    Parameters
    ----------
    lon : numpy.ndarray
        2D array of longitude values.
    lat : numpy.ndarray
        2D array of latitude values.
    z : numpy.ndarray
        2D array of elevation data.
    wb : tuple
        Coordinates (longitude, latitude) of a known water body point.
    FloodDepth : numpy.ndarray
        2D array of flood depth values.
    rfgrid : numpy.ndarray
        2D array of reduction factors corresponding to each grid point.
    prevcalc : dict, optional
        Dictionary containing previously calculated 'coastline' and 'distmat_int' to avoid recalculation. Default is None.

    Returns
    -------
    SF_red : dict
        Dictionary containing adjusted flood depth values and other relevant outputs.

    Examples
    --------
    >>> lon = np.array([[...], [...]])
    >>> lat = np.array([[...], [...]])
    >>> z = np.array([[...], [...]])
    >>> wb = (-95.3698, 29.7604)
    >>> FloodDepth = np.array([[...], [...]])
    >>> rfgrid = np.array([[...], [...]])
    >>> SF_red = red_fac_appl(lon, lat, z, wb, FloodDepth, rfgrid)
    >>> print(SF_red['FloodDepthRF'])
    """
    # Start the timer
    start_time = time.time()
    
    if prevcalc is None:
        # 1. Reduce spatial resolution
        res = 50
        rres, cres = np.round(np.array(z.shape) / res).astype(int)

        lat2 = resize(lat, (rres, cres), anti_aliasing=True)
        lon2 = resize(lon, (rres, cres), anti_aliasing=True)
        z2 = resize(z, (rres, cres), anti_aliasing=True)

        # 2. Find coastline using the initial spatial resolution
        coastline, fBW = find_coastline(lon, lat, z, wb)

        # 3. Calculate distances from every point to the coast using DEM
        # Assuming coastline is correctly computed and is an array of points
        coastline_points = coastline[:, :2]  # Extracting just the lon/lat from coastline

        # Now, call the function with the flattened lon/lat arrays and the coastline points
        MinDists = calc_distance_to_coast(lon2.flatten(), lat2.flatten(), coastline_points)
        distmat = MinDists.reshape(z2.shape)  # Reshape if necessary for your use case

        # 4. Transfer distances to the original resolution
        distmat_int = griddata((lon2.flatten(), lat2.flatten()), distmat.flatten(), (lon, lat), method='cubic')
    else:
        coastline = prevcalc['coastline']
        distmat_int = prevcalc['distmat_int']

    # 5. Applying reduction factor
    wl_depth_red = FloodDepth + (rfgrid * distmat_int)
    wl_depth_red[wl_depth_red > 0] = np.nan

    # 6. Correction: remove flooded areas not connected to the main water body
    z_flood = z.copy()
    z_flood[~np.isnan(wl_depth_red)] = wl_depth_red[~np.isnan(wl_depth_red)]
    SF = static_flooding(lon, lat, z_flood, 0, wb)

    FloodDepthRF = SF['FloodDepthAndSea'].copy()
    FloodDepthRF[z <= 0] = np.nan

    # Prepare output
    FloodDepthVecRF = np.column_stack((lon.flatten(), lat.flatten(), FloodDepthRF.flatten()))
    FloodDepthVecRF = FloodDepthVecRF[~np.isnan(FloodDepthVecRF[:, 2])]

    SF_red = {
        'FloodDepthAndSea': SF['FloodDepthAndSea'],
        'FloodDepthVecRF': FloodDepthVecRF,
        'FloodDepthRF': FloodDepthRF,
        'prevcalc': {
            'distmat_int': distmat_int,
            'coastline': coastline
        }
    }

    # End the timer and calculate the duration
    end_time = time.time()
    duration = end_time - start_time
    print(f"red_fac_appl execution time: {duration} seconds")
    
    return SF_red

def find_coastline(lon, lat, z, wb):
    # Start the timer
    start_time = time.time()
    
    # 1. Identify sea body under present day conditions
    # Black and white bathymetry: water as 1, land as 0
    zBW = np.where(z > 0, np.nan, 1)  # Water
    zBW = np.where(np.isnan(zBW), 0, zBW)  # Land
    
    Lc = bwboundaries_sfm(zBW)
    
    # Find the polygon with the sea boundary (This requires calc_distance function in Python)
    mindist, sea_bound = calc_distance_matrix(wb, lon, lat)

    
    Lwb1 = Lc['objects'][sea_bound]
    Lwb2 = Lc['holes'][sea_bound]

    # Create matrices for polygons and holes
    sea = np.zeros_like(z)
    sea[Lc['objects'] == Lwb1] = 1

    holes = np.zeros_like(z)
    holes[Lc['holes'] == Lwb2] = 0
    
    sea[holes == 1] = 0
    
    # 2. Define coastline using edge detection on the binary sea mask
    coastline_mask = canny(sea)
    fBW = np.argwhere(coastline_mask)
    
    coastline = np.column_stack((lon.flatten(), lat.flatten(), z.flatten()))[coastline_mask.flatten()]
    
    # End the timer and calculate the duration
    end_time = time.time()
    duration = end_time - start_time
    print(f"find_coastline execution time: {duration} seconds")
    return coastline, fBW

def plot_coastline(coastline):
    """
    Plots coastline points with their elevations.

    Parameters
    ----------
    coastline : array-like, shape (M, 3)
        The coastline data, where each row is a point (longitude, latitude, elevation).

    Examples
    --------
    >>> coastline = np.array([[lon1, lat1, elev1], [lon2, lat2, elev2], ...])
    >>> plot_coastline(coastline)
    """
    # Splitting the coastline array into longitude, latitude, and elevation components
    x, y, z = coastline[:, 0], coastline[:, 1], coastline[:, 2]
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(x, y, c=z, cmap='jet', marker='o', edgecolor='k', s=0.01)
    plt.colorbar(scatter, label='Elevation (m)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Coastline Elevation')
    plt.show()

def apply_uniform_rf_to_model(lon, lat, z, wb, FloodDepth, distmat_int, uniform_rf):
    """
    Applies a uniform reduction factor to the flood depth values.

    Parameters
    ----------
    lon : numpy.ndarray
        2D array of longitude coordinates.
    lat : numpy.ndarray
        2D array of latitude coordinates.
    z : numpy.ndarray
        2D array of elevation data.
    wb : tuple
        Coordinates (longitude, latitude) of a known water body point.
    FloodDepth : numpy.ndarray
        2D array of flood depth values.
    distmat_int : numpy.ndarray
        2D array of distances from each point in the DEM to the coastline, interpolated to the original resolution.
    uniform_rf : float
        The uniform reduction factor to apply across all cells.

    Returns
    -------
    FloodDepthRF : numpy.ndarray
        2D array of adjusted flood depth values after applying the uniform reduction factor.

    Examples
    --------
    >>> lon = np.array([[...], [...]])
    >>> lat = np.array([[...], [...]])
    >>> z = np.array([[...], [...]])
    >>> wb = (-95.3698, 29.7604)
    >>> FloodDepth = np.array([[...], [...]])
    >>> distmat_int = np.array([[...], [...]])
    >>> uniform_rf = 0.5
    >>> FloodDepthRF = apply_uniform_rf_to_model(lon, lat, z, wb, FloodDepth, distmat_int, uniform_rf)
    >>> print(FloodDepthRF)
    """
    # Apply the uniform reduction factor across all cells
    wl_depth_red = FloodDepth + (uniform_rf * distmat_int)
    wl_depth_red[wl_depth_red > 0] = np.nan

    # Correction: remove flooded areas not connected to the main water body
    z_flood = z.copy()
    z_flood[~np.isnan(wl_depth_red)] = wl_depth_red[~np.isnan(wl_depth_red)]
    SF = static_flooding(lon, lat, z_flood, 0, wb)

    FloodDepthRF = SF['FloodDepthAndSea'].copy()
    FloodDepthRF[z <= 0] = np.nan

    return FloodDepthRF

def load_dem_data(dem_path):
    """
    Load DEM data from a .mat file.

    Parameters
    ----------
    dem_path : str
        Path to the .mat file containing DEM data.

    Returns
    -------
    z : numpy.ndarray
        The elevation data matrix.
    lon : numpy.ndarray
        The longitude coordinates of the DEM data points.
    lat : numpy.ndarray
        The latitude coordinates of the DEM data points.

    Examples
    --------
    >>> z, lon, lat = load_dem_data('DEM.mat')
    >>> print(z.shape)
    """
    # Start the timer
    start_time = time.time()
    
    # Load DEM data from the .mat file
    data = loadmat(dem_path)  # Load the data from the .mat file

    # Extract the elevation (z), longitude (lon), and latitude (lat) arrays from the loaded data
    z = data['z']  # Elevation data, representing the height at each point
    lon = data['lon']  # Longitude coordinates of the DEM data points
    lat = data['lat']  # Latitude coordinates of the DEM data points

    # Determine the dimensions of the elevation matrix to understand the size of the DEM
    rows, columns = z.shape  # 'shape' attribute returns the dimensions of the array

    # Print the dimensions of the DEM to provide an overview of its size
    print("DEM Dimensions:", rows, "rows x", columns, "columns")

    # End the timer and calculate the duration
    end_time = time.time()
    duration = end_time - start_time
    print(f"Function execution time: {duration} seconds")
    
    return z, lon, lat

def compute_interpolated_distance_matrix(lon, lat, z, coastline, res=50):
    """
    Computes and interpolates the distance matrix from each point in the DEM to the coastline.

    Parameters
    ----------
    lon : numpy.ndarray
        2D array of longitude coordinates.
    lat : numpy.ndarray
        2D array of latitude coordinates.
    z : numpy.ndarray
        2D array of elevation data.
    coastline : numpy.ndarray
        Array of coastline points where each row is [longitude, latitude, elevation].
    res : int, optional
        Resolution reduction factor for computational efficiency. Default is 50.

    Returns
    -------
    distmat_int : numpy.ndarray
        2D array of distances from each point in the DEM to the coastline, interpolated to the original resolution.

    Examples
    --------
    >>> lon = np.array([[...], [...]])
    >>> lat = np.array([[...], [...]])
    >>> z = np.array([[...], [...]])
    >>> coastline = np.array([[lon1, lat1, elev1], [lon2, lat2, elev2], ...])
    >>> distmat_int = compute_interpolated_distance_matrix(lon, lat, z, coastline)
    >>> print(distmat_int.shape)
    """
    # 1. Reduce spatial resolution
    rres, cres = np.round(np.array(z.shape) / res).astype(int)

    # Resize the latitude, longitude, and elevation matrices to the reduced resolution
    lat2 = resize(lat, (rres, cres), anti_aliasing=True)
    lon2 = resize(lon, (rres, cres), anti_aliasing=True)
    z2 = resize(z, (rres, cres), anti_aliasing=True)

    # 2. Extract the longitude and latitude from the coastline points
    coastline_points = coastline[:, :2]

    # 3. Calculate distances from every point to the coast using DEM
    MinDists = calc_distance_to_coast(lon2.flatten(), lat2.flatten(), coastline_points)
    distmat = MinDists.reshape(z2.shape)

    # 4. Transfer distances to the original resolution
    distmat_int = griddata((lon2.flatten(), lat2.flatten()), distmat.flatten(), (lon, lat), method='cubic')

    return distmat_int

# Load filtered data from the new CSV file
def load_filtered_csv(file_path):
    """
    Load filtered data from a CSV file and format it similarly to wl_s2.

    Parameters
    ----------
    file_path : str
        Path to the filtered CSV file.

    Returns
    -------
    filtered_stations : list of lists
        List of stations with format [longitude, latitude, predicted_value].
    """
    # Read the CSV
    df = pd.read_csv(file_path)
    
    # Format the data as list of lists
    filtered_stations = df[['x', 'y', 'Predicted']].values.tolist()
    
    return filtered_stations