'''
Direct Implementation of the Temporal Inversion using COmbination of displacements with Interpolation (TICOI) method.
This implementation can be divided in three parts:
    - Data Download : Download the data cube eventually considering a given subset or buffer to limit its size.
    - Inversion & Interpolation: For each pixel of the cube, solving a system AX = Y to produce Irregular Leap Frog time series using the IRLS method, 
    then interpolate the obtained ILF time series to Regular LF time series using interpolation.
    - Results saving: Save the result in a new netCDF file.
/!\ This implementation requires important memory ressources for big datacubes as it loads and computes TICOI on the whole cube at the same time. 
For big data cubes, we recommand using the ticoi_cube_demo_block_process approach instead.

Author : Laurane Charrier
Reference:
    Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an optimal temporal sampling from displacement
    observation networks. IEEE Transactions on Geoscience and Remote Sensing.
    Charrier, L., Yan, Y., Colin Koeniguer, E., Mouginot, J., Millan, R., & Trouvé, E. (2022). Fusion of multi-temporal and multi-sensor ice velocity observations.
    ISPRS annals of the photogrammetry, remote sensing and spatial information sciences, 3, 311-318.
'''

import time
import itertools
import warnings
import geopandas
import os
import numpy as np
import pandas as pd
import xarray as xr

from joblib import Parallel, delayed
from pyproj import CRS
from tqdm import tqdm
from rasterio.features import rasterize
from osgeo import gdal, osr

from ticoi.cube_data_classxr import cube_data_class
from ticoi.core import process
from ticoi.other_functions import points_in_polygon


# %%========================================================================= #
#                                    PARAMETERS                               #
# =========================================================================%% #

warnings.filterwarnings("ignore")

## ------------------------------- Data selection -------------------------- ##
# cube_name = '/media/tristan/Data3/Hala_lake/Landsat8/Hala_lake_diaplacement_LS7_subset.nc'  # Path where the Sentinel-2 IGE cubes are stored
# path_save = '/media/tristan/Data3/Hala_lake/Landsat8/ticoi_test/cube-demo/'  # Path where to stored the results
cube_names = ['nathan/Donnees/Cubes_de_donnees/cubes_Sentinel_2/c_x01225_y03920_all_filt-multi.nc',
               'nathan/Donnees/Cubes_de_donnees/stack_median_pleiades_alllayers_2012-2022_modiflaurane.nc']
# cube_names = ['nathan/Donnees/Cubes_de_donnees/cubes_Sentinel_2/c_x01225_y03920_all_filt-multi.nc',    
#               'nathan/Donnees/Cubes_de_donnees/stack_median_pleiades_alllayers_2012-2022_modiflaurane.nc']
path_save = 'nathan/Tests_MB/'
name_save = 'test'
path_mask = None
path_mask = 'nathan/Tests_MB/Areas/Full_MB/mask/Full_MB.shp'
proj = 'EPSG:32632'  # EPSG system of the coordinates given
# To select a specific period for the measurements, if you want to select all the dates put None, 
# else give an inteval of dates ['aaaa-mm-dd', 'aaaa-mm-dd'] ([min, max])
dates_input = ['2015-01-01', '2023-01-01']
# To select certain temporal baselines in the dataset, if you want to select all the temporal baselines put None, 
# else give two integers [min, max] to form an interval in days
temp_baseline = None
sensor = None # Select a specific sensor 
conf = False # If you want confidence indicators ranging between 0 and 1, with 1 the lowest errors
unit = 'm/y' # Unit in m/y or m/d
# If None, all the data are included ; if an integer, the data with an error higher than this integer are removed ;
# if 'median_average', the data with a direction 45° away compared to the averaged direction are removed
delete_outliers = None
load_interp = 'nearest' # Interpolation used to select which data to use when the pixel is not exactly a pixel of the dataset

subset = None
# subset = [331200, 334550, 5080000, 5084400] # Bossons central part
subset = [332700, 335200, 5071200, 5073000]

## -------------------------------- Inversion ------------------------------ ##
# Type of regularisation : 1, 2,'1accelnotnull','regu01' (1: Tikhonov first order, 2: Tikhonov second order,
# '1accelnotnull': minimisation of the difference between the acceleration of the time series and acceleration computed on a moving average
regu = '1accelnotnull'
coef = 100  # Coefficient of the regularisation (lambda in the paper)
smooth_method = 'gaussian' # 
apriori_weight = True  # Add a weight in the first step of the inversion, True ou False
solver = 'LSMR_ini'  # Solver for the inversion : 'LSMR', 'LSMR_ini', 'LS', 'LS_bounded', 'LSQR'
detect_temporal_decorrelation = True  # Detect temporal decorrelation by setting a weight of 0 at the beginning at the first inversion to all observation with a temporal baseline larger than 200

## ----------------------------- Interpolation ----------------------------- ##
option_interpol = 'spline'  # Type of interpolation : 'spline', 'nearest' or 'spline_smooth' for smoothing spline
interpolation_bas = 30  # Temporal sampling of the velocity time series
redundancy = 5
# result_quality = None  # None or list of str, which can contain 'Norm_residual' to determine the L2 norm of the residuals 
# from the last inversion, 'X_contribution' to determine the number of Y observations which have contributed to estimate 
# each value in X (it corresponds to A.dot(weight))
result_quality = ['X_contribution']

## -------------------------------- Process -------------------------------- ##
nb_cpu = 12
verbose = False
save = True
interpolation = True
linear_operator = None
# Among ['orginal_velocity_xy','original_magnitude','error','vv_good_quality','vv_quality','vxvy_quality','X','X_vxvy','X_magnitude','X_magnitude_Zoom',
# 'X_filter','X_filterZoom','X_magnitude_filter','Y_contribution','Residu','Residu_magnitude']
option_visual = ['orginal_velocity_xy', 'original_magnitude', 'X_magnitude_zoom', 'X',
                 'X_vxvy', 'vv_quality']

unit = 365 if unit == 'm/y' else 1

if not os.path.exists(path_save):
    os.mkdir(path_save)


# %%========================================================================= #
#                                 DATA DOWNLOAD                               #
# =========================================================================%% #

start = [time.time()]

# Load the data
cube = cube_data_class()
cube.load(cube_names[0], pick_date=dates_input, proj=proj, pick_temp_bas=temp_baseline,
          pick_sensor=sensor, subset=subset, chunks={})

# Several cubes have to be merged together
if len(cube_names) > 1:
    for n in range(1, len(cube_names)):
        cube2 = cube_data_class()
        res = cube.ds['x'].values[1] - cube.ds['x'].values[0]
        cube2.load(cube_names[n], pick_date=dates_input, proj=proj, pick_temp_bas=temp_baseline, 
                    subset=[subset[0]-res, subset[1]+res, subset[2]-res, subset[3]+res] if subset is not None else None, 
                    conf=conf, pick_sensor=sensor, chunks={})
        # Align the new cube to the main one (interpolate the coordinate and/or reproject it)
        cube2 = cube.align_cube(cube2, reproj_vel=False, reproj_coord=True, interp_method='nearest')
        cube.merge_cube(cube2) # Merge the new cube to the main one
    del cube2

stop = [time.time()]
print(f'[ticoi_cube_demo] Cube of dimension (nz, nx, ny): ({cube.nz}, {cube.nx}, {cube.ny}) ')
print(f'[ticoi_cube_demo] Data loading took {round(stop[0] - start[0], 3)} s')

start.append(time.time())

mask = None
if path_mask is not None:  
    if path_mask[-3:] == 'shp': # Convert the shp file to an xarray dataset (rasterize the shapefile) 
        polygon = geopandas.read_file(path_mask).to_crs(epsg=int(proj.split(':')[1]))
        raster = rasterize([polygon.geometry[0]], out_shape=cube.ds.rio.shape, transform=cube.ds.rio.transform(), fill=0, dtype='int16')
        mask = xr.DataArray(data=raster.T, dims=['x', 'y'], coords=cube.ds[['x', 'y']].coords)
    else:
        mask = xr.open_dataarray(path_mask)
    mask.load()

obs_filt = cube.filter_cube(smooth_method=smooth_method, s_win=3, t_win=90, sigma=3, order=3, proj=proj, regu=regu,
                            delete_outliers=None, mask=mask, velo_or_disp='velo', verbose=True) # Compute the rolling mean

# Borders of the temporal interpolation
cube_date1 = cube.date1_().tolist()
cube_date1.remove(np.min(cube_date1))
merged = None
start_date_interpol = np.min(cube_date1)
last_date_interpol = np.max(cube.date2_())

stop.append(time.time())    
print(f'[ticoi_cube_demo] Filtering the cube took {round(stop[1] - start[1], 3)} s')


# %%========================================================================= #
#                                INITIALISATION                               #
# =========================================================================%% #

start.append(time.time())

# Some informations to be writen within the netCDF file
if save:
    sensor_array = np.unique(cube.ds['sensor'])
    sensor_strings = [str(sensor) for sensor in sensor_array]
    sensor = ', '.join(sensor_strings)
    if len(cube_names) > 1:
        source = f'Temporal inversion on cubes [{" ; ".join(cube_names)}] using TICOI with a selection of dates between {dates_input[0]} and {dates_input[1]}, with a selection of temporal baselines {temp_baseline}'
    else:
        source = f'Temporal inversion on cube {cube.filename} using TICOI with a selection of dates between {dates_input[0]} and {dates_input[1]}, with a selection of temporal baselines {temp_baseline}, and a temporal spacing of {redundancy} days'
    if apriori_weight:
        source += ' and apriori weight'
    source += f'. The Tikhonov coef is {coef}.'
    if interpolation:
        source += f' The interpolation option is {option_interpol}.'
        if interpolation_bas:
            source += f' The interpolation baseline is {interpolation_bas} days.'
        source += f' The temporal spacing is every {redundancy} days.'

EPSG = f'EPSG:{CRS(cube.ds.proj4).to_epsg()}'

stop.append(time.time())    
print(f'[ticoi_cube_demo] Initialisation took {round(stop[2] - start[2], 3)} s')


# %%========================================================================= #
#                                      TICOI                                  #
# =========================================================================%% #

nb_points = mask.where(mask != 0).size if mask is not None else len(cube.ds['x'].values) * len(cube.ds['y'].values)
print(f'[ticoi_cube_demo] Number of CPU : {nb_cpu}')
print(f'[ticoi_cube_demo] {nb_points} points to be computed within the given subset')

start.append(time.time())

# Initialize the progress bar
xy_values = itertools.product(cube.ds['x'].values, cube.ds['y'].values)
xy_values_tqdm = tqdm(xy_values, total=nb_points, mininterval=0.5)

# With parallelization
result = Parallel(n_jobs=nb_cpu, verbose=0)(
    delayed(process)(cube, i, j, solver, coef, apriori_weight, path_save, 
                      obs_filt=obs_filt, interpolation_load_pixel=load_interp,
                      first_date_interpol=start_date_interpol, last_date_interpol=last_date_interpol, proj=EPSG,
                      conf=conf, regu=regu, interpolation_bas=interpolation_bas, option_interpol=option_interpol, 
                      redundancy=redundancy, detect_temporal_decorrelation=detect_temporal_decorrelation, 
                      unit=unit, result_quality=result_quality, delete_outliers=delete_outliers, verbose=verbose)
    for i, j in xy_values_tqdm)

# Without parallelization
# result = []
# for i, j in xy_values_tqdm:
#     result.append(process(cube, i, j, solver, coef, apriori_weight, path_save, 
#                           obs_filt=obs_filt, interpolation_load_pixel=load_interp,
#                           first_date_interpol=start_date_interpol, last_date_interpol=last_date_interpol, proj=EPSG,
#                           conf=conf, regu=regu, interpolation_bas=interpolation_bas, option_interpol=option_interpol, 
#                           redundancy=redundancy, detect_temporal_decorrelation=detect_temporal_decorrelation, 
#                           unit=unit, result_quality=result_quality, delete_outliers=delete_outliers, verbose=verbose))

stop.append(time.time())
print(f'[ticoi_cube_demo] TICOI processing took {round(stop[3] - start[3], 3)} s')


# %%========================================================================= #
#                                WRITING RESULTS                              #
# =========================================================================%% #

start.append(time.time())

# Create a new netCDF file with TICOI results
cubenew = cube.write_result_ticoi(result, source, sensor, filename=name_save, savepath=path_save, 
                        result_quality=result_quality, verbose=verbose)

# Compute mean velocities as an example
mean_vv = np.sqrt(cubenew.ds['vx'].mean(dim='mid_date') ** 2 + cubenew.ds['vy'].mean(dim='mid_date') ** 2).to_numpy()
mean_vv = mean_vv.astype(np.float32)

driver = gdal.GetDriverByName('GTiff')
srs = osr.SpatialReference()
srs.SetWellKnownGeogCS('EPSG:32632')

resolution = int(cube.ds['x'].values[1] - cube.ds['x'].values[0])
dst_ds_temp = driver.Create(f'{path_save}mean_velocity.tiff', mean_vv.shape[1], mean_vv.shape[0], 1, gdal.GDT_Float32)
dst_ds_temp.SetGeoTransform([np.min(cube.ds['x'].values), resolution, 0, np.min(cube.ds['y'].values), 0, resolution])
dst_ds_temp.GetRasterBand(1).WriteArray(np.flip(mean_vv, axis=0))
dst_ds_temp.SetProjection(srs.ExportToWkt())

dst_ds_temp = None
driver = None

stop.append(time.time())
print(f'[ticoi_cube_demo] Writing cube to netCDF file took {round(stop[4] - start[4], 3)} s')
print(f'[ticoi_cube_demo] Overall processing took {round(stop[4] - start[0], 0)} s')