'''
Implementation of the Temporal Inversion using COmbination of displacements with Interpolation (TICOI) method
The implementation can be divided in three parts:
    - Data Download : find and download the cube corresponding for the area pixel we want to study,
    - Inversion & Interpolation: for each pixel of the cube: solving a system AX = Y to produce Irregular Leap Frog time series using the IRLS method, then interpolate the obtained ILF time series to Regular LF time series using interpolation
    - Results saving: save the result in a new netdf file

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
import os
import numpy as np

from joblib import Parallel, delayed
from pyproj import CRS
from tqdm import tqdm

from ticoi.cube_data_classxr import cube_data_class
from ticoi.core import process

# %%========================================================================= #
#                                    PARAMETERS                               #
# =========================================================================%% #

warnings.filterwarnings("ignore")

####  Selection of data
# cube_name = '/media/tristan/Data3/Hala_lake/Landsat8/Hala_lake_diaplacement_LS7_subset.nc'  # Path where the Sentinel-2 IGE cubes are stored
# path_save = '/media/tristan/Data3/Hala_lake/Landsat8/ticoi_test/cube-demo/'  # Path where to stored the results
cube_name = 'nathan/Donnees/Cubes_de_donnees/cubes_Sentinel_2/c_x01225_y03675_all_filt-multi.nc'
path_save = 'nathan/Tests_MB/'
name_save = 'test'
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

subset = [333615, 334529, 5079318, 5081236] # Bossons upper central part (741 points)

####  Inversion
# Type of regularisation : 1, 2,'1accelnotnull','regu01' (1: Tikhonov first order, 2: Tikhonov second order,
# '1accelnotnull': minimisation of the difference between the acceleration of the time series and acceleration computed on a moving average
smooth_method = 'gaussian'
regu = '1accelnotnull'
coef = 100  # lambda : coef of the regularisation
apriori_weight = True  # Add a weight in the first step of the inversion, True ou False
solver = 'LSMR_ini'  # Solver for the inversion : 'LSMR', 'LSMR_ini', 'LS', 'LS_bounded', 'LSQR'
detect_temporal_decorrelation = True  # Detect temporal decorrelation by setting a weight of 0 at the beginning at the first inversion to all observation with a temporal baseline larger than 200

####  Interpolation
option_interpol = 'spline'  # Type of interpolation : 'spline', 'nearest' or 'spline_smooth' for smoothing spline
interpolation_bas = 30  # Temporal sampling of the velocity time series
redundancy = 5
# result_quality = None  # None or list of str, which can contain 'Norm_residual' to determine the L2 norm of the residuals 
# from the last inversion, 'X_contribution' to determine the number of Y observations which have contributed to estimate 
# each value in X (it corresponds to A.dot(weight))
result_quality = ['X_contribution']

####  Process
nb_cpu = 12
verbose = False
save = True
interpolation = True
linear_operator = None
option_visual = ['orginal_velocity_xy', 'original_magnitude', 'X_magnitude_zoom',
                 'X',
                 'X_vxvy',
                 'vv_quality']  # ['orginal_velocity_xy','original_magnitude','error','vv_good_quality','vv_quality','vxvy_quality','X','X_vxvy','X_magnitude','X_magnitude_Zoom','X_filter','X_filterZoom','X_magnitude_filter','Y_contribution','Residu','Residu_magnitude']

unit = 365 if unit == 'm/y' else 1

if not os.path.exists(path_save):
    os.mkdir(path_save)


# %%========================================================================= #
#                                 DATA DOWNLOAD                               #
# =========================================================================%% #

start = [time.time()]

cube = cube_data_class()
cube.load(cube_name, pick_date=dates_input, proj=proj, pick_temp_bas=temp_baseline,
          pick_sensor=sensor, subset=subset, chunks={})

stop = [time.time()]
print(f'[ticoi_cube_demo] Cube of dimension (nz, nx, ny) Sentinel-2 : ({cube.nz}, {cube.nx}, {cube.ny}) ')
print(f'[ticoi_cube_demo] Data loading took {round(stop[0] - start[0], 3)} s')


start.append(time.time())
obs_filt = cube.filter_cube(smooth_method=smooth_method, s_win=3, t_win=90, sigma=3, order=3, proj=proj, regu=regu,
                            delete_outliers=None, verbose=False, velo_or_disp='velo')

stop.append(time.time())    
print(f'[ticoi_cube_demo] Filtering the cube took {round(stop[1] - start[1], 3)} s')

cube_date1 = cube.date1_().tolist()
cube_date1.remove(np.min(cube_date1))
merged = None
start_date_interpol = np.min(cube_date1)
last_date_interpol = np.max(cube.date2_())


# %%========================================================================= #
#                                INITIALISATION                               #
# =========================================================================%% #

start.append(time.time())

if save:
    sensor_array = np.unique(cube.ds['sensor'])

    sensor_strings = [str(sensor) for sensor in sensor_array]
    sensor = ', '.join(sensor_strings)
    if merged is None:
        source = f'Temporal inversion on cube {cube.filename} using TICOI with a selection of the dates between: {dates_input}, with a selection of the temporal baselines {temp_baseline}'
    else:
        source = f'Temporal inversion on cube {cube.filename} using TICOI with a selection of the dates: {dates_input}, with a selection of the baseline {temp_baseline}, and a temporal spacing every {redundancy} days '
    if apriori_weight:
        source += ' and apriori weight'
    source += f'. The Tikhonov coef is: {coef}.'
    if interpolation:
        source += f'The interpolation option is: {option_interpol}.'
        if interpolation_bas:
            source += f'The interpolation baseline is: {interpolation_bas} days.'
        source += f'The temporal spacing every {redundancy} days.'

EPSG = f'EPSG:{CRS(cube.ds.proj4).to_epsg()}'

stop.append(time.time())    
print(f'[ticoi_cube_demo] Initialisation took {round(stop[2] - start[2], 3)} s')


# %%========================================================================= #
#                                      TICOI                                  #
# =========================================================================%% #

nb_points = len(cube.ds['x'].values) * len(cube.ds['y'].values)
print(f'[ticoi_cube_demo] Number of CPU : {nb_cpu}')
print(f'[ticoi_cube_demo] {nb_points} points to be computed within the given subset')

start.append(time.time())

xy_values = itertools.product(cube.ds['x'].values, cube.ds['y'].values)
xy_values_tqdm = tqdm(xy_values, total=len(cube.ds['x'].values)*len(cube.ds['y'].values), mininterval=0.1)

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

cube.write_result_ticoi(result, source, sensor, filename=name_save, savepath=path_save, 
                        result_quality=result_quality, verbose=verbose)

stop.append(time.time())
print(f'[ticoi_cube_demo] Writing cube to netCDF file took {round(stop[4] - start[4], 3)} s')
print(f'[ticoi_cube_demo] Overall processing took {round(stop[4] - start[0], 0)} s')
