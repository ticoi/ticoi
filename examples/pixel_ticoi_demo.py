'''
Implementation of the Temporal Inversion using COmbination of displacements with Interpolation (TICOI) method
for one pixel.
Author: Laurane Charrier
Reference:
    Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an optimal temporal sampling from displacement
    observation networks. IEEE Transactions on Geoscience and Remote Sensing.
    Charrier, L., Yan, Y., Colin Koeniguer, E., Mouginot, J., Millan, R., & Trouvé, E. (2022). Fusion of multi-temporal and multi-sensor ice velocity observations.
    ISPRS annals of the photogrammetry, remote sensing and spatial information sciences, 3, 311-318.
'''

import time
import os
import numpy as np

from ticoi.core import inversion_core, visualisation, interpolation_core
from ticoi.cube_data_classxr import cube_data_class


# %%========================================================================= #
#                                    PARAMETERS                               #
# =========================================================================%% #

####  Selection of data
# Paths to the data cubes
# cube_names = [f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_data"))}/ITS_LIVE_Lowell_Lower_test.nc']
# path_save = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples", "results"))}'  # Path where to store the results
cube_names = ['nathan/Donnees/Cubes_de_donnees/cubes_Sentinel_2/c_x01225_y03675_all_filt-multi.nc', # Sentinel-2 cube
               'nathan/Donnees/Cubes_de_donnees/stack_median_pleiades_alllayers_2012-2022_modiflaurane.nc'] # Pleiade cube
path_save = 'nathan/Tests_MB/useless/'
i, j = 332100, 5080350 # Point (pixel) where to carry on the computation
proj = 'EPSG:32632' # Projection of the given coordinates
buffer_size = 500 # Size of the buffer to be loaded around the pixel
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

####  Inversion
# Type of regularisation : 1, 2,'1accelnotnull','regu01' (1: Tikhonov first order, 2: Tikhonov second order,
# '1accelnotnull': minimisation of the difference between the acceleration of the time series and acceleration computed on a moving average
regu = '1accelnotnull'
coef = 100  # lambda : coef of the regularisation
apriori_weight = True  # Add a weight in the first step of the inversion, True ou False
solver = 'LSMR_ini'  # Solver for the inversion : 'LSMR', 'LSMR_ini', 'LS', 'LS_bounded', 'LSQR'
detect_temporal_decorrelation = True  # Detect temporal decorrelation by setting a weight of 0 at the beginning at the first inversion to all observation with a temporal baseline larger than 200

####  Interpolation
option_interpol = 'spline'  # Type of interpolation : 'spline', 'nearest' or 'spline_smooth' for smoothing spline
interpolation_bas = 30  # Temporal sampling of the velocity time series
redundancy = 5
result_quality = None

####  Visualization
visual = True  # Plot some results or not
verbose = False  # Print informations during the process or not
save = True  # Save the results or not
vmax = [50, 170]  # vmin and vmax of the legend
visual_inversion = False  # Visualize the different iterations of the inversion
# Visualisation options
option_visual = ['original_velocity_xy', 'original_magnitude',
                 'X_magnitude_zoom', 'X_magnitude', 'X_zoom', 'X',
                 'vv_quality', 'vxvy_quality',
                 'Residu_magnitude', 'Residu',
                 'X_z', 'Y_contribution',
                 'direction']

# Create a subfolder if it doesnt exist
if not os.path.exists(path_save):
    os.mkdir(path_save)

unit = 365 if unit == 'm/y' else 1


# %% ======================================================================== #
#                                DATA DOWNLOAD                                #
# =========================================================================%% #

start = [time.time()]

cube = cube_data_class()
cube.load(cube_names[0], pick_date=dates_input, proj=proj, pick_temp_bas=temp_baseline, 
          buffer=[i, j, buffer_size], conf=conf, pick_sensor=sensor, chunks={})

# Several cubes have to be merged together
if len(cube_names) > 1:
    for n in range(1, len(cube_names)):
        cube2 = cube_data_class()
        cube2.load(cube_names[n], pick_date=dates_input, proj=proj, pick_temp_bas=temp_baseline, 
                   buffer=[i, j, buffer_size], conf=conf, pick_sensor=sensor, chunks={})
        cube2 = cube.align_cube(cube2, reproj_vel=False, reproj_coord=True, interp_method='nearest')
        cube.merge_cube(cube2)

stop = [time.time()]
print(f'[ticoi_pixel_demo] Loading the data cube.s took {round((stop[0] - start[0]), 4)} s')
print(f'[ticoi_pixel_demo] Cube of dimension (nz,nx,ny) : ({cube.nz}, {cube.nx}, {cube.ny}) ')

start.append(time.time())

obs_filt = cube.filter_cube(s_win=3, t_win=90, unit=unit, proj=proj, regu=regu, delete_outliers=delete_outliers, 
                             velo_or_disp='velo', verbose=verbose)
data, mean, dates_range = cube.load_pixel(i, j, proj=proj, interp=load_interp, solver=solver, regu=regu, 
                                          rolling_mean=obs_filt, visual=visual, verbose=verbose)

cube2_date1 = cube.date1_().tolist()
cube2_date1.remove(np.min(cube2_date1))
start_date_interpol = np.min(cube2_date1)
last_date_interpol = np.max(cube.date2_())

date1 = None

stop.append(time.time())
print(f'[ticoi_pixel_demo] Loading the pixel took {round((stop[1] - start[1]), 4)} s')


# %% ======================================================================== #
#                                 INVERSION                                   #
# =========================================================================%% #

start.append(time.time())
A, result, dataf = inversion_core(data, i, j, dates_range=dates_range, solver=solver, coef=coef, weight=apriori_weight,
                                  unit=unit, conf=conf, regu=regu, mean=mean,
                                  detect_temporal_decorrelation=detect_temporal_decorrelation,
                                  linear_operator=None, result_quality=result_quality, 
                                  visual=visual, verbose=verbose)

stop.append(time.time())
print(f'[ticoi_pixel_demo] Inversion took {round((stop[2] - start[2]), 4)} s')

if visual: visualisation(dataf, result, option_visual, path_save, A=A, dataf=dataf, unit=unit, show=True, figsize=(12, 6))
if save: result.to_csv(f'{path_save}/ILF_result.csv')


# %% ======================================================================== #
#                              INTERPOLATION                                  #
# =========================================================================%% #

start.append(time.time())

if interpolation_bas == False: interpolation_bas = 1
start_date_interpol = np.min(np.min(cube.date2_()))
last_date_interpol = np.max(np.max(cube.date2_()))
dataf_lp = interpolation_core(result, interpolation_bas,
                              path_save, option_interpol=option_interpol,
                              first_date_interpol=start_date_interpol, last_date_interpol=last_date_interpol,
                              visual=visual, data=dataf, unit=unit, redundancy=redundancy,
                              result_quality=result_quality,
                              verbose=verbose, vmax=vmax)

stop.append(time.time())
print(f'[ticoi_pixel_demo] Interpolation took {round((stop[3] - start[3]), 4)} s')

if save: dataf_lp.to_csv(f'{path_save}/RLF_result.csv')

print(f'[ticoi_pixel_demo] Overall processing took {round((stop[3] - start[0]), 4)} s')
