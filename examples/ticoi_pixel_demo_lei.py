'''
Implementation of the Temporal Inversion using COmbination of displacements with Interpolation (TICOI) method
For one cube of pixel
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
from ticoi.core import inversion, visualisation, interpolation_post
from ticoi.cube_data_classxr import cube_data_class

# %%========================================================================= #
#                                    PARAMETERS                               #
# =========================================================================%% #

####  Selection of data
# cube_name = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_data"))}/ITS_LIVE_Lowell_Lower_test.nc'  # Path where the Sentinel-2 IGE cubes are stored
# path_save = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples", "results"))}'  # Path where to stored the results
cube_name = '/media/tristan/Data3/Hala_lake/Landsat8/Hala_lake_diaplacement_LS7_subset.nc'  # Path where the Sentinel-2 IGE cubes are stored
path_save = '/media/tristan/Data3/Hala_lake/Landsat8/ticoi_test/ewma-30/'  # Path where to stored the results

####  Point (pixel) where to carry on the computation
i, j = 396343, 4259420
proj = 'EPSG:32647'  # EPSG system of the coordinates given
# To select a specific period for the measurements, if you want to select all the dates put None, else give an inteval of dates ['aaaa-mm-dd', 'aaaa-mm-dd'] ([min, max])
dates_input = ['2000-01-01', '2014-12-31']
# To select certain temporal baselines in the dataset, if you want to select all the temporal put None, else give two integers [min, max] to form an interval in days
temp_baseline = None
sensor = None
# If you want confidence indicators ranging between 0 and 1, with 1 the lowest errors
conf = False
# Unit in m/y or m/d
unit = 'm/y'
# If None, all the data are included ; if an integer, the data with an error higher than this integer are removed ;
# if 'median_average', the data with a direction 45° away compared to the averaged direction are removed
delete_outliers = None

####  Inversion
# Variables to play with
smooth_method = 'gaussian' # Type of smoothing : 'gaussian', 'savgol', 'median', 'ewma'
coef = 200  # lambda : coef of the regularisation
# Type of regularisation : 1, 2,'1accelnotnull','regu01' (1: Tikhonov first order, 2: Tikhonov second order,
# '1accelnotnull': minization of the difference between the acceleration of the time series and acceleration computed on a moving average
regu = '1accelnotnull'
apriori_weight = True  # Add a weight in the first step of the inversion, True ou False
# Varibales which can stay stable for the moment
solver = 'LSMR_ini'  # Solver for the inversion : 'LSMR', 'LSMR_ini', 'LS', 'LS_bounded', 'LSQR'
detect_temporal_decorrelation = True  # Detect temporal decorrelation by setting a weight of 0 at the beginning at the first inversion to all observation with a temporal baseline larger than 200
result_quality = ['X_contribution']



####  Interpolation
option_interpol = 'spline'  # Type of interpolation : 'spline', 'nearest' or 'spline_smooth' for smoothing spline
interpolation_bas = 90  # Temporal sampling of the velocity time series
redundancy = 30

####  Visualization
visual = True  # Plot some results or not
verbose = False  # Print informations during the process or not
save = True  # Save the results or not
vmax = [0, 150]  # vmin and vmax of the legend
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
#                    DATA DOWNLOAD (ITS_LIVE AND IGE)                         #
# =========================================================================%% #

# %% Data download ITS_LIVE and IGE

start = time.time()

cube = cube_data_class()
cube.load(cube_name, pick_date=dates_input,
           proj=proj, pick_temp_bas=temp_baseline, conf=conf, pick_sensor=sensor,
           chunks={})
print(f'Time download cube {round((time.time() - start), 4)} sec')
print(f'Cube of dimesion (nz,nx,ny) : ({cube.nz},{cube.nx},{cube.ny}) ')

obs_filt = cube.preData_np(smooth_method=smooth_method, s_win=3, t_win=90, sigma=3, order=3, proj=proj, regu=regu,
                            delete_outliers=None, verbose=True,
                            velo_or_disp='velo')

start = time.time()
data, mean, dates_range = cube.load_pixel(i, j, proj=proj, interp='nearest', solver=solver,
                                                 visual=visual,
                                           regu=regu, rolling_mean=obs_filt)

cube2_date1 = cube.date1_().tolist()
cube2_date1.remove(np.min(cube2_date1))
start_date_interpol = np.min(cube2_date1)
last_date_interpol = np.max(cube.date2_())

date1 = None
print(date1)
print(f'Time download pixel {round((time.time() - start), 4)} sec')


# %% Inversion
start = time.time()
A, result, dataf = inversion(data, i, j, dates_range=dates_range, solver=solver, coef=coef, weight=apriori_weight,
                             visual=visual,
                             verbose=verbose, unit=unit,
                             conf=conf, regu=regu, mean=mean, visual_inversion=visual_inversion,
                             detect_temporal_decorrelation=detect_temporal_decorrelation,
                             linear_operator=None, result_quality=result_quality)
print(f'Time inversion {round((time.time() - start), 4)} sec')

if visual: visualisation(dataf, result, option_visual, path_save, A=A, dataf=dataf, unit=unit, show=True,
                         figsize=(12, 6))

if save: result.to_csv(f'{path_save}/ILF_result.csv')

# start = time.time()
if interpolation_bas == False: interpolation_bas = 1
start_date_interpol = np.min(np.min(cube.date2_()))
last_date_interpol = np.max(np.max(cube.date2_()))
dataf_lp = interpolation_post(result, interpolation_bas,
                              path_save, option_interpol=option_interpol,
                              first_date_interpol=start_date_interpol, last_date_interpol=last_date_interpol,
                              visual=visual, data=dataf, unit=unit, redundancy=redundancy,
                              result_quality=result_quality,
                              verbose=verbose, vmax=vmax)
if save: dataf_lp.to_csv(f'{path_save}/RLF_result.csv')
# print(f'Time interpolation {round((time.time() - start) / 60, 4)} min')

end = time.time()
print(f'{(end - start)} s')
