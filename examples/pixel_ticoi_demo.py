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
from ticoi.interpolation_functions import prepare_interpolation_date
from ticoi.cube_data_classxr import cube_data_class


# %%========================================================================= #
#                                    PARAMETERS                               #
# =========================================================================%% #

####  Selection of data

cube_name = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_data"))}/ITS_LIVE_Lowell_Lower_test.nc'  # Path where the Sentinel-2 IGE cubes are stored
path_save = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples", "results","pixel"))}/'  # Path where to stored the results

save = True
visual = True
proj = 'EPSG:3413'  # EPSG system of the given coordinates
i, j = -138.18069, 60.29076
vmax = [50, 170]  # vmin and vmax of the legend
# Visualisation options
option_visual = ['original_velocity_xy', 'original_magnitude',
                 'X_magnitude_zoom', 'X_magnitude', 'X_zoom', 'X',
                 'vv_quality', 'vxvy_quality',
                 'Residu_magnitude', 'Residu',
                 'X_z', 'Y_contribution',
                 'direction']
## ---------------------------- Loading parameters ------------------------- ##
load_kwargs = {'chunks': {},
               'conf': False, # If True, confidence indicators will be put between 0 and 1, with 1 the lowest errors
               'subset': None, # Subset of the data to be loaded ([xmin, xmax, ymin, ymax] or None)
               'buffer': None, # Area to be loaded around the pixel ([longitude, latitude, buffer size] or None)
               'pick_date': ['2015-01-01', '2023-01-01'], # Select dates ([min, max] or None to select all)
               'pick_sensor': None, # Select sensors (None to select all)
               'pick_temp_bas': None, # Select temporal baselines ([min, max] in days or None to select all)
               'proj': proj, # EPSG system of the given coordinates
               'verbose': False # Print information throughout the loading process
               }

## ----------------------- Data preparation parameters --------------------- ##
#For the folling part we advice the user to change only the following parameter, the other paramaters stored in a dictionary can be kept as it is for a first use
regu = '1accelnotnull' # Regularization method.s to be used (for each flag if flags is not None) : 1 minimize the acceleration, '1accelnotnull' minize the distance with an apriori on the acceleration computed over a spatio-temporal filtering of the cube
coef = 150  #Regularization coefficient.s to be used (for each flag if flags is not None)
delete_outlier = 'median_angle'
apriori_weight = False
interpolation_bas = 90

preData_kwargs = {'smooth_method': 'gaussian', # Smoothing method to be used to smooth the data in time ('gaussian', 'median', 'emwa', 'savgol')
                  's_win': 3, # Size of the spatial window
                  't_win': 90, # Time window size for 'ewma' smoothing
                  'sigma': 3, # Standard deviation for 'gaussian' filter
                  'order': 3, # Order of the smoothing function
                  'unit': 365, # 365 if the unit is m/y, 1 if the unit is m/d
                  'delete_outliers': delete_outlier, # Delete data with a poor quality indicator (if int), or with aberrant direction ('vvc_angle')
                  'flags': None, # Divide the data in several areas where different methods should be used
                  'regu': regu, # Regularization method.s to be used (for each flag if flags is not None) : 1 minimize the acceleration, '1accelnotnull' minize the distance with an apriori on the acceleration computed over a spatio-temporal filtering of the cube
                  'solver': 'LSMR_ini', # Solver for the inversion
                  'proj': proj, # EPSG system of the given coordinates
                  'velo_or_disp': 'velo', # Type of data contained in the data cube ('disp' for displacements, and 'velo' for velocities)
                  'verbose': True # Print information throughout the filtering process
                  }

## ---------------- Inversion and interpolation parameters ----------------- ##
inversion_kwargs = {'coef': coef, # Regularization coefficient.s to be used (for each flag if flags is not None)
                    'conf': False, # If True, confidence indicators are set between 0 and 1, with 1 the lowest errors
                    'interpolation_load_pixel': 'nearest', # Interpolation method used to load the pixel when it is not in the dataset
                    'iteration': True, # Allow the inversion process to make several iterations
                    'nb_max_iteration': 10, # Maximum number of iteration during the inversion process
                    'threshold_it': 0.1, # Threshold to test the stability of the results between each iteration, used to stop the process
                    'apriori_weight': apriori_weight, # If True, use apriori weights
                    'detect_temporal_decorrelation': True, # If True, the first inversion will use only velocity observations with small temporal baselines, to detect temporal decorelation
                    'linear_operator': None, # Perform the inversion using this specific linear operator
                    'interpolation': True, # If True, perform the temporal interpolation step of TICOI
                    'interval_output': 1,
                    'interpolation_bas': interpolation_bas, # Temporal baseline of the time series resulting from TICOI (after interpolation)
                    'option_interpol': 'spline', # Type of interpolation ('spline', 'spline_smooth', 'nearest')
                    'redundancy': 30, # Redundancy in the interpolated time series in number of days, no redundancy if None
                    'result_quality': 'X_contribution', # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')
                    'visual': False, # Plot results along the way
                    'path_save': path_save, # Path where to store the results
                    'verbose': False # Print information throughout TICOI processing
                    }


#Update of dictionary with common parameteres
for common_parameter in ['flags','proj','delete_outliers','regu','solver','unit']: inversion_kwargs[common_parameter] = preData_kwargs[common_parameter]

# Create a subfolder if it doesnt exist
if not os.path.exists(path_save):
    os.mkdir(path_save)

# %% ======================================================================== #
#                                DATA DOWNLOAD                                #
# =========================================================================%% #

start = [time.time()]

cube = cube_data_class()
cube.load(cube_name, **load_kwargs)

stop = [time.time()]
print(f'[ticoi_pixel_demo] Loading the data cube.s took {round((stop[0] - start[0]), 4)} s')
print(f'[ticoi_pixel_demo] Cube of dimension (nz,nx,ny) : ({cube.nz}, {cube.nx}, {cube.ny}) ')

start.append(time.time())
obs_filt = cube.filter_cube(**preData_kwargs)
data, mean, dates_range = cube.load_pixel(i, j, proj=proj, interp=inversion_kwargs['interpolation_load_pixel'], solver=inversion_kwargs['solver'], regu=regu,
                                          rolling_mean=obs_filt, visual=visual, verbose=load_kwargs['verbose'])

# Prepare interpolation dates
first_date_interpol,last_date_interpol = prepare_interpolation_date(cube)
inversion_kwargs.update({'first_date_interpol': first_date_interpol, 'last_date_interpol': last_date_interpol})

stop.append(time.time())
print(f'[ticoi_pixel_demo] Loading the pixel took {round((stop[1] - start[1]), 4)} s')


# %% ======================================================================== #
#                                 INVERSION                                   #
# =========================================================================%% #

start.append(time.time())
A, result, dataf = inversion_core(data, i, j, dates_range=dates_range, solver=inversion_kwargs['solver'], coef=coef, weight=apriori_weight,
                                  unit=inversion_kwargs['unit'], conf=inversion_kwargs['conf'], regu=regu, mean=mean,
                                  detect_temporal_decorrelation=inversion_kwargs['detect_temporal_decorrelation'],
                                  linear_operator=None, result_quality=inversion_kwargs['result_quality'],
                                  visual=visual, verbose=inversion_kwargs['verbose'])
from ticoi.visualization_functions import pixel_class
pixel_object = pixel_class()
# pixel_object.load(dataf, type_data = 'obs',dataformat='df',save=False,show=True,figsize = (10,6),unit='m/y')
# pixel_object.plot_vv()
# pixel_object.plot_vx_vy()
# pixel_object.plot_vx_vy_quality()
#
# pixel_object.load(result, type_data = 'invert',dataformat='df',save=False,show=True,figsize = (10,6))
# pixel_object.plot_vv()
# pixel_object.plot_vx_vy()
# pixel_object.plot_vv()
# stop.append(time.time())


# pixel_object.load_two_dataset([dataf,result],save=False,show=True)
# pixel_object.plot_vx_vy_overlayed()
# pixel_object.plot_vv_overlayed()
# pixel_object.plot_xcount_vv(pixel_object.data2)
# pixel_object.plot_xcount_vx_vy(pixel_object.data2)

print(f'[ticoi_pixel_demo] Inversion took {round((stop[2] - start[2]), 4)} s')

if visual: visualisation(dataf, result, option_visual, path_save, A=A, dataf=dataf, unit=preData_kwargs['unit'], show=True, figsize=(12, 6))
if save: result.to_csv(f'{path_save}/ILF_result.csv')

# %% ======================================================================== #
#                              INTERPOLATION                                  #
# =========================================================================%% #

start.append(time.time())
if interpolation_bas == False: interpolation_bas = 1
start_date_interpol = np.min(np.min(cube.date2_()))
last_date_interpol = np.max(np.max(cube.date2_()))
dataf_lp = interpolation_core(result, interpolation_bas,
                              path_save, option_interpol=inversion_kwargs['option_interpol'],
                              first_date_interpol=start_date_interpol, last_date_interpol=last_date_interpol,
                              visual=visual, data=dataf, unit=inversion_kwargs['unit'], redundancy=inversion_kwargs['redundancy'],
                              result_quality=inversion_kwargs['result_quality'],
                              verbose=inversion_kwargs['verbose'], vmax=vmax)

stop.append(time.time())
print(f'[ticoi_pixel_demo] Interpolation took {round((stop[3] - start[3]), 4)} s')

if save: dataf_lp.to_csv(f'{path_save}/RLF_result.csv')

print(f'[ticoi_pixel_demo] Overall processing took {round((stop[3] - start[0]), 4)} s')
