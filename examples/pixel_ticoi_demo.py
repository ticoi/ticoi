'''
Implementation of the Temporal Inversion using COmbination of displacements with Interpolation (TICOI) method for one pixel.
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

## ------------------------------ Data selection --------------------------- ##
# Path.s to the data cube.s (can be a list of str to merge several cubes, or a single str)
cube_name = 'test_data/Alps_Mont-Blanc_Argentiere_example.nc'
path_save = 'examples/results/pixel/' # Path where to store the results
proj = 'EPSG:32632'  # EPSG system of the given coordinates

i, j = 342537.1,5092253.3 # Point (pixel) where to carry on the computation

## --------------------------- Main parameters ----------------------------- ##
regu = '1accelnotnull' # Regularization method to be used
coef = 100 # Regularization coefficient to be used
solver = 'LSMR_ini' # Solver for the inversion
unit = 365 # 1 for m/d, 365 for m/y
result_quality = 'X_contribution' # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')

## ----------------------- Visualization parameters ------------------------ ##
verbose = False # Print information throughout TICOI processing
visual = True # Plot informations along the way
save = True # Save the results or not
# Visualisation options
option_visual = ['original_velocity_xy', 'original_magnitude',
                 'X_magnitude_zoom', 'X_magnitude', 'X_zoom', 'X',
                 'vv_quality', 'vxvy_quality',
                 'Residu_magnitude', 'Residu',
                 'X_z', 'Y_contribution',
                 'direction']
vmax = [False, False] # Vertical limits for the plots

## ---------------------------- Loading parameters ------------------------- ##
load_kwargs = {'chunks': {}, 
               'conf': False, # If True, confidence indicators will be put between 0 and 1, with 1 the lowest errors
               'buffer': [i, j, 500], # Area to be loaded around the pixel ([longitude, latitude, buffer size] or None)
               'pick_date': ['2015-01-01', '2023-01-01'], # Select dates ([min, max] or None to select all)
               'pick_sensor': None, # Select sensors (None to select all)
               'pick_temp_bas': None, # Select temporal baselines ([min, max] in days or None to select all)
               'proj': proj, # EPSG system of the given coordinates
               'verbose': verbose} # Print information throughout the loading process 

## ----------------------- Data preparation parameters --------------------- ##
preData_kwargs = {'smooth_method': 'gaussian', # Smoothing method to be used to smooth the data in time ('gaussian', 'median', 'emwa', 'savgol')
                  's_win': 3, # Size of the spatial window
                  't_win': 90, # Time window size for 'ewma' smoothing
                  'sigma': 3, # Standard deviation for 'gaussian' filter
                  'order': 3, # Order of the smoothing function
                  'unit': unit, # 365 if the unit is m/y, 1 if the unit is m/d
                  'delete_outliers': 'vvc_angle', # Delete data with a poor quality indicator (if int), or with aberrant direction ('vvc_angle') 
                  'regu': regu, # Regularization method to be used
                  'solver': solver, # Solver for the inversion
                  'proj': proj, # EPSG system of the given coordinates
                  'velo_or_disp': 'velo', # Type of data contained in the data cube ('disp' for displacements, and 'velo' for velocities)
                  'verbose': verbose} # Print information throughout the filtering process 
                  
## ---------------- Parameters for the pixel loading part ------------------ ##
load_pixel_kwargs = {'regu': regu, # Regularization method to be used
                     'coef': coef,
                     'solver': solver, # Solver for the inversion
                     'proj': proj, # EPSG system of the given coordinates
                     'interp': 'nearest', # Interpolation method used to load the pixel when it is not in the dataset
                     'visual': visual, # Plot results along the way
                     'verbose':verbose} # Print information throughout TICOI processing
                     
## --------------------------- Inversion parameters ------------------------ ##
inversion_kwargs = {'regu': regu, # Regularization method to be used
                    'coef': coef, # Regularization coefficient to be used
                    'solver': solver, # Solver for the inversion
                    'conf': False, # If True, confidence indicators are set between 0 and 1, with 1 the lowest errors
                    'unit': unit, # 365 if the unit is m/y, 1 if the unit is m/d

                    'iteration': True, # Allow the inversion process to make several iterations
                    'nb_max_iteration': 10, # Maximum number of iteration during the inversion process
                    'threshold_it': 0.1, # Threshold to test the stability of the results between each iteration, used to stop the process
                    'weight': True, # If True, use apriori weights
                    'detect_temporal_decorrelation': True, # If True, the first inversion will use only velocity observations with small temporal baselines, to detect temporal decorelation
                    'linear_operator': None, # Perform the inversion using this specific linear operator
                    'result_quality': result_quality, # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')
                    
                    'visual': visual, # Plot results along the way
                    'verbose': verbose} # Print information throughout TICOI processing
                    
## ----------------------- Interpolation parameters ------------------------ ##
interpolation_kwargs = {'interval_output': 30, # Temporal baseline of the time series resulting from TICOI (after interpolation)
                        'redundancy': 5, # Redundancy in the interpolated time series in number of days, no redundancy if None
                        'option_interpol': 'spline', # Type of interpolation ('spline', 'spline_smooth', 'nearest')                
                        'result_quality': result_quality,  # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')
                        'unit': unit, # 365 if the unit is m/y, 1 if the unit is m/d
                        
                        'visual': visual, # Plot results along the way
                        'vmax': vmax, # vmin and vmax of the legend
                        'verbose': verbose} # Print information throughout TICOI processing

# Create a subfolder if it doesnt exist
if not os.path.exists(path_save):
    os.mkdir(path_save)

if type(cube_name) == str:
    cube_name = [cube_name]

# %% ======================================================================== #
#                                DATA LOADING                                 #
# =========================================================================%% #

start = [time.time()]

# Load the main cube
cube = cube_data_class()
cube.load(cube_name[0], **load_kwargs)

# Several cubes have to be merged together
if len(cube_name) > 1:
    for n in range(1, len(cube_name)):
        cube2 = cube_data_class()
        cube2.load(cube_name[n], **load_kwargs)
        cube2 = cube.align_cube(cube2, reproj_vel=False, reproj_coord=True, interp_method='nearest')
        cube.merge_cube(cube2)

stop = [time.time()]
print(f'[Data loading] Loading the data cube.s took {round((stop[0] - start[0]), 4)} s')
print(f'[Data loading] Cube of dimension (nz,nx,ny) : ({cube.nz}, {cube.nx}, {cube.ny}) ')

start.append(time.time())

# Filter the cube (compute rolling_mean for regu=1accelnotnull)
obs_filt = cube.filter_cube(**preData_kwargs)
# Load pixel data
data, mean, dates_range = cube.load_pixel(i, j, rolling_mean=obs_filt, **load_pixel_kwargs)

cube2_date1 = cube.date1_().tolist()
cube2_date1.remove(np.min(cube2_date1))
start_date_interpol = np.min(cube2_date1)
last_date_interpol = np.max(cube.date2_())

stop.append(time.time())
print(f'[Data loading] Loading the pixel took {round((stop[1] - start[1]), 4)} s')


# %% ======================================================================== #
#                                 INVERSION                                   #
# =========================================================================%% #

start.append(time.time())

# Proceed to inversion
A, result, dataf = inversion_core(data, i, j, dates_range=dates_range, mean=mean, **inversion_kwargs)

stop.append(time.time())
print(f'[Inversion] Inversion took {round((stop[2] - start[2]), 4)} s')

# Plot the results of the inversion
if visual: visualisation(dataf, result, option_visual, path_save, A=A, dataf=dataf, unit=inversion_kwargs['unit'], 
                         show=True, figsize=(12, 6))
if save: result.to_csv(f'{path_save}/ILF_result.csv')


# %% ======================================================================== #
#                              INTERPOLATION                                  #
# =========================================================================%% #

start.append(time.time())

if interpolation_kwargs['interval_output'] == False: 
    interpolation_kwargs['interval_output'] = 1
start_date_interpol = np.min(np.min(cube.date2_()))
last_date_interpol = np.max(np.max(cube.date2_()))

# Proceed to interpolation
dataf_lp = interpolation_core(result, path_save=path_save, data=dataf, first_date_interpol=start_date_interpol, 
                              last_date_interpol=last_date_interpol, **interpolation_kwargs)

stop.append(time.time())
print(f'[Interpolation] Interpolation took {round((stop[3] - start[3]), 4)} s')

if save: 
    dataf_lp.to_csv(f'{path_save}/RLF_result.csv')

print(f'[Overall] Overall processing took {round((stop[3] - start[0]), 4)} s')