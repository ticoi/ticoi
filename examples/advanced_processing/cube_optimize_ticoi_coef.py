'''
Coefficient optimization of the TICOI post-processing method, according to "ground truth" given data (GPS, more
precise satellitarian data...). A range of coefficients is tested for a given regularisation method, by computing
the RMSE between TICOI results for the tested coefficient, interpolated to the ground truth dates, and compared
to those ground truth dates using the Root Mean Square Error (RMSE).
This code computes a RMSE-coefficient curve for every pixel of a given subset (or whole data cube).
'''

import os
import time
import itertools
import warnings
import asyncio
import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from tqdm import tqdm

from ticoi.core import chunk_to_block, load_block
from ticoi.cube_data_classxr import cube_data_class
from ticoi.other_functions import optimize_coef


# %%========================================================================= #
#                                    PARAMETERS                               #
# =========================================================================%% #
    
warnings.filterwarnings("ignore")

## ---------------------------- Data selection ------------------------- ##
# List of the paths where the data cubes are stored
cube_name = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "test_data"))}/Alps_Mont-Blanc_Argentiere_example.nc'
mask_file = 'nathan/Tests_MB/Areas/Full_MB/mask/Full_MB.shp' # Path where the mask file is stored
mask_file = None
path_save = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))}/optimize_coef' # Path where to store the results
proj = 'EPSG:32632'  # EPSG system of the given coordinates

## ------------------------- Main parameters --------------------------- ##
regu = '1accelnotnull' # Regularization method to be used
solver = 'LSMR_ini' # Solver for the inversion
unit = 365 # 1 for m/d, 365 for m/y
result_quality = 'X_contribution' # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')

## --------------------- Optimization parameters ----------------------- ##
# 'block_process' or 'direct_process' 
optimization_process = 'block_process'
# Path to the "ground truth" cube used to optimize the regularisation
cube_gt_name = 'nathan/Donnees/Cubes_de_donnees/stack_median_pleiades_alllayers_2012-2022_modiflaurane.nc' 
# Specify the coefficients you want to test
coefs = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 240, 280, 320, 360, 400, 450, 500, 550, 600, 700, 800, 900, 1000]
coef_min = 10 # If coefs=None, start point of the range of coefs to be tested 
coef_max = 1000 # If coefs=None, stop point of the range of coefs to be tested
step = 10 # If coefs=None, step for the range of coefs to be tested
stats = True # Compute some statistics on raw data and GT data
# Visualisation options
save = True
plot_them_all = True

## ------------------------ Loading parameters ------------------------- ##
load_kwargs = {'chunks': {}, 
               'conf': False, # If True, confidence indicators will be put between 0 and 1, with 1 the lowest errors
               'subset': [343448.4, 343779.4, 5091223.5, 5091454.3], # Area to be loaded around the pixel ([longitude, latitude, buffer size] or None)
               'pick_date': ['2015-01-01', '2023-01-01'], # Select dates ([min, max] or None to select all)
               'pick_sensor': None, # Select sensors (None to select all)
               'pick_temp_bas': None, # Select temporal baselines ([min, max] in days or None to select all)
               'proj': proj, # EPSG system of the given coordinates
               'mask': mask_file, # Path to mask file (.shp file) to mask some of the data on cube
               'verbose': False} # Print information throughout the loading process 

## ------------------- Data preparation parameters --------------------- ##
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
                  'verbose': False} # Print information throughout the filtering process 
                  
## -------------- Parameters for the pixel loading part ---------------- ##
load_pixel_kwargs = {'regu': regu, # Regularization method to be used
                     'solver': solver, # Solver for the inversion
                     'proj': proj, # EPSG system of the given coordinates
                     'interp': 'nearest', # Interpolation method used to load the pixel when it is not in the dataset
                     'visual': False} # Plot results along the way
                     
## ----------------------- Inversion parameters ------------------------ ##
inversion_kwargs = {'regu': regu, # Regularization method to be used
                    'solver': solver, # Solver for the inversion
                    'conf': False, # If True, confidence indicators are set between 0 and 1, with 1 the lowest errors
                    'unit': unit, # 365 if the unit is m/y, 1 if the unit is m/d

                    'iteration': True, # Allow the inversion process to make several iterations
                    'nb_max_iteration': 10, # Maximum number of iteration during the inversion process
                    'threshold_it': 0.1, # Threshold to test the stability of the results between each iteration, used to stop the process
                    'apriori_weight': True, # If True, use apriori weights
                    'detect_temporal_decorrelation': True, # If True, the first inversion will use only velocity observations with small temporal baselines, to detect temporal decorelation
                    'linear_operator': None, # Perform the inversion using this specific linear operator
                    'result_quality': result_quality, # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')
                    
                    'visual': False, # Plot results along the way
                    'verbose': False} # Print information throughout TICOI processing
                    
## --------------------- Interpolation parameters ---------------------- ##
interpolation_kwargs = {'option_interpol': 'spline', # Type of interpolation ('spline', 'spline_smooth', 'nearest')                
                        'result_quality': result_quality,  # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')
                        'unit': unit} # 365 if the unit is m/y, 1 if the unit is m/d

## ----------------------- Parallelization parameters ---------------------- ##
nb_cpu = 8 # Number of CPU to be used for parallelization
block_size = 0.5 # Maximum sub-block size (in GB) for the 'block_process' TICOI processing method

if not os.path.exists(path_save):
    os.mkdir(path_save)


# %%========================================================================= #
#                                 DATA LOADING                                #
# =========================================================================%% #

start = [time.time()]

# In the first place, we load the data
cube = cube_data_class()
cube.load(cube_name, **load_kwargs)


# Then we load the "ground truth"
cube_gt = cube_data_class()
cube_gt.load(cube_gt_name, **load_kwargs)

# Mask some of the data
if mask_file is not None:
    cube.mask_cube(mask_file)
    cube_gt.mask_cube(mask_file)

stop = [time.time()]
print(f'[Data loading] Loading the data cube.s took {round((stop[0] - start[0]), 4)} s')
print(f'[Data loading] Data cube of dimension (nz,nx,ny) : ({cube.nz}, {cube.nx}, {cube.ny}) ')
print(f'[Data loading] Ground Truth cube of dimension (nz,nx,ny) : ({cube_gt.nz}, {cube_gt.nx}, {cube_gt.ny})')


# %% ======================================================================== #
#                         COEFFICIENT OPTIMIZATION                            #
# =========================================================================%% #

async def process_block(block, cube_gt, load_pixel_kwargs, inversion_kwargs, interpolation_kwargs,
                        cmin=10, cmax=1000, step=10, coefs=None, stats=False, nb_cpu=8):
    
    # Progression bar
    xy_values = itertools.product(block.ds['x'].values, block.ds['y'].values)
    xy_values_tqdm = tqdm(xy_values, total=(block.nx * block.ny))
    
    # Filter cube
    obs_filt = block.filter_cube(**preData_kwargs)

    # Optimization of the coefficient for every pixels of the block
    result_block = Parallel(n_jobs=nb_cpu, verbose=0)(delayed(optimize_coef)(block, cube_gt, i, j, obs_filt, load_pixel_kwargs, 
                           inversion_kwargs, interpolation_kwargs, cmin=cmin, cmax=cmax, step=step, coefs=coefs, 
                           stats=stats, visual=False) for i, j in xy_values_tqdm)
    
    return result_block

async def process_blocks_main(cube, cube_gt, load_pixel_kwargs, inversion_kwargs, interpolation_kwargs,
                              cmin=10, cmax=1000, step=10, coefs=None, stats=False, nb_cpu=8, block_size=0.5, 
                              verbose=False):
    
    blocks = chunk_to_block(cube, block_size=block_size, verbose=True)
    
    dataf_list = [None] * ( cube.nx * cube.ny )

    loop = asyncio.get_event_loop()

    for n in range(len(blocks)):
        print(f'Processing block {n+1}/{len(blocks)}')

        # Load the first block and start the loop
        if n == 0:
            x_start, x_end, y_start, y_end = blocks[0]
            future = loop.run_in_executor(None, load_block, cube, x_start, x_end, y_start, y_end)

        block, flags_block, duration = await future
        print(f'Block {n+1} loaded in {duration:.2f} s')
        if verbose: print(f'Block {n+1} loaded in {duration:.2f} s')

        if n < len(blocks) - 1:
            # Load the next block while processing the current block
            x_start, x_end, y_start, y_end = blocks[n+1]
            future = loop.run_in_executor(None, load_block, cube, x_start, x_end, y_start, y_end)

        block_result = await process_block(block, cube_gt, load_pixel_kwargs, inversion_kwargs, interpolation_kwargs,
                                           cmin=cmin, cmax=cmax, step=step, coefs=coefs, stats=stats, nb_cpu=nb_cpu)

        for i in range(len(block_result)):
            row = i % block.ny + blocks[n][2]
            col = np.floor( i / block.ny ) + blocks[n][0]
            idx = int( col * cube.ny + row )

            dataf_list[idx] = block_result[i]

        del block_result, block, flags_block

    return dataf_list


nb_points = len(cube.ds['x'].values) * len(cube.ds['y'].values)
print(f'[Coef optimization] Number of CPU : {nb_cpu}')
print(f'[Coef optimization] {nb_points} points to be computed within the given subset')

start.append(time.time())

if optimization_process == 'block_process':
    result = asyncio.run(process_blocks_main(cube, cube_gt, load_pixel_kwargs, inversion_kwargs, interpolation_kwargs,
                                             cmin=coef_min, cmax=coef_max, step=step, coefs=coefs, stats=stats, 
                                             nb_cpu=nb_cpu, block_size=block_size, verbose=False))

elif optimization_process == 'direct_process':
    obs_filt = cube.filter_cube(**preData_kwargs)
    
    # Progression bar
    xy_values = itertools.product(cube.ds['x'].values, cube.ds['y'].values)
    xy_values_tqdm = tqdm(xy_values, total=len(cube.ds['x'].values)*len(cube.ds['y'].values), mininterval=0.5)

    result = Parallel(n_jobs=nb_cpu, verbose=0)(delayed(optimize_coef)(cube, cube_gt, i, j, obs_filt, load_pixel_kwargs, 
                        inversion_kwargs, interpolation_kwargs, cmin=coef_min, cmax=coef_max, step=step, coefs=coefs, 
                        stats=stats, visual=False) for i, j in xy_values_tqdm)

    stop.append(time.time())
    print(f'[Coef optimization] Whole coefficient optimization took {round((stop[1] - start[1]), 1)} s')


# %% ======================================================================== #
#                                  RESULTS                                    #
# =========================================================================%% #

# Informations about raw and GT datas over this cube
nb_datas = (np.array([result[i]['nb_data'][0][0] if not result[i].empty else 0 for i in range(len(result))]),
            np.array([result[i]['nb_data'][0][1] if not result[i].empty else 0 for i in range(len(result))]))
mean_nb_data = (np.mean(nb_datas[0]), np.mean(nb_datas[1]))
mean_baselines = (np.nanmean([result[i]['temporal_baseline'][0][0] if not result[i].empty else np.nan for i in range(len(result))]),
                  np.nanmean([result[i]['temporal_baseline'][0][1] if not result[i].empty else np.nan for i in range(len(result))]))
mean_std = (np.nanmean([result[i]['std'][0][0] if not result[i].empty else np.nan for i in range(len(result))]),
            np.nanmean([result[i]['std'][0][1] if not result[i].empty else np.nan for i in range(len(result))]),
            np.nanmean([result[i]['std'][0][2] if not result[i].empty else np.nan for i in range(len(result))]),
            np.nanmean([result[i]['std'][0][3] if not result[i].empty else np.nan for i in range(len(result))]))
mean_std_p = np.sqrt(mean_std[2] ** 2 + mean_std[3] ** 2)
mean_v = (np.nanmean([result[i]['mean_v'][0][0] if not result[i].empty else np.nan for i in range(len(result))]),
          np.nanmean([result[i]['mean_v'][0][1] if not result[i].empty else np.nan for i in range(len(result))]),
          np.nanmean([result[i]['mean_v'][0][2] if not result[i].empty else np.nan for i in range(len(result))]),
          np.nanmean([result[i]['mean_v'][0][3] if not result[i].empty else np.nan for i in range(len(result))]))
mean_std_all = (np.nanmedian([result[i]['std_all'][0][0] if not result[i].empty else np.nan for i in range(len(result))]),
                np.nanmedian([result[i]['std_all'][0][1] if not result[i].empty else np.nan for i in range(len(result))]),
                np.nanmedian([result[i]['std_all'][0][2] if not result[i].empty else np.nan for i in range(len(result))]),
                np.nanmedian([result[i]['std_all'][0][3] if not result[i].empty else np.nan for i in range(len(result))]))

print(f'[Coef optimization] Pixel with the greatest amount of GT data : {result[np.argmax(nb_datas[1])]["position"][0]} '
      f'with {np.max(nb_datas[1])} available GT data')

# Because tested coefficients are the same for each pixel (methods 'constant' and 'given_coef'), we compute the average RMSE 
# for each coefficient and then select which one is the best (by plotting the curve showing the evolution of the RMSE with 
# the regularisation coefficient)
if coefs is None:
    coefs = np.arange(coef_min, coef_max, step)
else:
    coefs = np.array(coefs)

RMSEs_result = np.array([result[i]['RMSEs'] if not result[i].empty else [np.nan for _ in range(len(coefs))] for i in range(len(result))])
# Compute the RMSE over the entire area
RMSEs = np.array([np.sqrt(1/np.sum(nb_datas[1]) * np.sum(nb_datas[1][~(nb_datas[1] == 0)] * RMSEs_result[~np.isnan(RMSEs_result).any(axis=1)][:, i] ** 2)) for i in range(len(coefs))])

best_coef = coefs[np.argmin(RMSEs)]
best_RMSE = np.min(RMSEs)
good_RMSE = max(1.05 * best_RMSE, best_RMSE + mean_std_p)
good_coefs = coefs[RMSEs < good_RMSE]

# Plot result
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(coefs[~(RMSEs < good_RMSE)], RMSEs[~(RMSEs < good_RMSE)], marker='x', markersize=7, linestyle='', markeredgecolor='darkred')
ax.plot(coefs[RMSEs < good_RMSE], RMSEs[RMSEs < good_RMSE], marker='x', markersize=7, linestyle='', markeredgecolor='darkgreen')
ax.plot([coefs[0], coefs[-1]], [good_RMSE, good_RMSE], linestyle='--', color='midnightblue')
ax.set_xlabel('Regularisation coefficient value')
ax.set_xlim([coef_min if type(coef_min) == int else coef_min[0], coef_max if type(coef_min) == int else coef_min[-1]])
ax.set_ylabel('Average RMSE between TICOI results and GT data [m/y]')
fig.suptitle(f'Average RMSE between TICOI results and GT data when changing the regularisation coefficient\nBest for coef = {best_coef} (RMSE = {best_RMSE})')
plt.show()

if save: fig.savefig(f'{path_save}RMSE_coef_{regu}.png')

if plot_them_all:
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))
    ax[0].set_xlabel('Regularisation coefficient value')
    ax[1].set_xlabel('Regularisation coefficient value')
    ax[0].set_xlim([coef_min if type(coef_min) == int else coef_min[0], coef_max])
    ax[1].set_xlim([coef_min if type(coef_min) == int else coef_min[0], coef_max])
    ax[0].set_ylabel('RMSE [m/y]')
    ax[1].set_ylabel('mean-substracted RMSE [m/y]')
    
    Q1 = np.percentile(nb_datas[1][nb_datas[1] > 0], 25)
    median = np.median(nb_datas[1][nb_datas[1] > 0])
    Q3 = np.percentile(nb_datas[1][nb_datas[1] > 0], 75)
    for r in range(len(result)):
        if nb_datas[1][r] > 0:
            if nb_datas[1][r] > Q3: 
                color = 'green'
                alpha = 0.6
            elif nb_datas[1][r] > median: 
                color = 'blue'
                alpha = 0.5
            elif nb_datas[1][r] > Q1: 
                color = 'red'
                alpha = 0.4
            else: 
                color = 'black'
                alpha = 0.3
                
            ax[0].plot(result[r]['coefs'], result[r]['RMSEs'], linestyle='dashed', color=color, alpha=alpha)
            ax[1].plot(result[r]['coefs'], result[r]['RMSEs'] - result[r]['RMSEs'].mean(), linestyle='dashed', color=color, alpha=alpha)
    
    ax[0].plot(coefs, RMSEs, linestyle='dashdot', linewidth=2, color='blueviolet')
    ax[1].plot(coefs, RMSEs - np.mean(RMSEs), linestyle='dashdot', linewidth=2, color='blueviolet')
        
    fig.suptitle('RMSE between TICOI results and GT data when changing the regularisation coefficient')
    plt.show()
    
    if save: fig.savefig(f'{path_save}RMSE_coef_{regu}_allplots.png')

stop.append(time.time())
print(f'[Overall] Overall processing took {round(stop[-1] - start[0], 1)} s')