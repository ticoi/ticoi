'''
Implementation of the Temporal Inversion using COmbination of displacements with Interpolation (TICOI) method to compute entire data cubes.
It can be divided in three parts:
    - Data Download : Download one or several data cube.s, eventually considering a given subset or buffer to limit its size. Additionnal data
    cubes are aligned and merged to the main cube.
    - TICOI : Compute TICOI on the selection of data using the given method (split in blocks or direct processing, think of reading the comments
    about those methods) to get a list of the results.
    - Save the results : Format the data to a new data cube, which can be saved to a netCDF file. The mean velocity can also be saved as an example.

Author : Laurane Charrier, Lei Guo, Nathan Lioret
Reference:
    Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an optimal temporal sampling from displacement
    observation networks. IEEE Transactions on Geoscience and Remote Sensing.
    Charrier, L., Yan, Y., Colin Koeniguer, E., Mouginot, J., Millan, R., & Trouvé, E. (2022). Fusion of multi-temporal and multi-sensor ice velocity observations.
    ISPRS annals of the photogrammetry, remote sensing and spatial information sciences, 3, 311-318.
'''

import time
import os
import warnings
import itertools
import xarray as xr
import numpy as np
import pandas as pd

from osgeo import gdal, osr
from joblib import Parallel, delayed
from tqdm import tqdm

from ticoi.core import process_blocks_refine, process
from ticoi.cube_data_classxr import cube_data_class

# %%========================================================================= #
#                                   PARAMETERS                                #
# =========================================================================%% #

warnings.filterwarnings("ignore")

## ------------------- Choose TICOI cube processing method ----------------- ##
# Choose the TICOI cube processing method you want to use :
#    - 'block_process' (recommended) : This implementation divides the data in smaller data cubes processed one after the other in a synchronous manner,
# in order to avoid memory overconsumption and kernel crashing. Computations within the blocks are parallelized so this method goes way faster
# than every other TICOI processing methods.
#      /!\ This implementation uses asyncio (way faster) which requires its own event loop to run : if you launch this code from a raw terminal, 
# there should be no problem, but if you try to launch it from an IDE (PyCharm, VSCode, Spyder...), think of specifying to your IDE to launch it 
# in a raw terminal instead of the default console (which leads to a RuntimeError)
#    - 'direct_process' : No subdivisition of the data is made beforehand which generally leads to memory overconsumption and kernel crashes
# if the amount of pixel to compute is too high (depending on your available memory). If you want to process big amount of data, you should use
# 'block_process', which is also faster. This method is essentially used for debug purposes.
#   - 'load' : The TICOI cube was already calculated before, load it using the load_file variable to indicate the path to the .nc file

TICOI_process = 'direct_process'

save = True # If True, save TICOI results to a netCDF file
save_mean_velocity = True # Save a .tiff file with the mean reulting velocities, as an example

## ------------------------------ Data selection --------------------------- ##
# List of the paths where the data cubes are stored
cube_names = ['/media/tristan/Data3/Hala_lake/Landsat8/Hala_lake_displacement_LS7_subset.nc'] #['test_data/c_x18620_y08085_2016-2022_crop_GPS_Lower.nc',]
               # 'nathan/Donnees/Cubes_de_donnees/stack_median_pleiades_alllayers_2012-2022_modiflaurane.nc']
flag_shp = '/media/tristan/Data3/Hala_lake/glacier_outline/hala_lake_RGI7.shp' # Path of the shape file for flags generating
mask_file = None # Path where the mask file is stored (.shp file)
load_file = None # If TICOI was already computed on this cube/subset, you can directly load it (TICOI_process='load')
path_save = 'Tests_MB/' # Path where to store the results
result_fn = 'test'# Name of the netCDF file to be created

if not os.path.exists(path_save):
    os.mkdir(path_save)

proj = 'EPSG:32647'  # EPSG system of the given coordinates

## ----------------------- Regularization parameters ---------------------- ##
# Divide the data in several areas where different methods should be used
assign_flag = True
regu = {0: 1, 1: 2, 2: "1accelnotnull"} # Regularization method.s to be used (for each flag if flags is not None)
coef = {0: 100, 1:150, 2:300} # Regularization coefficient.s to be used (for each flag if flags is not None)

solver = 'LSMR_ini' # Solver for the inversion

# What results must be returned from TICOI processing (can be a list of both)
#   - 'invert' for the results of the inversion
#   - 'interp' for the results of the interpolation
returned = ['invert', 'interp']


## ----------------------- Parallelization parameters ---------------------- ##
nb_cpu = 20 # Number of CPU to be used for parallelization
block_size = 0.5 # Maximum sub-block size (in GB) for the 'block_process' TICOI processing method


## ---------------------------- Loading parameters ------------------------- ##
load_kwargs = {'chunks': {}, 
               'conf': False, # If True, confidence indicators will be put between 0 and 1, with 1 the lowest errors
               'subset': None, # Subset of the data to be loaded ([xmin, xmax, ymin, ymax] or None)
               'buffer': None, # Area to be loaded around the pixel ([longitude, latitude, buffer size] or None)
               'pick_date': None, # Select dates ([min, max] or None to select all)
               'pick_sensor': None, # Select sensors (None to select all)
               'pick_temp_bas': None, # Select temporal baselines ([min, max] in days or None to select all)
               'proj': proj, # EPSG system of the given coordinates
               'verbose': False # Print information throughout the loading process 
               }

## ----------------------- Data preparation parameters --------------------- ##
preData_kwargs = {'smooth_method': 'gaussian', # Smoothing method to be used to smooth the data in time ('gaussian', 'median', 'emwa', 'savgol')
                  's_win': 3, # Size of the spatial window
                  't_win': 90, # Time window size for 'ewma' smoothing
                  'sigma': 3, # Standard deviation for 'gaussian' filter
                  'order': 3, # Order of the smoothing function
                  'unit': 365, # 365 if the unit is m/y, 1 if the unit is m/d
                  'delete_outliers': 'vvc_angle', # Delete data with a poor quality indicator (if int), or with aberrant direction ('vvc_angle') 
                  'flag': flag_shp, # shape file to divide the data into several areas where different methods should be used
                  'regu': regu, # Regularization method.s to be used (for each flag if flags is not None)
                  'solver': solver, # Solver for the inversion
                  'proj': proj, # EPSG system of the given coordinates
                  'velo_or_disp': 'disp', # Type of data contained in the data cube ('disp' for displacements, and 'velo' for velocities)
                  'verbose': True # Print information throughout the filtering process 
                  }

## ---------------- Inversion and interpolation parameters ----------------- ##
inversion_kwargs = {'regu': regu, # Regularization method.s to be used (for each flag if flags is not None)
                    'coef': coef, # Regularization coefficient.s to be used (for each flag if flags is not None)
                    'solver': solver, # Solver for the inversion
                    'flag': flag_shp, # shape file to divide the data into several areas where different methods should be used
                    'conf': False, # If True, confidence indicators are set between 0 and 1, with 1 the lowest errors
                    'unit': 365, # 365 if the unit is m/y, 1 if the unit is m/d
                    'delete_outliers': 'vvc_angle', # Delete data with a poor quality indicator (if int), or with aberrant direction ('vvc_angle') 
                    'proj': proj, # EPSG system of the given coordinates
                    'interpolation_load_pixel': 'nearest', # Interpolation method used to load the pixel when it is not in the dataset
                    
                    'iteration': True, # Allow the inversion process to make several iterations
                    'nb_max_iteration': 10, # Maximum number of iteration during the inversion process
                    'threshold_it': 0.1, # Threshold to test the stability of the results between each iteration, used to stop the process
                    'apriori_weight': True, # If True, use apriori weights
                    'detect_temporal_decorrelation': True, # If True, the first inversion will use only velocity observations with small temporal baselines, to detect temporal decorelation
                    'linear_operator': None, # Perform the inversion using this specific linear operator
                    
                    'interval_output': 30, 
                    'option_interpol': 'spline', # Type of interpolation ('spline', 'spline_smooth', 'nearest')
                    'redundancy': 30, # Redundancy in the interpolated time series in number of days, no redundancy if None
                    
                    'result_quality': 'X_contribution', # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')
                    'visual': False, # Plot results along the way
                    'path_save': path_save, # Path where to store the results
                    'verbose': False # Print information throughout TICOI processing
                    }

# %%========================================================================= #
#                                 DATA LOADING                                #
# =========================================================================%% #

start = [time.time()]
if TICOI_process != 'load':    
    # Load the first cube
    cube = cube_data_class()
    cube.load(cube_names[0], **load_kwargs)
    
    # Several cubes have to be merged together
    filenames = [cube.filename]
    if len(cube_names) > 1:
        for n in range(1, len(cube_names)):
            cube2 = cube_data_class()
            subset = load_kwargs['subset']
            res = cube.ds['x'].values[1] - cube.ds['x'].values[0] # Resolution of the main data
            load_kwargs.update({'subset': [subset[0]-res, subset[1]+res, subset[2]-res, subset[3]+res]})
            cube2.load(cube_names[n], **load_kwargs)
            filenames.append(cube2.filename)
            # Align the new cube to the main one (interpolate the coordinate and/or reproject it)
            cube2 = cube.align_cube(cube2, reproj_vel=False, reproj_coord=True, interp_method='nearest')
            cube.merge_cube(cube2) # Merge the new cube to the main one
        del cube2
    
    # Mask some of the data
    if mask_file is not None:
        cube.mask_cube(mask_file)

    stop = [time.time()]
    print(f'[Data loading] Cube of dimension (nz, nx, ny): ({cube.nz}, {cube.nx}, {cube.ny}) ')
    print(f'[Data loading] Data loading took {round(stop[0] - start[0], 3)} s')
    
else:
    stop = [time.time()]
    

# Prepare interpolation dates
cube_date1 = cube.date1_().tolist()
cube_date1.remove(np.min(cube_date1))
first_date_interpol = np.min(cube_date1)
last_date_interpol = np.max(cube.date2_())

inversion_kwargs.update({'first_date_interpol': first_date_interpol, 'last_date_interpol': last_date_interpol})

# %%========================================================================= #
#                                      TICOI                                  #
# =========================================================================%% #

start.append(time.time())

# The data cube is subdivided in smaller cubes computed one after the other in a synchronous manner (uses async)
# TICOI computation is then parallelized among those cubes
if TICOI_process == 'block_process':
    result = process_blocks_refine(cube, nb_cpu=nb_cpu, block_size=block_size, returned=returned, 
                                   preData_kwargs=preData_kwargs, inversion_kwargs=inversion_kwargs)

# Direct computation of the whole TICOI cube
elif TICOI_process == 'direct_process':
    # Preprocessing of the data (compute rolling mean for regu='1accelnotnull', delete outliers...)
    obs_filt, flag = cube.filter_cube(**preData_kwargs)
    inversion_kwargs.update({'flag': flag})
    
    # Progression bar
    xy_values = itertools.product(cube.ds['x'].values, cube.ds['y'].values)
    xy_values_tqdm = tqdm(xy_values, total=len(cube.ds['x'].values)*len(cube.ds['y'].values), mininterval=0.5)
    
    # Main processing of the data with TICOI algorithm, individually for each pixel
    result = Parallel(n_jobs=nb_cpu, verbose=0)(delayed(process)(cube, i, j, returned=returned, obs_filt=obs_filt,
                                                                 **inversion_kwargs) for i, j in xy_values_tqdm)

elif TICOI_process == 'load':
    cubenew = cube_data_class()
    cubenew.load(load_file, **load_kwargs)
    
    # Mask some of the data
    if mask_file is not None:
        cubenew.mask_cube(mask_file)
        
    result = process_blocks_refine(cubenew, nb_cpu=nb_cpu, block_size=block_size, returned='raw', preData_kwargs=preData_kwargs, inversion_kwargs=inversion_kwargs)
    result = [pd.DataFrame(data={'First_date': r[0][0][:, 0], 'Second_date': r[0][0][:, 1],
                                  'vx': r[0][1][:, 0], 'vy': r[0][1][:, 1],
                                  'errorx': r[0][1][:, 2], 'errory': r[0][1][:, 3],
                                  'temporal_baseline': r[0][1][:, 4]}) for r in result]

stop.append(time.time())
print(f'[TICOI processing] TICOI {"processing" if TICOI_process != "load" else "loading"} took {round(stop[1] - start[1], 0)} s')


# %%========================================================================= #
#                                INITIALISATION                               #
# =========================================================================%% #

start.append(time.time())
if TICOI_process != 'load':
    # Write down some informations about the data and the TICOI processing performed
    if save:
        sensor_array = np.unique(cube.ds['sensor'])
        sensor_strings = [str(sensor) for sensor in sensor_array]
        sensor = ', '.join(sensor_strings)
        
        if len(cube_names) > 1:
            source = f'Temporal inversion on cubes {", ".join(filenames)} using TICOI'
        else:
            source = f'Temporal inversion on cube {filenames[0]} using TICOI'
        source += f' with a selection of dates among {load_kwargs["pick_date"]},' if load_kwargs['pick_date'] is not None else '' + \
                  f' with a selection of the temporal baselines among {load_kwargs["pick_temp_bas"]}' if load_kwargs['pick_temp_bas'] is not None else ''
        
        if inversion_kwargs['apriori_weight']:
            source += ' and apriori weight'
        source += f'. The regularisation coefficient is {inversion_kwargs["coef"]}.'
        if 'interp' in returned:
            source_interp = source + f'The interpolation method used is {inversion_kwargs["option_interpol"]}.'
            source_interp += f'The interpolation baseline is {inversion_kwargs["interval_output"]} days.'
            source_interp += f'The temporal spacing (redundancy) is {inversion_kwargs["redundancy"]} days.'
    
    stop.append(time.time())    
    print(f'[Writing results] Initialisation took {round(stop[2] - start[2], 3)} s')
    
else:
    stop.append(time.time())


# %%========================================================================= #
#                                WRITING RESULTS                              #
# =========================================================================%% #

start.append(time.time())
if TICOI_process != 'load':
    # Save TICO.I results to a netCDF file, thus obtaining a new data cube
    cube_interp = None
    several = (type(returned) == list and len(returned) >= 2)
    if 'invert' in returned:
        cube_invert = cube.write_result_tico([result[i][0] for i in range(len(result))] if several else result, source, sensor, 
                                             filename=f'{result_fn}_invert' if several else result_fn, 
                                             savepath=path_save if save else None, 
                                             result_quality=inversion_kwargs['result_quality'], verbose=inversion_kwargs['verbose'])
    if 'interp' in returned:
        cube_interp = cube.write_result_ticoi([result[i][1] for i in range(len(result))] if several else result, source_interp, sensor, 
                                              filename=f'{result_fn}_invert' if several else result_fn, 
                                              savepath=path_save if save else None, 
                                              result_quality=inversion_kwargs['result_quality'], verbose=inversion_kwargs['verbose'])

# Plot the mean velocity as an example
# TODO format the data cube with inversion results to the format of the interpolation resulting cube
if save_mean_velocity and cube_interp is not None:
    cubenew = cube_interp
    mean_vv = np.sqrt(cubenew.ds['vx'].mean(dim='mid_date') ** 2 + cubenew.ds['vy'].mean(dim='mid_date') ** 2).to_numpy().astype(np.float32)
    mean_vv = np.flip(mean_vv, axis=0)
    
    driver = gdal.GetDriverByName('GTiff')
    srs = osr.SpatialReference()
    srs.SetWellKnownGeogCS('EPSG:32632')
    
    resolution = int(cube.ds['x'].values[1] - cube.ds['x'].values[0])
    dst_ds_temp = driver.Create(f'{path_save}mean_velocity.tiff', mean_vv.shape[1], mean_vv.shape[0], 1, gdal.GDT_Float32)
    dst_ds_temp.SetGeoTransform([np.min(cube.ds['x'].values), resolution, 0, np.min(cube.ds['y'].values), 0, resolution])
    dst_ds_temp.GetRasterBand(1).WriteArray(mean_vv)
    dst_ds_temp.SetProjection(srs.ExportToWkt())
    
    dst_ds_temp = None
    driver = None
        
if save or save_mean_velocity:
    print(f'[Writing results] Results saved at {path_save}')

stop.append(time.time())
if TICOI_process != 'load':
    print(f'[Writing results] Writing cube to netCDF file took {round(stop[3] - start[3], 3)} s')    
print(f'[Overall] Overall processing took {round(stop[3] - start[0], 0)} s')