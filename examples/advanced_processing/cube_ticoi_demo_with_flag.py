#!/usr/bin/env python3

"""
Implementation of the Temporal Inversion using COmbination of displacements with Interpolation (TICOI) method to compute entire data cubes.
It can be divided in three parts:
    - Data loading : Load one or several data cube.s, eventually considering a given subset or buffer to limit its size. additional data
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
"""

import itertools
import os
import time
import warnings

import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from tqdm import tqdm

from ticoi.core import process, process_blocks_refine, save_cube_parameters
from ticoi.cube_data_classxr import cube_data_class
from ticoi.interpolation_functions import prepare_interpolation_date

# %%========================================================================= #
#                                   PARAMETERS                                #
# =========================================================================%% #

warnings.filterwarnings("ignore")

## ------------------- Choose TICOI cube processing method ----------------- ##
# Choose the TICOI cube processing method you want to use :
#    - 'block_process' (recommended) : This implementation divides the data in smaller data cubes processed one after the other in a synchronous manner,
# in order to avoid memory overconsumption and kernel crashing. Computations within the blocks are parallelized so this method goes way faster
# than the 'direct_process' method.
#      /!\ This implementation uses asyncio (way faster) which requires its own event loop to run : if you launch this code from a raw terminal,
# there should be no problem, but if you try to launch it from some IDE (like Spyder), think of specifying to your IDE to launch it
# in a raw terminal instead of the default console (which leads to a RuntimeError)
#    - 'direct_process' : No subdivisition of the data is made beforehand which generally leads to memory overconsumption and kernel crashes
# if the amount of pixel to compute is too high (depending on your available memory). If you want to process big amount of data, you should use
# 'block_process', which is also faster. This method is essentially used for debug purposes.
#   - 'load' : The  TICOI cube was already calculated before, load it by giving the cubes to be loaded in a dictionary like {name: path} (name can be
# 'interp', 'invert' or 'raw' as for returned, path can be a single str or a list of str to merge cubes) in cube_name, or a single str to a TICOI cube

TICOI_process = "direct_process"

save = True  # If True, save TICOI results to a netCDF file
save_mean_velocity = True  # Save a .tiff file with the mean reulting velocities, as an example

# For TICOI_process = 'load', generate a 'result' list with raw data and/or TICOI results as pandas dataframe for further processing
compute_result_load = False

## ------------------------------ Data selection --------------------------- ##
# Path.s to the data cube.s (can be a list of str to merge several cubes, or a single str,
cube_name = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..", "test_data"))}/Alps_Mont-Blanc_Argentiere_S2.nc'
# If TICOI_process is 'load', it can be a dictionary like {name: path} to load existing cubes and name them (path can be a list of str or a single str)
# If it is an str (or list of str), we suppose we want to load TICOI results (like 'interp' in the dict)
# cube_name = {'raw': f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "test_data"))}/Alps_Mont-Blanc_Argentiere_S2.nc',
#              'invert': f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "cube"))}/Argentiere_example_invert.nc',
#              'interp': f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "cube"))}/Argentiere_example_interp.nc'}
flag_file = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..", "test_data"))}/Alps_Mont-Blanc_flags.nc'  # Path to flags file
mask_file = None  # Path to mask file (.shp file) to mask some of the data on cube
path_save = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "cube"))}/'  # Path where to store the results
result_fn = "Argentiere_example"  # Name of the netCDF file to be created (if save is True)

proj = "EPSG:32632"  # EPSG system of the given coordinates

# Divide the data in several areas where different methods should be used
assign_flag = True
if assign_flag:
    flag = xr.open_dataset(flag_file)
    flag.load()
else:
    flag = None

# Regularization method.s to be used (for each flag if flag is not None)
regu = {0: 1, 1: "1accelnotnull"}  # With flag (0: stable ground, 1: glaciers)
# regu = '1accelnotnull' # Without flag
# Regularization coefficient.s to be used (for each flag if flag is not None)
coef = {0: 500, 1: 200}  # With flag (0: stable ground, 1: glaciers)
# coef = 200 # Without flag
solver = "LSMR_ini"  # Solver for the inversion

# What results must be returned from TICOI processing (not used for TICOI_process='load')
#   - 'raw' for loading raw data at pixels too
#   - 'invert' for the results of the inversion
#   - 'interp' for the results of the interpolation
returned = ["invert", "interp"]

## ---------------------------- Loading parameters ------------------------- ##
load_kwargs = {
    "chunks": {},
    "conf": False,  # If True, confidence indicators will be put between 0 and 1, with 1 the lowest errors
    "subset": None,  # Subset of the data to be loaded ([xmin, xmax, ymin, ymax] or None)
    "buffer": None,  # Area to be loaded around the pixel ([longitude, latitude, buffer size] or None)
    "pick_date": ["2015-01-01", "2023-01-01"],  # Select dates ([min, max] or None to select all)
    "pick_sensor": None,  # Select sensors (None to select all)
    "pick_temp_bas": None,  # Select temporal baselines ([min, max] in days or None to select all)
    "proj": proj,  # EPSG system of the given coordinates
    "mask": mask_file,  # Path to mask file (.shp file) to mask some of the data on cube
    "verbose": False,
}  # Print information throughout the loading process

## ----------------------- Data preparation parameters --------------------- ##
preData_kwargs = {
    "smooth_method": "savgol",  # Smoothing method to be used to smooth the data in time ('gaussian', 'median', 'emwa', 'savgol')
    "s_win": 3,  # Size of the spatial window
    "t_win": 90,  # Time window size for 'ewma' smoothing
    "sigma": 3,  # Standard deviation for 'gaussian' filter
    "order": 3,  # Order of the smoothing function
    "unit": 365,  # 365 if the unit is m/y, 1 if the unit is m/d
    "delete_outliers": "vvc_angle",  # Delete data with a poor quality indicator (if int), or with aberrant direction ('vvc_angle')
    "flag": flag,  # Divide the data in several areas where different methods should be used
    "regu": regu,  # Regularization method.s to be used (for each flag if flag is not None)
    "solver": solver,  # Solver for the inversion
    "proj": proj,  # EPSG system of the given coordinates
    "velo_or_disp": "velo",  # Type of data contained in the data cube ('disp' for displacements, and 'velo' for velocities)
    "verbose": True,
}  # Print information throughout the filtering process

## ---------------- Inversion and interpolation parameters ----------------- ##
inversion_kwargs = {
    "regu": regu,  # Regularization method.s to be used (for each flag if flag is not None)
    "coef": coef,  # Regularization coefficient.s to be used (for each flag if flag is not None)
    "solver": solver,  # Solver for the inversion
    "conf": False,  # If True, confidence indicators are set between 0 and 1, with 1 the lowest errors
    "unit": 365,  # 365 if the unit is m/y, 1 if the unit is m/d
    "delete_outliers": "vvc_angle",  # Delete data with a poor quality indicator (if int), or with aberrant direction ('vvc_angle')
    "proj": proj,  # EPSG system of the given coordinates
    "interpolation_load_pixel": "nearest",  # Interpolation method used to load the pixel when it is not in the dataset
    "iteration": True,  # Allow the inversion process to make several iterations
    "nb_max_iteration": 10,  # Maximum number of iteration during the inversion process
    "threshold_it": 0.1,  # Threshold to test the stability of the results between each iteration, used to stop the process
    "apriori_weight": True,  # If True, use apriori weights
    "detect_temporal_decorrelation": True,  # If True, the first inversion will use only velocity observations with small temporal baselines, to detect temporal decorelation
    "linear_operator": None,  # Perform the inversion using this specific linear operator
    "interval_output": 30,
    "option_interpol": "spline",  # Type of interpolation ('spline', 'spline_smooth', 'nearest')
    "redundancy": 30,  # Redundancy in the interpolated time series in number of days, no redundancy if None
    "result_quality": "X_contribution",  # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')
    "visual": False,  # Plot results along the way
    "path_save": path_save,  # Path where to store the results
    "verbose": False,
}  # Print information throughout TICOI processing

## ----------------------- Parallelization parameters ---------------------- ##
nb_cpu = 6  # Number of CPU to be used for parallelization
block_size = 0.1  # Maximum sub-block size (in GB) for the 'block_process' TICOI processing method

if not os.path.exists(path_save):
    os.mkdir(path_save)


# %%========================================================================= #
#                                 DATA LOADING                                #
# =========================================================================%% #

start, stop = [], []

if TICOI_process != "load" or (TICOI_process == "load" and type(cube_name) == dict and "raw" in cube_name.keys()):
    start.append(time.time())

    # Load the cube.s
    cube = cube_data_class()

    if TICOI_process == "load" and type(cube_name) == dict:
        cube.load(cube_name["raw"], **load_kwargs)
    elif TICOI_process != "load":
        cube.load(cube_name, **load_kwargs)

    # Load raw data at pixels if required
    if (TICOI_process == "load" and "raw" in cube_name.keys()) and compute_result_load:
        print("[Data loading] Loading raw data...")
        data_raw = process_blocks_refine(
            cube, nb_cpu=nb_cpu, block_size=block_size, returned=["raw"], inversion_kwargs=inversion_kwargs
        )
        data_raw = [
            pd.DataFrame(
                data={
                    "date1": raw[0][0][:, 0],
                    "date2": raw[0][0][:, 1],
                    "vx": raw[0][1][:, 0],
                    "vy": raw[0][1][:, 1],
                    "errorx": raw[0][1][:, 2],
                    "errory": raw[0][1][:, 3],
                    "temporal_baseline": raw[0][1][:, 4],
                }
            )
            for raw in data_raw
        ]

    # Prepare interpolation dates
    first_date_interpol, last_date_interpol = prepare_interpolation_date(cube)
    inversion_kwargs.update({"first_date_interpol": first_date_interpol, "last_date_interpol": last_date_interpol})

    stop.append(time.time())
    print(f"[Data loading] Cube of dimension (nz, nx, ny): ({cube.nz}, {cube.nx}, {cube.ny}) ")
    print(f"[Data loading] Data loading took {round(stop[-1] - start[-1], 3)} s")


# %%========================================================================= #
#                                      TICOI                                  #
# =========================================================================%% #

start.append(time.time())

cube_interp, cube_invert = None, None

# The data cube is subdivided in smaller cubes computed one after the other in a synchronous manner (uses async)
# TICOI computation is then parallelized among those cubes
if TICOI_process == "block_process":
    result = process_blocks_refine(
        cube,
        nb_cpu=nb_cpu,
        block_size=block_size,
        returned=returned,
        preData_kwargs=preData_kwargs,
        inversion_kwargs=inversion_kwargs,
    )

# Direct computation of the whole TICOI cube
elif TICOI_process == "direct_process":
    # Preprocessing of the data (compute rolling mean for regu='1accelnotnull', delete outliers...)
    obs_filt, flag = cube.filter_cube(**preData_kwargs)
    inversion_kwargs.update({"flag": flag})

    # Progression bar
    xy_values = itertools.product(cube.ds["x"].values, cube.ds["y"].values)
    xy_values_tqdm = tqdm(xy_values, total=len(cube.ds["x"].values) * len(cube.ds["y"].values), mininterval=0.5)

    # Main processing of the data with TICOI algorithm, individually for each pixel
    result = Parallel(n_jobs=nb_cpu, verbose=0)(
        delayed(process)(cube, i, j, returned=returned, obs_filt=obs_filt, **inversion_kwargs)
        for i, j in xy_values_tqdm
    )

elif TICOI_process == "load":
    # Load inversion results
    if type(cube_name) == dict and "invert" in cube_name.keys():
        cube_invert = cube_data_class()
        cube_invert.load(cube_name["invert"], **load_kwargs)

    # Load interpolation results
    if (type(cube_name) == dict and "interp" in cube_name.keys()) or type(cube_name) == str:
        cube_interp = cube_data_class()
        cube_interp.load(cube_name["interp"] if type(cube_name) == dict else cube_name, **load_kwargs)

        if compute_result_load:
            print("[TICOI processing] Loading TICOI data...")
            result = process_blocks_refine(
                cube_interp, nb_cpu=nb_cpu, block_size=block_size, returned=["raw"], inversion_kwargs=inversion_kwargs
            )
            result = [
                pd.DataFrame(
                    data={
                        "date1": r[0][0][:, 0],
                        "date2": r[0][0][:, 1],
                        "vx": r[0][1][:, 0],
                        "vy": r[0][1][:, 1],
                        "errorx": r[0][1][:, 2],
                        "errory": r[0][1][:, 3],
                        "temporal_baseline": r[0][1][:, 4],
                    }
                )
                for r in result
            ]

stop.append(time.time())
print(
    f'[TICOI processing] TICOI {"processing" if TICOI_process != "load" else "loading"} took {round(stop[-1] - start[-1], 0)} s'
)


# %%========================================================================= #
#                                INITIALISATION                               #
# =========================================================================%% #

if TICOI_process != "load" and save:
    # Write down some information about the data and the TICOI processing performed
    if save:
        if "invert" in returned:
            source, sensor = save_cube_parameters(
                cube, load_kwargs, preData_kwargs, inversion_kwargs, returned=["invert"]
            )
        if "interp" in returned:
            source_interp, sensor = save_cube_parameters(
                cube, load_kwargs, preData_kwargs, inversion_kwargs, returned=["interp"]
            )
        stop.append(time.time())
        print(f"[Writing results] Initialisation took {round(stop[-1] - start[-1], 3)} s")


# %%========================================================================= #
#                                WRITING RESULTS                              #
# =========================================================================%% #

start.append(time.time())
if TICOI_process != "load":
    # Save TICO.I results to a netCDF file, thus obtaining a new data cube
    several = type(returned) == list and len(returned) >= 2
    j = 1 if "raw" in returned else 0
    if "invert" in returned:
        cube_invert = cube.write_result_tico(
            [result[i][j] for i in range(len(result))] if several else result,
            source,
            sensor,
            filename=f"{result_fn}_invert" if several else result_fn,
            savepath=path_save if save else None,
            result_quality=inversion_kwargs["result_quality"],
            verbose=inversion_kwargs["verbose"],
        )
    if "interp" in returned:
        cube_interp = cube.write_result_ticoi(
            [result[i][j + 1] for i in range(len(result))] if several else result,
            source_interp,
            sensor,
            filename=f"{result_fn}_interp" if several else result_fn,
            savepath=path_save if save else None,
            result_quality=inversion_kwargs["result_quality"],
            verbose=inversion_kwargs["verbose"],
        )

# Plot the mean velocity as an example
if save_mean_velocity and cube_interp is not None:
    cube_interp.average_cube(return_format="geotiff", return_variable=["vv"], save=True, path_save=path_save)

if save or save_mean_velocity:
    print(f"[Writing results] Results saved at {path_save}")

stop.append(time.time())
if TICOI_process != "load":
    print(f"[Writing results] Writing cube to netCDF file took {round(stop[-1] - start[-1], 3)} s")
print(f"[Overall] Overall processing took {round(stop[-1] - start[0], 0)} s")
