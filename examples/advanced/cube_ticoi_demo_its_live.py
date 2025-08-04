#!/usr/bin/env python3
"""
Implementation of the Temporal Inversion using COmbination of displacements with Interpolation (TICOI) method to compute entire data cubes.
It can be divided in three parts:
    - Data Download : Download one or several data cube.s, eventually considering a given subset or buffer to limit its size. additional data
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

from joblib import Parallel, delayed
from tqdm import tqdm

from ticoi.core import process, process_blocks_refine, save_cube_parameters
from ticoi.cube_data_classxr import CubeDataClass
from ticoi.cube_writer import CubeResultsWriter

warnings.filterwarnings("ignore")

# %%========================================================================= #
#                                   PARAMETERS                                #
# =========================================================================%% #

## ------------------- Choose TICOI cube processing method ----------------- ##
# Choose the TICOI cube processing method you want to use :
#    - 'block_process' (recommended) : This implementation divides the data in smaller data cubes processed one after the other in a synchronous manner,
# in order to avoid memory overconsumption and kernel crashing. Computations within the blocks are parallelized so this method goes way faster
# than the 'direct_process' method.
#      /!\ This implementation uses asyncio (way faster) which requires its own event loop to run : if you launch this code from a raw terminal,
# there should be no problem, but if you try to launch it from an IDE (PyCharm, VSCode, Spyder...), think of specifying to your IDE to launch it
# in a raw terminal instead of the default console (which leads to a RuntimeError)
#    - 'direct_process' : No subdivisition of the data is made beforehand which generally leads to memory overconsumption and kernel crashes
# if the amount of pixel to compute is too high (depending on your available memory). If you want to process big amount of data, you should use
# 'block_process', which is also faster. This method is essentially used for debug purposes.

TICOI_process = "block_process"

save = False  # If True, save TICOI results to a netCDF file
save_mean_velocity = False  # Save a .tiff file with the mean resulting velocities, as an example

## ------------------------------ Data selection --------------------------- ##
# List of the paths where the data cubes are stored
# List of the paths where the data cubes are stored
cube_name = "http://its-live-data.s3.amazonaws.com/datacubes/v2/N60W130/ITS_LIVE_vel_EPSG3413_G0120_X-3250000_Y150000.zarr"  # Path where the Sentinel-2 IGE cubes are stored
path_save = "to_file"  # Path where to stored the results
result_fn = "Lowell_example"  # Name of the netCDF file to be created
subset = [-138.28962881999922274, -138.279, 60.25934205396930565, 60.261]
proj = "EPSG:3413"  # EPSG system of the given coordinates

# What results must be returned from TICOI processing (can be a list of both)
#   - 'invert' for the results of the inversion
#   - 'interp' for the results of the interpolation
returned = ["interp"]
## ---------------------------- Loading parameters ------------------------- ##
load_kwargs = {
    "chunks": {},
    "conf": False,  # If True, confidence indicators will be put between 0 and 1, with 1 the lowest errors
    "subset": subset,  # Subset of the data to be loaded ([xmin, xmax, ymin, ymax] or None)
    "buffer": None,  # Area to be loaded around the pixel ([longitude, latitude, buffer size] or None)
    "pick_date": ["2015-01-01", "2023-01-01"],  # Select dates ([min, max] or None to select all)
    "pick_sensor": None,  # Select sensors (None to select all)
    "pick_temp_bas": None,  # Select temporal baselines ([min, max] in days or None to select all)
    "proj": "EPSG:4326",  # EPSG system of the given coordinates
    "verbose": False,  # Print information throughout the loading process
}

## ----------------------- Data preparation parameters --------------------- ##
# For the following parts we advice the user to change only the following parameter, the other parameters stored in a dictionary can be kept as it is for a first use
regu = "1accelnotnull"  # Regularization method.s to be used (for each flag if flag is not None) : 1 minimize the acceleration, '1accelnotnull' minize the distance with an apriori on the acceleration computed over a spatio-temporal filtering of the cube
coef = 100  # Regularization coefficient.s to be used (for each flag if flag is not None)
delete_outlier = "vvc_angle"
apriori_weight = True

preData_kwargs = {
    "smooth_method": "gaussian",  # Smoothing method to be used to smooth the data in time ('gaussian', 'median', 'savgol', 'lowess')
    "s_win": 3,  # Size of the spatial window
    "t_win": 90,  # Time window size for 'ewma' smoothing
    "sigma": 3,  # Standard deviation for 'gaussian' filter
    "order": 3,  # Order of the smoothing function
    "unit": 365,  # 365 if the unit is m/y, 1 if the unit is m/d
    "delete_outliers": delete_outlier,  # Delete data with a poor quality indicator (if int), or with aberrant direction ('vvc_angle')
    "flag": None,  # Divide the data in several areas where different methods should be used
    "dem_file": None,
    "regu": regu,  # Regularization method.s to be used (for each flag if flag is not None) : 1 minimize the acceleration, '1accelnotnull' minize the distance with an apriori on the acceleration computed over a spatio-temporal filtering of the cube
    "solver": "LSMR_ini",  # Solver for the inversion
    "proj": proj,  # EPSG system of the given coordinates
    "velo_or_disp": "velo",  # Type of data contained in the data cube ('disp' for displacements, and 'velo' for velocities)
    "verbose": True,  # Print information throughout the filtering process
}

## ---------------- Inversion and interpolation parameters ----------------- ##
inversion_kwargs = {
    "coef": coef,  # Regularization coefficient.s to be used (for each flag if flag is not None)
    "conf": False,  # If True, confidence indicators are set between 0 and 1, with 1 the lowest errors
    "unit": 365,  # 365 if the unit is m/y, 1 if the unit is m/d
    "interpolation_load_pixel": "nearest",  # Interpolation method used to load the pixel when it is not in the dataset
    "iteration": True,  # Allow the inversion process to make several iterations
    "nb_max_iteration": 10,  # Maximum number of iteration during the inversion process
    "threshold_it": 0.1,  # Threshold to test the stability of the results between each iteration, used to stop the process
    "apriori_weight": apriori_weight,  # If True, use apriori weights
    "detect_temporal_decorrelation": True,  # If True, the first inversion will use only velocity observations with small temporal baselines, to detect temporal decorelation
    "linear_operator": None,  # Perform the inversion using this specific linear operator
    "interval_output": 30,
    "option_interpol": "spline",  # Type of interpolation ('spline', 'spline_smooth', 'nearest')
    "redundancy": 5,  # Redundancy in the interpolated time series in number of days, no redundancy if None
    "result_quality": "X_contribution",  # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')
    "visual": False,  # Plot results along the way
    "path_save": path_save,  # Path where to store the results
    "verbose": False,  # Print information throughout TICOI processing
}

## ----------------------- Parallelization parameters ---------------------- ##
nb_cpu = 12  # Number of CPU to be used for parallelization
block_size = 0.5  # Maximum sub-block size (in GB) for the 'block_process' TICOI processing method

if not os.path.exists(path_save):
    os.mkdir(path_save)

# Update of dictionary with common parameters
for common_parameter in ["proj", "delete_outliers", "regu", "solver"]:
    inversion_kwargs[common_parameter] = preData_kwargs[common_parameter]


# %%========================================================================= #
#                                 DATA LOADING                                #
# =========================================================================%% #

start = [time.time()]
# Load the first cube
cube = CubeDataClass()
cube.load(cube_name, **load_kwargs)

# Prepare interpolation dates
first_date_interpol, last_date_interpol = cube.prepare_interpolation_date()
inversion_kwargs.update({"first_date_interpol": first_date_interpol, "last_date_interpol": last_date_interpol})

stop = [time.time()]
print(f"[cube_ticoi_demo] Cube of dimension (nz, nx, ny): ({cube.nz}, {cube.nx}, {cube.ny}) ")
print(f"[cube_ticoi_demo] Data loading took {round(stop[-1] - start[-1], 3)} s")


# %%========================================================================= #
#                                      TICOI                                  #
# =========================================================================%% #

start.append(time.time())

# The data cube is subdivided in smaller cubes computed one after the other in a synchronous manner (uses async)
# TICOI computation is then parallelized among those cubes
if TICOI_process == "block_process":
    result = process_blocks_refine(
        cube,
        nb_cpu=nb_cpu,
        block_size=block_size,
        preData_kwargs=preData_kwargs,
        inversion_kwargs=inversion_kwargs,
        returned=returned,
    )

# Direct computation of the whole TICOI cube
elif TICOI_process == "direct_process":
    # Preprocessing of the data (compute rolling mean for regu='1accelnotnull', delete outliers...)
    obs_filt, flag = cube.filter_cube_before_inversion(**preData_kwargs)

    # Progression bar
    xy_values = itertools.product(cube.ds["x"].values, cube.ds["y"].values)
    xy_values_tqdm = tqdm(xy_values, total=len(cube.ds["x"].values) * len(cube.ds["y"].values), mininterval=0.5)

    # Main processing of the data with TICOI algorithm, individually for each pixel
    result = Parallel(n_jobs=nb_cpu, verbose=0)(
        delayed(process)(cube, i, j, obs_filt=obs_filt, returned=returned, **inversion_kwargs)
        for i, j in xy_values_tqdm
    )

    result = {"raw": [result[i][0] for i in range(len(result))], "interp": [result[i][1] for i in range(len(result))]}

else:
    raise NameError("Please enter either direct_process or block_process")

stop.append(time.time())
print(f"[cube_ticoi_demo] TICOI processing took {round(stop[-1] - start[-1], 0)} s")


# %%========================================================================= #
#                           INITIALISATION FOR SAVING                         #
# =========================================================================%% #

start.append(time.time())
# Write down some information about the data and the TICOI processing performed
if save:
    if "invert" in returned:
        source, sensor = save_cube_parameters(cube, load_kwargs, preData_kwargs, inversion_kwargs, returned="invert")
    if "interp" in returned:
        source_interp, sensor = save_cube_parameters(
            cube, load_kwargs, preData_kwargs, inversion_kwargs, returned="interp"
        )
    stop.append(time.time())
    print(f"[cube_ticoi_demo] Initialisation took {round(stop[-1] - start[-1], 3)} s")


# %%========================================================================= #
#                                WRITING RESULTS                              #
# =========================================================================%% #

start.append(time.time())

if save:  # Save TICOI results to a netCDF file, thus obtaining a new data cube
    several = isinstance(returned, list) and len(returned) >= 2
    writer = CubeResultsWriter(cube)

    if "invert" in returned:
        cube_invert = writer.write_result_tico(
            result["invert"] if several else result,
            source,
            sensor,
            filename=f"{result_fn}_invert" if several else result_fn,
            savepath=path_save if save else None,
            result_quality=inversion_kwargs["result_quality"],
            verbose=inversion_kwargs["verbose"],
        )
    if "interp" in returned:
        cube_interp = writer.write_result_ticoi(
            result["interp"] if several else result,
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

stop.append(time.time())
if save or save_mean_velocity:
    print(f"[cube_ticoi_demo] Results saved at {path_save}")
    print(f"[cube_ticoi_demo] Writing cube to netCDF file took {round(stop[-1] - start[-1], 3)} s")

print(f"[cube_ticoi_demo] Overall processing took {round(stop[-1] - start[0], 0)} s")
