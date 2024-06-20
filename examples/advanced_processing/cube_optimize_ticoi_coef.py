#!/usr/bin/env python3

"""
Coefficient optimization of the TICOI post-processing method, according to "ground truth" given data (GPS, more
precise satellitarian data...). A range of coefficients is tested for a given regularisation method, by computing
the RMSE between TICOI results for the tested coefficient, interpolated to the ground truth dates, and compared
to those ground truth dates using the Root Mean Square Error (RMSE).
This code computes a RMSE-coefficient curve for every pixel of a given subset (or whole data cube).
"""

import asyncio
import itertools
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from osgeo import gdal, osr
from tqdm import tqdm

from ticoi.core import chunk_to_block, load_block
from ticoi.cube_data_classxr import cube_data_class
from ticoi.other_functions import optimize_coef

# %%========================================================================= #
#                                    PARAMETERS                               #
# =========================================================================%% #

warnings.filterwarnings("ignore")

## ------------------------------ Data selection --------------------------- ##
# List of the paths where the data cubes are stored
# cube_name = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "test_data"))}/Alps_Mont-Blanc_Argentiere_S2.nc'
cube_name = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..", "test_data", "cubes_Sentinel_2_2022_2023"))}/c_x01470_y03675.nc'
# Path to the "ground truth" cube used to optimize the regularisation
# cube_gt_name = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "test_data"))}/Alps_Mont-Blanc_Argentiere_Pleiades.nc'
cube_gt_name = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..", "test_data", "cubes_Pleiades"))}/stack_median_pleiades_alllayers_2012-2022_modiflaurane.nc'
flag_file = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "test_data"))}/Alps_Mont-Blanc_flags.nc'  # Path to flags file
mask_file = None  # Path where the mask file is stored
path_save = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "cube", "optimize_coef"))}/'  # Path where to store the results

proj = "EPSG:32632"  # EPSG system of the given coordinates

# Divide the data in several areas where different methods should be used
assign_flag = False
flag = None  # Do not put it in load_kwargs and/or preData_kwargs but pass it to optimize_coef directly
if assign_flag:
    flag = xr.open_dataset(flag_file)
    flag.load()
    if "flags" in list(flag.variables):
        flag = flag.rename({"flags": "flag"})

flag_name = {0: "stable ground", 1: "glacier"}

## --------------------------- Main parameters ----------------------------- ##
regu = "1accelnotnull"  # Regularization method to be used (don't put it in inversion_kwargs)
# regu = {0: 1, 1: "1accelnotnull"}
solver = "LSMR_ini"  # Solver for the inversion
unit = 365  # 1 for m/d, 365 for m/y
result_quality = (
    "X_contribution"  # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')
)

## ----------------------- Optimization parameters ------------------------- ##
# Choose optimization processing method you want to use:
#    - 'block_process' (recommended) : This implementation divides the data in smaller data cubes processed one after the other in a synchronous manner,
# in order to avoid memory overconsumption and kernel crashing. Computations within the blocks are parallelized so this method goes way faster
# than the 'direct_process' method
#      /!\ This implementation uses asyncio (way faster) which requires its own event loop to run : if you launch this code from a raw terminal,
# there should be no problem, but if you try to launch it from some IDE (like Spyder), think of specifying to your IDE to launch it
# in a raw terminal instead of the default console (which leads to a RuntimeError)
#    - 'direct_process' : No subdivisition of the data is made beforehand which generally leads to memory overconsumption and kernel crashes
# if the amount of pixel to compute is too high (depending on your available memory). If you want to process big amount of data, you should use
# 'block_process', which is also faster. This method is essentially used for debug purposes.optimization_process = 'direct_process'
optimization_process = "block_process"
# Specify the coefficients you want to test
coefs = [
    20,
    40,
    60,
    80,
    100,
    120,
    140,
    160,
    180,
    200,
    240,
    280,
    320,
    360,
    400,
    450,
    500,
    550,
    600,
    700,
    800,
    900,
    1000,
    1200,
    1400,
    1600,
    1800,
    2000,
    2500,
    3000,
    3500,
    4000,
    4500,
    5000,
    6000,
    7000,
    8000,
    9000,
    10000,
]
coef_min = 10  # If coefs=None, start point of the range of coefs to be tested
coef_max = 1000  # If coefs=None, stop point of the range of coefs to be tested
step = 10  # If coefs=None, step for the range of coefs to be tested
stats = False  # Compute some statistics on raw data and GT data
# Visualisation options
save = True
plot_them_all = True
coef_maps = ["best", "good"]

## ------------------------ Loading parameters ------------------------- ##
load_kwargs = {
    "chunks": {},
    "conf": False,  # If True, confidence indicators will be put between 0 and 1, with 1 the lowest errors
    "subset": [
        338703.2,
        339258.9,
        5081177.4,
        5081947.2,
    ],  # Area to be loaded around the pixel ([longitude, latitude, buffer size] or None)
    "pick_date": ["2015-01-01", "2024-01-01"],  # Select dates ([min, max] or None to select all)
    "pick_sensor": None,  # Select sensors (None to select all)
    "pick_temp_bas": None,  # Select temporal baselines ([min, max] in days or None to select all)
    "proj": proj,  # EPSG system of the given coordinates
    "mask": mask_file,
    "verbose": False,
}  # Print information throughout the loading process

## ------------------- Data preparation parameters --------------------- ##
preData_kwargs = {
    "smooth_method": "gaussian",  # Smoothing method to be used to smooth the data in time ('gaussian', 'median', 'emwa', 'savgol')
    "s_win": 3,  # Size of the spatial window
    "t_win": 90,  # Time window size for 'ewma' smoothing
    "sigma": 3,  # Standard deviation for 'gaussian' filter
    "order": 3,  # Order of the smoothing function
    "unit": unit,  # 365 if the unit is m/y, 1 if the unit is m/d
    "delete_outliers": "vvc_angle",  # Delete data with a poor quality indicator (if int), or with aberrant direction ('vvc_angle')
    "regu": regu,  # Regularization method to be used
    "solver": solver,  # Solver for the inversion
    "proj": proj,  # EPSG system of the given coordinates
    "velo_or_disp": "velo",  # Type of data contained in the data cube ('disp' for displacements, and 'velo' for velocities)
    "verbose": False,
}  # Print information throughout the filtering process

## -------------- Parameters for the pixel loading part ---------------- ##
load_pixel_kwargs = {
    "regu": regu,  # Regularization method to be used
    "coef": 0 if type(regu) != dict else {key: 0 for key in regu.keys()},  # Initialisation to a useless value
    "solver": solver,  # Solver for the inversion
    "proj": proj,  # EPSG system of the given coordinates
    "interp": "nearest",  # Interpolation method used to load the pixel when it is not in the dataset
    "visual": False,  # Plot results along the way
}

## ----------------------- Inversion parameters ------------------------ ##
inversion_kwargs = {
    "regu": regu,  # Regularization method to be used
    "solver": solver,  # Solver for the inversion
    "conf": False,  # If True, confidence indicators are set between 0 and 1, with 1 the lowest errors
    "unit": unit,  # 365 if the unit is m/y, 1 if the unit is m/d
    "iteration": True,  # Allow the inversion process to make several iterations
    "nb_max_iteration": 10,  # Maximum number of iteration during the inversion process
    "threshold_it": 0.1,  # Threshold to test the stability of the results between each iteration, used to stop the process
    "apriori_weight": True,  # If True, use apriori weights
    "detect_temporal_decorrelation": True,  # If True, the first inversion will use only velocity observations with small temporal baselines, to detect temporal decorelation
    "linear_operator": None,  # Perform the inversion using this specific linear operator
    "result_quality": result_quality,  # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')
    "visual": False,  # Plot results along the way
    "verbose": False,
}  # Print information throughout TICOI processing

## --------------------- Interpolation parameters ---------------------- ##
interpolation_kwargs = {
    "option_interpol": "spline",  # Type of interpolation ('spline', 'spline_smooth', 'nearest')
    "result_quality": result_quality,  # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')
    "unit": unit,
}  # 365 if the unit is m/y, 1 if the unit is m/d

## ----------------------- Parallelization parameters ---------------------- ##
nb_cpu = 8  # Number of CPU to be used for parallelization
block_size = 0.5  # Maximum sub-block size (in GB) for the 'block_process' TICOI processing method

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
print(f"[Data loading] Loading the data cube.s took {round((stop[-1] - start[-1]), 4)} s")
print(f"[Data loading] Data cube of dimension (nz,nx,ny) : ({cube.nz}, {cube.nx}, {cube.ny}) ")
print(f"[Data loading] Ground Truth cube of dimension (nz,nx,ny) : ({cube_gt.nz}, {cube_gt.nx}, {cube_gt.ny})")


# %% ======================================================================== #
#                         COEFFICIENT OPTIMIZATION                            #
# =========================================================================%% #


async def process_block(
    block,
    cube_gt,
    load_pixel_kwargs,
    inversion_kwargs,
    interpolation_kwargs,
    regu=None,
    flag=None,
    cmin=10,
    cmax=1000,
    step=10,
    coefs=None,
    stats=False,
    nb_cpu=8,
):

    # Progression bar
    xy_values = itertools.product(block.ds["x"].values, block.ds["y"].values)
    xy_values_tqdm = tqdm(xy_values, total=(block.nx * block.ny))

    # Filter cube
    obs_filt, flag_block = block.filter_cube(**preData_kwargs, flag=flag)

    # Optimization of the coefficient for every pixels of the block
    #    (faster using parallelization here and sequential processing in optimize_coef)
    result_block = Parallel(n_jobs=nb_cpu, verbose=0)(
        delayed(optimize_coef)(
            block,
            cube_gt,
            i,
            j,
            obs_filt,
            load_pixel_kwargs,
            inversion_kwargs,
            interpolation_kwargs,
            regu=regu,
            flag=flag_block,
            cmin=cmin,
            cmax=cmax,
            step=step,
            coefs=coefs,
            stats=stats,
            parallel=False,
            visual=False,
        )
        for i, j in xy_values_tqdm
    )

    return result_block


async def process_blocks_main(
    cube,
    cube_gt,
    load_pixel_kwargs,
    inversion_kwargs,
    interpolation_kwargs,
    regu=None,
    flag=None,
    cmin=10,
    cmax=1000,
    step=10,
    coefs=None,
    stats=False,
    nb_cpu=8,
    block_size=0.5,
    verbose=False,
):

    blocks = chunk_to_block(cube, block_size=block_size, verbose=True)

    dataf_list = [None] * (cube.nx * cube.ny)

    loop = asyncio.get_event_loop()

    for n in range(len(blocks)):
        print(f"[Block process] Processing block {n+1}/{len(blocks)}")
        print(f"Processing block {n+1}/{len(blocks)}")

        # Load the first block and start the loop
        if n == 0:
            x_start, x_end, y_start, y_end = blocks[0]
            future = loop.run_in_executor(None, load_block, cube, x_start, x_end, y_start, y_end)

        block, duration = await future
        print(f"[Block process] Block {n+1} loaded in {duration:.2f} s")
        if verbose:
            print(f"[Block process] Block {n+1} loaded in {duration:.2f} s")

        if n < len(blocks) - 1:
            # Load the next block while processing the current block
            x_start, x_end, y_start, y_end = blocks[n + 1]
            future = loop.run_in_executor(None, load_block, cube, x_start, x_end, y_start, y_end)

        block_result = await process_block(
            block,
            cube_gt,
            load_pixel_kwargs,
            inversion_kwargs,
            interpolation_kwargs,
            regu=regu,
            flag=flag,
            cmin=cmin,
            cmax=cmax,
            step=step,
            coefs=coefs,
            stats=stats,
            nb_cpu=nb_cpu,
        )

        for i in range(len(block_result)):
            row = i % block.ny + blocks[n][2]
            col = np.floor(i / block.ny) + blocks[n][0]
            idx = int(col * cube.ny + row)

            dataf_list[idx] = block_result[i]

        del block_result, block

    return dataf_list


nb_points = len(cube.ds["x"].values) * len(cube.ds["y"].values)
print(f"[Coef optimization] Number of CPU : {nb_cpu}")
print(f"[Coef optimization] {nb_points} points to be computed within the given subset")

start.append(time.time())

if optimization_process == "block_process":
    result = asyncio.run(
        process_blocks_main(
            cube,
            cube_gt,
            load_pixel_kwargs,
            inversion_kwargs,
            interpolation_kwargs,
            regu=regu,
            flag=flag,
            cmin=coef_min,
            cmax=coef_max,
            step=step,
            coefs=coefs,
            stats=stats,
            nb_cpu=nb_cpu,
            block_size=block_size,
            verbose=False,
        )
    )

elif optimization_process == "direct_process":
    obs_filt = cube.filter_cube(**preData_kwargs, flag=flag)

    # Progression bar
    xy_values = itertools.product(cube.ds["x"].values, cube.ds["y"].values)
    xy_values_tqdm = tqdm(xy_values, total=len(cube.ds["x"].values) * len(cube.ds["y"].values), mininterval=0.5)

    result = Parallel(n_jobs=nb_cpu, verbose=0)(
        delayed(optimize_coef)(
            cube,
            cube_gt,
            i,
            j,
            obs_filt,
            load_pixel_kwargs,
            inversion_kwargs,
            interpolation_kwargs,
            regu=regu,
            flag=flag,
            cmin=coef_min,
            cmax=coef_max,
            step=step,
            coefs=coefs,
            stats=stats,
            visual=False,
        )
        for i, j in xy_values_tqdm
    )

stop.append(time.time())
print(f"[Coef optimization] Whole coefficient optimization took {round((stop[-1] - start[-1]), 1)} s")


# %% ======================================================================== #
#                                  COEF MAPS                                  #
# =========================================================================%% #

start.append(time.time())

# Converting coefs to an array
if coefs is None:
    coefs = np.arange(coef_min, coef_max, step)
else:
    coefs = np.array(coefs)

driver = gdal.GetDriverByName("GTiff")
srs = osr.SpatialReference()
srs.ImportFromEPSG(int(proj.split(":")[1]))

# Remove pixels with no data
empty = list(filter(bool, [d if result[d] is not None else False for d in range(len(result))]))
positions = np.array(list(itertools.product(cube.ds["x"].values, cube.ds["y"].values)))[empty, :]
result = [result[i] for i in empty]

# Coordinates information
resolution = int(cube.ds["x"].values[1] - cube.ds["x"].values[0])
long_data = (positions[:, 0] - np.min(cube.ds["x"].values)).astype(int) // resolution
lat_data = (positions[:, 1] - np.min(cube.ds["y"].values)).astype(int) // resolution

# Map with the coefficient giving the best RMSE with GT data in band 1 and the value of this RMSE in band 2
if "best" in coef_maps:
    best_coef_map = np.empty([cube.nx, cube.ny])
    best_coef_map[:, :] = np.nan
    best_RMSE_map = np.empty([cube.nx, cube.ny])
    best_RMSE_map[:, :] = np.nan

    best_coef_map[long_data, lat_data] = [coefs[np.argmin(result[i].values)] for i in range(len(result))]
    best_RMSE_map[long_data, lat_data] = [np.min(result[i].values) for i in range(len(result))]

    tiff = driver.Create(
        f"{path_save}best_map.tiff", best_coef_map.shape[0], best_coef_map.shape[1], 2, gdal.GDT_Float32
    )
    tiff.SetGeoTransform([np.min(cube.ds["x"].values), resolution, 0, np.max(cube.ds["y"].values), 0, -resolution])
    tiff.GetRasterBand(1).WriteArray(np.flip(best_coef_map.T, axis=0))
    tiff.GetRasterBand(2).WriteArray(np.flip(best_RMSE_map.T, axis=0))
    tiff.SetProjection(srs.ExportToWkt())

    tiff = None

# Map with the minimum and maximum value of coefficients giving a "good" RMSE (relatively to the best RMSE) in band 1 and 2
# and the RMSE value considered as "good" in band 3. A strong hypothesis is done by considering that every coefficient values
# in the given range give a "good" RMSE : it is often the case (plateau, minimum) but not always (local minimum).
if "good" in coef_maps:
    min_good_coef_map = np.empty([cube.nx, cube.ny])
    min_good_coef_map[:, :] = np.nan
    max_good_coef_map = np.empty([cube.nx, cube.ny])
    max_good_coef_map[:, :] = np.nan
    good_RMSE_map = np.empty([cube.nx, cube.ny])
    good_RMSE_map[:, :] = np.nan

    good_RMSE_result = [1.05 * np.min(result[i].values) for i in range(len(result))]
    good_RMSE_map[long_data, lat_data] = good_RMSE_result

    min_good_coef_map[long_data, lat_data] = [
        np.min(coefs[result[i].values < good_RMSE_result[i]]) for i in range(len(result))
    ]
    max_good_coef_map[long_data, lat_data] = [
        np.max(coefs[result[i].values < good_RMSE_result[i]]) for i in range(len(result))
    ]

    tiff = driver.Create(
        f"{path_save}good_map.tiff", good_RMSE_map.shape[0], good_RMSE_map.shape[1], 3, gdal.GDT_Float32
    )
    tiff.SetGeoTransform([np.min(cube.ds["x"].values), resolution, 0, np.max(cube.ds["y"].values), 0, -resolution])
    tiff.GetRasterBand(1).WriteArray(np.flip(min_good_coef_map.T, axis=0))
    tiff.GetRasterBand(2).WriteArray(np.flip(max_good_coef_map.T, axis=0))
    tiff.GetRasterBand(3).WriteArray(np.flip(good_RMSE_map.T, axis=0))
    tiff.SetProjection(srs.ExportToWkt())

    tiff = None

driver = None

stop.append(time.time())
print(f"[Coef maps] Coefficients maps saved in {path_save}")
print(f"[Coef maps] Generating the maps took {round(stop[-1] - start[-1], 1)} s")


# %% ======================================================================== #
#                             RMSE-COEF CURVES                                #
# =========================================================================%% #

start.append(time.time())

nb_res = len(result)
nb_data = np.array([result[i].nb_data if result[i] is not None else 0 for i in range(nb_res)])
RMSEs_result = np.array(
    [result[i].values if result[i] is not None else [np.nan for _ in range(len(coefs))] for i in range(nb_res)]
)

# For each areas of the cube we compute the RMSE-coef curves
if flag is None:
    regu = {0: regu}
for key in regu.keys():
    mask_regu = [result[i].regu == regu[key] for i in range(nb_res)]
    if any(mask_regu) is True:
        nb_data_regu = nb_data[mask_regu]
        RMSEs_result_regu = RMSEs_result[mask_regu]

        # Average RMSE on the area
        RMSEs = np.array(
            [
                np.sqrt(
                    1
                    / np.sum(nb_data_regu[:, 1])
                    * np.sum(
                        nb_data_regu[:, 1][~(nb_data_regu[:, 1] == 0)]
                        * RMSEs_result_regu[~np.isnan(RMSEs_result_regu).any(axis=1)][:, i] ** 2
                    )
                )
                for i in range(len(coefs))
            ]
        )

        best_coef = coefs[np.argmin(RMSEs)]
        best_RMSE = np.min(RMSEs)
        # good_RMSE = max(1.05 * best_RMSE, best_RMSE + mean_std_p)
        good_RMSE = 1.05 * best_RMSE

        # Plot result
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            coefs[~(RMSEs < good_RMSE)],
            RMSEs[~(RMSEs < good_RMSE)],
            marker="x",
            markersize=7,
            linestyle="",
            markeredgecolor="darkred",
        )
        ax.plot(
            coefs[RMSEs < good_RMSE],
            RMSEs[RMSEs < good_RMSE],
            marker="x",
            markersize=7,
            linestyle="",
            markeredgecolor="darkgreen",
        )
        ax.plot([coefs[0], coefs[-1]], [good_RMSE, good_RMSE], linestyle="--", color="midnightblue")
        ax.set_xlabel("Regularisation coefficient value", fontsize=14)
        ax.set_xlim([np.min(coefs), np.max(coefs)])
        ax.set_ylabel("Average RMSE between TICOI results and GT data [m/y]", fontsize=14)
        fig.suptitle(
            f'RMSE-coef average curve for the {flag_name[key] if type(flag_name) == dict else ""} area with regu={regu[key]}\n'
            + f"Best for coef = {best_coef} (RMSE = {best_RMSE})",
            fontsize=16,
        )

        if save and flag is None:
            fig.savefig(f"{path_save}RMSE_coef_{regu[key]}.png")
        elif flag is not None:
            fig.savefig(f"{path_save}RMSE_coef_{flag_name[key]}_{regu[key]}.png")

        # Plot every RMSE-coef curves which
        if plot_them_all:
            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))
            ax[0].set_xlabel("Regularisation coefficient value", fontsize=14)
            ax[1].set_xlabel("Regularisation coefficient value", fontsize=14)
            ax[0].set_xlim([np.min(coefs), np.max(coefs)])
            ax[1].set_xlim([np.min(coefs), np.max(coefs)])
            ax[0].set_ylabel("RMSE [m/y]", fontsize=14)
            ax[1].set_ylabel("mean-substracted RMSE [m/y]", fontsize=14)

            Q1 = np.percentile(nb_data[:, 1][nb_data[:, 1] > 0], 25)
            median = np.median(nb_data[:, 1][nb_data[:, 1] > 0])
            Q3 = np.percentile(nb_data[:, 1][nb_data[:, 1] > 0], 75)
            for r in range(len(result)):
                if nb_data[:, 1][r] > 0:
                    if nb_data[:, 1][r] > Q3:
                        color = "green"
                        alpha = 0.6
                    elif nb_data[:, 1][r] > median:
                        color = "blue"
                        alpha = 0.5
                    elif nb_data[:, 1][r] > Q1:
                        color = "red"
                        alpha = 0.4
                    else:
                        color = "black"
                        alpha = 0.3

                    ax[0].plot(coefs, RMSEs_result[r, :], linestyle="dashed", color=color, alpha=alpha)
                    ax[1].plot(
                        coefs,
                        RMSEs_result[r, :] - np.min(RMSEs_result[r, :]),
                        linestyle="dashed",
                        color=color,
                        alpha=alpha,
                    )

            ax[0].plot(coefs, RMSEs, linestyle="dashdot", linewidth=2, color="blueviolet")
            ax[1].plot(coefs, RMSEs - np.min(RMSEs), linestyle="dashdot", linewidth=2, color="blueviolet")

            fig.suptitle(
                f'RMSE between TICOI results and GT data for the {flag_name[key] if type(flag_name) == dict else ""} area\n'
                + "when changing the regularisation coefficient",
                y=0.95,
                fontsize=16,
            )

            if save and flag is None:
                fig.savefig(f"{path_save}RMSE_coef_{regu[key]}_allplots.png")
            elif flag is not None:
                fig.savefig(f"{path_save}RMSE_coef_{flag_name[key]}_{regu[key]}_allplots.png")

        plt.show()

stop.append(time.time())
print(f"[Overall] Overall processing took {round(stop[-1] - start[0], 1)} s")
