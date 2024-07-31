#!/usr/bin/env python3

"""
Optimization of the regularization coefficient value for the TICOI post-processing method, either by comparing the results
to a "ground truth" (method='ground_truth') or a zero velocity in stable ground ('stable_ground'), or by computing the
Velocity Vector Coherence of the results.

RMSE-coef and VVC-coef curves are plotted and best_coef and good_coef maps are generated.
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
from ticoi.optimize_coefficient_functions import *
# %%========================================================================= #
#                                    PARAMETERS                               #
# =========================================================================%% #

warnings.filterwarnings("ignore")

## ------------------------------ Data selection --------------------------- ##
# List of the paths where the data cubes are stored
cube_name = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "test_data"))}/Alps_Mont-Blanc_Argentiere_S2.nc'
# Path to the "ground truth" cube used to optimize the regularisation
cube_gt_name = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "test_data"))}/Alps_Mont-Blanc_Argentiere_Pleiades.nc'
flag_file = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "test_data"))}/Alpes_RGI7.shp'  # Path to flags file
mask_file = None  # Path where the mask file is stored
path_save = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "cube", "optimize_coef"))}/'  # Path where to store the results

proj = "EPSG:32632"  # EPSG system of the given coordinates

# Divide the data in several areas where different methods should be used
assign_flag = True
flag_name = {0: "stable ground", 1: "glacier"}

## --------------------------- Main parameters ----------------------------- ##
# regu = "1accelnotnull"  # Regularization method to be used (don't put it in inversion_kwargs)
regu = {0: "1accelnotnull", 1: "1accelnotnull"}
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

# Optimization method:
#    - 'ground_truth': Compare the TICOI results interpolated to a ground truth, and the ground truth (cube_name_gt) for different coefficients (coefs)
#    - 'stable_ground': Compare TICOI results to a zero velocity over stable ground for different coefficients (coefs)
#    - 'vvc': Compute the Velocity Vector Coherence of the results for different coefficients (coefs)
optimization_method = "vvc"

# Method to select the "good coef":
#    - 'curvature': Compute the second derivative of the data to find the greatest curvature (coefs must be uniformally distributed)
#    - 'min-max relative': (RECOMMENDED) Put the limit at thresh% of max(RMSE or VVC) - min(RMSE or VVC), position of the "elbow" for 1 - e^{-x} function
#    - 'max relative': Put the limit at thresh% of max(RMSE or VVC)
#    - 'absolute': Put the limit at best_RMSE + thresh or bset_VVC - thresh
#    - 'vvc_angle_thresh': Compute the incertainty of the VVC for variation of angle thresh around angle_limit, which is passed to an 'absolute' selection
#    - 'vvc_disp_thresh': Compute the incertainty of the angle for an incertainty of thresh on the displacements, which is passed to a 'vvc_angle_thresh' selection
select_method = "min-max relative"

# Parameters for the selection method
thresh = 95
angle_limit = None

# Specify the coefficients you want to test
coefs = [
    10,
    50,
    100,
    150,
    200,
    250,
    300,
    350,
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
coef_min = 0  # If coefs=None, start point of the range of coefs to be tested
coef_max = 5000  # If coefs=None, stop point of the range of coefs to be tested
step = 100  # If coefs=None, step for the range of coefs to be tested

# Visualisation options
save = True
plot_them_all = True
coef_maps = ["best", "good"]

## ------------------------ Loading parameters ------------------------- ##
load_kwargs = {
    "chunks": {},
    "conf": False,  # If True, confidence indicators will be put between 0 and 1, with 1 the lowest errors
    "subset": [342899.30, 343449.60, 5091549.77, 5092101.18],  # Subset to be loaded
    "pick_date": ["2015-01-01", "2024-01-01"],  # Select dates ([min, max] or None to select all)
    "pick_sensor": None,  # Select sensors (None to select all)
    "pick_temp_bas": None,  # Select temporal baselines ([min, max] in days or None to select all)
    "proj": proj,  # EPSG system of the given coordinates
    "mask": mask_file,  # Mask part of the data
    "verbose": False,  # Print information throughout the loading process
}

## ------------------- Data preparation parameters --------------------- ##
preData_kwargs = {
    "smooth_method": "gaussian",  # Smoothing method to be used to smooth the data in time ('gaussian', 'median', 'emwa', 'savgol')
    "s_win": 3,  # Size of the spatial window
    "t_win": 90,  # Time window size for 'ewma' smoothing
    "sigma": 3,  # Standard deviation for 'gaussian' filter
    "order": 3,  # Order of the smoothing function
    "unit": unit,  # 365 if the unit is m/y, 1 if the unit is m/d
    "delete_outliers": {
        "median_angle": 45,
        "mz_score": 3.5,
    },  # Delete data with a poor quality indicator (if int), or with aberrant direction ('vvc_angle')
    "regu": regu,  # Regularization method to be used
    "solver": solver,  # Solver for the inversion
    "proj": proj,  # EPSG system of the given coordinates
    "velo_or_disp": "velo",  # Type of data contained in the data cube ('disp' for displacements, and 'velo' for velocities)
    "verbose": False,  # Print information throughout the filtering process
}

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
    "verbose": False,  # Print information throughout TICOI processing
}

## --------------------- Interpolation parameters ---------------------- ##
if optimization_method == "ground_truth":  # The results of the inversion are interpolated to GT dates
    interpolation_kwargs = {
        "option_interpol": "spline",  # Type of interpolation ('spline', 'spline_smooth', 'nearest')
        "result_quality": result_quality,  # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')
        "unit": unit,  # 365 if the unit is m/y, 1 if the unit is m/d
    }
elif (
    optimization_method == "stable_ground" or optimization_method == "vvc"
):  # The results of the inversion are interpolated as usual
    interpolation_kwargs = {
        "interval_output": 30,  # Temporal baseline of the time series resulting from TICOI (after interpolation)
        "redundancy": 5,  # Redundancy in the interpolated time series in number of days, no redundancy if None
        "option_interpol": "spline",  # Type of interpolation ('spline', 'spline_smooth', 'nearest')
        "result_quality": result_quality,  # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')
        "unit": unit,  # 365 if the unit is m/y, 1 if the unit is m/d
    }
else:
    raise ValueError("The optimization_method must be 'ground_truth', 'stable_ground' or 'vvc'")

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

flag = None
if assign_flag:
    flag = cube.create_flag(flag_file)

cube_gt = None
if optimization_method == "ground_truth":
    # We load the "ground truth" cube
    cube_gt = cube_data_class()
    cube_gt.load(cube_gt_name, **load_kwargs)

stop = [time.time()]
print(f"Loading the data cube.s took {round((stop[-1] - start[-1]), 4)} s")
print(f"Data cube of dimension (nz,nx,ny) : ({cube.nz}, {cube.nx}, {cube.ny}) ({cube.nx * cube.ny} pixels) ")
if optimization_method == "ground_truth_data":
    print(f"Ground Truth cube of dimension (nz,nx,ny) : ({cube_gt.nz}, {cube_gt.nx}, {cube_gt.ny})")


# %% ======================================================================== #
#                         COEFFICIENT OPTIMIZATION                            #
# =========================================================================%% #

nb_points = len(cube.ds["x"].values) * len(cube.ds["y"].values)
print(f"{nb_points} points to be computed within the given subset")

start.append(time.time())

if optimization_method == "stable_ground":  # We only compute stable ground pixels
    xy_values = list(
        filter(
            bool,
            [
                (x, y) if flag.sel(x=x, y=y)["flag"].values == 0 else False
                for x in flag["x"].values
                for y in flag["y"].values
            ],
        )
    )
else:
    xy_values = list(itertools.product(cube.ds["x"].values, cube.ds["y"].values))

if optimization_process == "block_process":
    print(f"Number of CPU : {nb_cpu}")

    result = asyncio.run(
        process_blocks_main(
            cube,
            cube_gt,
            load_pixel_kwargs,
            inversion_kwargs,
            preData_kwargs,
            interpolation_kwargs,
            optimization_method=optimization_method,
            regu=regu,
            flag=flag,
            cmin=coef_min,
            cmax=coef_max,
            step=step,
            coefs=coefs,
            nb_cpu=nb_cpu,
            block_size=block_size,
            verbose=False,
        )
    )

elif optimization_process == "direct_process":
    obs_filt, flag = cube.filter_cube(**preData_kwargs, flag=flag)

    # Progression bar
    xy_values_tqdm = tqdm(xy_values, total=len(xy_values), mininterval=0.5)

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
            method=optimization_method,
            regu=regu,
            flag=flag,
            cmin=coef_min,
            cmax=coef_max,
            step=step,
            coefs=coefs,
            stats=True,
            visual=False,
        )
        for i, j in xy_values_tqdm
    )

stop.append(time.time())
print(f"Whole coefficient optimization took {round((stop[-1] - start[-1]), 1)} s")


# %% ======================================================================== #
#                                  COEF MAPS                                  #
# =========================================================================%% #

start.append(time.time())

# Converting coefs to an array
if coefs is None:
    coefs = np.arange(coef_min, coef_max + 1, step)
else:
    coefs = np.array(coefs)

driver = gdal.GetDriverByName("GTiff")
srs = osr.SpatialReference()
srs.ImportFromEPSG(int(proj.split(":")[1]))

# Remove pixels with no data
empty = list(filter(bool, [d if result[d] is not None else False for d in range(len(result))]))
positions = np.array(list(itertools.product(cube.ds["x"].values, cube.ds["y"].values)))[empty, :]
result = [result[i] for i in empty]
xy_values = [xy_values[i] for i in empty]

nb_res = len(result)
nb_data = np.array([result[i].nb_data if result[i] is not None else 0 for i in range(nb_res)])
mean_baseline = np.array([result[i].mean_temporal_baseline if result[i] is not None else np.nan for i in range(nb_res)])
mean_disp = np.array([result[i].mean_disp if result[i] is not None else np.nan for i in range(nb_res)])
mean_angle_to_median = np.array(
    [result[i].mean_angle_to_median if result[i] is not None else np.nan for i in range(nb_res)]
)
mean_v = np.array([result[i].mean_v if result[i] is not None else np.nan for i in range(nb_res)])
std_v = np.array([result[i].std_raw_data if result[i] is not None else np.nan for i in range(nb_res)])
measures_result = np.array(
    [result[i].values if result[i] is not None else [np.nan for _ in range(len(coefs))] for i in range(nb_res)]
)

# Coordinates information
resolution = int(cube.ds["x"].values[1] - cube.ds["x"].values[0])
long_data = (positions[:, 0] - np.min(cube.ds["x"].values)).astype(int) // resolution
lat_data = (positions[:, 1] - np.min(cube.ds["y"].values)).astype(int) // resolution

goods = [
    find_good_coefs(
        coefs,
        result[i].values,
        method=optimization_method,
        select_method=select_method,
        thresh=thresh,
        mean_disp=(np.nanmean(mean_disp[:, 0]), np.nanmean(mean_disp[:, 1])),
        mean_angle=angle_limit if angle_limit is not None else np.nanmax(mean_angle_to_median),
    )
    for i in range(len(result))
]

# Map with the coefficient giving the best RMSE with GT data in band 1 and the value of this RMSE in band 2
if "best" in coef_maps:
    best_coef_map = np.empty([cube.nx, cube.ny])
    best_coef_map[:, :] = np.nan
    best_measure_map = np.empty([cube.nx, cube.ny])
    best_measure_map[:, :] = np.nan

    best_coef_map[long_data, lat_data] = [goods[i][3] for i in range(len(result))]
    best_measure_map[long_data, lat_data] = [goods[i][2] for i in range(len(result))]

    tiff = driver.Create(
        f"{path_save}best_map.tiff", best_coef_map.shape[0], best_coef_map.shape[1], 2, gdal.GDT_Float32
    )
    tiff.SetGeoTransform([np.min(cube.ds["x"].values), resolution, 0, np.max(cube.ds["y"].values), 0, -resolution])
    tiff.GetRasterBand(1).WriteArray(np.flip(best_coef_map.T, axis=0))
    tiff.GetRasterBand(2).WriteArray(np.flip(best_measure_map.T, axis=0))
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
    good_measure_map = np.empty([cube.nx, cube.ny])
    good_measure_map[:, :] = np.nan

    good_measure_map[long_data, lat_data] = [goods[i][0] for i in range(len(result))]
    min_good_coef_map[long_data, lat_data] = [np.min(goods[i][1]) for i in range(len(result))]
    max_good_coef_map[long_data, lat_data] = [np.max(goods[i][1]) for i in range(len(result))]

    tiff = driver.Create(
        f"{path_save}good_map.tiff", good_measure_map.shape[0], good_measure_map.shape[1], 3, gdal.GDT_Float32
    )
    tiff.SetGeoTransform([np.min(cube.ds["x"].values), resolution, 0, np.max(cube.ds["y"].values), 0, -resolution])
    tiff.GetRasterBand(1).WriteArray(np.flip(min_good_coef_map.T, axis=0))
    tiff.GetRasterBand(2).WriteArray(np.flip(max_good_coef_map.T, axis=0))
    tiff.GetRasterBand(3).WriteArray(np.flip(good_measure_map.T, axis=0))
    tiff.SetProjection(srs.ExportToWkt())

    tiff = None

driver = None

stop.append(time.time())
print(f"Coefficients maps saved in {path_save}")
print(f"Generating the maps took {round(stop[-1] - start[-1], 1)} s")


# %% ======================================================================== #
#                          RMSE/VVC-COEF CURVES                               #
# =========================================================================%% #

start.append(time.time())

# For each areas of the cube we compute the RMSE-coef curves
if flag is None:
    regu = {0: regu}
for key in regu.keys():
    if flag is not None:  # Select the corresponding area
        mask_regu = [flag.sel(x=x, y=y)["flag"].values == key for x, y in xy_values]
    else:
        mask_regu = [True for i in range(nb_res)]

    if any(mask_regu):
        nb_data_regu = nb_data[mask_regu]
        mean_baseline_regu = mean_baseline[mask_regu]
        mean_angle_to_median_regu = mean_angle_to_median[mask_regu]
        mean_disp_regu = mean_disp[mask_regu]
        mean_v_regu = mean_v[mask_regu]
        std_v_regu = std_v[mask_regu]
        measures_result_regu = measures_result[mask_regu]

        print(f"Area {flag_name[key]} :")
        print(f"{len(nb_data_regu)} pixels in the area")

        print(f"Nb data median S2 : {np.median(nb_data_regu[:, 0])}")
        if optimization_method == "ground_truth":
            print(f"Nb data median Pleiades : {np.median(nb_data_regu[:, 1])}")

        print(f"Mean temporal baseline : {np.nanmean(mean_baseline_regu)}")
        print(f"Mean disp S2 : {np.nanmean(np.sqrt(mean_disp_regu[:, 0] ** 2 + mean_disp_regu[:, 1] ** 2))} m")
        print(f"Mean velocity S2 : {np.nanmean(np.sqrt(mean_v_regu[:, 0] ** 2 + mean_v_regu[:, 1] ** 2))} m/y")
        print(f"Mean std dev S2 : {np.nanmean(np.sqrt(std_v_regu[:, 0] ** 2 + std_v_regu[:, 1] ** 2))} m/y")
        if optimization_method == "ground_truth":
            print(f"Mean velocity Pleiades : {np.nanmean(np.sqrt(mean_v_regu[:, 2] ** 2 + mean_v_regu[:, 3] ** 2))}")

        if optimization_method == "ground_truth" or optimization_method == "stable_ground":
            # Average RMSE on the area
            measures = np.array(
                [
                    np.sqrt(
                        1
                        / np.sum(nb_data_regu[:, 1])
                        * np.sum(
                            nb_data_regu[:, 1][~(nb_data_regu[:, 1] == 0)]
                            * measures_result_regu[~np.isnan(measures_result_regu).any(axis=1)][:, i] ** 2
                        )
                    )
                    for i in range(len(coefs))
                ]
            )
        else:
            measures = np.array(
                [
                    np.nanmean(measures_result_regu[~np.isnan(measures_result_regu).any(axis=1)][:, i])
                    for i in range(len(coefs))
                ]
            )

        # Compute median good_coef, best_coef and best_measure over the area
        coefs = np.array(coefs)
        good_coef = np.nanmedian(min_good_coef_map)
        good_measure = measures[np.argmin(np.abs(coefs - good_coef))]
        best_coef = np.nanmedian(best_coef_map)
        best_measure = measures[np.argmin(np.abs(coefs - best_coef))]
        term = "RMSE" if optimization_method in ["ground_truth", "stable_ground"] else "VVC"

        # Plot result
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            coefs[~(measures < good_measure)],
            measures[~(measures < good_measure)],
            marker="x",
            markersize=7,
            linestyle="",
            markeredgecolor="darkred" if optimization_method in ["ground_truth", "stable_ground"] else "darkgreen",
        )
        ax.plot(
            coefs[measures < good_measure],
            measures[measures < good_measure],
            marker="x",
            markersize=7,
            linestyle="",
            markeredgecolor="darkgreen" if optimization_method in ["ground_truth", "stable_ground"] else "darkred",
        )

        ax.plot([coefs[0], coefs[-1]], [good_measure, good_measure], linestyle="--", color="midnightblue")
        ax.set_xlabel("Regularisation coefficient value", fontsize=14)
        ax.set_xlim([np.min(coefs), np.max(coefs)])
        if optimization_method == "ground_truth":
            ax.set_ylabel("Average RMSE between TICOI results and GT data [m/y]", fontsize=14)
        elif optimization_method == "stable_ground":
            ax.set_ylabel("Average RMSE between TICOI results and zero velocity [m/y]", fontsize=14)
        elif optimization_method == "VVC":
            ax.set_ylabel("Average VVC of TICOI results", fontsize=14)
        fig.suptitle(
            f'{term}-coef average curve for the {flag_name[key] if type(flag_name) == dict else ""} area with regu={regu[key]}\n'
            + f"Best for coef = {best_coef} ({term} = {best_measure})\n"
            + f"Good for coef = {good_coef}",
            fontsize=16,
        )
        plt.subplots_adjust(top=0.85)

        if save and flag is None:
            fig.savefig(f"{path_save}{term}_coef_{regu[key]}.png")
        elif flag is not None:
            fig.savefig(f"{path_save}{term}_coef_{flag_name[key]}_{regu[key]}.png")

        # Plot every RMSE-coef curves which
        if plot_them_all:
            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 12))
            ax[0].set_xlabel("Regularisation coefficient value", fontsize=14)
            ax[1].set_xlabel("Regularisation coefficient value", fontsize=14)
            ax[0].set_xlim([np.min(coefs), np.max(coefs)])
            ax[1].set_xlim([np.min(coefs), np.max(coefs)])
            ax[0].set_ylabel(f"{term}{' [m/y]' if term == 'RMSE' else ''}", fontsize=14)
            ax[1].set_ylabel(f"mean-substracted {term}{' [m/y]' if term == 'RMSE' else ''}", fontsize=14)

            Q1 = np.percentile(nb_data[:, 1][nb_data[:, 1] > 0], 25)
            median = np.median(nb_data[:, 1][nb_data[:, 1] > 0])
            Q3 = np.percentile(nb_data[:, 1][nb_data[:, 1] > 0], 75)
            for r in range(len(measures_result)):
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

                    ax[0].plot(coefs, measures_result[r, :], linestyle="dashed", color=color, alpha=alpha)
                    ax[1].plot(
                        coefs,
                        measures_result[r, :] - np.min(measures_result[r, :]),
                        linestyle="dashed",
                        color=color,
                        alpha=alpha,
                    )

            ax[0].plot(coefs, measures, linestyle="dashdot", linewidth=2, color="blueviolet")
            ax[1].plot(coefs, measures - np.min(measures), linestyle="dashdot", linewidth=2, color="blueviolet")

            fig.suptitle(
                f'RMSE between TICOI results and GT data for the {flag_name[key] if type(flag_name) == dict else ""} area\n'
                + "when changing the regularisation coefficient",
                y=0.95,
                fontsize=16,
            )

            if save and flag is None:
                fig.savefig(f"{path_save}RMSE_coef_{regu[key]}_allplots.png")
            elif save:
                fig.savefig(f"{path_save}RMSE_coef_{flag_name[key]}_{regu[key]}_allplots.png")

        plt.show()

stop.append(time.time())
print(f"Overall processing took {round(stop[-1] - start[0], 1)} s")
