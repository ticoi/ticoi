#!/usr/bin/env python3

"""
Coefficient optimization of the TICOI post-processing method, according to a "ground truth" given data cube (GPS,
more precise satellitarian data...). A range of coefficients is tested for a given regularisation method, by computing
the RMSE between TICOI results for the tested coefficient, interpolated to the ground truth dates, and compared
to those ground truth data computing the Root Mean Square Error (RMSE). A RMSE-coefficient curve is then plotted.
"""

import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as sm
import xarray as xr

from ticoi.core import interpolation_to_data, inversion_core
from ticoi.cube_data_classxr import cube_data_class
from ticoi.utilis import optimize_coef

# %%===================================================================== #
#                               PARAMETERS                                #
# =====================================================================%% #

warnings.filterwarnings("ignore")

## ---------------------------- Data selection ------------------------- ##
# List of the paths where the data cubes are stored
cube_name = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "test_data"))}/Alps_Mont-Blanc_Argentiere_S2.nc'
# Path to the "ground truth" cube used to optimize the regularisation
cube_gt_name = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "test_data"))}/Alps_Mont-Blanc_Argentiere_Pleiades.nc'
path_save = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "pixel", "optimize_coef"))}/'  # Path where to store the results
proj = "EPSG:32632"  # EPSG system of the given coordinates

i, j = 342537.1, 5092253.3  # Point (pixel) where to carry on the computation
# i, j = 343309.23, 5091987.07

## ------------------------- Main parameters --------------------------- ##
regu = "1accelnotnull"  # Regularization method to be used
solver = "LSMR_ini"  # Solver for the inversion
unit = 365  # 1 for m/d, 365 for m/y
result_quality = (
    "X_contribution"  # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')
)

## --------------------- Visualization parameters ---------------------- ##
verbose = False  # Print information throughout TICOI processing
visual = False  # Plot information along the way
save = False  # Save the results or not
# Visualisation options for
option_visual = [
    "original_velocity_xy",
    "original_magnitude",
    "X_magnitude_zoom",
    "X_magnitude",
    "X_zoom",
    "X",
    "vv_quality",
    "vxvy_quality",
    "Residu_magnitude",
    "Residu",
    "X_z",
    "Y_contribution",
    "direction",
]
vmax = [False, False]  # Vertical limits for the plots

## --------------------- Optimization parameters ----------------------- ##
# Specify the coefficients you want to test
# To specify the coefficients to be tested, if None, range(cmin, cmax, step) coefs will be tested
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
stats = True  # Compute some statistics on raw data and GT data
# Visualization during optimization (for each coefficient) /!\ Can generate a lot of figures...
visual_opt = False  # Plot interpolated and GT velocities
plot_raw = True  # Add raw data to the plot
vminmax = None  # Specify the vertical limits of the plot
savedir = path_save  # Save the figure to this location

## ------------------------ Loading parameters ------------------------- ##
load_kwargs = {
    "chunks": {},
    "conf": False,  # If True, confidence indicators will be put between 0 and 1, with 1 the lowest errors
    "buffer": [i, j, 500],  # Area to be loaded around the pixel ([longitude, latitude, buffer size] or None)
    "pick_date": ["2015-01-01", "2023-01-01"],  # Select dates ([min, max] or None to select all)
    "pick_sensor": None,  # Select sensors (None to select all)
    "pick_temp_bas": None,  # Select temporal baselines ([min, max] in days or None to select all)
    "proj": proj,  # EPSG system of the given coordinates
    "verbose": verbose,
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
    "verbose": verbose,
}  # Print information throughout the filtering process

## -------------- Parameters for the pixel loading part ---------------- ##
load_pixel_kwargs = {
    "regu": regu,  # Regularization method to be used
    "solver": solver,  # Solver for the inversion
    "proj": proj,  # EPSG system of the given coordinates
    "interp": "nearest",  # Interpolation method used to load the pixel when it is not in the dataset
    "visual": visual,
}  # Plot results along the way

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
    "visual": visual,  # Plot results along the way
    "verbose": verbose,
}  # Print information throughout TICOI processing

## --------------------- Interpolation parameters ---------------------- ##
interpolation_kwargs = {
    "option_interpol": "spline",  # Type of interpolation ('spline', 'spline_smooth', 'nearest')
    "result_quality": result_quality,  # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')
    "unit": unit,
}  # 365 if the unit is m/y, 1 if the unit is m/d

## -------------------- Parallelization parameters --------------------- ##
parallel = True  # Should the computation of the results for different coefficient be done using parallelization ?
nb_cpu = 8  # If parallel is True, the number of CPUs to use for parallelization

# Create a subfolder if it does not exist
if not os.path.exists(path_save):
    os.mkdir(path_save)

# %% ======================================================================== #
#                                DATA LOADING                                 #
# =========================================================================%% #

start = [time.time()]

# In the first place, we load the data
cube = cube_data_class()
cube.load(cube_name, **load_kwargs)

# Then we load the "ground truth"
cube_gt = cube_data_class()
cube_gt.load(cube_gt_name, **load_kwargs)

stop = [time.time()]
print(f"[Data loading] Loading the data cube.s took {round((stop[-1] - start[-1]), 4)} s")
print(f"[Data loading] Data cube of dimension (nz,nx,ny) : ({cube.nz}, {cube.nx}, {cube.ny}) ")
print(f"[Data loading] Ground Truth cube of dimension (nz,nx,ny) : ({cube_gt.nz}, {cube_gt.nx}, {cube_gt.ny})")

start.append(time.time())

# Filter the data cube (compute rolling_mean for regu=1accelnotnull)
obs_filt, _ = cube.filter_cube_before_inversion(**preData_kwargs)

stop.append(time.time())
print(f"[Data loading] Filtering the cube took {round((stop[-1] - start[-1]), 4)} s")


# %%========================================================================= #
#                             OPTIMIZE COEFS                                  #
# =========================================================================%% #

start.append(time.time())

# Optimize the regularisation coefficient
result = optimize_coef(
    cube,
    cube_gt,
    i,
    j,
    obs_filt,
    load_pixel_kwargs,
    inversion_kwargs,
    interpolation_kwargs,
    unit=unit,
    cmin=coef_min,
    cmax=coef_max,
    step=step,
    coefs=coefs,
    stats=stats,
    parallel=parallel,
    nb_cpu=nb_cpu,
    visual=visual_opt,
    plot_raw=plot_raw,
    vminmax=vminmax,
    savedir=savedir,
)

stop.append(time.time())
print(f"[Coef optimization] Coefficient optimization took {round((stop[-1] - start[-1]), 1)} s")


# %%========================================================================= #
#                                 PLOT COEFS                                  #
# =========================================================================%% #

start.append(time.time())

if result is not None:
    if coefs is None:
        coefs = np.arrange(coef_min, coef_max + 1, step)
    else:
        coefs = np.array(coefs)

    RMSEs = result.values
    mean_nb_data = result.nb_data

    if stats:
        mean_baselines = result.mean_temporal_baseline
        mean_v = result.mean_v_similar_data
        mean_std = result.std_v_similar_data
        mean_std_gt = np.sqrt(mean_std[2] ** 2 + mean_std[3] ** 2)
        mean_std_all = result.std_raw_data

    best_coef = coefs[np.argmin(RMSEs)]
    best_RMSE = np.min(RMSEs)
    good_RMSE = max(1.05 * best_RMSE, best_RMSE + mean_std_gt)
    good_coefs = coefs[RMSEs < good_RMSE]

    if verbose:
        print(f"[Coef optimization] Best RMSE {best_RMSE} obtained for coef = {best_coef}")
        print(
            f"[Coef optimization] Good RMSE {good_RMSE} obtained for coef in [{np.min(good_coefs)}, {np.max(good_coefs)}]"
        )

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
    ax.set_xlabel("Regularisation coefficient value")
    if coefs is None:
        ax.set_xlim(coef_min - int(step / 2), coef_max + int(step / 2))
    else:
        ax.set_xlim(min(coefs), max(coefs))
    ax.set_ylabel("RMSE between TICOI results and GT data [m/y]")
    fig.suptitle(
        f"RMSE between TICOI results and GT data when changing the regularisation coefficient (regu={regu}) at point ({round(i, 5)}, {round(j, 5)})\nBest for coef = {best_coef} (RMSE = {best_RMSE})"
    )
    plt.show()

    if save:
        fig.savefig(f"{path_save}RMSE_coef_{round(i,5)}_{round(j,5)}_{regu}.png")

else:
    print(f"Impossible to optimize the coef at point ({i}, {j}), there is not enough data.")

stop.append(time.time())
print(f"[Overall] Overall processing took {round((stop[-1] - start[0]), 1)} s")
