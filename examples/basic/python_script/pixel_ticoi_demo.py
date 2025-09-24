#!/usr/bin/env python3

"""
Implementation of the Temporal Inversion using COmbination of displacements with Interpolation (TICOI) method
for one pixel.
Author: Laurane Charrier
Reference:
    Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an optimal temporal sampling from displacement
    observation networks. IEEE Transactions on Geoscience and Remote Sensing.
    Charrier, L., Yan, Y., Colin Koeniguer, E., Mouginot, J., Millan, R., & Trouvé, E. (2022). Fusion of multi-temporal and multi-sensor ice velocity observations.
    ISPRS annals of the photogrammetry, remote sensing and spatial information sciences, 3, 311-318.
"""

import os
import time

import numpy as np

from ticoi import example
from ticoi.core import interpolation_core, inversion_core, visualization_core
from ticoi.cube_data_classxr import CubeDataClass
from ticoi.interpolation_functions import visualisation_interpolation

# %%========================================================================= #
#                                    PARAMETERS                               #
# =========================================================================%% #

###  Selection of data
cube_name = example.get_path("ITS_LIVE_Lowell_Lower")
path_save = None  # path where to save our results if save = True

i, j = -138.17069, 60.29076  # coordinate in pixel
proj = "EPSG:4326"  # EPSG system of the given coordinates

## --------------------------- Main parameters ----------------------------- ##
# For the following part we advice the user to change only the following parameter, the other parameters stored in a dictionary can be kept as it is for a first use
regu = "1accelnotnull"  # Regularization method.s to be used (for each flag if flags is not None) : 1 minimize the acceleration, '1accelnotnull' minize the distance with an apriori on the acceleration computed over a spatio-temporal filtering of the cube
coef = 100  # Regularization coefficient.s to be used (for each flag if flags is not None)
delete_outliers = None

apriori_weight = False  # Use the error as apriori
interval_output = 30  # temporal sampling of the output results
unit = 365  # 1 for m/d, 365 for m/y
result_quality = [
    "Error_propagation",
    "X_contribution",
]  # Criterium used to evaluate the quality of the results ('X_count', 'Norm_residual', 'X_contribution', 'Error_propagation')

## ----------------------- Visualization parameters ------------------------ ##
verbose = False  # Print information throughout TICOI processing
save = False  # Save the results and figures
show = True  # Plot some figures

vminmax = [0, 4700]

option_visual = ["obs_magnitude", "invertvv_overlaid", "quality_metrics"]


## ---------------------------- Loading parameters ------------------------- ##
load_kwargs = {
    "chunks": {},
    "conf": False,  # If True, confidence indicators will be put between 0 and 1, with 1 as the lowest errors
    "subset": None,  # Subset of the data to be loaded ([xmin, xmax, ymin, ymax] or None)
    "buffer": [i, j, 0.05],  # Area to be loaded around the pixel ([longitude, latitude, buffer size] or None)
    "pick_date": ["2015-01-01", "2024-01-01"],  # Select dates ([min, max] or None to select all)
    "pick_sensor": None,  # Select sensors (None to select all)
    "pick_temp_bas": None,  # Select temporal baselines ([min, max] in days or None to select all)
    "proj": "EPSG:4326",  # EPSG system of the given coordinates
    "verbose": False,  # Print information throughout the loading process
}

## ----------------------- Data preparation parameters --------------------- ##
preData_kwargs = {
    "smooth_method": "savgol",  # Smoothing method to be used to smooth the data in time ('gaussian', 'median', 'emwa', 'savgol')
    "s_win": 3,  # Size of the spatial window
    "t_win": 90,  # Time window size for smoothing
    "sigma": 3,  # Standard deviation for 'gaussian' filter
    "order": 3,  # Order of the smoothing function
    "unit": 365,  # 365 if the unit is m/y, 1 if the unit is m/d
    "delete_outliers": delete_outliers,  # Delete the outliers from the data according to one (int or str) or several (dict) criteriums
    "flag": None,  # Divide the data in several areas where different methods should be used
    "regu": regu,  # Regularization method.s to be used (for each flag if flags is not None) : 1 minimize the acceleration, '1accelnotnull' minize the distance with an apriori on the acceleration computed over a spatio-temporal filtering of the cube
    "solver": "LSMR_ini",  # Solver for the inversion
    "proj": proj,  # EPSG system of the given coordinates
    "velo_or_disp": "velo",  # Type of data contained in the data cube ('disp' for displacements, and 'velo' for velocities)
    "verbose": True,  # Print information throughout the filtering process
}

## ---------------- Parameters for the pixel loading part ------------------ ##
load_pixel_kwargs = {
    "regu": regu,  # Regularization method to be used
    "coef": coef,  # Regularization coefficient to be used
    "solver": "LSMR_ini",  # Solver for the inversion
    "proj": proj,  # EPSG system of the given coordinates
    "interp": "nearest",  # Interpolation method used to load the pixel when it is not in the dataset
    "visual": show | save,  # If the observations data need to be returned
}

## --------------------------- Inversion parameters ------------------------ ##
inversion_kwargs = {
    "regu": regu,  # Regularization method to be used
    "coef": coef,  # Regularization coefficient to be used
    "solver": "LSMR_ini",  # Solver for the inversion
    "conf": False,  # If True, confidence indicators are set between 0 and 1, with 1 the lowest errors
    "unit": unit,  # 365 if the unit is m/y, 1 if the unit is m/d
    "iteration": True,  # Allow the inversion process to make several iterations
    "nb_max_iteration": 10,  # Maximum number of iteration during the inversion process
    "threshold_it": 0.1,  # Threshold to test the stability of the results between each iteration, used to stop the process
    "apriori_weight": True,  # If True, use apriori weights
    "apriori_weight_in_second_iteration": True,  # it True use the error to weight each of the iterations, if not use it only in the first iteration
    "detect_temporal_decorrelation": True,  # If True, the first inversion will use only velocity observations with small temporal baselines, to detect temporal decorelation
    "linear_operator": None,  # Perform the inversion using this specific linear operator
    "result_quality": result_quality,  # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')
    "visual": show | save,  # If the observations data need to be returned
    "verbose": verbose,  # Print information throughout TICOI processing
}

## ----------------------- Interpolation parameters ------------------------ ##
interpolation_kwargs = {
    "interval_output": interval_output,  # Temporal baseline of the time series resulting from TICOI (after interpolation)
    "redundancy": 5,  # Redundancy in the interpolated time series in number of days, no redundancy if None
    "option_interpol": "spline",  # Type of interpolation ('spline', 'spline_smooth', 'nearest')
    "result_quality": result_quality,  # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')
    "unit": unit,  # 365 if the unit is m/y, 1 if the unit is m/d
}

# Update of dictionary with common parameters
for common_parameter in ["regu", "solver", "unit"]:
    inversion_kwargs[common_parameter] = preData_kwargs[common_parameter]

# Create a subfolder if it does not exist
if path_save is not None and os.path.exists(path_save):
    os.mkdir(path_save)


# %% ======================================================================== #
#                                DATA LOADING                                 #
# =========================================================================%% #

start = [time.time()]

# Load the main cube
cube = CubeDataClass()
cube.load(cube_name, **load_kwargs)

stop = [time.time()]
print(f"[Data loading] Loading the data cube.s took {round((stop[0] - start[0]), 4)} s")
print(f"[Data loading] Cube of dimension (nz,nx,ny) : ({cube.nz}, {cube.nx}, {cube.ny}) ")

start.append(time.time())

# Filter the cube (compute rolling_mean for regu=1accelnotnull)
obs_filt, flag = cube.filter_cube_before_inversion(**preData_kwargs)

# Load pixel data
data, mean, dates_range = cube.load_pixel(i, j, rolling_mean=obs_filt, **load_pixel_kwargs)

# Prepare interpolation dates
first_date_interpol, last_date_interpol = cube.prepare_interpolation_date()
interpolation_kwargs.update({"first_date_interpol": first_date_interpol, "last_date_interpol": last_date_interpol})

stop.append(time.time())
print(f"[Data loading] Loading the pixel took {round((stop[1] - start[1]), 4)} s")


# %% ======================================================================== #
#                                 INVERSION                                   #
# =========================================================================%% #

start.append(time.time())

# Proceed to inversion
A, result, dataf = inversion_core(data, i, j, dates_range=dates_range, mean=mean, **inversion_kwargs)

stop.append(time.time())
print(f"[Inversion] Inversion took {round((stop[2] - start[2]), 4)} s")
if save:
    result.to_csv(f"{path_save}/ILF_result.csv")


# %% ======================================================================== #
#                              INTERPOLATION                                  #
# =========================================================================%% #

start.append(time.time())

if not interpolation_kwargs["interval_output"]:
    interpolation_kwargs["interval_output"] = 1
start_date_interpol = np.min(np.min(cube.date2_()))
last_date_interpol = np.max(np.max(cube.date2_()))

# Proceed to interpolation
dataf_lp = interpolation_core(result, **interpolation_kwargs)

if save:
    dataf_lp.to_csv(f"{path_save}/ILF_result.csv")


stop.append(time.time())
print(f"[Interpolation] Interpolation took {round((stop[3] - start[3]), 4)} s")

if save:
    dataf_lp.to_csv(f"{path_save}/RLF_result.csv")
if show or save:  # plot some figures
    visualization_core(
        [dataf, result],
        option_visual=option_visual,
        save=save,
        show=show,
        path_save=path_save,
        A=A,
        log_scale=False,
        cmap="rainbow",
        colors=["orange", "blue"],
        vminmax=vminmax,
    )
    visualisation_interpolation(
        [dataf, dataf_lp],
        option_visual=option_visual,
        save=save,
        show=show,
        path_save=path_save,
        colors=["orange", "blue"],
        vminmax=vminmax,
    )

print(f"[Overall] Overall processing took {round((stop[3] - start[0]), 4)} s")
