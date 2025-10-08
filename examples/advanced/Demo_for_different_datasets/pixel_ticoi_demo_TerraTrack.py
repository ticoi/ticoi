#!/usr/bin/env python3

"""
The package is based on the methodological developments published in:

- Charrier, L., Dehecq, A., Guo, L., Brun, F., Millan, R., Lioret, N., ... & Halas, P. (2025). TICOI: an operational
  Python package to generate regular glacier velocity time series. EGUsphere, 2025, 1-40.

- Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an
  optimal temporal sampling from displacement observation networks. IEEE Transactions on Geoscience and Remote Sensing,
  60, 1-10.
"""

import os
import time


from ticoi.core import interpolation_core, inversion_core, visualization_core
from ticoi.cube_data_classxr import CubeDataClass
from ticoi.interpolation_functions import visualisation_interpolation

# %%========================================================================= #
#                                    PARAMETERS                               #
# =========================================================================%% #

###  Selection of data
cube_name = "/home/charriel/Documents/Collaborations/Maximilian/wetransfer_fixed-data-cubes_2025-09-26_1038/datacubefixed_shrt_bsln.nc"
path_save = (
    os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")),
        "examples",
        "result",
    )
    + "/"
)  # path where to save our results
result_fn = "TerraTrack_example"  # Name of the netCDF file to be created
# path_save = None  # path where to save our results if save = True

i, j = 55, 117  # coordinate in pixel
proj = "int"  # EPSG system of the given coordinates

## --------------------------- Main parameters ----------------------------- ##
# For the following part we advice the user to change only the following parameter, the other parameters stored in a dictionary can be kept as it is for a first use
regu = "1"  # Regularization method.s to be used (for each flag if flags is not None) : 1 minimize the acceleration, '1accelnotnull' minize the distance with an apriori on the acceleration computed over a spatio-temporal filtering of the cube
coef = 100  # Regularization coefficient.s to be used (for each flag if flags is not None)
delete_outliers = None
solver = "LSMR"

apriori_weight = False  # Use the error as apriori
interval_output = 180  # temporal sampling of the output results
result_quality = [
    "Error_propagation",
    "X_contribution",
]  # Criterium used to evaluate the quality of the results ('X_count', 'Norm_residual', 'X_contribution', 'Error_propagation')

## ----------------------- Visualization parameters ------------------------ ##
verbose = False  # Print information throughout TICOI processing
save = True  # Save the results and figures
show = True  # Plot some figures

vminmax = [-2, 10]

option_visual = ["obs_magnitude", "invertvv_overlaid", "quality_metrics", "cumulative_dv"]


## ---------------------------- Loading parameters ------------------------- ##
load_kwargs = {
    "subset": None,  # Subset of the data to be loaded ([xmin, xmax, ymin, ymax] or None)
    "buffer": None,  # Area to be loaded around the pixel ([longitude, latitude, buffer size] or None)
    "pick_date": None,  # Select dates ([min, max] or None to select all)
    "pick_sensor": None,  # Select sensors (None to select all)
    "pick_temp_bas": None,  # Select temporal baselines ([min, max] in days or None to select all)
    "proj": "int",  # EPSG system of the given coordinates
}

## ----------------------- Data preparation parameters --------------------- ##
preData_kwargs = {
    "delete_outliers": delete_outliers,  # Delete the outliers from the data according to one (int or str) or several (dict) criteriums
    "regu": regu,  # Regularization method.s to be used (for each flag if flags is not None) : 1 minimize the acceleration, '1accelnotnull' minize the distance with an apriori on the acceleration computed over a spatio-temporal filtering of the cube
    "solver": solver,  # Solver for the inversion
    "proj": proj,  # EPSG system of the given coordinates
}

## ---------------- Parameters for the pixel loading part ------------------ ##
load_pixel_kwargs = {
    "regu": regu,  # Regularization method to be used
    "coef": coef,  # Regularization coefficient to be used
    "solver": solver,  # Solver for the inversion
    "proj": proj,  # EPSG system of the given coordinates
    "interp": "nearest",  # Interpolation method used to load the pixel when it is not in the dataset
    "visual": show | save,  # If the observations data need to be returned
}

## --------------------------- Inversion parameters ------------------------ ##
inversion_kwargs = {
    "regu": regu,  # Regularization method to be used
    "coef": coef,  # Regularization coefficient to be used
    "solver": solver,  # Solver for the inversion
    "conf": False,  # If True, confidence indicators are set between 0 and 1, with 1 the lowest errors
    "apriori_weight": apriori_weight,  # If True, use apriori weights
    "result_quality": result_quality,  # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')
    "visual": show | save,  # If the observations data need to be returned
    "verbose": verbose,  # Print information throughout TICOI processing
}

## ----------------------- Interpolation parameters ------------------------ ##
interpolation_kwargs = {
    "interval_output": interval_output,  # Temporal baseline of the time series resulting from TICOI (after interpolation)
    "option_interpol": "spline",  # Type of interpolation ('spline', 'spline_smooth', 'nearest')
    "result_quality": result_quality,  # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')
}

# Update of dictionary with common parameters
for common_parameter in ["regu", "solver"]:
    inversion_kwargs[common_parameter] = preData_kwargs[common_parameter]

# Create a subfolder if it does not exist
if path_save is not None and not os.path.exists(path_save):
    os.mkdir(path_save)


# %% ======================================================================== #
#                                DATA LOADING                                 #
# =========================================================================%% #

start = time.time()

# Load the main cube
cube = CubeDataClass()
cube.load(cube_name, **load_kwargs)

stop = time.time()
print(f"[Data loading] Loading the data cube took {round((stop - start), 4)} s")
print(f"[Data loading] Cube of dimension (nz,nx,ny) : ({cube.nz}, {cube.nx}, {cube.ny}) ")


# %% ======================================================================== #
#                                DATA FILTERING                                 #
# =========================================================================%% #
start = time.time()

# Filter the cube (compute rolling_mean for regu=1accelnotnull)
obs_filt, flag = cube.filter_cube_before_inversion(**preData_kwargs)

stop = time.time()

print(f"[Data filtering] Filtering the data cube took {round((stop - start), 4)} s")

# %% ======================================================================== #
#                                PIXEL LOADING                                 #
# =========================================================================%% #

start = time.time()
# Load pixel data
data, mean, dates_range = cube.load_pixel(i, j, rolling_mean=obs_filt, **load_pixel_kwargs)
stop = time.time()
print(f"[Data loading] Loading the pixel took {round((stop - start), 4)} s")


# Prepare interpolation dates
first_date_interpol, last_date_interpol = cube.prepare_interpolation_date()
interpolation_kwargs.update({"first_date_interpol": first_date_interpol, "last_date_interpol": last_date_interpol})


# %% ======================================================================== #
#                                 INVERSION                                   #
# =========================================================================%% #

start = time.time()
# Proceed to inversion
A, result, dataf = inversion_core(data, i, j, dates_range=dates_range, mean=mean, **inversion_kwargs)
stop = time.time()

print(f"[Inversion] Inversion took {round((stop - start), 4)} s")
if save:
    result.to_csv(f"{path_save}/ILF_result.csv")


# %% ======================================================================== #
#                              INTERPOLATION                                  #
# =========================================================================%% #

start = time.time()

# Proceed to interpolation
dataf_lp = interpolation_core(result, **interpolation_kwargs)

if save:
    dataf_lp.to_csv(f"{path_save}/ILF_result.csv")

stop = time.time()
print(f"[Interpolation] Interpolation took {round((stop - start), 4)} s")

if save:
    dataf_lp.to_csv(f"{path_save}/RLF_result.csv")


# %% ======================================================================== #
#                              PLOT FIGURES                                   #
# =========================================================================%% #

# removing interpolated estimation which are not constrained by any observations
dataf_lp = dataf_lp[dataf_lp["xcount_x"] > 0]

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

print(f"[Overall] Overall processing took {round((stop - start), 4)} s")
