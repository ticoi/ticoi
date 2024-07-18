"""
Implementation of the Temporal Inversion using COmbination of displacements with Interpolation (TICOI) method
For one cube of pixel
Author: Laurane Charrier
Reference:
    Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an optimal temporal sampling from displacement
    observation networks. IEEE Transactions on Geoscience and Remote Sensing.
    Charrier, L., Yan, Y., Colin Koeniguer, E., Mouginot, J., Millan, R., & Trouvé, E. (2022). Fusion of multi-temporal and multi-sensor ice velocity observations.
    ISPRS annals of the photogrammetry, remote sensing and spatial information sciences, 3, 311-318.
"""

import json
import os
import time
import urllib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ticoi.core import interpolation_core, inversion_core, visualization_core
from ticoi.cube_data_classxr import cube_data_class
from ticoi.interpolation_functions import (
    prepare_interpolation_date,
    visualisation_interpolation,
)

warnings.filterwarnings("ignore")

# %%========================================================================= #
#                                    PARAMETERS                               #
# =========================================================================%% #

# List of the paths where the data cubes are stored

cube_name = (
    f'{os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..", "test_data"))}/ITS_LIVE_Lowell_Lower_test.nc'
)
path_save = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples", "results","pixel"))}/'  # Path where to store the results
dem_file = None
proj = "EPSG:32632"  # EPSG system of the given coordinates

# i, j = 343617.7, 5091275.0  # Pixel coordinates
i,j = 138.28962881999922274,60.25934205396930565

## --------------------------- Main parameters ----------------------------- ##
# For the following part we advice the user to change only the following parameter, the other parameters stored in a dictionary can be kept as it is for a first use
regu = "1accelnotnull"  # Regularization method.s to be used (for each flag if flags is not None) : 1 minimize the acceleration, '1accelnotnull' minize the distance with an apriori on the acceleration computed over a spatio-temporal filtering of the cube
coef = 200  # Regularization coefficient.s to be used (for each flag if flags is not None)
delete_outlier = "flow_angle"  # delete outliers, based on the angle between the median vector and the observations, recommended:: vvc_angle or None
apriori_weight = True  # Use the error as apriori
interval_output = 30  # temporal sampling of the output results
unit = 365  # 1 for m/d, 365 for m/y
result_quality = [
    "X_contribution"
]  # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')

interpolation = True  # Interpolate the results

## ----------------------- Visualization parameters ------------------------ ##
verbose = True  # Print information throughout TICOI processing
visual = True  # Plot information along the way
save = True  # Save the results or not
# Visualisation options
option_visual = [
    "obs_xy",
    "obs_magnitude",
    "obs_vxvy_quality",
    "invertxy_overlaid",
    "invertvv_overlaid",
    "residuals",
    "xcount_xy",
    "xcount_vv",
    "invert_weight",
    "interp_xy_overlaid",
    "interp_xy_overlaid_zoom",
    "invertvv_overlaid",
    "invertvv_overlaid_zoom",
    "direction_overlaid",
]  # see README_visualization_pixel_output.md

parameter = "coef"

## ---------------------------- Loading parameters ------------------------- ##
load_kwargs = {
    "chunks": {},
    "conf": False,  # If True, confidence indicators will be put between 0 and 1, with 1 the lowest errors
    "subset": None,  # Subset of the data to be loaded ([xmin, xmax, ymin, ymax] or None)
    "buffer": [i, j, 100],  # Area to be loaded around the pixel ([longitude, latitude, buffer size] or None)
    "pick_date": None,  # Select dates ([min, max] or None to select all)
    "pick_sensor": None,  # Select sensors (None to select all)
    "pick_temp_bas": None,  # Select temporal baselines ([min, max] in days or None to select all)
    "proj": "EPSG:4326",  # EPSG system of the given coordinates
    "verbose": verbose,  # Print information throughout the loading process
}
## ----------------------- Data preparation parameters --------------------- ##
preData_kwargs = {
    "smooth_method": "gaussian",  # Smoothing method to be used to smooth the data in time ('gaussian', 'median', 'emwa', 'savgol')
    "s_win": 3,  # Size of the spatial window
    "t_win": 90,  # Time window size for 'ewma' smoothing
    "sigma": 3,  # Standard deviation for 'gaussian' filter
    "order": 3,  # Order of the smoothing function
    "unit": 365,  # 365 if the unit is m/y, 1 if the unit is m/d
    "delete_outliers": delete_outlier,  # Delete data with a poor quality indicator (if int), or with aberrant direction ('vvc_angle')
    "flag": None,
    "regu": regu,  # Regularization method to be used
    "solver": "LSMR_ini",  # Solver for the inversion
    "proj": proj,  # EPSG system of the given coordinates
    "velo_or_disp": "disp",  # Type of data contained in the data cube ('disp' for displacements, and 'velo' for velocities)
    "select_baseline": 120,
    "verbose": verbose,
}  # Print information throughout the filtering process

## ---------------- Parameters for the pixel loading part ------------------ ##
load_pixel_kwargs = {
    "regu": regu,  # Regularization method to be used
    "coef": coef,
    "solver": "LSMR_ini",  # Solver for the inversion
    "proj": proj,  # EPSG system of the given coordinates
    "interp": "nearest",  # Interpolation method used to load the pixel when it is not in the dataset
    "visual": visual,  # Plot results along the way
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
    "apriori_weight_in_second_iteration": True,
    "detect_temporal_decorrelation": True,  # If True, the first inversion will use only velocity observations with small temporal baselines, to detect temporal decorelation
    "linear_operator": None,  # Perform the inversion using this specific linear operator
    "result_quality": result_quality,  # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')
    "visual": visual,  # Plot results along the way
    "verbose": verbose,
}  # Print information throughout TICOI processing

## ----------------------- Interpolation parameters ------------------------ ##
interpolation_kwargs = {
    "interval_output": interval_output,  # Temporal baseline of the time series resulting from TICOI (after interpolation)
    "redundancy": 30,  # Redundancy in the interpolated time series in number of days, no redundancy if None
    "option_interpol": "spline",  # Type of interpolation ('spline', 'spline_smooth', 'nearest')
    "result_quality": result_quality,  # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')
}  # Print information throughout TICOI processing

# Create a subfolder if it does not exist
if not os.path.exists(path_save):
    os.makedirs(path_save)

if type(cube_name) == str:
    cube_name = [cube_name]

list_parameter = [
    50,
    70,
    90,
    110,
    130,
    150,
    170,
    190,
    210,
    230,
    250,
    270,
    290,
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
    220,
    240,
    260,
    280,
    300,
    500,
    1000,
    5000,
    10000,
]


# initialization
start = [time.time()]
options = ""
if inversion_kwargs["iteration"]:
    options += "_it"
if inversion_kwargs["apriori_weight"]:
    options += "_weighted"
if inversion_kwargs["solver"][-1] == "u":
    options += "_regu"

if interpolation:
    options += "interpol"


list_Coh_vector_before, list_Coh_vector_after, list_NCoh_vector_before, list_NCoh_vector_after = [], [], [], []
if interpolation:
    list_NCoh_vector_after_interpol = []
if "L_curve" in option_visual:
    Residux, Residuy, Regux, Reguy, Rx, Ry, Rmeanx, Rmeany = [], [], [], [], [], [], [], []


cube = cube_data_class()
cube.load(cube_name[0], **load_kwargs)

# Prepare interpolation dates
first_date_interpol, last_date_interpol = prepare_interpolation_date(cube)
interpolation_kwargs.update({"first_date_interpol": first_date_interpol, "last_date_interpol": last_date_interpol})


stop = [time.time()]
print(f"[Data loading] Loading the data cube.s took {round((stop[0] - start[0]), 4)} s")
print(f"[Data loading] Cube of dimension (nz,nx,ny) : ({cube.nz}, {cube.nx}, {cube.ny}) ")

# Filter the cube (compute rolling_mean for regu=1accelnotnull)
start.append(time.time())
obs_filt, flag = cube.filter_cube(**preData_kwargs)
stop.append(time.time())
print(f"[Data filtering] Loading the pixel took {round((stop[1] - start[1]), 4)} s")


# %% For each parameter
f = open(f"{path_save}parameters_{options}_{parameter}.txt", "w")
for param_value in list_parameter:
    print(f"{parameter}: {param_value}")

    f.write(f"\n###   {param_value}\n")
    if parameter == "coef":
        load_pixel_kwargs.update({"coef": param_value}), inversion_kwargs.update({"coef": param_value})

    elif parameter == "interval_output":
        interpolation_kwargs.update({"interval_output": param_value})
    else:
        raise NameError('Please enter as parameter "coef" or "interval_output"')

    if verbose:
        # print(f'Cube of dimesion (nz,nx,ny): ({cube.nz},{cube.nx},{cube.ny} ')
        print(f"Inversion for pixel {i, j}")

    # %% ======================================================================== #
    #                                Inversion                                    #
    # =========================================================================%% #

    # Load pixel data
    start.append(time.time())
    data, mean, dates_range = cube.load_pixel(i, j, rolling_mean=obs_filt, **load_pixel_kwargs)

    stop.append(time.time())
    print(f"[Data loading] Loading the pixel took {round((stop[2] - start[2]), 4)} s")

    # inversion
    start.append(time.time())
    A, result, dataf = inversion_core(data, i, j, dates_range=dates_range, mean=mean, **inversion_kwargs)
    stop.append(time.time())
    print(f"[Inversion] Inversion took {round((stop[3] - start[3]), 4)} s")

    if interpolation or save:
        save_path = f"{path_save}/{parameter}_{param_value}/"
    if not os.path.exists(save_path):  # cree un sous dossier
        os.mkdir(save_path)

    # Save the results
    if save:
        result.to_csv(f"{save_path}/ILF_result.csv")

    # %% ======================================================================== #
    #                              INTERPOLATION                                  #
    # =========================================================================%% #
    start.append(time.time())

    if interpolation_kwargs["interval_output"] == False:
        interpolation_kwargs["interval_output"] = 1

    if interpolation:
        # Proceed to interpolation
        dataf_lp = interpolation_core(
            result,
            **interpolation_kwargs,
        )

    if save:
        dataf_lp.to_csv(f"{save_path}/RLF_result.csv")

    stop.append(time.time())
    print(f"[Interpolation] Interpolation took {round((stop[4] - start[4]), 4)} s")

    # %% ======================================================================== #
    #                              Visualization                                  #
    # =========================================================================%% #
    if visual:
        visualization_core(
            [dataf, result],
            option_visual=option_visual,
            save=True,
            show=True,
            path_save=save_path,
            A=A,
            log_scale=False,
            cmap="rainbow",
            colors=["orange", "blue"],
        )
        visualisation_interpolation(
            [dataf, dataf_lp],
            option_visual=option_visual,
            save=True,
            show=True,
            path_save=save_path,
            colors=["orange", "blue"],
        )

    # %% ======================================================================== #
    #                                Update vvc                                   #
    # =========================================================================%% #

    if "L_curve" in option_visual:
        Residux.append(dataf["NormR"][0])
        Residuy.append(dataf["NormR"][2])
        Regux.append(dataf["NormR"][1])
        Reguy.append(dataf["NormR"][3])

    vv_after_inv = np.array([np.sqrt(el) for el in (dataf_lp["vx"] ** 2 + dataf_lp["vy"] ** 2)])
    Normalized_Coh_vector_after = (
        np.sqrt(np.nansum(dataf_lp["vx"] / vv_after_inv) ** 2 + np.nansum(dataf_lp["vy"] / vv_after_inv) ** 2)
        / dataf_lp.shape[0]
    )
    list_NCoh_vector_after.append(Normalized_Coh_vector_after)

    f.write(f"\n A shape {A.shape}")
    f.write(f"\n Normalized Coh_vector after inversion {Normalized_Coh_vector_after}")

end = time.time()
print(f"{end - start[0]} ms")

# %% ======================================================================== #
#                            VVC visualization                                #
# =========================================================================%% #
dataf = pd.DataFrame({"param": list_parameter, "VCC": list_NCoh_vector_after})
dataf.sort_values(by="param", inplace=True)
VCC_max99 = dataf["VCC"].max() * 0.99
VCC_max95 = dataf["VCC"].max() * 0.95
# list_interval_output=[72,60,24,12,36,48]
plt.style.use("seaborn-v0_8-whitegrid")
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(dataf["param"], dataf["VCC"], linestyle="-", marker="o", color="orange", label=f"Velocity observations")

ax1.plot(
    [dataf[dataf[f"VCC"] > VCC_max99].iloc[0]["param"]],
    [dataf[dataf[f"VCC"] > VCC_max99].iloc[0]["VCC"]],
    linestyle="-",
    marker="o",
    color="r",
    label=f"99% of maximal VCC",
    markersize=5,
)
ax1.annotate(
    f"{dataf[dataf['VCC'] > VCC_max99].iloc[0]['param']}",
    (dataf[dataf[f"VCC"] > VCC_max99].iloc[0]["param"], dataf[dataf[f"VCC"] > VCC_max99].iloc[0]["VCC"]),
)
ax1.plot(
    [dataf[dataf[f"VCC"] > VCC_max95].iloc[0]["param"]],
    [dataf[dataf[f"VCC"] > VCC_max95].iloc[0]["VCC"]],
    linestyle="-",
    marker="+",
    color="r",
    label=f"95% of maximal VCC",
    markersize=10,
)
ax1.annotate(
    f"{dataf[dataf['VCC'] > VCC_max95].iloc[0]['param']}",
    (dataf[dataf[f"VCC"] > VCC_max95].iloc[0]["param"], dataf[dataf[f"VCC"] > VCC_max95].iloc[0]["VCC"]),
)

ax1.set_ylabel("VVC", fontsize=25)
ax1.set_xlabel("Regularization coefficient", fontsize=25)
plt.subplots_adjust(bottom=0.25)
ax1.legend(loc="lower left", bbox_to_anchor=(0.12, -0.03), bbox_transform=fig1.transFigure, ncol=3, fontsize=16)
fig1.savefig(f"{path_save}/Compa_VVC_coef")
plt.show()

diff = pd.DataFrame({})
diff["diff"] = np.diff(dataf["VCC"]) / np.diff(dataf["param"])
diff["param"] = dataf["param"].values[1:]

plt.style.use("seaborn-v0_8-whitegrid")
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(diff["param"], diff["diff"] * 100, linestyle="-", marker="o", color="orange", label=f"Velocity observations")
threshold = 0.025
ax1.plot(
    [diff["param"][diff["diff"] < threshold / 100].iloc[0]],
    [diff["diff"][diff["diff"] < threshold / 100].iloc[0] * 100],
    linestyle="-",
    marker="o",
    color="r",
    label=f"<{threshold}",
    markersize=5,
)
ax1.annotate(
    f"{diff['param'][diff['diff'] < threshold / 100].iloc[0]}",
    (diff["param"][diff["diff"] < threshold / 100].iloc[0], diff["diff"][diff["diff"] < threshold / 100].iloc[0] * 100),
)

threshold = 0.01
ax1.plot(
    [diff["param"][diff["diff"] < threshold / 100].iloc[0]],
    [diff["diff"][diff["diff"] < threshold / 100].iloc[0] * 100],
    linestyle="-",
    marker="+",
    color="r",
    label=f"<{threshold}",
    markersize=5,
)
ax1.annotate(
    f"{diff['param'][diff['diff'] < threshold / 100].iloc[0]}",
    (diff["param"][diff["diff"] < threshold / 100].iloc[0], diff["diff"][diff["diff"] < threshold / 100].iloc[0] * 100),
)
ax1.set_ylim(-0.02, 0.25)
ax1.set_ylabel("VVC derivative", fontsize=25)
ax1.set_xlabel("Regularization coefficient", fontsize=25)
plt.subplots_adjust(bottom=0.25)
ax1.legend(loc="lower left", bbox_to_anchor=(0.12, -0.03), bbox_transform=fig1.transFigure, ncol=3, fontsize=16)
fig1.savefig(f"{path_save}/Compa_VVC_coef_diff2")
plt.show()

list_coef = [0.1, 1, 10, 100, 150, 1000, 5000, 10000]
dataf = dataf[dataf["param"].isin(list_coef)]

VCC_max99 = dataf["VCC"].max() * 0.99
VCC_max95 = dataf["VCC"].max() * 0.95
# list_interval_output=[72,60,24,12,36,48]
plt.style.use("seaborn-v0_8-whitegrid")
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(dataf["param"], dataf["VCC"], linestyle="-", marker="o", color="orange", label=f"Velocity observations")

ax1.plot(
    [dataf[dataf[f"VCC"] > VCC_max99].iloc[0]["param"]],
    [dataf[dataf[f"VCC"] > VCC_max99].iloc[0]["VCC"]],
    linestyle="-",
    marker="o",
    color="r",
    label=f"99% of maximal VCC",
    markersize=5,
)
ax1.annotate(
    f"{dataf[dataf['VCC'] > VCC_max99].iloc[0]['param']}",
    (dataf[dataf[f"VCC"] > VCC_max99].iloc[0]["param"], dataf[dataf[f"VCC"] > VCC_max99].iloc[0]["VCC"]),
)
ax1.plot(
    [dataf[dataf[f"VCC"] > VCC_max95].iloc[0]["param"]],
    [dataf[dataf[f"VCC"] > VCC_max95].iloc[0]["VCC"]],
    linestyle="-",
    marker="+",
    color="r",
    label=f"95% of maximal VCC",
    markersize=10,
)
ax1.annotate(
    f"{dataf[dataf['VCC'] > VCC_max95].iloc[0]['param']}",
    (dataf[dataf[f"VCC"] > VCC_max95].iloc[0]["param"], dataf[dataf[f"VCC"] > VCC_max95].iloc[0]["VCC"]),
)

ax1.set_ylabel("VVC", fontsize=25)
ax1.set_xlabel("Regularization coefficient", fontsize=25)
plt.subplots_adjust(bottom=0.25)
ax1.legend(loc="lower left", bbox_to_anchor=(0.12, -0.03), bbox_transform=fig1.transFigure, ncol=3, fontsize=16)
fig1.savefig(f"{path_save}/Compa_VVC_coef_selectedlist")
plt.show()

diff = pd.DataFrame({})
diff["diff"] = np.diff(dataf["VCC"]) / np.diff(dataf["param"])
diff["param"] = dataf["param"].values[1:]

plt.style.use("seaborn-v0_8-whitegrid")
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(diff["param"], diff["diff"] * 100, linestyle="-", marker="o", color="orange", label=f"Velocity observations")
threshold = 0.025
ax1.plot(
    [diff["param"][diff["diff"] < threshold / 100].iloc[0]],
    [diff["diff"][diff["diff"] < threshold / 100].iloc[0] * 100],
    linestyle="-",
    marker="o",
    color="r",
    label=f"<{threshold}",
    markersize=5,
)
ax1.annotate(
    f"{diff['param'][diff['diff'] < threshold / 100].iloc[0]}",
    (diff["param"][diff["diff"] < threshold / 100].iloc[0], diff["diff"][diff["diff"] < threshold / 100].iloc[0] * 100),
)

threshold = 0.01
ax1.plot(
    [diff["param"][diff["diff"] < threshold / 100].iloc[0]],
    [diff["diff"][diff["diff"] < threshold / 100].iloc[0] * 100],
    linestyle="-",
    marker="+",
    color="r",
    label=f"<{threshold}",
    markersize=5,
)
ax1.annotate(
    f"{diff['param'][diff['diff'] < threshold / 100].iloc[0]}",
    (diff["param"][diff["diff"] < threshold / 100].iloc[0], diff["diff"][diff["diff"] < threshold / 100].iloc[0] * 100),
)
ax1.set_ylim(-0.02, 0.25)
ax1.set_ylabel("VVC derivative", fontsize=25)
ax1.set_xlabel("Regularization coefficient", fontsize=25)
plt.subplots_adjust(bottom=0.25)
ax1.legend(loc="lower left", bbox_to_anchor=(0.12, -0.03), bbox_transform=fig1.transFigure, ncol=3, fontsize=16)
fig1.savefig(f"{path_save}/Compa_VVC_coef_diff2_selectedlist")
plt.show()
