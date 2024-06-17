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

cube_name = "/media/tristan/Data3/Hala_lake/Landsat7_refine/Hala_lake_disp_refine_LS7.nc"


proj = "EPSG:32647"
# catalog_cubeige = '/home/charriel/Documents/Bettik/Yukon/cube_grid.shp'

# selection of data
# i, j = 388767, 4257984 # centeral part of Gangnalou glacier
# i, j = 395800, 4259037 # centeral part of upper surging glacier
i, j = 396343, 4259420  # middle lower part of upper surging glacier
# To select a specific period for the measurements, if you want to select all the dates put None, else give an interval of dates ['aaaa-mm-dd', 'aaaa-mm-dd'] ([min, max])
path_save = f"/media/tristan/Data3/Hala_lake/Landsat7_refine/{i}-{j}/"  # Path where to store the results

## --------------------------- Main parameters ----------------------------- ##
regu = "1accelnotnull"  # Regularization method to be used
solver = "LSMR_ini"  # Solver for the inversion
unit = 365  # 1 for m/d, 365 for m/y
result_quality = (
    "X_contribution"  # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')
)
interpolation = True

## ----------------------- Visualization parameters ------------------------ ##
verbose = False  # Print information throughout TICOI processing
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
]  # see README_visualization_pixel_output.md
vmax = [False, False]  # Vertical limits for the plots

parameter = "coef"
threshold_vcc = 0.01
zone_stable = False  # if the considered point is supposed to be a stable point


## ---------------------------- Loading parameters ------------------------- ##
load_kwargs = {
    "chunks": {},
    "conf": False,  # If True, confidence indicators will be put between 0 and 1, with 1 the lowest errors
    "buffer": [i, j, 100],  # Area to be loaded around the pixel ([longitude, latitude, buffer size] or None)
    "pick_date": None,  # Select dates ([min, max] or None to select all)
    "pick_sensor": None,  # Select sensors (None to select all)
    "pick_temp_bas": None,  # Select temporal baselines ([min, max] in days or None to select all)
    "proj": proj,  # EPSG system of the given coordinates
    "verbose": verbose,
}  # Print information throughout the loading process

## ----------------------- Data preparation parameters --------------------- ##
preData_kwargs = {
    "smooth_method": "gaussian",  # Smoothing method to be used to smooth the data in time ('gaussian', 'median', 'emwa', 'savgol')
    "s_win": 3,  # Size of the spatial window
    "t_win": 90,  # Time window size for 'ewma' smoothing
    "sigma": 3,  # Standard deviation for 'gaussian' filter
    "order": 3,  # Order of the smoothing function
    "unit": unit,  # 365 if the unit is m/y, 1 if the unit is m/d
    "delete_outliers": None,  # Delete data with a poor quality indicator (if int), or with aberrant direction ('vvc_angle')
    "regu": regu,  # Regularization method to be used
    "solver": solver,  # Solver for the inversion
    "proj": proj,  # EPSG system of the given coordinates
    "velo_or_disp": "disp",  # Type of data contained in the data cube ('disp' for displacements, and 'velo' for velocities)
    "verbose": verbose,
}  # Print information throughout the filtering process

## ---------------- Parameters for the pixel loading part ------------------ ##
load_pixel_kwargs = {
    "regu": regu,  # Regularization method to be used
    "coef": 100,
    "solver": solver,  # Solver for the inversion
    "proj": proj,  # EPSG system of the given coordinates
    "interp": "nearest",  # Interpolation method used to load the pixel when it is not in the dataset
    "visual": visual,  # Plot results along the way
}

## --------------------------- Inversion parameters ------------------------ ##
inversion_kwargs = {
    "regu": regu,  # Regularization method to be used
    "coef": 100,  # Regularization coefficient to be used
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

## ----------------------- Interpolation parameters ------------------------ ##
interpolation_kwargs = {
    "interval_output": 90,  # Temporal baseline of the time series resulting from TICOI (after interpolation)
    "redundancy": 30,  # Redundancy in the interpolated time series in number of days, no redundancy if None
    "option_interpol": "spline",  # Type of interpolation ('spline', 'spline_smooth', 'nearest')
    "result_quality": result_quality,  # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')
    "unit": unit,  # 365 if the unit is m/y, 1 if the unit is m/d
    "visual": visual,  # Plot results along the way
    "vmax": vmax,  # vmin and vmax of the legend
    "verbose": verbose,
}  # Print information throughout TICOI processing

# Create a subfolder if it does not exist
if not os.path.exists(path_save):
    os.mkdir(path_save)

if type(cube_name) == str:
    cube_name = [cube_name]

# list_parameter = [150]
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
if zone_stable:
    list_RMSE_before, list_RMSE_after, list_std_before, list_std_after = [], [], [], []
if "L_curve" in option_visual:
    Residux, Residuy, Regux, Reguy, Rx, Ry, Rmeanx, Rmeany = [], [], [], [], [], [], [], []


cube = cube_data_class()
cube.load(cube_name[0], **load_kwargs)

# Several cubes have to be merged together
if len(cube_name) > 1:
    for n in range(1, len(cube_name)):
        cube2 = cube_data_class()
        cube2.load(cube_name[n], **load_kwargs)
        cube2 = cube.align_cube(cube2, reproj_vel=False, reproj_coord=True, interp_method="nearest")
        cube.merge_cube(cube2)
stop = [time.time()]
print(f"[Data loading] Loading the data cube.s took {round((stop[0] - start[0]), 4)} s")
print(f"[Data loading] Cube of dimension (nz,nx,ny) : ({cube.nz}, {cube.nx}, {cube.ny}) ")

cube2_date1 = cube.date1_().tolist()
cube2_date1.remove(np.min(cube2_date1))
start_date_interpol = np.min(cube2_date1)
last_date_interpol = np.max(cube.date2_())

# Filter the cube (compute rolling_mean for regu=1accelnotnull)
start.append(time.time())
obs_filt, _ = cube.filter_cube(**preData_kwargs)
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

    # %% Inversion

    # Load pixel data
    start.append(time.time())
    data, mean, dates_range = cube.load_pixel(i, j, rolling_mean=obs_filt, **load_pixel_kwargs)

    # Prepare interpolation dates
    first_date_interpol, last_date_interpol = prepare_interpolation_date(cube)
    interpolation_kwargs.update({"first_date_interpol": first_date_interpol, "last_date_interpol": last_date_interpol})
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
        
        
    # Save the results
    if save:
        result.to_csv(f"{save_path}/ILF_result.csv")
    stop.append(time.time())
    print(f"[Inversion] Inversion took {round((stop[3] - start[3]), 4)} s")

    # interpolation
    start.append(time.time())

    if interpolation_kwargs["interval_output"] == False:
        interpolation_kwargs["interval_output"] = 1

    if interpolation:
        # Proceed to interpolation
        dataf_lp = interpolation_core(
            result,
            path_save=save_path,
            data=dataf,
            first_date_interpol=start_date_interpol,
            last_date_interpol=last_date_interpol,
            **interpolation_kwargs,
        )

    if save:
        dataf_lp.to_csv(f"{path_save}/RLF_result.csv")
    
    if visual:
        visualisation_interpolation([dataf, dataf_lp], save=True, show=True, path_save=path_save, colors=["orange", "blue"])

    stop.append(time.time())
    print(f"[Interpolation] Interpolation took {round((stop[4] - start[3]), 4)} s")

    if "L_curve" in option_visual:
        Residux.append(dataf["NormR"][0])
        Residuy.append(dataf["NormR"][2])
        Regux.append(dataf["NormR"][1])
        Reguy.append(dataf["NormR"][3])

    # # cube2 = cube_list[0][0].deepcopy()  # for comparison with original data
    # # cube2.pick_offset(interval_output - 1, interval_output + 1)

    if zone_stable:  # RMSE over stable areas

        RMSE_before = m.sqrt(np.nansum(vv_befor_inv**2) / len(cube2.vy_()[:, j, i].data))
        RMSE_after = m.sqrt(np.nansum(vv_after_inv**2) / len(result_vx[:, 2]))
        list_RMSE_before.append(RMSE_before)
        list_RMSE_after.append(RMSE_after)

        std_before = np.std(vv_befor_inv)
        std_after = np.std(vv_after_inv)
        list_std_after.append(std_after)
        list_std_before.append(std_before)

        f.write(f"\n A shape {A.shape}")
        f.write(f"\n RMSE before inversion  {RMSE_before}")
        f.write(f"\n RMSE after inversion {RMSE_after}")
        f.write(f"\n std before inversion  {std_before}")
        f.write(f"\n std after inversion {std_after}")

    else:  # Velocity Vector Coherence

        vv_after_inv = np.array([np.sqrt(el) for el in (dataf_lp["vx"] ** 2 + dataf_lp["vy"] ** 2)])
        Normalized_Coh_vector_after = (
            np.sqrt(np.nansum(dataf_lp["vx"] / vv_after_inv) ** 2 + np.nansum(dataf_lp["vy"] / vv_after_inv) ** 2)
            / dataf_lp.shape[0]
        )
        list_NCoh_vector_after.append(Normalized_Coh_vector_after)

        f.write(f"\n A shape {A.shape}")
        # f.write(f'\n Coh_vector before inversion  {Coh_vector_before}')
        # f.write(f'\n Normalized Coh_vector before inversion {Normalized_Coh_vector_before}')
        f.write(f"\n Normalized Coh_vector after inversion {Normalized_Coh_vector_after}")
        # if interpolation: f.write(
        #     f'\n Normalized Coh_vector after inversion and interpolation {Normalized_Coh_vector_after_interpol}')

# vv_beforinv = np.sqrt(cube.vx_()[:, j, i].data ** 2 + cube.vy_()[:, j, i].data** 2)
# vv_afterinv = np.array([np.sqrt(el) for el in result_vx[:,2]**2+result_vy[:,2]**2])
# print('Vv RMSE (pour zone stable) avant inversion',m.sqrt(np.nansum(vv_beforinv**2) / cube.vx_()[:, j, i].data.shape[0]))
# print('Vv RMSE (pour zone stable) apres inversion',m.sqrt(np.nansum(vv_afterinv**2)/result_vx.shape[0]))
#
# print('Vy RMSE before inversion', m.sqrt(np.nansum(cube.vy_()[:, j, i].data ** 2) / cube.vx_()[:, j, i].data.shape[0]))
# print('Vy RMSE after inversion', m.sqrt(np.nansum(result_vy[:, 2] ** 2) / result_vy[:, 2].shape[0]))
#
# print('Vx RMSE before inversion',m.sqrt(np.nansum(cube.vx_()[:, j, i].data ** 2) / cube.vx_()[:, j, i].data.shape[0]))
# print('Vx RMSE after inversion', m.sqrt(np.nansum(result_vx[:,2] ** 2) / result_vx[:,2].shape[0]))

end = time.time()
print(f"{end - start[0]} ms")

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

#
# # %%
# if zone_stable:
#     # visualisation of the original error x and y according to the temporal baseline
#     fig1, ax1 = plt.subplots(figsize=(12, 4))
#     ax1.plot(list_parameter, list_NCoh_vector_before, linestyle='', marker='o', color='salmon',
#              label=f'Velocity observations')
#     ax1.plot(list_parameter, list_NCoh_vector_after, linestyle='', color='b', marker='o',
#              label=f'Leap Frog velocity time series')  # Display the vx components
#     # ax1.plot(list_coef, list_RMSE_before, linestyle='', marker='o', color='salmon',
#     #          label=f'Velocity observations')
#     # ax1.plot(list_coef, list_RMSE_after, linestyle='', color='b', marker='o',
#     #          label=f'Leap Frog velocity time series')  # Display the vx components
#     ax1.set_ylabel('RMSE', fontsize=14)
#     # ax1.set_xlabel('Temporal Baseline [days]', fontsize=14)
#     ax1.set_xlabel('Coef regularisation', fontsize=14)
#     plt.subplots_adjust(bottom=0.25)
#     ax1.legend(loc='lower left', bbox_to_anchor=(0.12, 0), bbox_transform=fig1.transFigure, ncol=2)
#     fig1.savefig(f'{path_save}coef{X_mode}_{options}_RMSE')
#     plt.show()
#
# else:
#
#     if 'L_curve' in option_visual:
#         fig1, ax1 = plt.subplots(figsize=(12, 4))
#         ax1.plot(Residux, Regux, linestyle='', marker='o', color='salmon',
#                  label=f'Vx')
#         for i in range(len(list_parameter)):
#             # i=0
#             ax1.annotate(f'{list_parameter[i]}', (Residux[i], Regux[i]))
#         # ax1.plot(Residuy, Reguy, linestyle='', color='b', marker='o',
#         #          label=f'Vy')  # Display the vx components
#         ax1.set_ylabel('Regularisation norm', fontsize=14)
#         # ax1.set_xlabel('Temporal Baseline [days]', fontsize=14)
#         ax1.set_xlabel('Residual norm', fontsize=14)
#         plt.subplots_adjust(bottom=0.25)
#         ax1.legend(loc='lower left', bbox_to_anchor=(0.12, 0), bbox_transform=fig1.transFigure)
#         plt.show()
#         fig1.savefig(f'{path_save}{parameter}{X_mode}_{options}_Lcurvex')
#
#         fig1, ax1 = plt.subplots(figsize=(12, 4))
#         ax1.plot(np.log(Residux), np.log(Regux), linestyle='', marker='o', color='salmon',
#                  label=f'Vx')
#         for i in range(len(list_parameter)):
#             # i=0
#             ax1.annotate(f'{list_parameter[i]}', (np.log(Residux[i]), np.log(Regux[i])))
#         # ax1.plot(Residuy, Reguy, linestyle='', color='b', marker='o',
#         #          label=f'Vy')  # Display the vx components
#         ax1.set_ylabel('Log Regularisation norm', fontsize=14)
#         # ax1.set_xlabel('Temporal Baseline [days]', fontsize=14)
#         ax1.set_xlabel('Log Residual norm', fontsize=14)
#         plt.subplots_adjust(bottom=0.25)
#         ax1.legend(loc='lower left', bbox_to_anchor=(0.12, 0), bbox_transform=fig1.transFigure)
#         plt.show()
#         fig1.savefig(f'{path_save}{parameter}{X_mode}_{options}_Lcurvexlog')
#
#         fig1, ax1 = plt.subplots(figsize=(12, 4))
#         ax1.plot(np.log(Residux)[1:], np.log(Regux)[1:], linestyle='', marker='o', color='salmon',
#                  label=f'Vx')
#         for i in range(len(list_parameter)):
#             # i=0
#             ax1.annotate(f'{list_parameter[i]}', (np.log(Residux[i]), np.log(Regux[i])))
#         # ax1.plot(Residuy, Reguy, linestyle='', color='b', marker='o',
#         #          label=f'Vy')  # Display the vx components
#         ax1.set_ylabel('Log Regularisation norm', fontsize=14)
#         # ax1.set_xlabel('Temporal Baseline [days]', fontsize=14)
#         ax1.set_xlabel('Log Residual norm', fontsize=14)
#         plt.subplots_adjust(bottom=0.25)
#         ax1.legend(loc='lower left', bbox_to_anchor=(0.12, 0), bbox_transform=fig1.transFigure)
#         plt.show()
#         fig1.savefig(f'{path_save}{parameter}{X_mode}_{options}_Lcurvexlog_zoom')
#
#
#
#         fig1, ax1 = plt.subplots(figsize=(12, 4))
#         ax1.plot(Residuy[1:], Reguy[1:], linestyle='', marker='o', color='salmon',
#                  label=f'Vy')
#         for i in range(len(list_parameter)):
#             # i=0
#             ax1.annotate(f'{list_parameter[i]}', (Residuy[i], Reguy[i]))
#         # ax1.plot(Residuy, Reguy, linestyle='', color='b', marker='o',
#         #          label=f'Vy')  # Display the vx components
#         ax1.set_ylabel('Regularisation norm', fontsize=14)
#         # ax1.set_xlabel('Temporal Baseline [days]', fontsize=14)
#         ax1.set_xlabel('Residual norm', fontsize=14)
#         plt.subplots_adjust(bottom=0.25)
#         ax1.legend(loc='lower left', bbox_to_anchor=(0.12, 0), bbox_transform=fig1.transFigure)
#         plt.show()
#         fig1.savefig(f'{path_save}{parameter}{X_mode}_{options}_Lcurvey')
#
#         fig1, ax1 = plt.subplots(figsize=(12, 4))
#         ax1.plot(np.log(Residuy), np.log(Reguy), linestyle='', marker='o', color='salmon',
#                  label=f'Vy')
#         for i in range(len(list_parameter)):
#             # i=0
#             ax1.annotate(f'{list_parameter[i]}', (np.log(Residuy[i]), np.log(Reguy[i])))
#         # ax1.plot(Residuy, Reguy, linestyle='', color='b', marker='o',
#         #          label=f'Vy')  # Display the vx components
#         ax1.set_ylabel('Log Regularisation norm', fontsize=14)
#         # ax1.set_xlabel('Temporal Baseline [days]', fontsize=14)
#         ax1.set_xlabel('Log Residual norm', fontsize=14)
#         plt.subplots_adjust(bottom=0.25)
#         ax1.legend(loc='lower left', bbox_to_anchor=(0.12, 0), bbox_transform=fig1.transFigure)
#         plt.show()
#         fig1.savefig(f'{path_save}coef{X_mode}_{options}_Lcurvey_log')
#
#         fig1, ax1 = plt.subplots(figsize=(12, 4))
#         ax1.plot(np.log(Residuy)[1:], np.log(Reguy)[1:], linestyle='', marker='o', color='salmon',
#                  label=f'Vy')
#         for i in range(len(list_parameter)):
#             # i=0
#             ax1.annotate(f'{list_parameter[i]}', (np.log(Residuy[i]), np.log(Reguy[i])))
#         # ax1.plot(Residuy, Reguy, linestyle='', color='b', marker='o',
#         #          label=f'Vy')  # Display the vx components
#         ax1.set_ylabel('Log Regularisation norm', fontsize=14)
#         # ax1.set_xlabel('Temporal Baseline [days]', fontsize=14)
#         ax1.set_xlabel('Log Residual norm', fontsize=14)
#         plt.subplots_adjust(bottom=0.25)
#         ax1.legend(loc='lower left', bbox_to_anchor=(0.12, 0), bbox_transform=fig1.transFigure)
#         plt.show()
#         fig1.savefig(f'{path_save}coef{X_mode}_{options}_Lcurvey_log_zoom')
#
#         # visualisation of the original error x and y according to the temporal baseline
#         fig1, ax1 = plt.subplots(2, 1, figsize=(12, 4))
#         # ax1[0].set_ylim(0.85, 1)
#         ax1[0].set_title(f'Strategy "{X_mode}"')
#         # ax1[0].plot(list_interval_output, list_Coh_vector_before, linestyle='-', color='r')
#         # ax1[0].plot(list_interval_output, list_Coh_vector_after, linestyle='', marker='o',
#         #             markersize=3)  # Display the vx components
#         # ax1[0].plot(list_parameter, list_Coh_vector_before, linestyle='-', color='r')
#         ax1[0].plot(list_parameter, list_Coh_vector_after, linestyle='', marker='o',
#                     markersize=3)  # Display the vx components
#         ax1[0].set_ylabel('CVV')
#         ax1[0].set_xlabel('Temporal Baseline')
#         # ax1[1].set_ylim(0.85, 1)
#         # ax1[1].set_title('Strategy "constant"')
#         # ax1[1].plot(list_interval_output, list_NCoh_vector_before, linestyle='-', color='r')
#         # ax1[1].plot(list_interval_output, list_NCoh_vector_after, linestyle='', marker='o',
#         #             markersize=3)  # Display the vx components
#         # ax1[1].plot(list_parameter, list_NCoh_vector_before, linestyle='-', color='r')
#         ax1[1].plot(list_parameter, list_NCoh_vector_after, linestyle='', marker='o',
#                     markersize=3)  # Display the vx components
#         ax1[1].set_ylabel('NCVV')
#         # ax1[1].set_xlabel('Temporal Baseline')
#         ax1[1].set_xlabel('Coef regularisation')
#         ax1[0].legend(loc='best')
#         # fig1.savefig(f'{savepath}{X_mode}_{options}_CVV')
#         fig1.savefig(f'{path_save}coef{X_mode}_{options}_CVV')
#         plt.show()
#
#         print('end')
#
#         dataf = pd.DataFrame({'VCC': list_NCoh_vector_after_interpol, 'param': list_parameter})
#         VCC_max99 = dataf['VCC'].max() * 0.99
#         VCC_max95 = dataf['VCC'].max() * 0.95
#         # list_interval_output=[72,60,24,12,36,48]
#         plt.style.use('seaborn-v0_8-whitegrid')
#         fig1, ax1 = plt.subplots(figsize=(12, 4))
#         ax1.plot(dataf['param'], dataf['VCC'], linestyle='-', marker='o', color='orange',
#                  label=f'Velocity observations')
#         ax1.plot([dataf[dataf[f'VCC'] > VCC_max99].iloc[0]['param']], [dataf[dataf[f'VCC'] > VCC_max99].iloc[0]['VCC']],
#                  linestyle='-', marker='o', color='r',
#                  label=f'99% of maximal VCC', markersize=5)
#         ax1.plot([dataf[dataf[f'VCC'] > VCC_max95].iloc[0]['param']], [dataf[dataf[f'VCC'] > VCC_max95].iloc[0]['VCC']],
#                  linestyle='-', marker='+', color='r',
#                  label=f'95% of maximal VCC', markersize=10)
#         ax1.set_ylabel('VVC', fontsize=25)
#         ax1.set_xlabel('Regularization coefficient', fontsize=25)
#         plt.subplots_adjust(bottom=0.25)
#         ax1.legend(loc='lower left', bbox_to_anchor=(0.12, -0.03), bbox_transform=fig1.transFigure, ncol=3, fontsize=16)
#         fig1.savefig(
#             f'{path_save}/Compa_VVC_coef')
#         plt.show()
#
# f.close()
#
# end = time.time()
# print(f'{(end - start) / 60} min')
