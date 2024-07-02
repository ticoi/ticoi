#!/usr/bin/env python3
r"""
Visualization of the cube results from TICO (without interpolation), or TICOI (with interpolation).
Additional visualisation functions related to the stufy of the seasonality of the data are available.
   /!\ A TICOI cube must be processed and saved to a netCDF file before calling this script (it can be done in cube_ticoi_demo.py
or cube_ticoi_demo_advanced.py)

Author : Laurane Charrier, Lei Guo, Nathan Lioret
Reference:
    Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an optimal temporal sampling from displacement
    observation networks. IEEE Transactions on Geoscience and Remote Sensing.
    Charrier, L., Yan, Y., Colin Koeniguer, E., Mouginot, J., Millan, R., & Trouvé, E. (2022). Fusion of multi-temporal and multi-sensor ice velocity observations.
    ISPRS annals of the photogrammetry, remote sensing and spatial information sciences, 3, 311-318.
"""

import os
import time
import warnings

from ticoi.cube_data_classxr import cube_data_class
from ticoi.pixel_class import pixel_class

# %%========================================================================= #
#                                   PARAMETERS                                #
# =========================================================================%% #

warnings.filterwarnings("ignore")

## ------------------------------ Data selection --------------------------- ##
# A TICOI cube must be processed before calling to this script

# cube_name must be a dictionary like {name: path} to load existing cubes and name them (path can be a list of str or a single str)
cube_name = {
    "raw": f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..", "test_data"))}/Alps_Mont-Blanc_Argentiere_S2.nc',
    "invert": f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "cube"))}/Argentiere_example_invert.nc',
    "interp": f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "cube"))}/Argentiere_example_interp.nc',
}
path_save = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "pixel"))}/'  # Path where to store the results
proj = "EPSG:32632"  # EPSG system of the given coordinates

i, j = 343686.3, 5091294.9  # Pixel coordinates

## ------------------------- Visualization parameters ---------------------- ##
save = True  # If True, save the figures to path_save
colors = ["orange", "blue"]
cmap = "rainbow"
log_scale = False
option_visual = [  # Visualization options for raw data and inverted results
    "obs_xy",
    "obs_magnitude",
    "obs_vxvy_quality",
]
option_visual_interp = [  # Visualization options for interpolated results
    "interp_xy_overlaid",
    "interp_xy_overlaid_zoom",
    "interpvv_overlaid",
    "interpvv_overlaid_zoom",
    "direction_overlaid",
]
option_visual_seasonality = [  # Visualization options to study the seasonality of the data
    "filtered_results",
    "TF",
    "best_matching_sinus",
    "annual_curves",
]

## ---------------------------- Loading parameters ------------------------- ##
load_kwargs = {
    "chunks": {},
    "conf": False,  # If True, confidence indicators will be put between 0 and 1, with 1 the lowest errors
    "subset": None,  # Subset of the data to be loaded ([xmin, xmax, ymin, ymax] or None)
    "buffer": [i, j, 250],  # Area to be loaded around the pixel ([longitude, latitude, buffer size] or None)
    "pick_date": ["2015-01-01", "2023-01-01"],  # Select dates ([min, max] or None to select all)
    "pick_sensor": None,  # Select sensors (None to select all)
    "pick_temp_bas": None,  # Select temporal baselines ([min, max] in days or None to select all)
    "proj": proj,  # EPSG system of the given coordinates
    "verbose": False,  # Print information throughout the loading process
}

## --------------------------- Filtering parameters ------------------------ ##
# Filter the raw data cube before plotting
filt_raw = True
delete_outliers = (
    "vvc_angle"  # Delete data with a poor quality indicator (if int), or with aberrant direction ('vvc_angle')
)

## ------------------- Parameters for seasonality analysis ----------------- ##
# Is the periodicity frequency imposed to 1/365.25 (one year seasonality) ?
impose_frequency = True
# Add several sinus at different freqs (1/365.25 and harmonics (2/365.25, 3/365.25...) if impose_frequency is True)
#   (only available for impose_frequency = True for now)
several_freq = 5
# Compute also the best matching sinus to raw data, for comparison
raw_seasonality = True
# Filter to use in the first place
# 'highpass' : apply a bandpass filter between low frequencies (reject variations over several years (> 1.5 y))
# and the Nyquist frequency to ensure Shanon theorem
# 'lowpass' : or apply a lowpass filter only (to Nyquist frequency) : risk of tackling an interannual trend (long period)
filt = None
# Method used to compute local variations
# 'rolling_7d' : median of the std of the data centered in +- 3 days around each central date
# 'uniform_7d' : median of the std of the data centered in +- 3 days around dates constantly distributed every redundnacy
# days -- BEST
# 'uniform_all' : median of the std of each data covering the dates, which are constantly distributed every redundancy days
# 'residu' : standard deviation of the data previously subtracted by TICOI results (ground truth) = standard deviation of the "noise"
local_var_method = "uniform_7d"
verbose = True  # Plot informations throughout the seasonality plotting process

# Parameters for annual curves plotting
normalize = True  # Normalize the annual velocities between 0 and 1
# Statistics to be computed, plotted and returned
statistics = ["min_max", "mean", "median", "std", "amplitude", "max_day", "nb_peaks", "relative_max"]

if not os.path.exists(path_save):
    os.mkdir(path_save)


# %%========================================================================= #
#                                 DATA LOADING                                #
# =========================================================================%% #

start, stop = [], []
start.append(time.time())

# Load the raw data
if type(cube_name) == dict and "raw" in cube_name.keys():
    cube = cube_data_class()
    cube.load(cube_name["raw"], **load_kwargs)

# Load inversion results if given
if type(cube_name) == dict and "invert" in cube_name.keys():
    cube_invert = cube_data_class()
    cube_invert.load(cube_name["invert"], **load_kwargs)

# Load interpolation results
if type(cube_name) == dict and "interp" in cube_name.keys():
    cube_interp = cube_data_class()
    cube_interp.load(cube_name["interp"] if type(cube_name) == dict else cube_name, **load_kwargs)

stop.append(time.time())
print(f"[Data loading] Cube of dimension (nz, nx, ny): ({cube.nz}, {cube.nx}, {cube.ny}) ")
print(f"[Data loading] Data loading took {round(stop[-1] - start[-1], 3)} s")


# %%========================================================================= #
#                                      TICOI                                  #
# =========================================================================%% #

dico_visual = {
    "obs_xy": (lambda pix: pix.plot_vx_vy(color=colors[0], type_data="obs")),
    "obs_magnitude": (lambda pix: pix.plot_vv(color=colors[0], type_data="obs")),
    "obs_vxvy_quality": (lambda pix: pix.plot_vx_vy_quality(cmap=cmap, type_data="obs")),
    "invertxy_overlaid": (lambda pix: pix.plot_vx_vy_overlaid(colors=colors)),
    "invertvv_overlaid": (lambda pix: pix.plot_vv_overlaid(colors=colors)),
    "residuals": (lambda pix: pix.plot_residuals(log_scale=log_scale)),
    "xcount_xy": (lambda pix: pix.plot_xcount_vx_vy(cmap=cmap)),
    "xcount_vv": (lambda pix: pix.plot_xcount_vv(cmap=cmap)),
    "invert_weight": (lambda pix: pix.plot_weights_inversion()),
}

dico_visual_interp = {
    "interp_xy_overlaid": (
        lambda pix: pix.plot_vx_vy_overlaid(type_data="interp", colors=colors, zoom_on_results=False)
    ),
    "interp_xy_overlaid_zoom": (
        lambda pix: pix.plot_vx_vy_overlaid(type_data="interp", colors=colors, zoom_on_results=True)
    ),
    "interpvv_overlaid": (lambda pix: pix.plot_vv_overlaid(type_data="interp", colors=colors, zoom_on_results=False)),
    "interpvv_overlaid_zoom": (
        lambda pix: pix.plot_vv_overlaid(type_data="interp", colors=colors, zoom_on_results=True)
    ),
    "direction_overlaid": (lambda pix: pix.plot_direction_overlaid(type_data="interp")),
}

dico_visual_seasonality = {
    "filtered_results": (lambda pix: pix.plot_filtered_results(filt=filt, impose_frequency=impose_frequency)),
    "TF": (lambda pix: pix.plot_TF(filt=filt, verbose=verbose)),
    "best_matching_sinus": (
        lambda pix: pix.plot_best_matching_sinus(
            filt=filt,
            impose_frequency=impose_frequency,
            raw_seasonality=raw_seasonality,
            several_freq=several_freq,
            verbose=verbose,
        )
    ),
    "annual_curves": (lambda pix: pix.plot_annual_curves()),
}

# Interpolated data cube
result = cube_interp.load_pixel(i, j, output_format="df", proj=proj, visual=True)[0]

# Filter and load raw data
if filt_raw:
    cube.filter_cube(delete_outliers=delete_outliers)
data_raw = cube.load_pixel(i, j, output_format="df", proj=proj, visual=True)[0]

pixel_object = pixel_class()
pixel_object.load(
    [result, data_raw],
    save=save,
    show=True,
    A=False,
    path_save=path_save,
    type_data=["interp", "obs_filt" if filt_raw else "obs"],
)

for option in option_visual:
    if option not in dico_visual.keys():
        raise ValueError(f"'{option}' is not a valid visual option, please choose among {list(dico_visual.keys())}")
    dico_visual[option](pixel_object)

for option in option_visual_interp:
    if option not in dico_visual_interp.keys():
        raise ValueError(
            f"'{option}' is not a valid visual option for interpolation results, please choose among {list(dico_visual_interp.keys())}"
        )
    dico_visual_interp[option](pixel_object)

for option in option_visual_seasonality:
    if option not in dico_visual_seasonality.keys():
        raise ValueError(
            f"'{option}' is not a valid visual option for interpolation results, please choose among {list(dico_visual_interp.keys())}"
        )
    dico_visual_seasonality[option](pixel_object)

stop.append(time.time())
print(f"[Overall] Overall processing took {round(stop[-1] - start[0], 0)} s")
