#!/usr/bin/env python3
"""
Visualization of the cube results from TICO (without interpolation), or TICOI (with interpolation)

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

## ------------------- Choose TICOI cube processing method ----------------- ##
# The  TICOI cube was already calculated before, load it by giving the cubes to be loaded in a dictionary like {name: path} (name can be
# 'interp', 'invert' or 'raw' as for returned, path can be a single str or a list of str to merge cubes) in cube_name, or a single str to a TICOI cube
# Plot figure for a given pixel

save = True  # If True, save TICOI results to a netCDF file
save_mean_velocity = True  # Save a .tiff file with the mean reulting velocities, as an example

# For TICOI_process = 'load', generate a 'result' list with raw data and/or TICOI results as pandas dataframe for further processing
compute_result_load = False

## ------------------------------ Data selection --------------------------- ##
# cube_name can be a dictionary like {name: path} to load existing cubes and name them (path can be a list of str or a single str)
# If it is an str (or list of str), we suppose we want to load TICOI results (like 'interp' in the dict)

cube_name = {
    "raw": f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..", "test_data"))}/Alps_Mont-Blanc_Argentiere_S2.nc',
    "invert": f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "cube"))}/Argentiere_example_invert.nc',
    "interp": f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "cube"))}/Argentiere_example_interp.nc',
}
path_save = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "pixel"))}/'  # Path where to store the results
result_fn = "Argentiere_example"  # Name of the netCDF file to be created (if save is True)
proj = "EPSG:32632"  # EPSG system of the given coordinates

i, j = 343686.3, 5091294.9  # Pixel coordinates

## ------------------------- Visualization parameters ---------------------- ##
colors = ["blueviolet", "orange"]
cmap = "rainbow"
log_scale = False
option_visual = [  # Visualization options for raw data and inverted results
    "obs_xy",
    "obs_magnitude",
    "obs_vxvy_quality",
    "xcount_xy",
    "xcount_vv",
]
option_visual_interp = [  # Visualization options for interpolated results
    "interp_xy_overlaid",
    "interp_xy_overlaid_zoom",
    "invertvv_overlaid",
    "invertvv_overlaid_zoom",
    "direction_overlaid",
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
filt = True
delete_outliers = (
    "vvc_angle"  # Delete data with a poor quality indicator (if int), or with aberrant direction ('vvc_angle')
)

if not os.path.exists(path_save):
    os.mkdir(path_save)


# %%========================================================================= #
#                                 DATA LOADING                                #
# =========================================================================%% #

start, stop = [], []

if type(cube_name) == dict and "raw" in cube_name.keys():
    start.append(time.time())

    # Load the cube.s
    cube = cube_data_class()
    cube.load(cube_name["raw"], **load_kwargs)

    stop.append(time.time())
    print(f"[Data loading] Cube of dimension (nz, nx, ny): ({cube.nz}, {cube.nx}, {cube.ny}) ")
    print(f"[Data loading] Data loading took {round(stop[-1] - start[-1], 3)} s")


# %%========================================================================= #
#                                      TICOI                                  #
# =========================================================================%% #

start.append(time.time())

cube_interp, cube_invert = None, None

#  Load inversion results
if type(cube_name) == dict and "invert" in cube_name.keys():
    cube_invert = cube_data_class()
    cube_invert.load(cube_name["invert"], **load_kwargs)

# Load interpolation results
if (type(cube_name) == dict and "interp" in cube_name.keys()) or type(cube_name) == str:
    cube_interp = cube_data_class()
    cube_interp.load(cube_name["interp"] if type(cube_name) == dict else cube_name, **load_kwargs)

stop.append(time.time())

# Plot the mean velocity as an example
if save_mean_velocity and cube_interp is not None:
    cube_interp.average_cube(return_format="geotiff", return_variable=["vv"], save=True, path_save=path_save)
if save or save_mean_velocity:
    print(f"[Writing results] Results saved at {path_save}")

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
    "invertvv_overlaid": (lambda pix: pix.plot_vv_overlaid(type_data="interp", colors=colors, zoom_on_results=False)),
    "invertvv_overlaid_zoom": (
        lambda pix: pix.plot_vv_overlaid(type_data="interp", colors=colors, zoom_on_results=True)
    ),
    "direction_overlaid": (lambda pix: pix.plot_direction_overlaid(type_data="interp")),
}

# Interpolated data cube
t = cube_interp.load_pixel(i, j, output_format="df", proj=proj, visual=True)[0]

# Filter and load raw data
if filt:
    cube.filter_cube(delete_outliers=delete_outliers)
t2 = cube.load_pixel(i, j, output_format="df", proj=proj, visual=True)[0]

pixel_object = pixel_class()
pixel_object.load(
    [t, t2], save=False, show=True, A=False, path_save=path_save, type_data=["interp", "obs_filt" if filt else "obs"]
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

stop.append(time.time())
print(f"[Overall] Overall processing took {round(stop[-1] - start[0], 0)} s")
