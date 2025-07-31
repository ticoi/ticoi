import csv
import os
from datetime import datetime

import glaft
import matplotlib.pyplot as plt
import numpy as np

from ticoi.cube_data_classxr import CubeDataClass
from ticoi.example import get_path

# ======== User Settings (please review this section every time) ========
current_dir = os.path.dirname(os.path.abspath(__file__))  # current file
package_root = os.path.abspath(os.path.join(current_dir, ".."))
cube_name = get_path("Argentiere_example_interp")  # path to our dataset
path_glaft_test = os.path.join(package_root, "test_data", "for_glaft", "Argentiere_example_interp")
path_save = os.path.join(current_dir, "..", "results", "cube", "r_")
save = True  # to save results

epsg_str = "epsg:32632"
static_area = get_path("Argentiere_static")
iceflow_area = get_path("Argentiere_iceflow")
output_csvfile = "glaft_argentiere_interp.csv"
output_png = "glaft_argentiere_interp.png"
# =======================================================================


def datetime64_to_datestr(dt_np64):
    """
    numpy.datetime64('2018-05-10T00:00:00.000000000') to datetime.date(2018, 5, 10)
    """
    dt = datetime.fromisoformat(str(dt_np64).split("T")[0])
    date_only = dt.date()
    return date_only


cube = CubeDataClass()
cube.load(cube_name)
ds = cube.ds.transpose("y", "x", ...)  # transpose coordinates

vx_file = "test_geotiff_vx.tif"  # This is a temporary file. Can be removed after running this script.
vy_file = "test_geotiff_vy.tif"  # This is a temporary file. Can be removed after running this script.
time_list = []
metric1_list = []
metric2_list = []

# Loop over the nc file and get both metrics for each velocity map
for i in range(ds["mid_date"].size):
    tmp = ds["vx"].isel(mid_date=i)
    tmp.rio.write_crs(cube.ds.rio.crs, inplace=True)
    tmp.rio.to_raster(vx_file)
    tmp = ds["vy"].isel(mid_date=i)
    tmp.rio.write_crs(cube.ds.rio.crs, inplace=True)
    tmp.rio.to_raster(vy_file)
    dt_np64 = ds.isel(mid_date=i)["mid_date"].values
    dt = datetime64_to_datestr(dt_np64)

    try:
        experiment = glaft.Velocity(vxfile=vx_file, vyfile=vy_file, static_area=static_area, on_ice_area=iceflow_area)
        experiment.static_terrain_analysis()
        experiment.longitudinal_shear_analysis()
        metric1 = np.sqrt(experiment.metric_static_terrain_x * experiment.metric_static_terrain_y)
        metric2 = experiment.metric_alongflow_shear
    except:
        continue

    time_list.append(dt)
    metric1_list.append(metric1)
    metric2_list.append(metric2)

headers = ["Mid_date", "Metric1(m/yr)", "Metric2(1/yr)"]
metric1_strlist = [f"{i:.4f}" for i in metric1_list]
metric2_strlist = [f"{i:.4f}" for i in metric2_list]

# Write to CSV
with open(output_csvfile, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    for row in zip(time_list, metric1_strlist, metric2_strlist):
        writer.writerow(row)


if save and not os.path.exists(path_save):
    os.mkdir(path_save)

# Visualize the metrics
fig, axs = plt.subplots(2, 1, figsize=(7, 7))
axs[0].plot(time_list, metric1_list)
axs[1].plot(time_list, metric2_list)

axs[0].set_ylabel(r"Metric 1 ($\sqrt{\delta_u  \delta_v}$, m/yr)")
axs[1].set_ylabel(r"Metric 2 ($\delta_{x'y'}$, 1/yr)")
if save:
    fig.savefig(f"{path_save}{output_png}")
