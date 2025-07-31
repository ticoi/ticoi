#!/usr/bin/env python
"""
Store geotiff files in a netcdf file in a format compatible with the loader used in TICOI
This script is only a template. The user need to modify the lines TO MODIFY according to their data.

"""

import glob
import time
from datetime import date, datetime

import rioxarray
import xarray as xr
from pyproj import CRS

start = time.time()

dst_nc = "save_path/netcdf_name.nc"

file_path = "geotiff_path/"

obs_mode = "displacement"  # 'displacement' or 'velocity', to decide if the conversion is needed
output_mode = "velocity"  # 'displacement' or 'velocity', to decide if the output is in displacement or velocity

unit = "m/y"  # need to specify the unit of the velocity 'm/y' or 'm/d'

source = "L. Charrier, L. Guo"  # TO MODIFY
sensor = "LS8"  # TO MODIFY

files = glob.glob(f"{file_path}*.tif")
files.sort()

datasets = []

for file in files:
    # read the geotiffs
    print(file)
    ds = rioxarray.open_rasterio(file, band_as_variable=True)

    # extract the date from the filename (MODIFY IT ACCORDING TO YOUR DATA)
    date1 = datetime.strptime(file.split("/")[-1].split("day")[-1][:8], "%Y%m%d")  # TO MODIFY
    date2 = datetime.strptime(file.split("/")[-1].split("day")[-1][9:17], "%Y%m%d")  # TO MODIFY
    mid_date = date1 + (date2 - date1) / 2

    # set the variable name
    ds = ds.assign_coords(time=mid_date)
    ds = ds.rename({"band_1": "vx", "band_2": "vy", "time": "mid_date"})  # TO MODIFY
    ds = ds.expand_dims("mid_date")

    # convert to displacement
    if output_mode == "displacement":
        if obs_mode == "velocity":
            velo_unit = 365 if unit == "m/y" else 1
            period = (date2 - date1).days
            ds["vx"] = ds["vx"] * period / velo_unit
            ds["vy"] = ds["vy"] * period / velo_unit
        elif obs_mode == "displacement":
            pass
    elif output_mode == "velocity":
        if obs_mode == "velocity":
            if unit == "m/y":
                pass
            elif unit == "m/d":
                velo_unit = 365
                ds["vx"] = ds["vx"] * velo_unit
                ds["vy"] = ds["vy"] * velo_unit
            else:
                raise ValueError(f"'{unit}' should be either 'm/d' or 'm/y'")
        elif obs_mode == "displacement":
            velo_unit = 365 if unit == "m/y" else 1
            period = (date2 - date1).days
            ds["vx"] = ds["vx"] * 365 / period
            ds["vy"] = ds["vy"] * 365 / period
    else:
        raise ValueError(f"'{output_mode}' should be either 'velocity' or 'displacement'")

    ds["date1"] = date1
    ds["date2"] = date2

    # add to the list
    datasets.append(ds)

ds_combined = xr.concat(datasets, dim="mid_date")

# set the variable description
ds_combined.x.attrs = {
    "description": "X coordinate of projection",
    "units": "m",
    "axis": "X",
}
ds_combined.y.attrs = {
    "description": "Y coordinate of projection",
    "units": "m",
    "axis": "Y",
}
ds_combined.mid_date.attrs = {
    "description": "Mid date of the image pair",
    "axis": "Time",
}
ds_combined.date1.attrs = {
    "description": "Start date of the image pair",
}
ds_combined.date2.attrs = {
    "description": "End date of the image pair",
}
ds_combined.vx.attrs = {
    "description": "X component of velocity or displacement",
    "units": "m/y",
    "axis": "X",
}
ds_combined.vy.attrs = {
    "description": "Y component of velocity or displacement",
    "units": "m/y",
    "axis": "Y",
}

proj4 = CRS(ds.spatial_ref.projected_crs_name).to_proj4()

ds_combined.attrs.update(
    {
        "Conventions": "CF-1.10",
        "title": "Image pair cube of glacier displacement",
        "institution": "IGE",
        "references": "Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an optimal temporal sampling from displacement observation networks. IEEE Transactions on Geoscience and Remote Sensing, 60, 1-10.\n Charrier, L., Yan, Y., Koeniguer, E. C., Trouve, E., Mouginot, J., & Millan, R. (2022, June). Fusion of multi-temporal and multi-sensor ice velocity observations. In International Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences.",
        "source": source,
        "sensor": sensor,
        "proj4": proj4,
        "author": "L. Charrier, L. Guo",
        "history": f"Created at {date.today()}",
    }
)

print(ds_combined)
output_mode = "disp" if output_mode == "displacement" else "velo"
print(f"obs_mode: {obs_mode}\noutput_mode: {output_mode}")
print(f"Please use disp_or_velo='{output_mode}' to load the data in TICOI")
ds_combined.to_netcdf(dst_nc)
print("time ", (time.time() - start), "seconds")
