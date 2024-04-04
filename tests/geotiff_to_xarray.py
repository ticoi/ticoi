#!/usr/bin/env python
import xarray as xr
import rioxarray
import glob
import time
from datetime import datetime, date
from pyproj import CRS


start = time.time()

dst_nc = "/media/tristan/Data3/Hala_lake/Landsat8/Hala_lake_diaplacement_LS7_subset.nc"

file_path = "/media/tristan/Data3/Hala_lake/Landsat8/Hala_displacement_LS7_subset/"

obs_mode = 'displacement' # 'displacement' or 'velocity', to decide if the conversion is needed

unit = 'm/y' # if obs_mode is 'velocity', need to specify the unit of the velocity 'm/y' or 'm/d'

files = glob.glob(f"{file_path}*filt.tif")
files.sort()

# demo_nc = "/media/tristan/Data3/Hala_lake/Landsat8/Hala_lake_velocities_LS7_subset.nc"
# demo_ds = xr.open_dataset(demo_nc)

datasets = []

for file in files:
    # read the geotiffs
    print(file)
    ds = rioxarray.open_rasterio(file, band_as_variable=True)

    # extract the date from the filename
    date1 = datetime.strptime(file.split("/")[-1].split("day")[-1][:8], "%Y%m%d")
    date2 = datetime.strptime(file.split("/")[-1].split("day")[-1][9:17], "%Y%m%d")
    mid_date = date1 + (date2 - date1) / 2
    
    # set the variable name
    ds = ds.assign_coords(time=mid_date)
    ds = ds.rename({"band_1": "vx", "band_2": "vy", "time": "mid_date"})
    ds = ds.expand_dims("mid_date")
    
    if obs_mode == 'displacement':
        period = (date2 - date1).days
        ds['vx'] = ds['vx'] / period * 365
        ds['vy'] = ds['vy'] / period * 365
    elif obs_mode == 'velocity':
        if unit == 'm/d':
            ds['vx'] = ds['vx'] * 365
            ds['vy'] = ds['vy'] * 365

    ds["date1"] = date1
    ds["date2"] = date2

    # add to the list
    datasets.append(ds)

ds_combined = xr.concat(datasets, dim="mid_date")
# set the variable description

ds_combined.x.attrs = {"description": "X coordinate of projection", "units": "m", 'axis': 'X'}
ds_combined.y.attrs = {"description": "Y coordinate of projection", "units": "m", 'axis': 'Y'}
ds_combined.mid_date.attrs = {
    "description": "Mid date of the imapge paier",
    'axis': 'Time'
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
    'axis': 'X'
}
ds_combined.vy.attrs = {
    "description": "Y component of velocity or displacement",
    "units": "m/y",
    'axis': 'Y'
}


proj4 = CRS(ds.spatial_ref.projected_crs_name).to_proj4()
source = "L. Charrier, L. Guo"
sensor = "LS7"

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
ds_combined.to_netcdf(dst_nc)

print("time ", (time.time() - start), "seconds")
