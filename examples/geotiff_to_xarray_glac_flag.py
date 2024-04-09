#!/usr/bin/env python
import xarray as xr
import rioxarray
import glob
import time
from datetime import datetime, date
from pyproj import CRS
from rasterio import features
import geopandas as gpd

start = time.time()

dst_nc = "/media/tristan/Data3/Hala_lake/Landsat8/Hala_lake_velocity_LS7.nc"

file_path = "/media/tristan/Data3/Hala_lake/Landsat8/Hala_displacement_LS7/"

obs_mode = "displacement"  # 'displacement' or 'velocity', to decide if the conversion is needed

unit = "m/y"  # if obs_mode is 'velocity', need to specify the unit of the velocity 'm/y' or 'm/d'

files = glob.glob(f"{file_path}*filt.tif")
files.sort()

assign_flag = True
flag_shp = "~/data/HMA_surging_glacier_inventory/HMA_surging_glacier_inventory_gamdam_v2_all.gpkg"

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

    if obs_mode == "displacement":
        period = (date2 - date1).days
        ds["vx"] = ds["vx"] / period * 365
        ds["vy"] = ds["vy"] / period * 365
    elif obs_mode == "velocity":
        if unit == "m/d":
            ds["vx"] = ds["vx"] * 365
            ds["vy"] = ds["vy"] * 365

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
    "description": "Mid date of the imapge paier",
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

if assign_flag:
    flag_shp = "~/data/HMA_surging_glacier_inventory/HMA_surging_glacier_inventory_gamdam_v2_all.gpkg"
    flag_shp = gpd.read_file(flag_shp).to_crs(proj4).clip(ds_combined.rio.bounds())
    
    flag_id = flag_shp['Surge_class'].apply(lambda x: 2 if x is not None else 1).astype("int16")
    geom_value = ((geom, value) for geom, value in zip(flag_shp.geometry, flag_id))
    
    flags = features.rasterize(
        geom_value,
        out_shape=ds_combined.rio.shape,
        transform=ds_combined.rio.transform(),
        all_touched=True,
        fill=0,  # background value
        dtype="int16",
    )
    
    flags = xr.Dataset(
                data_vars=dict(
                    flags=(["y", "x"], flags),
                ),
                coords=dict(
                    x=(["x"], ds_combined.x.data),
                    y=(["y"], ds_combined.y.data),))
    
    flags.to_netcdf('Hala_lake_velocity_LS7_flags.nc')
