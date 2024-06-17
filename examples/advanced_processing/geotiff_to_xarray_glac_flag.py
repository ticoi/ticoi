#!/usr/bin/env python
import glob
import time
from datetime import date, datetime

import geopandas as gpd
import rioxarray
import xarray as xr
from pyproj import CRS
from rasterio import features
import dask.array as da


def determine_optimal_chunk_size(
    cube,
    variable_name: str = "vx",
    x_dim: str = "x",
    y_dim: str = "y",
    time_dim: str = "mid_date",
    verbose: bool = False,
) -> (int, int, int):

    """
    A function to determine the optimal chunk size for a given time series array based on its size.

    :param variable_name: [str] [default is 'vx'] --- Name of the variable containing the time series array
    :param x_dim: [str] [default is 'x'] --- Name of the x dimension in the array
    :param y_dim: [str] [default is 'y'] --- Name of the y dimension in the array
    :param time_dim_name: [str] [default is 'mid_date'] --- Name of the z dimension within the original dataset cube
    :param verbose: [bool] [default is False] --- Boolean flag to control verbosity of output

    :return tc: [int] --- Chunk size along the time dimension
    :return yc: [int] --- Chunk size along the y dimension
    :return xc: [int] --- Chunk size along the x dimension
    """

    if verbose:
        print("[Data loading] Dask chunk size:")

    # set chunk size to 5 MB if single time series array < 1 MB in size, else increase to max of 1 GB chunk sizes.
    time_series_array_size = (
        cube[variable_name]
        .sel(
            {
                x_dim: cube[variable_name][x_dim].values[0],
                y_dim: cube[variable_name][y_dim].values[0],
            }
        )
        .nbytes
    )
    mb = 1048576
    if time_series_array_size < 1e6:
        chunk_size_limit = 50 * mb
    elif time_series_array_size < 1e7:
        chunk_size_limit = 100 * mb
    elif time_series_array_size < 1e8:
        chunk_size_limit = 200 * mb
    else:
        chunk_size_limit = 1000 * mb

    time_axis = cube[variable_name].dims.index(time_dim)
    x_axis = cube[variable_name].dims.index(x_dim)
    y_axis = cube[variable_name].dims.index(y_dim)
    axis_sizes = {i: -1 if i == time_axis else "auto" for i in range(3)}
    dask_array = da.from_array(cube[variable_name].data)
    arr = dask_array.rechunk(axis_sizes, block_size_limit=chunk_size_limit, balance=True)
    tc, yc, xc = arr.chunks[time_axis][0], arr.chunks[y_axis][0], arr.chunks[x_axis][0]
    chunksize = cube[variable_name][:tc, :yc, :xc].nbytes / 1e6
    if verbose:
        print("[Data loading] Chunk shape:", "(" + ",".join([str(x) for x in [tc, yc, xc]]) + ")")
        print(
            "[Data loading] Chunk size:",
            cube[variable_name][:tc, :yc, :xc].nbytes,
            "(" + str(round(chunksize, 1)) + "MB)",
        )
    return tc, yc, xc




start = time.time()

dst_nc = '/media/tristan/Data3/Hala_lake/Landsat7_refine/Hala_lake_disp_refine_LS7.nc'

file_path = "/media/tristan/Data3/Hala_lake/Landsat7_refine/Velo_refine/filtered/"

obs_mode = "displacement"  # 'displacement' or 'velocity', to decide if the conversion is needed

unit = "m/y"  # if obs_mode is 'velocity', need to specify the unit of the velocity 'm/y' or 'm/d'

files = glob.glob(f"{file_path}*filt.tif")
files.sort()

assign_flag = False
flag_shp = "~/data/HMA_surging_glacier_inventory/HMA_surging_glacier_inventory_gamdam_v2_all.gpkg"
dst_flag_nc = "/media/tristan/Data3/Hala_lake/Landsat8/Hala_lake_displacement_LS7_flags.nc"

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

    # convert to displacement
    if obs_mode == "velocity":
        velo_unit = 365 if unit == "m/y" else 1
        period = (date2 - date1).days
        ds["vx"] = ds["vx"] * period / velo_unit
        ds["vy"] = ds["vy"] * period / velo_unit
    elif obs_mode == "displacement":
        pass

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
    "description": "Mid date of the imape paier",
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
sensor = "LS8"

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

tc, yc, xc = determine_optimal_chunk_size(
    ds_combined, variable_name="vx", x_dim="x", y_dim="y", time_dim="mid_date", verbose=True
)

ds_combined = ds_combined.chunk({"mid_date": tc, "x": xc, "y": yc})

print(ds_combined)
ds_combined.to_netcdf(dst_nc)
print("time ", (time.time() - start), "seconds")

if assign_flag:
    flag_shp = gpd.read_file(flag_shp).to_crs(proj4).clip(ds_combined.rio.bounds())

    flag_id = flag_shp["Surge_class"].apply(lambda x: 2 if x is not None else 1).astype("int16")
    geom_value = ((geom, value) for geom, value in zip(flag_shp.geometry, flag_id))

    flag = features.rasterize(
        geom_value,
        out_shape=ds_combined.rio.shape,
        transform=ds_combined.rio.transform(),
        all_touched=True,
        fill=0,  # background value
        dtype="int16",
    )

    flag = xr.Dataset(
        data_vars=dict(
            flag=(["y", "x"], flag),
        ),
        coords=dict(
            x=(["x"], ds_combined.x.data),
            y=(["y"], ds_combined.y.data),
        ),
    )

    flag.to_netcdf(dst_flag_nc)
