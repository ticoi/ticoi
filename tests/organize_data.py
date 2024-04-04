import os

import geopandas as gpd
import pandas as pd
import xarray as xr
from shapely.geometry import Point, Polygon

# script to organize test data for testing
# assumes you have downloaded test data from googlr drive,
# placed in 'test_data' directory within the project directory
# with same structure as google drive (ie. Malaspina/, Lowell/)


def get_bounds(input_xr):

    xmin = input_xr.coords["x"].data.min()
    xmax = input_xr.coords["x"].data.max()

    ymin = input_xr.coords["y"].data.min()
    ymax = input_xr.coords["y"].data.max()
    return xmin, xmax, ymin, ymax


def make_bounds_poly(input_xr):
    """function to return a geopandas geodataframe representing the footprt"""

    xmin, xmax, ymin, ymax = get_bounds(input_xr)

    pts_ls = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]

    crs = input_xr.rio.crs

    polygon_geom = Polygon(pts_ls)
    polygon = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[polygon_geom])

    return polygon


def find_granule_by_point(input_point):
    """returns url for the granule (zarr datacube) containing a specified point. point must be passed in epsg:4326"""
    catalog = gpd.read_file("https://its-live-data.s3.amazonaws.com/datacubes/catalog_v02.json")

    # make shapely point of input point
    p = gpd.GeoSeries([Point(input_point[0], input_point[1])], crs="EPSG:4326")
    # make gdf of point
    gdf = gpd.GeoDataFrame({"label": "point", "geometry": p})
    # find row of granule
    granule = catalog.sjoin(gdf, how="inner")

    url = granule["zarr_url"].values[0]
    return url


def read_in_s3(http_url, chunks="auto"):

    datacube = xr.open_dataset(
        http_url,
        engine="zarr",
        # storage_options={'anon':True},
        chunks=chunks,
    )

    return datacube


class SingleLocationData:
    """this is a class to hold test data for a single location, malaspina or lowell
    it holds locally downloaded itslive data (its_nc), itslive data from s3 (its_zarr),
    and the gps data. this object class is used in creating the TestData object that
    will be used for tests

    Attributes:
        location name (str): name of location, either Malaspina or Lowell
        _test_data_dir (str): path to test data directory
        _nc_path (str): path to locally downloaded itslive data
        _csv_path (str): path to gps data
        its_nc (xr.Dataset): locally downloaded itslive data
        its_zarr (xr.Dataset): itslive data from s3
        s3_url (str): url to itslive data in s3
        gps_csv (pd.DataFrame): gps data"""

    def __init__(self, test_data_dir, location_name, nc_file, csv_file):
        self.location_name = location_name
        self._test_data_dir = test_data_dir
        self._nc_path = os.path.join(self._test_data_dir, f"{self.location_name}/{nc_file}")
        self._csv_path = os.path.join(self._test_data_dir, f"{location_name}/{csv_file}")
        self.its_nc = self.read_in_nc()
        self.its_zarr, self.s3_url = self.find_itslive_urls()
        self.gps_csv = self.read_in_gps_csv()

    def read_in_gps_csv(self):
        """Reads csv files stored in directory for each location
        returns:
            pd.DataFrame: gps data"""
        df = pd.read_csv(self._csv_path)
        return df

    def read_in_nc(self):
        """Reads in netcdf files stored in directory for each location,
        uses stored CRS info to set a crs attribute for the xarray object
        Returns:
            xr.Dataset: itslive data"""
        its_nc = xr.open_dataset(self._nc_path)

        if self.location_name == "Malaspina":
            its_nc = its_nc.rio.write_crs(its_nc.mapping.attrs["spatial_epsg"])
        elif self.location_name == "Lowell":
            its_nc = its_nc.rio.write_crs(its_nc.attrs["proj4"])
        return its_nc

    def find_itslive_urls(self):
        """Finds the url for the granule containing the centroid of the itslive data,
        Uses the locally downloaded itslive data, creatse a gpd.GeoDataFrame of the bounds
        of this object, uses the centroid of the gdf to query the itslive s3 catalog for the
        appropriate url to read in the corresponding zarr datacube. Writes the crs of the zarr
        datacube and clips the data cube to the bounds of the local itslive data
        Returns:
            xr.Dataset: clipped itslive data
            str: url to itslive data in s3"""

        vec_nc = make_bounds_poly(self.its_nc)
        vec_ll = vec_nc.to_crs("EPSG:4326")
        point_ls = [vec_ll.centroid.x[0], vec_ll.centroid.y[0]]
        url = find_granule_by_point(point_ls)
        ds_zarr = read_in_s3(url)
        ds_zarr_prj = ds_zarr.rio.write_crs(ds_zarr.mapping.attrs["spatial_epsg"])
        ds_zarr_clip = ds_zarr_prj.rio.clip(vec_nc.geometry, vec_nc.crs)

        return ds_zarr_clip, url


class TestData:
    """This class holds the test data for both locations, Malaspina and Lowell
    Attributes:
        malaspina (SingleLocationData): test data for Malaspina
        lowell (SingleLocationData): test data for Lowell"""

    def __init__(self, test_data_dir):
        self.malaspina = SingleLocationData(test_data_dir, "Malaspina", "mchi_itslive.nc", "mchi.csv")
        self.lowell = SingleLocationData(
            test_data_dir, "Lowell", "ITS_LIVE_Lowell_Lower_test.nc", "Lowell_Lower_v_shared.csv"
        )
