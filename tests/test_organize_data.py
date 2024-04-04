import geopandas as gpd
import organize_data
import pandas as pd
import pytest
import xarray as xr

# define kwargs to create single loc class objects
mala_test_kwargs = {
    "test_data_dir": "../ivel/test_data/",
    "location_name": "Malaspina",
    "nc_file": "mchi_itslive.nc",
    "csv_file": "mchi.csv",
}

lowell_test_kwargs = {
    "test_data_dir": "../ivel/test_data/",
    "location_name": "Lowell",
    "nc_file": "ITS_LIVE_Lowell_Lower_test.nc",
    "csv_file": "Lowell_Lower_v_shared.csv",
}


@pytest.fixture(params=[mala_test_kwargs, lowell_test_kwargs])
def test_kwargs(request):
    return request.param


class TestSingleLocationData:
    @pytest.fixture(autouse=True)
    def setup_method(self, test_kwargs):
        self.single_loc = organize_data.SingleLocationData(**test_kwargs)

    def test_read_in_gps_csv(self):
        self.single_loc.read_in_gps_csv()
        print(type(self.single_loc.gps_csv))
        assert isinstance(self.single_loc.gps_csv, pd.core.frame.DataFrame)

    def test_read_in_nc(self):
        self.single_loc.read_in_nc()
        assert isinstance(self.single_loc.its_nc, xr.Dataset)
        crs_nc = self.single_loc.its_nc.rio.crs
        crs_its = self.single_loc.its_zarr.rio.crs
        assert crs_nc == crs_its
        assert crs_nc == "EPSG:3413"

    def test_find_itslive_urls(self):
        # self.single_loc.find_itslive_urls()
        vec_nc = organize_data.make_bounds_poly(self.single_loc.its_nc)
        assert isinstance(vec_nc, gpd.GeoDataFrame)


@pytest.fixture
def single_loc(test_kwargs):
    test_data = organize_data.SingleLocationData(**test_kwargs)
    # test_data.setup_method(test_kwargs)
    return test_data


def test_get_bounds(single_loc):

    data = single_loc.its_nc
    xmin, xmax, ymin, ymax = organize_data.get_bounds(data)
    assert xmin < xmax
    assert ymin < ymax


def test_make_bounds_poly(single_loc):
    data = single_loc.its_nc

    result = organize_data.make_bounds_poly(data)

    assert isinstance(result, gpd.GeoDataFrame)
    assert result.crs == "EPSG:3413"
    assert result.geometry[0].geom_type == "Polygon"
