import geopandas as gpd

from ivel.io_ivel import obj_setup


@pytest.fixture
def test_data():
    test_point_data = obj_setup.Glacier_Point("Lowell", "label", " RGI60-01.16545", [-138.645, 60.283])
    return test_point_data


def test_swap_time_dim(test_data):
    data = test_data.cube_around_point
    assert isinstance(data, xr.Dataset)
    result = obj_setup.swap_time_dim(data)
    assert isinstance(result, xr.Dataset)
    assert "img_separation" in result.coords, "object does not have temporal baseline coord"
