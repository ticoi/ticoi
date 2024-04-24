import pytest
from ticoi.secondary_functions import Construction_dates_range_np
import numpy as np
# define kwargs to create single loc class objects
mala_test_kwargs = {
    "test_data_dir": "../ticoi/test_data/",
    "location_name": "Malaspina",
    "nc_file": "mchi_itslive.nc",
    "csv_file": "mchi.csv",
}

lowell_test_kwargs = {
    "test_data_dir": "../ticoi/test_data/",
    "location_name": "Lowell",
    "nc_file": "ITS_LIVE_Lowell_Lower_test.nc",
    "csv_file": "Lowell_Lower_v_shared.csv",
}


@pytest.fixture(params=[mala_test_kwargs, lowell_test_kwargs])
def test_kwargs(request):
    return request.param

def test_Construct_Dates_range():
    dates = np.array([['2013-03-14', '2013-03-30'], ['2013-03-14', '2013-03-30'], ['2013-03-14', '2013-04-15'], ['2013-03-30', '2013-04-15'],
     ['2013-03-30', '2013-04-15'], ['2013-03-14', '2013-08-13'], ['2013-03-14', '2013-10-16'], ['2013-06-19', '2013-07-13'],
     ['2013-03-14', '2013-10-24'], ['2013-03-14', '2013-11-01']]).astype('datetime64[D]')
    assert (Construction_dates_range_np(dates) == np.array(['2013-03-14', '2013-03-30', '2013-04-15', '2013-06-19', '2013-07-13', '2013-08-13', '2013-10-16', '2013-10-24', '2013-11-01']).astype('datetime64[D]')).all()

# def test_load():

# def test_load()
# print(test_kwargs())