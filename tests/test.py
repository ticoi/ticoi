import os
import xarray as xr
from ticoi.cube_data_classxr import cube_data_class
name= f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_data"))}/ITS_LIVE_Lowell_Lower_test.nc'
print(name)

t = xr.open_dataset(name)
print(t)

cube = cube_data_class()
cube.load(name)
print(cube)