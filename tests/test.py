import os
import xarray as xr
from ticoi.cube_data_classxr import cube_data_class
# name= f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_data"))}/c_x18620_y08085_2016-2022_crop_GPS_Lower.nc'
name = '/home/charriel/Documents/Bettik/Yukon/STACK/Data_2021/c_x18620_y08085_2016-2022_crop_GPS_Lower.nc'
print(name)

t = xr.open_dataset(name,engine='netcdf4')
print(t)
#
# cube = cube_data_class()
# cube.load(name)
# print(cube)