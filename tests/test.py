import os
import xarray as xr
from ticoi.cube_data_classxr import cube_data_class
import numpy as np
# name= f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_data"))}/c_x18620_y08085_2016-2022_crop_GPS_Lower.nc'
name = '/home/charriel/Documents/Bettik/Yukon/STACK/Data_2021/c_x18620_y08085_2016-2022_crop_GPS_Lower.nc'
print(name)
#
cube = cube_data_class()
cube.load(name)
data, mean, dates_range = cube.load_pixel(1,2)
data, mean, dates_range = cube.load_pixel(-138.18069, 60.29076)

# print(cube)