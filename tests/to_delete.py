import os
import xarray as xr
from ticoi.cube_data_classxr import cube_data_class
from ticoi.inversion_functions import construction_a_lf,inversion_one_component
from ticoi.core import mu_regularisation
import numpy as np
# name= f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_data"))}/c_x18620_y08085_2016-2022_crop_GPS_Lower.nc'
name = '/home/charriel/Documents/Bettik/Yukon/STACK/Data_2021/c_x18620_y08085_2016-2022_crop_GPS_Lower.nc'
print(name)
#
cube = cube_data_class()
cube.load(name)
data, mean, dates_range = cube.load_pixel(1,2)
data, mean, dates_range = cube.load_pixel(-138.18069, 60.29076)

dates = np.array([['2013-03-14', '2013-03-30'], ['2013-03-14', '2013-03-30'], ['2013-03-14', '2013-04-15'],
                  ['2013-03-30', '2013-04-15'],
                  ['2013-03-30', '2013-04-15'], ['2013-03-14', '2013-08-13'], ['2013-03-14', '2013-10-16'],
                  ['2013-06-19', '2013-07-13'],
                  ['2013-03-14', '2013-10-24'], ['2013-03-14', '2013-11-01']]).astype('datetime64[D]')
dates_range = np.array(
    ['2013-03-14', '2013-03-30', '2013-04-15', '2013-06-19', '2013-07-13', '2013-08-13', '2013-10-16', '2013-10-24',
     '2013-11-01']).astype('datetime64[D]')
A=construction_a_lf(dates,dates_range)
mu = mu_regularisation('1accelnotnull', A, dates_range, ini=None)
ini = np.array([-6, -8., 100., -100., -7., 10., 5., -10.]).astype('float32')
t2 = inversion_one_component(A,dates_range,1,data[1][:10,:],solver='LSMR_ini',Weight=1,mu=mu,ini =ini)
# print(cube)