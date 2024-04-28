'''
Implementation of the Temporal Inversion using COmbination of displacements with Interpolation (TICOI) method
The implementation can be divided in three parts:
    - Data Download : find and download the cube corresponding for the area pixel we want to study,
    - Inversion & Interpolation: for each pixel of the cube: solving a system AX = Y to produce Irregular Leap Frog time series using the IRLS method, then interpolate the obtained ILF time series to Regular LF time series using interpolation
    - Results saving: save the result in a new netdf file

Author : Laurane Charrier
Reference:
    Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an optimal temporal sampling from displacement
    observation networks. IEEE Transactions on Geoscience and Remote Sensing.
    Charrier, L., Yan, Y., Colin Koeniguer, E., Mouginot, J., Millan, R., & Trouvé, E. (2022). Fusion of multi-temporal and multi-sensor ice velocity observations.
    ISPRS annals of the photogrammetry, remote sensing and spatial information sciences, 3, 311-318.
'''
from ticoi.core import *
from joblib import Parallel, delayed
import time
from ticoi.cube_data_classxr import cube_data_class
import os
import xarray as xr
from pyproj import CRS
from tqdm import tqdm
import itertools
import warnings

warnings.filterwarnings("ignore")

# %%
# Selection of data
cube_name = '/media/tristan/Data3/Hala_lake/Landsat8/Hala_lake_displacement_LS8.nc'  # Path where the Sentinel-2 IGE cubes are stored
path_save = f'/media/tristan/Data3/Hala_lake/Landsat8/ticoi_test/cube-with-flag-region-test/'  # Path where to stored the results
flag_file = '/media/tristan/Data3/Hala_lake/Landsat8/Hala_lake_displacement_LS8_flags.nc'  # Path where the flag file is stored

result_fn = 'Hala_lake_velocity_LS8_block_test'

save = True
merged = None  # Path to the second cube to merge with the first one
sensor = None

proj = 'EPSG:32647'  # EPSG system of the coordinates given

assign_flag = True
if assign_flag:
    flags = xr.open_dataset(flag_file)
    flags.load()

coef = { 0: 100, 1: 150, 2:200}  # lambda : coef of the regularisation
regu = {0: 1, 1: 2, 2: '1accelnotnull'}  # Type of regularisation : 1, 2'1accelnotnull'  # Type of regularisation : 1, 2,'1accelnotnull' : 1 is Tikhonov first order, 2 is Tikhonov second order and '1accelnotnull is Tikhonov first order with an apriori on the acceleration

load_kwargs = {'filepath': cube_name, 
               'chunks': {}, 
               'conf': False, 
               'subset': None, 
               'buffer': None, 
               'pick_date': ['2013-06-01', '2023-12-31'],
               'pick_sensor': None, 
               'pick_temp_bas': None, 
               'proj': proj, 
               'verbose': False}

preData_kwargs = {'smooth_method': 'gaussian', 
                  's_win': 3,
                  't_win': 90,
                  'sigma': 3,
                  'order': 3,
                  'unit': 365,
                  'delete_outliers': None,
                  'flags': flags,
                  'regu': regu,
                  'solver': 'LSMR_ini',
                  'proj': proj,
                  'unit': 365, 
                  'velo_or_disp': 'disp',
                  'verbose': True}

inversion_kwargs = {'solver': 'LSMR_ini',
                    'coef': coef,
                    'apriori_weight': True,
                    'path_save': path_save,
                    'interpolation_load_pixel': 'nearest',
                    'iteration': True,
                    'interval_output': 1,
                    'treshold_it': 0.1,
                    'conf': False,
                    'flags': flags,
                    'regu': regu,
                    'interpolation_bas': 90,
                    'option_interpol': 'spline',
                    'redundancy': 30,
                    'proj': proj,
                    'detect_temporal_decorrelation': True,
                    'unit': 365,
                    'result_quality': ['X_contribution'],
                    'nb_max_iteration': 10,
                    'delete_outliers': None,
                    'interpolation': True,
                    'linear_operator': None,
                    'visual': False,
                    'verbose': False
}


if not os.path.exists(path_save):
    os.mkdir(path_save)

# lazy load the original data
start_process = time.time()
cube = cube_data_class()

cube.load(cube_name, pick_date=load_kwargs['pick_date'], chunks=load_kwargs['chunks'], conf=load_kwargs['conf'], 
          pick_sensor=load_kwargs['pick_sensor'], pick_temp_bas=load_kwargs['pick_temp_bas'], proj=load_kwargs['proj'], verbose=load_kwargs['verbose'])
print(f'Time download cube {round((time.time() - start_process), 4)} sec')


cube_date1 = cube.date1_().tolist()
cube_date1.remove(np.min(cube_date1))
first_date_interpol = np.min(cube_date1)
last_date_interpol = np.max(cube.date2_())

inversion_kwargs.update({'first_date_interpol': first_date_interpol, 'last_date_interpol': last_date_interpol})

start = time.time()
result = process_blocks(cube, nb_cpu=40, block_size=0.5, preData_kwargs=preData_kwargs, inversion_kwargs=inversion_kwargs)


print(f'Time inversion {round((time.time() - start), 4)} sec')


# %% Initialisation of the cube to store the data
start = time.time()
if save:
    sensor_array = np.unique(cube.ds['sensor'])

    sensor_strings = [str(sensor) for sensor in sensor_array]
    sensor = ', '.join(sensor_strings)
    if merged is None:
        source = f'Temporal inversion on cube {cube.filename} using TICOI with a selection of the dates between: {load_kwargs["pick_date"]}, with a selection of the temporal baselines {load_kwargs["pick_temp_bas"]}'
    else:
        source = f'Temporal inversion on cube {cube.filename} & {cube2.filename} using TICOI with a selection of the dates: {load_kwargs["pick_date"]}, with a selection of the baseline {load_kwargs["pick_temp_bas"]}, and a temporal spacing every {inversion_kwargs["redundancy"]} days '
    if inversion_kwargs['apriori_weight']:
        source += ' and apriori weight'
    source += f'. The Tikhonov coef is: {inversion_kwargs["coef"]}.'
    if inversion_kwargs['interpolation']:
        source += f'The interpolation option is: {inversion_kwargs["option_interpol"]}.'
        if inversion_kwargs['interpolation_bas']:
            source += f'The interpolation baseline is: {inversion_kwargs["interpolation_bas"]} days.'
        source += f'The temporal spacing every {inversion_kwargs["redundancy"]} days.'



# %% save the res
cube.write_result_TICOI(result, source, sensor, filename=result_fn,
                        savepath=path_save, result_quality=inversion_kwargs['result_quality'], verbose=True)
# cube.write_result_TICO(result, source, sensor, filename=name_result,
#                         savepath=path_save, result_quality=result_quality, verbose=verbose)
print(f'Total process {round((time.time() - start_process) / 60, 2)} min')
print('Finished')
