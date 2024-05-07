'''
Implementation of the Temporal Inversion using COmbination of displacements with Interpolation (TICOI) method for big data cube.
This implementation divides the data in smaller data cubes processed one after the other in order to avoid memory overconsumption and kernel crashing.
It can be divided in three parts:
    - Data Download : Download the data cube eventually considering a given subset or buffer to limit its size.
    - Inversion & Interpolation: For each pixel of the cube, solving a system AX = Y to produce Irregular Leap Frog time series using the IRLS method, 
    then interpolate the obtained ILF time series to Regular LF time series using interpolation.
    - Results saving: Save the result in a new netCDF file.
/!\ This implementation uses asyncio which requires its own event loop to run : if you launch this code from a raw terminal, there should be no
problem, but if you try to launch it from an IDE (PyCharm, VSCode, Spyder...), think of specifying to your IDE to launch it in a raw terminal
instead of the default console (which leads to a RuntimeError)

Author : Laurane Charrier
Reference:
    Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an optimal temporal sampling from displacement
    observation networks. IEEE Transactions on Geoscience and Remote Sensing.
    Charrier, L., Yan, Y., Colin Koeniguer, E., Mouginot, J., Millan, R., & Trouvé, E. (2022). Fusion of multi-temporal and multi-sensor ice velocity observations.
    ISPRS annals of the photogrammetry, remote sensing and spatial information sciences, 3, 311-318.
'''

import time
import os
import xarray as xr
import warnings
import geopandas

from ticoi.core import *
from ticoi.cube_data_classxr import cube_data_class


# %%========================================================================= #
#                                    PARAMETERS                               #
# =========================================================================%% #

warnings.filterwarnings("ignore")

## ------------------------------- Data selection -------------------------- ##
# cube_name = '/media/tristan/Data3/Hala_lake/Landsat8/Hala_lake_displacement_LS7.nc'  # Path where the Sentinel-2 IGE cubes are stored
# path_save = f'/media/tristan/Data3/Hala_lake/Landsat8/ticoi_test/cube-with-flag-region-test/'  # Path where to stored the results
# flag_file = '/media/tristan/Data3/Hala_lake/Landsat8/Hala_lake_displacement_LS7_flags.nc'  # Path where the flag file is stored

cube_names = ['nathan/Donnees/Cubes_de_donnees/cubes_Sentinel_2/c_x01225_y03675_all_filt-multi.nc']
path_save = 'nathan/Tests_MB/'
poly_path = None
poly_path = 'nathan/Tests_MB/Glaciers/Full_MB.shp'

result_fn = 'test'
# result_fn = 'Hala_lake_velocity_LS7_block_test_median_filt'
save = True
merged = None  # Path to the second cube to merge with the first one
sensor = None

proj = 'EPSG:32632'  # EPSG system of the given coordinates

assign_flag = False
flags = None
if assign_flag:
    flags = xr.open_dataset(flag_file)
    flags.load()

# coef = { 0: 100, 1: 150, 2:200}  # lambda : coef of the regularisation
# regu = {0: 1, 1: 2, 2: '1accelnotnull'}  # Type of regularisation : 1, 2'1accelnotnull'  # Type of regularisation : 1, 2,'1accelnotnull' : 1 is Tikhonov first order, 2 is Tikhonov second order and '1accelnotnull is Tikhonov first order with an apriori on the acceleration

regu = '1accelnotnull'
coef = 100

load_kwargs = {'filepath': None, 
               'chunks': {}, 
               'conf': False, 
               'subset': [333250, 334250, 5083100, 5084200],
               'buffer': None, 
               'pick_date': ['2015-01-01', '2023-01-01'],
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
                  'delete_outliers': 'vvc_angle',
                  'flags': flags,
                  'regu': regu,
                  'solver': 'LSMR_ini',
                  'proj': proj, 
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
                    'delete_outliers': 'vvc_angle',
                    'interpolation': True,
                    'linear_operator': None,
                    'visual': False,
                    'verbose': False
}

if not os.path.exists(path_save):
    os.mkdir(path_save)


# %%========================================================================= #
#                                 DATA DOWNLOAD                               #
# =========================================================================%% #

start = [time.time()]

cube = cube_data_class()
cube.load(cube_names[0], pick_date=load_kwargs['pick_date'], chunks=load_kwargs['chunks'], conf=load_kwargs['conf'], 
          pick_sensor=load_kwargs['pick_sensor'], pick_temp_bas=load_kwargs['pick_temp_bas'], proj=load_kwargs['proj'], 
          subset=load_kwargs['subset'], verbose=load_kwargs['verbose'])

# Several cubes have to be merged together
if len(cube_names) > 1:
    for n in range(1, len(cube_names)):
        cube2 = cube_data_class()
        cube2.load(cube_names[n], pick_date=load_kwargs['pick_date'], chunks=load_kwargs['chunks'], conf=load_kwargs['conf'], 
                   pick_sensor=load_kwargs['pick_sensor'], pick_temp_bas=load_kwargs['pick_temp_bas'], proj=load_kwargs['proj'], 
                   subset=load_kwargs['subset'], verbose=load_kwargs['verbose'])
        # Align the new cube to the main one (interpolate the coordinate and/or reproject it)
        cube2 = cube.align_cube(cube2, reproj_vel=False, reproj_coord=True, interp_method='nearest')
        cube.merge_cube(cube2) # Merge the new cube to the main one
    del cube2

cube_date1 = cube.date1_().tolist()
cube_date1.remove(np.min(cube_date1))
first_date_interpol = np.min(cube_date1)
last_date_interpol = np.max(cube.date2_())

inversion_kwargs.update({'first_date_interpol': first_date_interpol, 'last_date_interpol': last_date_interpol})

stop = [time.time()]
print(f'[ticoi_cube_demo_block_process] Cube of dimension (nz, nx, ny): ({cube.nz}, {cube.nx}, {cube.ny}) ')
print(f'[ticoi_cube_demo_block_process] Data loading took {round(stop[0] - start[0], 3)} s')


# %%========================================================================= #
#                                      TICOI                                  #
# =========================================================================%% #

start.append(time.time())

poly = None
if poly_path is not None: # Apply a shp mask on those points
    poly = geopandas.read_file(poly_path).to_crs(epsg=int(proj.split(':')[1])).geometry[0]

result = process_blocks_refine(cube, nb_cpu=12, block_size=0.5, mask=poly, preData_kwargs=preData_kwargs, inversion_kwargs=inversion_kwargs)

stop.append(time.time())
print(f'[ticoi_cube_demo_block_process] TICOI processing took {round(stop[1] - start[1], 0)} s')


# %%========================================================================= #
#                                INITIALISATION                               #
# =========================================================================%% #

start.append(time.time())
if save:
    sensor_array = np.unique(cube.ds['sensor'])

    sensor_strings = [str(sensor) for sensor in sensor_array]
    sensor = ', '.join(sensor_strings)
    if merged is None:
        source = f'Temporal inversion on cube {cube.filename} using TICOI with a selection of the dates between: {load_kwargs["pick_date"]}, with a selection of the temporal baselines {load_kwargs["pick_temp_bas"]}'
    else:
        source = f'Temporal inversion on cube {cube.filename} using TICOI with a selection of the dates: {load_kwargs["pick_date"]}, with a selection of the baseline {load_kwargs["pick_temp_bas"]}, and a temporal spacing every {inversion_kwargs["redundancy"]} days '
    if inversion_kwargs['apriori_weight']:
        source += ' and apriori weight'
    source += f'. The Tikhonov coef is: {inversion_kwargs["coef"]}.'
    if inversion_kwargs['interpolation']:
        source += f'The interpolation option is: {inversion_kwargs["option_interpol"]}.'
        if inversion_kwargs['interpolation_bas']:
            source += f'The interpolation baseline is: {inversion_kwargs["interpolation_bas"]} days.'
        source += f'The temporal spacing every {inversion_kwargs["redundancy"]} days.'

stop.append(time.time())    
print(f'[ticoi_cube_demo_block_process] Initialisation took {round(stop[2] - start[2], 3)} s')


# %%========================================================================= #
#                                WRITING RESULTS                              #
# =========================================================================%% #

start.append(time.time())
cube.write_result_ticoi(result, source, sensor, filename=result_fn, savepath=path_save, result_quality=inversion_kwargs['result_quality'], 
                        verbose=inversion_kwargs['verbose'])

stop.append(time.time())
print(f'[ticoi_cube_demo_block_process] Writing cube to netCDF file took {round(stop[3] - start[3], 3)} s')
print(f'[ticoi_cube_demo_block_process] Overall processing took {round(stop[3] - start[0], 0)} s')
