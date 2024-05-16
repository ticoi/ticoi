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
import numpy as np

from osgeo import gdal, osr
from rasterio.features import rasterize

from ticoi.core import process_blocks_refine
from ticoi.cube_data_classxr import cube_data_class


# %%========================================================================= #
#                                    PARAMETERS                               #
# =========================================================================%% #

warnings.filterwarnings("ignore")

## ------------------------------- Data selection -------------------------- ##
# cube_name = '/media/tristan/Data3/Hala_lake/Landsat8/Hala_lake_displacement_LS7.nc'  # Path where the Sentinel-2 IGE cubes are stored
# path_save = f'/media/tristan/Data3/Hala_lake/Landsat8/ticoi_test/cube-with-flag-region-test/'  # Path where to stored the results
flag_file = '/media/tristan/Data3/Hala_lake/Landsat8/Hala_lake_displacement_LS7_flags.nc'  # Path where the flag file is stored

cube_names = ['nathan/Donnees/Cubes_de_donnees/cubes_Sentinel_2/c_x01225_y03920_all_filt-multi.nc',
              'nathan/Donnees/Cubes_de_donnees/stack_median_pleiades_alllayers_2012-2022_modiflaurane.nc']
path_save = 'nathan/Tests_MB/'
path_mask = None
path_mask = 'nathan/Tests_MB/Areas/Full_MB/mask/Full_MB.shp'

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
               'subset': None,
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
                  'mask': path_mask,
                  'proj': proj, 
                  'velo_or_disp': 'velo',
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
                    'mask': path_mask,
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
                    'verbose': False}

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
        subset = load_kwargs['subset']
        res = cube.ds['x'].values[1] - cube.ds['x'].values[0]
        cube2.load(cube_names[n], pick_date=load_kwargs['pick_date'], chunks=load_kwargs['chunks'], 
                   conf=load_kwargs['conf'], pick_sensor=load_kwargs['pick_sensor'], pick_temp_bas=load_kwargs['pick_temp_bas'], 
                   proj=load_kwargs['proj'], verbose=load_kwargs['verbose'],
                   subset=[subset[0]-res, subset[1]+res, subset[2]-res, subset[3]+res] if subset is not None else None)
        # Align the new cube to the main one (interpolate the coordinate and/or reproject it)
        cube2 = cube.align_cube(cube2, reproj_vel=False, reproj_coord=True, interp_method='nearest')
        cube.merge_cube(cube2) # Merge the new cube to the main one
    del cube2

cube_date1 = cube.date1_().tolist()
cube_date1.remove(np.min(cube_date1))
first_date_interpol = np.min(cube_date1)
last_date_interpol = np.max(cube.date2_())

inversion_kwargs.update({'first_date_interpol': first_date_interpol, 'last_date_interpol': last_date_interpol})
 
if path_mask is not None:  
    if path_mask[-3:] == 'shp': # Convert the shp file to an xarray dataset (rasterize the shapefile) 
        polygon = geopandas.read_file(path_mask).to_crs(epsg=int(proj.split(':')[1]))
        raster = rasterize([polygon.geometry[0]], out_shape=cube.ds.rio.shape, transform=cube.ds.rio.transform(), fill=0, dtype='int16')
        mask = xr.DataArray(data=raster.T, dims=['x', 'y'], coords=cube.ds[['x', 'y']].coords)
    else:
        mask = xr.open_dataarray(path_mask)
    mask.load()
    preData_kwargs['mask'] = mask
    inversion_kwargs['mask'] = mask

stop = [time.time()]
print(f'[ticoi_cube_demo_block_process] Cube of dimension (nz, nx, ny): ({cube.nz}, {cube.nx}, {cube.ny}) ')
print(f'[ticoi_cube_demo_block_process] Data loading took {round(stop[0] - start[0], 3)} s')


# %%========================================================================= #
#                                      TICOI                                  #
# =========================================================================%% #

start.append(time.time())

result = process_blocks_refine(cube, nb_cpu=12, block_size=0.5, preData_kwargs=preData_kwargs, inversion_kwargs=inversion_kwargs)

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
cubenew = cube.write_result_ticoi(result, source, sensor, filename=result_fn, savepath=path_save, result_quality=inversion_kwargs['result_quality'], 
                        verbose=inversion_kwargs['verbose'])

# Plot the mean velocity as an example
mean_vv = np.sqrt(cubenew.ds['vx'].mean(dim='mid_date') ** 2 + cubenew.ds['vy'].mean(dim='mid_date') ** 2).to_numpy()
mean_vv = mean_vv.astype(np.float32)
mean_vv = np.flip(mean_vv, axis=0)

driver = gdal.GetDriverByName('GTiff')
srs = osr.SpatialReference()
srs.SetWellKnownGeogCS('EPSG:32632')

resolution = int(cube.ds['x'].values[1] - cube.ds['x'].values[0])
dst_ds_temp = driver.Create(f'{path_save}mean_velocity.tiff', mean_vv.shape[1], mean_vv.shape[0], 1, gdal.GDT_Float32)
dst_ds_temp.SetGeoTransform([np.min(cube.ds['x'].values), resolution, 0, np.min(cube.ds['y'].values), 0, resolution])
dst_ds_temp.GetRasterBand(1).WriteArray(mean_vv)
dst_ds_temp.SetProjection(srs.ExportToWkt())

dst_ds_temp = None
driver = None

stop.append(time.time())
print(f'[ticoi_cube_demo_block_process] Writing cube to netCDF file took {round(stop[3] - start[3], 3)} s')
print(f'[ticoi_cube_demo_block_process] Overall processing took {round(stop[3] - start[0], 0)} s')
