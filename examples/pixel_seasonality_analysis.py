'''
Implementation of the Temporal Inversion using COmbination of displacements with Interpolation (TICOI) method
for one pixel. An additionnal seasonality analysis is implemented, the idea is to match a sine to TICOI results
(fixed frequency or not).
Author: Laurane Charrier, Nathan Lioret
Reference:
    Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an optimal temporal sampling from displacement
    observation networks. IEEE Transactions on Geoscience and Remote Sensing.
    Charrier, L., Yan, Y., Colin Koeniguer, E., Mouginot, J., Millan, R., & Trouvé, E. (2022). Fusion of multi-temporal and multi-sensor ice velocity observations.
    ISPRS annals of the photogrammetry, remote sensing and spatial information sciences, 3, 311-318.
'''

import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.fft as fft
import scipy.signal as signal

from scipy.optimize import curve_fit
from sklearn.metrics import root_mean_squared_error

from ticoi.core import inversion_core, visualisation, interpolation_core
from ticoi.cube_data_classxr import cube_data_class


# %%========================================================================= #
#                                    PARAMETERS                               #
# =========================================================================%% #

####  Selection of data
# Paths to the data cubes
# cube_names = [f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_data"))}/ITS_LIVE_Lowell_Lower_test.nc']
# path_save = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples", "results"))}'  # Path where to store the results
cube_names = ['nathan/Donnees/Cubes_de_donnees/cubes_Sentinel_2/c_x01470_y03675_all_filt-multi.nc'] # Sentinel-2 cube
              # 'nathan/Donnees/Cubes_de_donnees/stack_median_pleiades_alllayers_2012-2022_modiflaurane.nc'] # Pleiade cube
path_save = 'nathan/Tests_MB/useless/'
i, j = 330639.4,5077007.4 # Point (pixel) where to carry on the computation
i, j = 340672.9,5085365.2
# i, j = 339989.7,5084463.3
proj = 'EPSG:32632' # Projection of the given coordinates
buffer_size = 500 # Size of the buffer to be loaded around the pixel
# To select a specific period for the measurements, if you want to select all the dates put None, 
# else give an inteval of dates ['aaaa-mm-dd', 'aaaa-mm-dd'] ([min, max])
dates_input = ['2016-01-01', '2023-01-01']
# To select certain temporal baselines in the dataset, if you want to select all the temporal baselines put None, 
# else give two integers [min, max] to form an interval in days
temp_baseline = None
sensor = None # Select a specific sensor 
conf = False # If you want confidence indicators ranging between 0 and 1, with 1 the lowest errors
unit = 'm/y' # Unit in m/y or m/d
# If None, all the data are included ; if an integer, the data with an error higher than this integer are removed ;
# if 'median_average', the data with a direction 45° away compared to the averaged direction are removed
delete_outliers = None
load_interp = 'nearest' # Interpolation used to select which data to use when the pixel is not exactly a pixel of the dataset

####  Inversion
# Type of regularisation : 1, 2,'1accelnotnull','regu01' (1: Tikhonov first order, 2: Tikhonov second order,
# '1accelnotnull': minimisation of the difference between the acceleration of the time series and acceleration computed on a moving average
regu = '1accelnotnull'
coef = 500  # lambda : coef of the regularisation
apriori_weight = True  # Add a weight in the first step of the inversion, True ou False
solver = 'LSMR_ini'  # Solver for the inversion : 'LSMR', 'LSMR_ini', 'LS', 'LS_bounded', 'LSQR'
detect_temporal_decorrelation = True  # Detect temporal decorrelation by setting a weight of 0 at the beginning at the first inversion to all observation with a temporal baseline larger than 200

####  Interpolation
option_interpol = 'spline'  # Type of interpolation : 'spline', 'nearest' or 'spline_smooth' for smoothing spline
interpolation_bas = 30  # Temporal sampling of the velocity time series
redundancy = 5
result_quality = None

####  Visualization
visual = False  # Plot some results or not
verbose = False  # Print informations during the process or not
save = True  # Save the results or not
vmax = [50, 170]  # vmin and vmax of the legend
visual_inversion = False  # Visualize the different iterations of the inversion
# Visualisation options
option_visual = ['original_velocity_xy', 'original_magnitude',
                  'X_magnitude_zoom', 'X_magnitude', 'X_zoom', 'X',
                  'vv_quality', 'vxvy_quality',
                  'Residu_magnitude', 'Residu',
                  'X_z', 'Y_contribution',
                  'direction']

# Create a subfolder if it doesnt exist
if not os.path.exists(path_save):
    os.mkdir(path_save)

unit = 365 if unit == 'm/y' else 1


# %% ======================================================================== #
#                                DATA DOWNLOAD                                #
# =========================================================================%% #

start = [time.time()]

cube = cube_data_class()
cube.load(cube_names[0], pick_date=dates_input, proj=proj, pick_temp_bas=temp_baseline, 
          buffer=[i, j, buffer_size], conf=conf, pick_sensor=sensor, chunks={})

# Several cubes have to be merged together
if len(cube_names) > 1:
    for n in range(1, len(cube_names)):
        cube2 = cube_data_class()
        cube2.load(cube_names[n], pick_date=dates_input, proj=proj, pick_temp_bas=temp_baseline, 
                   buffer=[i, j, buffer_size], conf=conf, pick_sensor=sensor, chunks={})
        cube2 = cube.align_cube(cube2, reproj_vel=False, reproj_coord=True, interp_method='nearest')
        cube.merge_cube(cube2)

stop = [time.time()]
print(f'[ticoi_pixel_demo] Loading the data cube.s took {round((stop[0] - start[0]), 4)} s')
print(f'[ticoi_pixel_demo] Cube of dimension (nz,nx,ny) : ({cube.nz}, {cube.nx}, {cube.ny}) ')

start.append(time.time())

obs_filt = cube.filter_cube(s_win=9, t_win=90, unit=unit, proj=proj, regu=regu, delete_outliers=delete_outliers, 
                             velo_or_disp='velo', verbose=verbose)
data, mean, dates_range = cube.load_pixel(i, j, proj=proj, interp=load_interp, solver=solver, regu=regu, 
                                          rolling_mean=obs_filt, visual=visual, verbose=verbose)

cube2_date1 = cube.date1_().tolist()
start_date_interpol = np.min(cube2_date1)
last_date_interpol = np.max(cube.date2_())

date1 = None

stop.append(time.time())
print(f'[ticoi_pixel_demo] Loading the pixel took {round((stop[1] - start[1]), 4)} s')


# %% ======================================================================== #
#                                 INVERSION                                   #
# =========================================================================%% #

start.append(time.time())
A, result, dataf = inversion_core(data, i, j, dates_range=dates_range, solver=solver, coef=coef, weight=apriori_weight,
                                  unit=unit, conf=conf, regu=regu, mean=mean,
                                  detect_temporal_decorrelation=detect_temporal_decorrelation,
                                  linear_operator=None, result_quality=result_quality, 
                                  visual=visual, verbose=verbose)

stop.append(time.time())
print(f'[ticoi_pixel_demo] Inversion took {round((stop[2] - start[2]), 4)} s')

if visual: visualisation(dataf, result, option_visual, path_save, A=A, dataf=dataf, unit=unit, show=True, figsize=(12, 6))
if save: result.to_csv(f'{path_save}/ILF_result.csv')


# %% ======================================================================== #
#                              INTERPOLATION                                  #
# =========================================================================%% #

start.append(time.time())

if interpolation_bas == False: interpolation_bas = 1
start_date_interpol = np.min(np.min(cube.date2_()))
last_date_interpol = np.max(np.max(cube.date2_()))
dataf_lp = interpolation_core(result, interpolation_bas,
                              path_save, option_interpol=option_interpol,
                              first_date_interpol=start_date_interpol, last_date_interpol=last_date_interpol,
                              visual=visual, data=dataf, unit=unit, redundancy=redundancy,
                              result_quality=result_quality,
                              verbose=verbose, vmax=vmax)

stop.append(time.time())
print(f'[ticoi_pixel_demo] Interpolation took {round((stop[3] - start[3]), 4)} s')

if save: dataf_lp.to_csv(f'{path_save}/RLF_result.csv')


# %% ======================================================================== #
#                             FOURIER ANALYSIS                                #
# =========================================================================%% #
# Compute the best periodicity of the signal (frequency with the highest TF)
# If 1/best_freq is around 365 days, there might be an annual periodicity (the
# significance is evaluated in the next section MATCH SINE CURVE)

# Is the periodicity frequency imposed to 1/365.25 (one year seasonality) ?
impose_frequency = True
# Filter to use in the first place
# 'highpass' : apply a bandpass filter between low frequencies (reject variations over several years (> 1.5 y))
# and the Nyquist frequency to ensure Shanon theorem
# 'lowpass' : or apply a lowpass filter only (to Nyquist frequency) : risk of tackling an interannual trend (long period)
filt = 'highpass'
# Method used to compute local variations
# 'rolling_7d' : median of the std of the data centered in +- 3 days around each central date
# 'uniform_7d' : median of the std of the data centered in +- 3 days around dates constantly distributed every redundnacy 
# days -- BEST
# 'uniform_all' : median of the std of each data covering the dates, which are constantly distributed every redundancy days
# 'residu' : standard deviation of the data previously substracted by TICOI results (ground truth) = standard deviation of the "noise"
local_var_method = 'uniform_7d'

start.append(time.time())

if not os.path.exists(f'{path_save}Fourier/'):
    os.mkdir(f'{path_save}Fourier/')
    
dataf_lp = dataf_lp.dropna()

dates_c = dataf_lp['date1'] + (dataf_lp['Second_date'] - dataf_lp['date1']) // 2
dates = (dates_c - dataf_lp['date1'].min()).dt.days.to_numpy()
vv = np.sqrt(dataf_lp['vx']**2 + dataf_lp['vy']**2).to_numpy()
vv_c = vv - np.mean(vv)

N = len(dates)
Ts = dates[1] - dates[0]
print(f'[ticoi_pixel_demo] Sampling period after interpolation : {Ts} days')

if filt == 'highpass':
    b, a = signal.butter(4, [1/(1.5*365), 1/(2.001*Ts)], 'bandpass', fs=1/Ts, output='ba')
    vv_filt = signal.filtfilt(b, a, vv_c)
elif filt == 'lowpass':
    sos = signal.butter(4, 1/(2.001*Ts), 'lowpass', fs=1/Ts, output='sos')
    vv_filt = signal.sosfilt(sos, vv_c)
else:
    print('[ticoi_pixel_demo] filt must be "bandpass" or "lowpass"')

if impose_frequency:
    fig, axe = plt.subplots(figsize=(12, 6))
else:
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 6))
    axe = ax[0]
axe.plot(dates_c, vv_c, 'blue', label='Before filtering')
axe.plot(dates_c, vv_filt, 'red', label='After filtering')
axe.set_xlabel('Centered velocity [m/y]', fontsize=16)
axe.set_ylabel('Central date', fontsize=16)
axe.set_title('Effect of filtering', fontsize=16)
axe.legend(loc='lower left')

def right_phi(phi):
    if phi >= 0 and phi < np.pi/2: return np.pi/2 - phi
    elif phi >= np.pi/2 and phi < 3*np.pi/2: return 5*np.pi/2 - phi
    return 5*np.pi/2 - phi

if impose_frequency:
    def sine_fconst(t, A, phi, off):
        f = 1/365.25 # One year
        return A * np.sin(2*np.pi*f*t + phi) + off
    
    guess = [np.max(vv_filt) - np.min(vv_filt), 0, 0]
    popt, pcov = curve_fit(sine_fconst, dates, vv_filt, p0=guess)
    A, phi, off = popt
    f = 1/365.25
    sine = sine_fconst(dates, A, phi, off)
    
    if phi < 0: phi += 2*np.pi

    print(f'[ticoi_pixel_demo] Amplitude of the best matching sinus: {round(abs(popt[0]), 1)} m/y')
    first_max_day  = pd.Timedelta(int((right_phi(phi) + (np.pi if A < 0 else 0)) / (2*np.pi*f)), 'D') +  dataf_lp['date1'].min()
    max_day = (first_max_day - pd.Timestamp(year=first_max_day.year, month=1, day=1))
    print(f'[ticoi_pixel_demo] Maximum at day {max_day.days}')
    print(f'[ticoi_pixel_demo] RMSE : {round(root_mean_squared_error(sine, vv_filt))} m/y')
    
else:
    # Apply a Hanning window
    window = signal.windows.hann(N)
    ax[1].plot(dates_c, vv_filt * window, 'blue', label='With Hanning windowing')
    ax[1].plot(dates_c, vv_filt, 'black', label='Without windowing')
    ax[1].set_xlabel('Centered velocity [m/y]', fontsize=16)
    ax[1].set_ylabel('Central date', fontsize=16)
    ax[1].set_title('Effect of Hanning windowing', fontsize=16)
    ax[1].legend(loc='best')
    
    fig.tight_layout()
    fig.savefig(f'{path_save}Fourier/Windowing_Filtering.png')
    
    # TFD
    n = 64*N
    vv_tf = fft.rfft(vv_filt, n=n)
    vv_win_tf = fft.rfft(vv_filt * window, n=n)
    freq = fft.rfftfreq(n, d=Ts)
    
    plt.figure(figsize=(12,6))
    plt.plot(freq, 2/N*np.abs(vv_tf), 'blue', label='TF without windowing')
    plt.plot(freq, 2/N*np.abs(vv_win_tf), 'red', label='TF after Hanning windowing')
    plt.vlines([i/365 for i in range(1, 4)], 
               0, 1.1*2/N*max(np.max(np.abs(vv_tf)), np.max(np.abs(vv_win_tf))), 
               color='black', label='365d periodicity')
    plt.xlim([0, 0.01])
    plt.ylim([0, 1.1*2/N*max(np.max(np.abs(vv_tf)), np.max(np.abs(vv_win_tf)))])
    plt.xlabel('Frequency [day-1]', fontsize=16)
    plt.ylabel('Amplitude [m/y]', fontsize=16)
    plt.legend(loc='best')
    plt.title(f'Fourier Transform of the TICOI-resulting velocities at point ({i}, {j})', fontsize=16)
    plt.savefig(f'{path_save}Fourier/TF.png')
    
    # Best matching sinus
    def sine(t, A, f, phi, off):
        return A * np.sin(2*np.pi*f*t + phi) + off
    
    # Initial guess from the TF
    guess = np.array([np.max(2/N*np.abs(vv_win_tf)), freq[np.argmax(np.abs(vv_win_tf))], 
                      np.angle(vv_win_tf)[np.argmax(np.abs(vv_win_tf))], np.mean(vv_win_tf)], dtype='float')
    
    popt, pcov = curve_fit(sine, dates, vv_filt, p0=guess)
    A, f, phi, off = popt
    sine = sine(dates, A, f, phi, off)    

    if phi < 0: phi += 2*np.pi
    
    print(f'[ticoi_pixel_demo] Period of the best matching sinus : {round(1/f, 0)} days')
    print(f'[ticoi_pixel_demo] Amplitude : {round(abs(A), 1)} m/y')
    first_max_day  = pd.Timedelta(int((right_phi(phi) + (np.pi if A < 0 else 0)) / (2*np.pi*f)), 'D') +  dataf_lp['date1'].min()
    max_day = (first_max_day - pd.Timestamp(year=first_max_day.year, month=1, day=1))
    print(f'[ticoi_pixel_demo] Maximum at day {max_day.days}')
    print(f'[ticoi_pixel_demo] RMSE : {round(root_mean_squared_error(sine, vv_filt))} m/y')

dataff = pd.DataFrame(data={'date1': data[0][:, 0], 'date2': data[0][:, 1],
                            'vx': data[1][:, 0], 'vy': data[1][:, 1],
                            'errorx': data[1][:, 2], 'errory': data[1][:, 3],
                            'temporal_baseline': data[1][:, 4]})
dataff['vx'] = dataff['vx'] * unit / dataff['temporal_baseline']
dataff['vy'] = dataff['vy'] * unit / dataff['temporal_baseline']
dataff['vv'] = np.sqrt(dataff['vx'] ** 2 + dataff['vy'] ** 2)
dataff.index = dataff['date1'] + (dataff['date2'] - dataff['date1']) // 2

plt.figure(figsize=(12, 6))
plt.plot(dataff.index, dataff['vv'], linestyle='', marker='x', markersize=2, color='orange', label='Raw data')
plt.plot(dates_c, vv, 'black', alpha=0.5, label='TICOI velocities')
plt.plot(dates_c, vv_filt + np.mean(vv), 'red', alpha=0.7, label='Filtered TICOI velocities')
plt.plot(dates_c, sine + np.mean(vv), 'cyan', linewidth=3, label='Best matching sinus')
plt.vlines(pd.date_range(start=first_max_day, end=dataf_lp['Second_date'].max(), freq=f'{int(1/f)}D'), 
                         np.min(vv), np.max(vv), 'black', label='Maximum')
plt.xlabel('Central dates', fontsize=16)
plt.ylabel('Velocity', fontsize=16)
plt.legend(loc='best')
plt.title('Best matching sinus around an annual seasonality')
plt.savefig(f'{path_save}Fourier/matching_sine.png')

# Compute local variations
if local_var_method == 'rolling_7d':
    var = dataff['vv'].rolling(window='7D', center=True).std(ddof=0).drop_duplicates().dropna().median().item()
    
elif local_var_method.split('_')[0] == 'uniform':
    period_between_dates = np.diff(np.sort(np.concatenate([dataff['date1'], dataff['date2']]))).astype('timedelta64[D]').astype('int')
    min_period = np.min(period_between_dates[period_between_dates > 0])
    var_dates = pd.date_range(start=dataff['date1'].min(), end=dataff['date2'].max(), freq=f'{min_period}D')
    local_var = pd.Series(index=var_dates)
    
    if local_var_method == 'uniform_7d':
        for date in var_dates:
            local_var[date] = dataff.loc[(dataff.index > date - pd.Timedelta('3D')) & (dataff.index < date + pd.Timedelta('3D')), 'vv'].std(ddof=0)
    elif local_var_method == 'uniform_all':
        for date in var_dates:
            local_var[date] = dataff.loc[(dataff['date1'] < date) & (dataff['date2'] > date), 'vv'].std(ddof=0)
            
    var = local_var[local_var > 0].dropna().median()  
    
elif local_var_method == 'residu':
    dataf_lp.index = dataf_lp['date1'] + (dataf_lp['Second_date'] - dataf_lp['date1']) // 2
    dataf_lp['vv'] = np.sqrt(dataf_lp['vx'] ** 2 + dataf_lp['vy'] ** 2)
    dataf_lp = dataf_lp.reindex(index=np.unique(dataff.index)).interpolate().dropna()
    dataff = dataff[dataff.index >= dataf_lp.index[0]]
    dataff_vv_c = dataff['vv'] - dataf_lp['vv']
    var = dataff_vv_c.std(ddof=0)

    plt.figure(figsize=(12, 6))
    plt.plot(dataff.index, dataff['vv'], linestyle='', marker='x', markersize=2, color='orange')
    plt.plot(dataff.index, dataff_vv_c + dataf_lp['vv'].mean(), linestyle='', marker='x', markersize=2, color='red')
    plt.plot(dataf_lp.index, dataf_lp['vv'], linestyle='', marker='x', markersize=2, color='blue')
    plt.hlines([np.mean(vv) + var, np.mean(vv) - var, np.mean(vv)], np.min(dataff.index), np.max(dataff.index), color='black')
    plt.savefig(f'{path_save}Fourier/residu.png')

# Amplitude to median local variations factor
AtoVar = max(0, 1 - var / abs(popt[0]))

print(f'[ticoi_pixel_demo] Local variations : {round(var, 2)} m/y')
print(f'[ticoi_pixel_demo] Amplitude to local variations factor : {round(AtoVar, 2)}')

plt.show()

stop.append(time.time())
print(f'[ticoi_pixel_demo] Fourier analysis took {round((stop[4] - start[4]), 4)} s')