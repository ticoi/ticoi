#!/usr/bin/env python3

"""
Implementation of the Temporal Inversion using COmbination of displacements with Interpolation (TICOI) method to compute entire data cubes.
An additional seasonality analysis is implemented, by matching a sinus to TICOI results for each pixel of the considered cube/subset,
thus generating maps with the amplitude of the best matching sinus, the position of its first maximum and an index comparing its amplitude
to the local variations of the raw data.s

Author : Laurane Charrier, Lei Guo, Nathan Lioret
Reference:
    Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an optimal temporal sampling from displacement
    observation networks. IEEE Transactions on Geoscience and Remote Sensing.
    Charrier, L., Yan, Y., Colin Koeniguer, E., Mouginot, J., Millan, R., & Trouvé, E. (2022). Fusion of multi-temporal and multi-sensor ice velocity observations.
    ISPRS annals of the photogrammetry, remote sensing and spatial information sciences, 3, 311-318.
"""

import itertools
import os
import time
import warnings

import numpy as np
import pandas as pd
import scipy.fft as fft
import scipy.signal as signal
import xarray as xr
from joblib import Parallel, delayed
from osgeo import gdal, osr
from scipy.optimize import curve_fit
from tqdm import tqdm

from ticoi.core import process, process_blocks_refine
from ticoi.cube_data_classxr import cube_data_class
from ticoi.interpolation_functions import prepare_interpolation_date

# %%========================================================================= #
#                                   PARAMETERS                                #
# =========================================================================%% #

warnings.filterwarnings("ignore")

## ------------------- Choose TICOI cube processing method ----------------- ##
# Choose the TICOI cube processing method you want to use :
#    - 'block_process' (recommended) : This implementation divides the data in smaller data cubes processed one after the other in a synchronous manner,
# in order to avoid memory overconsumption and kernel crashing. Computations within the blocks are parallelized so this method goes way faster
# than every other TICOI processing methods.
#      /!\ This implementation uses asyncio (way faster) which requires its own event loop to run : if you launch this code from a raw terminal,
# there should be no problem, but if you try to launch it from some IDE (like Spyder), think of specifying to your IDE to launch it
# in a raw terminal instead of the default console (which leads to a RuntimeError)
#    - 'direct_process' : No subdivisition of the data is made beforehand which generally leads to memory overconsumption and kernel crashes
# if the amount of pixel to compute is too high (depending on your available memory). If you want to process big amount of data, you should use
# 'block_process', which is also faster. This method is essentially used for debug purposes.
#   - 'load' : The  TICOI cube was already calculated before, load it by giving the cubes to be loaded in a dictionary like {name: path} (at least
# 'raw' and 'interp' must be given)

TICOI_process = "load"

save = True  # If True, save TICOI results to a netCDF file

## ------------------------------ Data selection --------------------------- ##
# Path.s to the data cube.s (can be a list of str to merge several cubes, or a single str,
# cube_name = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "test_data"))}/Alps_Mont-Blanc_Argentiere_S2.nc'
# If TICOI_process is 'load', must be a dictionary like {name: path} to load existing cubes and name them (path can be a list of str or a single str)
cube_name = {
    "raw": f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "test_data"))}/Alps_Mont-Blanc_Argentiere_S2.nc',
    "interp": f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "cube"))}/Argentiere_example_interp.nc',
}
flag_file = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "test_data"))}/Alps_Mont-Blanc_flags.nc'  # Path to flags file
mask_file = None  # Path to mask file (.shp file) to mask some of the data on cube
# path_save = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "cube", "seasonality"))}/'  # Path where to store the results
path_save = f'{os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "cube", "seasonality"))}/'  # Path where to store the results
result_fn = "c_x01470_y03675_x_0_y_1"  # Name of the netCDF file to be created (if save is True)

proj = "EPSG:32632"  # EPSG system of the given coordinates

# Divide the data in several areas where different methods should be used
assign_flag = False
if not assign_flag:
    flag_file = None

# Regularization method.s to be used (for each flag if flag is not None)
# regu = {0: 1, 1: "1accelnotnull"}  # With flag (0: stable ground, 1: glaciers)
regu = "1accelnotnull"
# regu = '1accelnotnull' # Without flag
# Regularization coefficient.s to be used (for each flag if flag is not None)
# coef = {0: 500, 1: 200}  # With flag (0: stable ground, 1: glaciers)
coef = 200
# coef = 200 # Without flag
solver = "LSMR_ini"  # Solver for the inversion
delete_outlier = {'mz_score':3.5,'median_angle':45}
## ---------------------------- Loading parameters ------------------------- ##
load_kwargs = {
    "chunks": {},
    "conf": False,  # If True, confidence indicators will be put between 0 and 1, with 1 the lowest errors
    "subset": None,  # Subset of the data to be loaded ([xmin, xmax, ymin, ymax] or None)
    "buffer": None,  # Area to be loaded around the pixel ([longitude, latitude, buffer size] or None)
    "pick_date": ["2015-01-01", "2023-01-01"],  # Select dates ([min, max] or None to select all)
    "pick_sensor": None,  # Select sensors (None to select all)
    "pick_temp_bas": None,  # Select temporal baselines ([min, max] in days or None to select all)
    "proj": proj,  # EPSG system of the given coordinates
    "mask": mask_file,  # Path to mask file (.shp file) to mask some of the data on cube
    "verbose": False,  # Print information throughout the loading process
}

## ----------------------- Data preparation parameters --------------------- ##
preData_kwargs = {
    "smooth_method": "gaussian",  # Smoothing method to be used to smooth the data in time ('gaussian', 'median', 'emwa', 'savgol')
    "s_win": 3,  # Size of the spatial window
    "t_win": 90,  # Time window size for 'ewma' smoothing
    "sigma": 3,  # Standard deviation for 'gaussian' filter
    "order": 3,  # Order of the smoothing function
    "unit": 365,  # 365 if the unit is m/y, 1 if the unit is m/d
    "delete_outliers": "vvc_angle",  # Delete data with a poor quality indicator (if int), or with aberrant direction ('vvc_angle')
    "flag": flag_file,  # Divide the data in several areas where different methods should be used
    "regu": regu,  # Regularization method.s to be used (for each flag if flag is not None)
    "solver": solver,  # Solver for the inversion
    "proj": proj,  # EPSG system of the given coordinates
    "velo_or_disp": "velo",  # Type of data contained in the data cube ('disp' for displacements, and 'velo' for velocities)
    "verbose": True,  # Print information throughout the filtering process
}

## ---------------- Inversion and interpolation parameters ----------------- ##
inversion_kwargs = {
    "regu": regu,  # Regularization method.s to be used (for each flag if flag is not None)
    "coef": coef,  # Regularization coefficient.s to be used (for each flag if flag is not None)
    "solver": solver,  # Solver for the inversion
    "flag": flag_file,  # Divide the data in several areas where different methods should be used
    "conf": False,  # If True, confidence indicators are set between 0 and 1, with 1 the lowest errors
    "unit": 365,  # 365 if the unit is m/y, 1 if the unit is m/d
    "delete_outliers": "vvc_angle",  # Delete data with a poor quality indicator (if int), or with aberrant direction ('vvc_angle')
    "proj": proj,  # EPSG system of the given coordinates
    "interpolation_load_pixel": "nearest",  # Interpolation method used to load the pixel when it is not in the dataset
    "iteration": True,  # Allow the inversion process to make several iterations
    "nb_max_iteration": 10,  # Maximum number of iteration during the inversion process
    "threshold_it": 0.1,  # Threshold to test the stability of the results between each iteration, used to stop the process
    "apriori_weight": True,  # If True, use apriori weights
    "detect_temporal_decorrelation": True,  # If True, the first inversion will use only velocity observations with small temporal baselines, to detect temporal decorelation
    "linear_operator": None,  # Perform the inversion using this specific linear operator
    "interval_output": 30,
    "option_interpol": "spline",  # Type of interpolation ('spline', 'spline_smooth', 'nearest')
    "redundancy": 30,  # Redundancy in the interpolated time series in number of days, no redundancy if None
    "result_quality": "X_contribution",  # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')
    "visual": False,  # Plot results along the way
    "path_save": path_save,  # Path where to store the results
    "verbose": False,  # Print information throughout TICOI processing
}

smooth_res = False  # Smooth TICOI results (to limit the noise)
smooth_window_size = 3  # Size of the window for the average filter used to smooth the cube
smooth_filt = (
    None  # Specify here the filter you want to use to smooth the cube (if None, an average filter will be used)
)

## ----------------------- Parallelization parameters ---------------------- ##
nb_cpu = 6  # Number of CPU to be used for parallelization
block_size = 0.1  # Maximum sub-block size (in GB) for the 'block_process' TICOI processing method

## ------------------- Parameters for seasonality analysis ----------------- ##
# Is the periodicity frequency imposed to 1/365.25 (one year seasonality) ?
impose_frequency = True
# Add several sinus at different freqs (1/365.25 and harmonics (2/365.25, 3/365.25...) if impose_frequency is True)
#   (only available for impose_frequency = True for now)
several_freq = 5
# Compute also the best matching sinus to raw data, for comparison
raw_seasonality = True
# Filter to use in the first place
# 'highpass' : apply a bandpass filter between low frequencies (reject variations over several years (> 1.5 y))
# and the Nyquist frequency to ensure Shanon theorem
# 'lowpass' : or apply a lowpass filter only (to Nyquist frequency) : risk of tackling an interannual trend (long period)
filt = None
# Method used to compute local variations
# 'rolling_7d' : median of the std of the data centered in +- 3 days around each central date
# 'uniform_7d' : median of the std of the data centered in +- 3 days around dates constantly distributed every redundnacy
# days -- BEST
# 'uniform_all' : median of the std of each data covering the dates, which are constantly distributed every redundancy days
# 'residu' : standard deviation of the data previously subtracted by TICOI results (ground truth) = standard deviation of the "noise"
local_var_method = "uniform_7d"

if not os.path.exists(path_save):
    os.mkdir(path_save)


# %%========================================================================= #
#                                 DATA LOADING                                #
# =========================================================================%% #

start, stop = [], []
start.append(time.time())

# Load the cube.s
cube = cube_data_class()
cube.load(cube_name if TICOI_process != "load" else cube_name["raw"], **load_kwargs)

# Load raw data at pixels if required
if TICOI_process == "load":
    print("[Data loading] Loading raw data...")
    data_raw = process_blocks_refine(
        cube, nb_cpu=nb_cpu, block_size=block_size, returned=["raw"], inversion_kwargs=inversion_kwargs
    )
    data_raw = [
        pd.DataFrame(
            data={
                "date1": raw[0][0][:, 0],
                "date2": raw[0][0][:, 1],
                "vx": raw[0][1][:, 0],
                "vy": raw[0][1][:, 1],
                "errorx": raw[0][1][:, 2],
                "errory": raw[0][1][:, 3],
                "temporal_baseline": raw[0][1][:, 4],
            }
        )
        for raw in data_raw
    ]

# Prepare interpolation dates
first_date_interpol, last_date_interpol = prepare_interpolation_date(cube)
inversion_kwargs.update({"first_date_interpol": first_date_interpol, "last_date_interpol": last_date_interpol})

stop.append(time.time())
print(f"[Data loading] Cube of dimension (nz, nx, ny): ({cube.nz}, {cube.nx}, {cube.ny}) ")
print(f"[Data loading] Data loading took {round(stop[-1] - start[-1], 3)} s")


# %%========================================================================= #
#                                      TICOI                                  #
# =========================================================================%% #

start.append(time.time())

cube_interp, cube_invert = None, None

# The data cube is subdivided in smaller cubes computed one after the other in a synchronous manner (uses async)
# TICOI computation is then parallelized among those cubes
if TICOI_process == "block_process":
    result = process_blocks_refine(
        cube,
        nb_cpu=nb_cpu,
        block_size=block_size,
        returned=["raw", "interp"],
        preData_kwargs=preData_kwargs,
        inversion_kwargs=inversion_kwargs,
    )

# Direct computation of the whole TICOI cube
elif TICOI_process == "direct_process":
    # Preprocessing of the data (compute rolling mean for regu='1accelnotnull', delete outliers...)
    obs_filt, flag = cube.filter_cube(**preData_kwargs)
    inversion_kwargs.update({"flag": flag})

    # Progression bar
    xy_values = itertools.product(cube.ds["x"].values, cube.ds["y"].values)
    xy_values_tqdm = tqdm(xy_values, total=len(cube.ds["x"].values) * len(cube.ds["y"].values), mininterval=0.5)

    # Main processing of the data with TICOI algorithm, individually for each pixel
    result = Parallel(n_jobs=nb_cpu, verbose=0)(
        delayed(process)(cube, i, j, returned=["raw", "interp"], obs_filt=obs_filt, **inversion_kwargs)
        for i, j in xy_values_tqdm
    )

    result = {"raw": [result[i][0] for i in range(len(result))], "interp": [result[i][1] for i in range(len(result))]}

elif TICOI_process == "load":
    cube_interp = cube_data_class()
    cube_interp.load(cube_name["interp"], **load_kwargs)

    print("[TICOI processing] Loading TICOI data...")
    result = process_blocks_refine(
        cube_interp, nb_cpu=nb_cpu, block_size=block_size, returned="raw", inversion_kwargs=inversion_kwargs
    )
    result = [
        pd.DataFrame(
            data={
                "date1": r[0][0][:, 0],
                "date2": r[0][0][:, 1],
                "vx": r[0][1][:, 0],
                "vy": r[0][1][:, 1],
                "errorx": r[0][1][:, 2],
                "errory": r[0][1][:, 3],
                "temporal_baseline": r[0][1][:, 4],
            }
        )
        for r in result
    ]

if TICOI_process == "block_process" or TICOI_process == "direct_process":
    # Raw data
    data_raw = [
        pd.DataFrame(
            data={
                "date1": result["raw"][r][0][0][:, 0],
                "date2": result["raw"][r][0][0][:, 1],
                "vx": result["raw"][r][0][1][:, 0],
                "vy": result["raw"][r][0][1][:, 1],
                "errorx": result["raw"][r][0][1][:, 2],
                "errory": result["raw"][r][0][1][:, 3],
                "temporal_baseline": result["raw"][r][0][1][:, 4],
            }
        )
        for r in range(len(result))
    ]
    result = result["interp"]  # Result of the interpolation

stop.append(time.time())
print(
    f'[TICOI processing] TICOI {"processing" if TICOI_process != "load" else "loading"} took {round(stop[-1] - start[-1], 0)} s'
)


# %%========================================================================= #
#                                INITIALISATION                               #
# =========================================================================%% #

if TICOI_process != "load":
    # Write down some information about the data and the TICOI processing performed
    if save:
        if "invert" in returned:
            source, sensor = save_cube_parameters(cube, load_kwargs, preData_kwargs, inversion_kwargs,
                                                  returned="invert")
        if "interp" in returned:
            source_interp, sensor = save_cube_parameters(
                cube, load_kwargs, preData_kwargs, inversion_kwargs, returned="interp"
            )
        stop.append(time.time())
        print(f"[cube_ticoi_demo] Initialisation took {round(stop[-1] - start[-1], 3)} s")


# %%========================================================================= #
#                                WRITING RESULTS                              #
# =========================================================================%% #

start.append(time.time())
if TICOI_process != "load" and save:
    # Save TICOI results to a netCDF file, thus obtaining a new data cube
    cube_interp = cube.write_result_ticoi(
        result,
        source_interp,
        sensor,
        filename=f"{result_fn}_interp",
        savepath=path_save if save else None,
        result_quality=inversion_kwargs["result_quality"],
        verbose=inversion_kwargs["verbose"],
    )

    stop.append(time.time())
    print(f"[Writing results] Results saved at {path_save}")
    print(f"[Writing results] Writing cube to netCDF file took {round(stop[-1] - start[-1], 3)} s")


# %%========================================================================= #
#                               PERIODICITY MAPS                              #
# =========================================================================%% #
# Match a sinus to the data (frequency which can be fixed to 1/365.25, amplitude, phase which gives the date of the f)

start.append(time.time())


def match_sine(
    d: pd.DataFrame,
    filt: str | None = None,
    impose_frequency: bool = True,
    several_freq: int | None = None,
    raw_seasonality: bool = False,
    d_raw: pd.DataFrame | None = None,
):

    """
       Match a sine curve to TICOI results to look for a periodicity among the velocities. The period can either
    be set to 365.25 days, or estimated along with the other parameters (amplitude, phase, offset).

       :param d: [pd dataframe] --- pandas dataframe of the data at the considered pixel (TICOI results)
       :param filt: which filter to use before processing the sinus ('highpass', 'lowpass' or None, default None)
       :param impose_frequency: [bool] [default is True] --- Whether we should impose the frequency to 1/365.25 or not (default True)
       :param several_freq: [int | None] [default is None] --- If > 1, a signal made of several frequencies (at n*f) is matched to the data
       :param raw_seasonality: [bool] [default is False] --- If True, we also match a sinus to the raw data
       :param d_raw: [pd dataframe | None] [default is None] --- If raw_seasonality is True, must be the dataframe of the raw velocity data (not displacements)
    """

    d = d.dropna()
    dates = (d["date1"] + (d["date2"] - d["date1"]) // 2 - d["date1"].min()).dt.days.to_numpy()
    N = len(dates)
    if N <= 4:
        if raw_seasonality:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        return np.nan, np.nan, np.nan
    vv = np.sqrt(d["vx"] ** 2 + d["vy"] ** 2).to_numpy()
    Ts = dates[1] - dates[0]

    # Filtering
    try:
        if filt == "highpass":
            b, a = signal.butter(4, [1 / (1.5 * 365), 1 / (2.001 * Ts)], "bandpass", fs=1 / Ts, output="ba")
            vv_filt = signal.filtfilt(b, a, vv - np.mean(vv))
        elif filt == "lowpass":
            sos = signal.butter(4, 1 / (2.001 * Ts), "lowpass", fs=1 / Ts, output="sos")
            vv_filt = signal.sosfilt(sos, vv - np.mean(vv))
        else:
            vv_filt = vv
    except:
        vv_filt = vv

    # Frequency is set to 1/365.25 (one year)
    if impose_frequency:

        def sine_fconst(t, *args, freqs=1, f=1 / 365.25):
            sine = args[0] * np.sin(2 * np.pi * f * t + args[1])
            for freq in range(1, freqs):
                sine += args[2 * freq] * np.sin(2 * np.pi * (freq + 1) * f * t + args[2 * freq + 1])
            return sine + args[-1]

        try:
            # Find the best matching sinus to TICOI results
            if several_freq is None:
                several_freq = 1
            guess = np.concatenate(
                [np.concatenate([[np.max(vv_filt) - np.min(vv_filt), 0] for _ in range(several_freq)]), [0]]
            )
            popt, pcov = curve_fit(lambda t, *args: sine_fconst(t, *args, freqs=several_freq), dates, vv_filt, p0=guess)

            sine_year = sine_fconst(np.linspace(1, 365, 365), *popt, freqs=several_freq)
            A = np.max(sine_year) - popt[-1]
            f = 1 / 365.25
            first_max_day = pd.Timedelta(np.argmax(sine_year), "D") + d["date1"].min()
            max_day = (first_max_day - pd.Timestamp(year=first_max_day.year, month=1, day=1)).days
            del sine_year

            if raw_seasonality:
                #  Find the best matching sinus to raw data
                dates_raw = (d_raw.index - d["date1"].min()).days.to_numpy()
                raw_c = d_raw["vv"] - d_raw["vv"].mean()
                guess_raw = np.concatenate(
                    [np.concatenate([[np.max(raw_c) - np.min(raw_c), 0] for _ in range(several_freq)]), [0]]
                )
                popt_raw, pcov_raw = curve_fit(
                    lambda t, *args: sine_fconst(t, *args, freqs=several_freq), dates_raw, raw_c, p0=guess_raw
                )

                sine_raw_year = sine_fconst(np.linspace(1, 365, 365), *popt_raw, freqs=several_freq)
                A_raw = np.max(sine_raw_year) - popt_raw[-1]
                first_max_day_raw = pd.Timedelta(np.argmax(sine_raw_year), "D") + d["date1"].min()
                max_day_raw = (first_max_day_raw - pd.Timestamp(year=first_max_day.year, month=1, day=1)).days
                del sine_raw_year

        except RuntimeError:
            A, f, max_day = np.nan, np.nan, np.nan
            if raw_seasonality:
                A_raw, max_day_raw = np.nan, np.nan

    # Frequency is to be found too
    else:
        n = 64 * N
        window = signal.windows.hann(N)
        vv_win_tf = fft.rfft(vv_filt * window, n=n)
        freq = fft.rfftfreq(n, d=Ts)

        # Match a sinus to the data
        def sine_fvar(t, A, f, phi, off):
            return A * np.sin(2 * np.pi * f * t + phi) + off

        # Initial guess of the best matching sinus parameters
        guess = np.array(
            [
                np.max(2 / N * np.abs(vv_win_tf)),
                freq[np.argmax(np.abs(vv_win_tf))],
                np.angle(vv_win_tf)[np.argmax(np.abs(vv_win_tf))],
                np.mean(vv),
            ],
            dtype="float",
        )

        try:
            popt, pcov = curve_fit(sine_fvar, dates, vv, p0=guess)

            sine_year = sine_fvar(np.linspace(1, 365, 365), *popt, freqs=several_freq)
            A = np.max(sine_year) - popt[-1]
            first_max_day = pd.Timedelta(np.argmax(sine_year), "D") + d["date1"].min()
            max_day = (first_max_day - pd.Timestamp(year=first_max_day.year, month=1, day=1)).days
            del sine_year

        except RuntimeError:
            A, f, max_day = np.nan, np.nan, np.nan

    # Return Period, amplitude and phase of the periodicity
    if impose_frequency and raw_seasonality:
        return 1 / f, A, max_day, A_raw, max_day_raw
    else:
        return 1 / f, A, max_day


def AtoVar(A, raw, dataf_lp, local_var_method="uniform_7d"):

    """
       Compute Amplitude to local VARiations index, which compares the amplitude of the best matching sinus to the standard
    deviation of the noise using one of the four given methods.

       :param A: float, amplitude of the best matchning sinus
       :param raw: list, raw data
       :param dataf_lp: list of pandas dataframes, TICOI results
       :param local_var_method: str, method to be used to process the local variations
    """

    if A == np.nan:
        return np.nan

    # Compute local variations
    if local_var_method == "rolling_7d":
        var = raw["vv"].rolling(window="7D", center=True).std(ddof=0).drop_duplicates().dropna().median().item()

    elif local_var_method.split("_")[0] == "uniform":
        period_between_dates = (
            np.diff(np.sort(np.concatenate([raw["date1"], raw["date2"]]))).astype("timedelta64[D]").astype("int")
        )
        min_period = np.min(period_between_dates[period_between_dates > 0])
        var_dates = pd.date_range(start=raw["date1"].min(), end=raw["date2"].max(), freq=f"{min_period}D")
        local_var = pd.Series(index=var_dates)

        if local_var_method == "uniform_7d":
            for date in var_dates:
                local_var[date] = raw.loc[
                    (raw.index > date - pd.Timedelta("3D")) & (raw.index < date + pd.Timedelta("3D")), "vv"
                ].std(ddof=0)
        elif local_var_method == "uniform_all":
            for date in var_dates:
                local_var[date] = raw.loc[(raw["date1"] < date) & (raw["date2"] > date), "vv"].std(ddof=0)

        var = local_var[local_var > 0].dropna().median()

    elif local_var_method == "residu":
        dataf_lp.index = dataf_lp["First_date"] + (dataf_lp["Second_date"] - dataf_lp["First_date"]) // 2
        dataf_lp["vv"] = np.sqrt(dataf_lp["vx"] ** 2 + dataf_lp["vy"] ** 2)
        dataf_lp = dataf_lp.reindex(index=np.unique(raw.index)).interpolate().dropna()
        dataf = raw[raw.index >= dataf_lp.index[0]]
        dataff_vv_c = dataf["vv"] - dataf_lp["vv"]
        var = dataff_vv_c.std(ddof=0)

    return max(0, 1 - var / abs(A))


driver = gdal.GetDriverByName("GTiff")
srs = osr.SpatialReference()
srs.ImportFromEPSG(int(proj.split(":")[1]))

# Remove pixels with no data
empty = list(
    filter(
        bool,
        [
            d if not (result[d].empty and result[d][result[d]["vx"] == 0].shape[0] == 0) else False
            for d in range(len(result))
        ],
    )
)
positions = np.array(list(itertools.product(cube.ds["x"].values, cube.ds["y"].values)))[empty, :]
useful_result = [result[i] for i in empty]
useful_data_raw = [data_raw[i] for i in empty]

# Coordinates information
resolution = int(cube.ds["x"].values[1] - cube.ds["x"].values[0])
long_data = (positions[:, 0] - np.min(cube.ds["x"].values)).astype(int) // resolution
lat_data = (positions[:, 1] - np.min(cube.ds["y"].values)).astype(int) // resolution

# Format raw data to velocities
for raw in data_raw:
    raw["vx"] = raw["vx"] * preData_kwargs["unit"] / raw["temporal_baseline"]
    raw["vy"] = raw["vy"] * preData_kwargs["unit"] / raw["temporal_baseline"]
    raw["vv"] = np.sqrt(raw["vx"] ** 2 + raw["vy"] ** 2)
    raw.index = raw["date1"] + (raw["date2"] - raw["date1"]) // 2

##  Best matching sinus map (amplitude and phase, and period if not fixed)
print("[Fourier analysis] Computing periodicity maps...")
if not impose_frequency:
    period_map = np.empty([cube.nx, cube.ny])
    period_map[:, :] = np.nan
amplitude_map = np.empty([cube.nx, cube.ny])
amplitude_map[:, :] = np.nan
AtoVar_map = np.empty([cube.nx, cube.ny])
AtoVar_map[:, :] = np.nan
peak_map = np.empty([cube.nx, cube.ny])
peak_map[:, :] = np.nan
if raw_seasonality:
    amplitude_raw_map = np.empty([cube.nx, cube.ny])
    amplitude_raw_map[:, :] = np.nan
    peak_raw_map = np.empty([cube.nx, cube.ny])
    peak_raw_map[:, :] = np.nan

result_tqdm = tqdm(zip(useful_result, useful_data_raw), total=len(useful_result), mininterval=0.5)
match_res = np.array(
    Parallel(n_jobs=nb_cpu, verbose=0)(
        delayed(match_sine)(d, filt=filt, impose_frequency=impose_frequency, raw_seasonality=raw_seasonality, d_raw=raw)
        for d, raw in result_tqdm
    )
)
if not impose_frequency:
    period = np.abs(match_res[:, 0])
    period_map[long_data, lat_data] = np.sign(period - 365) * (1 - np.minimum(period, 365) / np.maximum(period, 365))
amplitude_map[long_data, lat_data] = np.abs(match_res[:, 1])
peak_map[long_data, lat_data] = match_res[:, 2]
raw_tqdm = tqdm(zip(match_res[:, 1], useful_data_raw, useful_result), total=len(useful_data_raw), mininterval=0.5)
AtoVar_map[long_data, lat_data] = Parallel(n_jobs=nb_cpu, verbose=0)(
    delayed(AtoVar)(A, raw, dataf_lp, local_var_method) for A, raw, dataf_lp in raw_tqdm
)
if raw_seasonality:
    amplitude_raw_map[long_data, lat_data] = np.abs(match_res[:, 3])
    peak_raw_map[long_data, lat_data] = match_res[:, 4]

# Save the maps to a .tiff file with two bands (one for period, and one for amplitude)
if impose_frequency:
    tiff = driver.Create(
        f"{path_save}matching_sine_map_fconst_{local_var_method}.tiff",
        amplitude_map.shape[0],
        amplitude_map.shape[1],
        3 if not raw_seasonality else 5,
        gdal.GDT_Float32,
    )
    tiff.SetGeoTransform([np.min(cube.ds["x"].values), resolution, 0, np.max(cube.ds["y"].values), 0, -resolution])
    tiff.GetRasterBand(1).WriteArray(np.flip(amplitude_map.T, axis=0))
    tiff.GetRasterBand(2).WriteArray(np.flip(peak_map.T, axis=0))
    tiff.GetRasterBand(3).WriteArray(np.flip(AtoVar_map.T, axis=0))
    if raw_seasonality:
        tiff.GetRasterBand(4).WriteArray(np.flip(amplitude_raw_map.T, axis=0))
        tiff.GetRasterBand(5).WriteArray(np.flip(peak_raw_map.T, axis=0))
else:
    tiff = driver.Create(
        f"{path_save}matching_sine_map_{filt}_{local_var_method}.tiff",
        period_map.shape[0],
        period_map.shape[1],
        4,
        gdal.GDT_Float32,
    )
    tiff.SetGeoTransform([np.min(cube.ds["x"].values), resolution, 0, np.max(cube.ds["y"].values), 0, -resolution])
    tiff.GetRasterBand(1).WriteArray(np.flip(period_map.T, axis=0))
    tiff.GetRasterBand(2).WriteArray(np.flip(amplitude_map.T, axis=0))
    tiff.GetRasterBand(3).WriteArray(np.flip(peak_map.T, axis=0))
    tiff.GetRasterBand(4).WriteArray(np.flip(AtoVar_map.T, axis=0))
tiff.SetProjection(srs.ExportToWkt())

# Needed to effectively save the .tiff file
tiff = None
driver = None

stop.append(time.time())
print(f"[Fourier analysis] Computing periodicity maps took {round(stop[-1] - start[-1], 0)} s")
print(f"[Overall] Overall processing took {round(stop[-1] - start[0], 0)} s")
