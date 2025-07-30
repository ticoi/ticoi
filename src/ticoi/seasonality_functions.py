import numpy as np
import pandas as pd
import scipy.fft as fft
import scipy.signal as signal
from scipy.optimize import curve_fit


def match_sine(
    d: pd.DataFrame,
    filt: str | None = None,
    impose_frequency: bool = True,
    several_freq: int | None = None,
    raw_seasonality: bool = False,
    d_raw: pd.DataFrame | None = None,
    variable: str = "vv",
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
       :param variable: [str] [default is 'vv'] --- variable used to fit the sinus (vx, vy or vv). vv correspond to the velocity magnitude
    """

    d = d.dropna()
    dates = (d["date1"] + (d["date2"] - d["date1"]) // 2 - d["date1"].min()).dt.days.to_numpy()

    N = len(dates)
    if N <= 4:  # do not compute anything
        if raw_seasonality:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        return np.nan, np.nan, np.nan
    if variable == "vv":
        vv = np.sqrt(d["vx"] ** 2 + d["vy"] ** 2).to_numpy()
    elif variable == "direction":
        vv = np.arctan2(d["vy"], d["vx"]).to_numpy()
    else:
        vv = d[variable]

    Ts = dates[1] - dates[0]

    # Filtering to remove inter-annual variations
    if filt == "highpass":
        b, a = signal.butter(4, [1 / (1.5 * 365), 1 / (2.001 * Ts)], "bandpass", fs=1 / Ts, output="ba")
        vv_filt = signal.filtfilt(b, a, vv - np.mean(vv))
    elif filt == "lowpass":
        sos = signal.butter(4, 1 / (2.001 * Ts), "lowpass", fs=1 / Ts, output="sos")
        vv_filt = signal.sosfilt(sos, vv - np.mean(vv))
    else:
        vv_filt = vv

    # Frequency is set to 1/365.25 (one year)
    if impose_frequency:

        def sine_fconst(t, *args, freqs=1, f=1 / 365.25):
            sine = args[0] * np.sin(
                2 * np.pi * f * t + args[1]
            )  # args[0] amplitude of the signal, and args[1] the phase of the signal
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
            first_max_day = pd.Timedelta(np.argmax(sine_year), "D") + d["date1"].min()  # date of the maximum
            max_day = (first_max_day - pd.Timestamp(year=first_max_day.year, month=1, day=1)).days  # day of the year
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
    else:  # use fft, with an hanning window
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


def rolling_std(raw, dataf_lp, local_var_method="uniform_7d"):
    """
       Compute Amplitude to local VARiations index, which compares the amplitude of the best matching sinus to the standard
    deviation of the noise using one of the four given methods.

       :param A: float, amplitude of the best matchning sinus
       :param raw: list, raw data
       :param dataf_lp: list of pandas dataframes, TICOI results
       :param local_var_method: str, method to be used to process the local variations
    """

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

    return var


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

    var = rolling_std(raw, dataf_lp, local_var_method)

    return max(0, 1 - var / abs(A))
