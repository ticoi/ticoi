# Parameters to play with

There are several parameters that you can adjust.
By carrying out a sensitivity analysis on six GNSS stations, we found that the parameters with the greatest impact are the
regularization type and coefficient.
While other parameters introduce small differences, testing them can improve the results at pixel level.
A detailed sensitivity analysis of these parameters can be found in Appendix B.
of [Charrier et al. 2025](https://egusphere.copernicus.org/preprints/2025/egusphere-2024-3409/)

### 'regu': the regularization type

- '1accelnotnull': use a first order Tikhonov regularisation, using a first guess on the velocity. This first guess is
  inferred from a spatio-temporal smoothing. The regularisation term is the form of || G (X - X_0) ||² with X_0 the
  initial guess on the velocity, and G the first order derivative operator.
- 1: use a first order Tikhonov regularisation

### 'coef': the regularization coefficient

It can be selected using the Velocity Vector Coherence, or GLAFT
metrics. It has to be large enough to have a good signal-to-noise ratio. The optimal value is usually 100 or 500.

### 'delete_outliers': Rejection of outliers before the inversion

- 'mz_score': Remove the observations if it is 3.5 time the MAD from the median of observations over this pixel
- 'median_angle': Remove the observation if its direction is angle_thres away from the direction of the median vector
- 'z_score':  Remove the observations if it is 3 time the standard deviation from the average of observations over this
  pixel
- 'vvc_angle':  Combine median_angle and z_score. If the Velocity Vector Coherence is lower than a given threshold,
  outliers are filtered out according to the zscore, else to the median angle filter, i.e. pixels are filtered out if
  the angle with the observation is angle_thres away from the median vector
- 'magnitude':  Remove the observations which a magnitude larger than a given threshold
- 'median_magnitude':   Remove the observation if it median_magnitude_thres times bigger than the mean velocity at
  pixel, or if it is
  1/median_magnitude_thres times smaller than the mean velocity at pixel
- 'error': Remove the observations if its error variable is larger than a given threshold

See the comparison of mz_score and median_angle in Appendix B
of [Charrier et al. 2025](https://egusphere.copernicus.org/preprints/2025/egusphere-2024-3409/)

### 'smooth_method': spatio-temporal filter

As an initial guess, we used filtered time series, using different spatio-temporal filters. Here are the options:

-'gaussian': Gaussian
-'median': Median
-'savgol': Savitzky–Golay
-'lowess': Locally Weighted Scatterplot Smoothing (LOWESS)

The choice of the filter results in a standard deviation in RMSE of about 2.6 m yr−1 on
average (i.e., 8% of the averaged RMSE). We note that both the LOWESS and median filters can provide slightly better
results for non-surge type glaciers, with
improvements ranging from 0.8 to 4.5 m yr−1 (i.e., 3% to 10%). However, they can also lead to over-smoothing and
LOWESS require 1.5 times more computational time than the other filters. Therefore, we recommend using the
Savitzky–Golay filter, which offers
a good balance between computational efficiency and accuracy in general scenarios. But it is worth testing LOWESS or median in some spatial cases.
See [Charrier et al. 2022](https://ieeexplore.ieee.org/document/9618734) for more details. 

### 'interval_output': temporal sampling of the output time series

The larger it is, the smoothes the time series will be.
See [Charrier et al. 2022](https://ieeexplore.ieee.org/document/9618734) for more details.

### 'solver': the type of solver used in the inversion

- 'LSMR': it corresponds to the function lsmr of scipy.linalg.
- 'LSMR_ini': same as above with an initialization.
- 'LSQR': it corresponds to the function lsqr of scipy.linalg.
- 'LS': Least Square solver, it corresponds to the function lstsq of numpy.linalg.
- 'L1': Norm L1

L1 and LS will lead to larger computation time.




