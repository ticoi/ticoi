Different parameters can be used:

#### regu

- '1accelnotnull': use a first order Tikhonov regularisation, using a first guess on the velocity. This first guess is
  inferred from a spatio-temporal smoothing. The regularisation term is the form of || G (X - X_0) ||² with X_0 the
  initial guess on the velocity, and G the first order derivative operator.
- 1: use a first order Tikhonov regularisation

#### coef

It corresponds to the coefficient of this regression, it can be selected using the Velocity Vector Coherence, or GLAFT
metrics.

#### delete_outliers: Rejection of outliers before the inversion

- 'z_score':  Remove the observations if it is 3 time the standard deviation from the average of observations over this
  pixel
- 'median_angle': Remove the observation if its direction is angle_thres away from the direction of the median vector
- 'vvc_angle':  Combine median_angle and z_score. If the Velocity Vector Coherence is lower than a given threshold,
  outliers are filtered out according to the zscore, else to the median angle filter, i.e. pixels are filtered out if
  the angle with the observation is angle_thres away from the median vector
- 'magnitude':  Remove the observations which a magnitude larger than a given threshold
- 'median_magnitude':   Remove the observation if it median_magnitude_thres times bigger than the mean velocity at
  pixel, or if it is
  1/median_magnitude_thres times smaller than the mean velocity at pixel
- 'error': Remove the observations if its error variable is larger than a given threshold


- 
