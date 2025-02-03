
Different parameters can be used. Here some explanation of the some parameters

#### delete_outliers: Rejection of outliers before the inversion
- 'z_score':  Remove the observations if it is 3 time the standard deviation from the average of observations over this pixel
- 'median_angle': Remove the observation if it is angle_thres away from the median vector
- 'vvc_angle':  Combine median_angle and z_score. If the Velocity Vector Coherence is lower than a given threshold, outliers are filtered out according to the zscore, else to the median angle filter, i.e. pixels are filtered out if the angle with the observation is angle_thres away from the median vector
- 'magnitude':  Remove the observations which a magnitude larger than a given threshold
- 'median_magnitude':   Remove the observation if it median_magnitude_thres times bigger than the mean velocity at pixel, or if it is
    1/median_magnitude_thres times smaller than the mean velocity at pixel
- 'error': Remove the observations if its error variable is larger than a given threshold
