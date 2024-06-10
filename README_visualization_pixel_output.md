
Some figures can be displayed by filling the option_visual parameter with different strings:

#### Original data
option_visual = ['obs_xy','obs_magnitude','obs_vxvy_quality','invertxy_overlayed','invertvv_overlayed','residuals','xcount_xy','xcount_vv']

- 'obs_xy' -> vx_vy.png : Original velocities (central date and temporal baselines) according to x and y.
- 'obs_magnitude' -> vxvy_quality_bas.png : Original velocities with measurement error (central date and colour).
- 'obs_vxvy_quality' -> vv_quality.png : Magnitude of the original velocity with measurement error (central date and colour).
- 'original_magnitude' -> vv.png : Magnitude of the original velocities (central date and temporal baselines).

#### Results after inversion of the AX = Y system (ILF: varying temporal sampling)

- 'invertxy_overlayed' -> vx_vy_overlayed.png : Superposition of estimated and observed velocities (vx and vy). The limits of the figure
  correspond to the estimated velocities.
- 'invertvv_overlayed' -> vv_overlayed.png : Superposition of the magnitude of estimated velocities and the magnitude of observed
  velocities. The limits of the figure correspond to the estimated velocities.
- 'xcount_xy' -> X_dates_contribution_vx_vy.png : Number of displacements of Y (observed displacements) used to
  calculate a displacement of X (estimated displacements) for vx and vy.
- 'xcount_vv' -> X_dates_contribution_vv.png : Norm of the number of displacements of Y (observed displacements) used to
  calculate a displacement of X (estimated displacements) for vx and vy.
- 'residuals' -> statistics about the residuals from the inversion:
    - Plot of the final residuals overlayed in colors on vx and vy measurements ('residuals_vx_vy_final_residual.png').
    - Plot of the recontructed velocity observations (from AX) overlayed on the original velocity observations ('residuals_vx_vy_mismatch.png').
    - Comparison of residuals according to the temporal baseline (residuals_tempbaseline.png),
    - the type of sensor and authors (residuals_author_abs.png,residuals_vy_author.png,residuals_vx_author_abs.png), 
    - and the quality indicators (residuals_quality.png).
- 'invert_weight' -> weight_ini_vx_vy.png; weightlast_vx_vy.png: weight given in the first and last iteration of the inversion, respectively