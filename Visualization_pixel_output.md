
Some figures can be displayed by filling the option_visual parameter with different strings:

#### Original data

- 'original_velocity_xy' -> vx_vy.png : Original velocities (central date and temporal baselines) according to x and y.
- 'vxvy_quality' -> vxvy_quality_bas.png : Original velocities with measurement error (central date and colour).
- 'vv_good_quality' -> vv_good_quality.png : Original velocities with a quality index greater than 0.5.
- 'original_magnitude' -> vv.png : Magnitude of the original velocities (central date and temporal baselines).
- 'vv_quality' -> vv_quality.png : Magnitude of the original velocity with measurement error (central date and colour).

#### Results after inversion of the AX = Y system (ILF: varying temporal sampling)

- 'X' -> X_velocity.png : Superposition of estimated and observed velocities (vx and vy). The limits of the figure
  correspond to the estimated velocities.
- 'X_zoom' -> X_velocity_Zoom.png : Superposition of estimated and observed velocities (vx and vy). The limits of the
  figure correspond to the observed velocities.
- 'X_magnitude' -> Xvv.png : Superposition of the magnitude of estimated velocities and the magnitude of observed
  velocities. The limits of the figure correspond to the estimated velocities.
- 'X_magnitude_zoom' -> Xvv_Zoom.png : Zoom on the ordinate of Xvv. The limits of the figure correspond to the observed
  velocities.
- 'Y_contribution' -> X_dates_contribution_vx_vy.png : Number of displacements of Y (observed displacements) used to
  calculate a displacement of X (estimated displacements) for vx and vy.
