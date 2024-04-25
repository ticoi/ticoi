# ticoi

![Python Logo](https://www.python.org/static/community_logos/python-logo.png "Sample inline image")

Post-processing method based on the temporal closure to fuse multi-temporal and multi-sensor velocity measurements,
which may have been computed from
different processing chains.

## Get started

### INSTALLATION

To clone the git repository and set up the conda environment:

```
git clone git@github.com:ticoi/ticoi.git
cd ticoi/   
conda env create -f ticoi_env.yml 
conda activate ticoi      
pip install -e .
```

*Note:* To be speed-up the environment setup, you may use mamba. Simply run before the previous lines:

```
conda install mamba -n base -c conda-forge
conda activate base
```

Then replace all conda commands by mamba.

### STRUCTURE

#### Main:

* **core.py**: Main functions to process the temporal inversion of glacier's surface velocity using
  the TICOI method. The inversion is solved using an Iterative Reweighted Least Square, and a robust downweighted
  function (Tukey's biweight).
* **secondary_functions.py**: Auxillary functions to process the temporal inversion.
* **cube_data_classxr.py**: Class object to store and manipulate velocity observation data
* **examples/ticoi_cube_demo.py**: Processing of one cube
* **examples/ticoi_pixel_demo.py**: Processing of one pixel

### OUTPUTS

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

[packaging guide]: https://packaging.python.org

[distribution tutorial]: https://packaging.python.org/tutorials/packaging-projects/

[src]: https://github.com/pypa/sampleproject

[rst]: http://docutils.sourceforge.net/rst.html

[md]: https://tools.ietf.org/html/rfc7764#section-3.5 "CommonMark variant"

[md use]: https://packaging.python.org/specifications/core-metadata/#description-content-type-optional
