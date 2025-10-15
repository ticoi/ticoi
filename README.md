# TICOI

[![Language](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![Python test](https://github.com/ticoi/ticoi/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/ticoi/ticoi/actions/workflows/python-app.yml)
[![License](https://img.shields.io/badge/license-GPLv3+-blue.svg?style=flat-square)](https://github.com/ticoi/ticoi/blob/main/LICENSE)
[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

TICOI is a tool to postprocess surface velocities estimated from remote sensing images (e.g., over glaciers, ice-sheets, landslides or earthquakes).
The method is based on the temporal closure principle. It fuses velocity measurements which are multi-temporal (with
different temporal baselines) and multi-sensor (from different satellite images),
and may have been computed by different processing chains. It takes as input NetCDF files containing the image-pair
velocities, that you may have generated yourself. It also natively supports data from the [NASA ITS_LIVE project](https://its-live.jpl.nasa.gov/) or from
[Millan et al. (2022)](https://www.theia-land.fr/en/blog/product/glacier-surface-flow-velocity/).

The package is based on the methodological developments published in:

- Charrier, L., Dehecq, A., Guo, L., Brun, F., Millan, R., Lioret, N., Copland, L., Maier, N., Dow, C., and Halas, P.: TICOI: an operational Python package to generate regular glacier velocity time series, The Cryosphere, 19, 4555–4583, https://doi.org/10.5194/tc-19-4555-2025, 2025.

- Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an
  optimal temporal sampling from displacement observation networks. IEEE Transactions on Geoscience and Remote Sensing,
  60, 1-10.

The main principle of TICOI relies on the temporal closure of the displacement measurement network.
Measured displacements with different temporal baselines are expressed as linear combinations of estimated
displacement (see the Figure below).
The aim is to take advantage of different types of information (displacement measured using different temporal
baselines,
on images from different types of satellite, using different processing chains) to extract glacier velocity time series, with a given temporal sampling.
This enable the
harmonization of various datasets, and the creation of standardized sub-annual velocity products.

<p align="center">
  <img src="examples/image/Temporal_closure.png" alt="Temporal_closure" width="800"/>
</p>

## INSTALLATION

### With `pip`

```bash
pip install ticoi
```

### With git clone and `mamba`

This option allow you to modify the code
Clone the git repo and create a `mamba` environment (see how to install `mamba` in
the [mamba documentation](https://mamba.readthedocs.io/en/latest/)):

```bash
git clone https://github.com/ticoi/ticoi.git
cd ticoi
mamba env create -f environment.yml -n ticoi_env  # change the name if you want
mamba activate ticoi_env  # Or any other name specified above
```


## TUTORIALS

### Basic examples

**- notebook**

* [How to process one pixel of a NetCDF file](examples/basic/notebook/pixel_demo_local_ncdata.ipynb)
* [How to process one pixel of ITS_LIVE data, stored on a cloud](examples/basic/notebook/pixel_demo_its_live_on_cloud.ipynb)

**- python_script**

* [How to process one cube](examples/basic/python_script/cube_ticoi_demo.py)
* [How to process one pixel](examples/basic/python_script/pixel_ticoi_demo.py)

### Advanced examples

* [How to process one ITS_LIVE cube directly from the cloud](/examples/advanced/demo_for_different_datasets/cube_ticoi_demo_its_live.py)
* [How to format several geotiff files into a netCDF file](examples/advanced/cube_prep_from_geotiff.py)
* [How to apply GLAFT on TICOI results](examples/advanced/quality_metrics/glaft_for_ticoi_results.py)
* [How to compute statistics in static areas](examples/advanced/quality_metrics/stats_in_static_areas.py)
* [How to process one IGE cube](examples/advanced/demo_for_different_datasets/cube_ticoi_demo_IGE_S2.py)

## TO USE YOUR OWN DATASET

### You have geotiff files

You need to convert them into netcdf, by
modifying [this script](examples/advanced/cube_prep_from_geotiff.py).

### You have netcdf files

If it is [ITS_LIVE data]((https://its-live.jpl.nasa.gov/)), or [Millan et al., 2022](https://www.theia-land.fr/en/blog/product/glacier-surface-flow-velocity/), you can directly use them!
If not, there are two options:
* you can modify your variables to match TICOI compatible format. 
For that, your dimensions need to be ("mid_date", "y", "x"), with "mid_date" your time dimension. 
Your variables need to be:
  *     vx, vy (necessary): "vx", "vy" are the velocity along your x and y axis in m/y
  *     date1, date2 (necessary): "date1" and "date2" correspond to the first and second date of acquisition of the image-pair
  *     errorx, errory (optional): "errorx", "errory" are the errors along your x and y axis in m/y
Your attributes:
  *     author (optional): if you want to specify the name of the author
  *     source (optional): if you want to specify the source of the dataset
* you can directly add your own data loader to convert your format in TICOI format, inside the TICOI package. For that see an example [here](examples/advanced/custom_loader.py)

## HYPERPARAMETERS AND OUTPUTS

* to understand the output of pixel_demo please
  check [README_output](README_output.md)
* to understand the parameters you can change, please
  check [README_possible_parameters](README_possible_parameters.md)

## TO CONTRIBUTE

We welcome any contribution to this package! See guidelines [here](CONTRIBUTING.md).

[packaging guide]: https://packaging.python.org

[distribution tutorial]: https://packaging.python.org/tutorials/packaging-projects/

[src]: https://github.com/pypa/sampleproject

[rst]: http://docutils.sourceforge.net/rst.html

[md]: https://tools.ietf.org/html/rfc7764#section-3.5 "CommonMark variant"

[md use]: https://packaging.python.org/specifications/core-metadata/#description-content-type-optional
