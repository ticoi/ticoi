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

#### Main code:

* **core.py**: Main functions to process the temporal inversion of glacier's surface velocity using
  the TICOI method. The inversion is solved using an Iterative Reweighted Least Square, and a robust downweighted
  function (Tukey's biweight).
* * **cube_data_classxr.py**: Class object to store and manipulate velocity observation data
* **inversion_functions.py**: Auxillary functions to process the temporal inversion.
* **interpolation_functions.py**: Auxillary functions to process the temporal interpolation.
* **filtering_functions.py**: Auxillary functions to process some filtering.
* **other_functions.py**: Two other functions for assesing ITS_LIVE data.
* **mjd2date.py**: Convert the dates from Modified Julian Date to Gregorian Date
* 
#### Examples:

* **test_data**: test data for demonstration
* 
* **examples/ticoi_cube_demo.py**: Demonstration of how to process one cube
* **examples/ticoi_pixel_demo.py**: Demonstration of how to process one pixel

* **examples/results/cube**: Expected results for the demo cube
* **examples/results/pixel**: Expected results for the demo pixel



### OUTPUTS
* to understand to output of pixel_demo please check Visualization_pixel_output.md

[packaging guide]: https://packaging.python.org

[distribution tutorial]: https://packaging.python.org/tutorials/packaging-projects/

[src]: https://github.com/pypa/sampleproject

[rst]: http://docutils.sourceforge.net/rst.html

[md]: https://tools.ietf.org/html/rfc7764#section-3.5 "CommonMark variant"

[md use]: https://packaging.python.org/specifications/core-metadata/#description-content-type-optional
