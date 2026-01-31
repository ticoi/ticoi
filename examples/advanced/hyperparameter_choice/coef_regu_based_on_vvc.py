import os

from ticoi import example
from ticoi.core import process_blocks_refine, save_cube_parameters
from ticoi.cube_data_classxr import CubeDataClass
from ticoi.cube_writer import CubeResultsWriter
import glob
import numpy as np
import pandas as pd
from ticoi.optimize_coefficient_functions import plot_vvc_time_series

path_save = "/home/charriel/Documents/Collaborations/Didal"
list_coef = [1, 10, 100, 500, 1000]


## ------------------------------ Data selection --------------------------- ##
cube_name = example.get_path("ITS_LIVE_Lowell_Lower")
path_save = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")),
    "examples",
    "results",
    "cube",
)  # path where to save our results
result_fn = "Lowell_example"  # Name of the netCDF file to be created
## ---------------------------- Loading parameters ------------------------- ##

## ----------------------- Data preparation parameters --------------------- ##
# For the following parts we advice the user to change only the following parameter, the other parameters stored in a dictionary can be kept as it is for a first use
regu = "1accelnotnull"  # Regularization method.s to be used (for each flag if flag is not None) : 1 minimize the acceleration, '1accelnotnull' minize the distance with an apriori on the acceleration computed over a spatio-temporal filtering of the cube
delete_outlier = None
apriori_weight = False
proj = "EPSG:3413"  # EPSG system of the given coordinates
nb_cpu = 12  # Number of CPU to be used for parallelization
block_size = 0.5  # Maximum sub-block size (in GB) for the 'block_process' TICOI processing method
compute_ticoi = True  # If True, save TICOI results to a netCDF file

load_kwargs = {
    "chunks": {},
    "proj": proj,  # EPSG system of the given coordinates
}

preData_kwargs = {
    "delete_outliers": delete_outlier,
    # Delete data with a poor quality indicator (if int), or with aberrant direction ('vvc_angle')
    "regu": regu,
    # Regularization method.s to be used (for each flag if flag is not None) : 1 minimize the acceleration, '1accelnotnull' minize the distance with an apriori on the acceleration computed over a spatio-temporal filtering of the cube
    "proj": proj,  # EPSG system of the given coordinates
}

if compute_ticoi:  # Apply TICOI for different regularization coefficients
    for coef in list_coef:
        ## ---------------- Inversion and interpolation parameters ----------------- ##
        inversion_kwargs = {
            "coef": coef,  # Regularization coefficient.s to be used (for each flag if flag is not None)
            "apriori_weight": apriori_weight,  # If True, use apriori weights
            "detect_temporal_decorrelation": True,
            # If True, the first inversion will use only velocity observations with small temporal baselines, to detect temporal decorelation
            "result_quality": "X_contribution",
            # Criterium used to evaluate the quality of the results ('Norm_residual', 'X_contribution')
            "path_save": path_save,  # Path where to store the results
            "proj": proj,
        }

        ## ----------------------- Parallelization parameters ---------------------- ##

        # Load the first cube
        cube = CubeDataClass()
        cube.load(cube_name, **load_kwargs)

        # The data cube is subdivided in smaller cubes computed one after the other in a synchronous manner (uses async)
        # TICOI computation is then parallelized among those cubes
        # Prepare interpolation dates
        first_date_interpol, last_date_interpol = cube.prepare_interpolation_date()
        inversion_kwargs.update({"first_date_interpol": first_date_interpol, "last_date_interpol": last_date_interpol})
        result = process_blocks_refine(
            cube,
            nb_cpu=nb_cpu,
            block_size=block_size,
            preData_kwargs=preData_kwargs,
            inversion_kwargs=inversion_kwargs,
            returned="interp",
        )

        source_interp, sensor = save_cube_parameters(
            cube, load_kwargs, preData_kwargs, inversion_kwargs, returned="interp"
        )

        writer = CubeResultsWriter(cube)
        cube_interp = writer.write_result_ticoi(
            result,
            source_interp,
            sensor,
            result_quality=inversion_kwargs["result_quality"],
            filename=f"{result_fn}_interp{coef}",
            savepath=path_save,
            verbose=True,
        )
        # Plot the mean velocity as an example
        if cube_interp is not None:
            cube_interp.average_cube(return_format="geotiff", return_variable=["vv"], save=True, path_save=path_save)

main_path = path_save
list_path = glob.glob(f"{main_path}/*.nc")
list_VVC, list_param = [], []

# Compute
dataflist = []
for cube_name in list_path:
    print(cube_name)
    cube = CubeDataClass()
    cube.load(cube_name)
    Coh_vector = cube.compute_vvc()
    list_VVC.append(np.nanmean(Coh_vector))
    list_param.append(int(cube_name.split("interp")[-1].split(".nc")[0]))

dataf = pd.DataFrame({"param": list_param, "VVC": list_VVC})
dataf.sort_values(by="param", inplace=True)
dataf.to_csv(f"{path_save}/dataf_vvc_coef.csv")
plot_vvc_time_series(dataf, path_save, name="VVC_curve")
