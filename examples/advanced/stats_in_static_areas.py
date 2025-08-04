from ticoi.cube_data_classxr import CubeDataClass
from ticoi.example import get_path


cube_name = get_path("Argentiere_example_interp")  # path to our dataset
stable_area = get_path("Argentiere_static")

cube = CubeDataClass()
cube.load(cube_name, pick_date=["2017-01-01", "2017-03-30"])
# Compute normalized median absolute deviation
nmad = cube.compute_nmad(shapefile_path=stable_area, return_as="dataframe")
# Compte median over stable areas
med = cube.compute_med_static_areas(shapefile_path=stable_area, return_as="dataframe")

# Note that we can add a variable invert
# If invert=False (default):
# The dataset is clipped inside the geometry — data outside the polygon becomes NaN.
# If invert=True:
# The dataset is clipped outside the geometry — data inside the polygon becomes NaN.
