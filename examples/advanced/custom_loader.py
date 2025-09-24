"""
Demonstration to show how to add your own data loader
"""

from ticoi.cube_data_classxr import CubeDataClass


def loader_example(self, conf):
    self.author = "to modify with the author names"
    self.source = "to modify with url or name of the dataset"
    self.ds.attrs["proj4"] = "to modify with proj4text"

    # standardize sensor names
    sensor_raw = (
        "to modify with str of list with the names of the sensors, could be in the form of S1, L8, L9, or Sentinel-1"
    )
    sensors = self._standardize_sensor_names(
        sensor_raw
    )  # convert to a long format type, like Sentinel-1, Landsat-8, etc

    # normalize error if needed
    errorx = "to modify with errors, in 1D (in time) or 3D (in space and time)"
    errory = "to modify with errors, in 1D (in time) or 3D (in space and time)"
    if conf:
        errorx = self._normalize_error_to_confidence(errorx)
        errory = self._normalize_error_to_confidence(errory)

    date1 = "to modify with the list or np.array of the first of acquisition"
    date2 = "to modify with the list or np.array of the first of acquisition"

    # if needed
    date1 = date1.astype("datetime64[ns]")
    date2 = date2.astype("datetime64[ns]")

    return {
        "date1": date1,
        "date2": date2,
        "sensor": sensors,
        "source": self.source,
        "errorx": errorx,
        "errory": errory,
    }


# add your loader to the CubeDataClass
CubeDataClass.register_loader("My Author", loader_example)
# after that you use all the functions of CubeDataClass with your own dataset!
