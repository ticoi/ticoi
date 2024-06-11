"""
Author : Laurane Charrier, Lei Guo, Nathan Lioret
Reference:
    Charrier, L., Yan, Y., Koeniguer, E. C., Leinss, S., & Trouvé, E. (2021). Extraction of velocity time series with an optimal temporal sampling from displacement
    observation networks. IEEE Transactions on Geoscience and Remote Sensing.
    Charrier, L., Yan, Y., Colin Koeniguer, E., Mouginot, J., Millan, R., & Trouvé, E. (2022). Fusion of multi-temporal and multi-sensor ice velocity observations.
    ISPRS annals of the photogrammetry, remote sensing and spatial information sciences, 3, 311-318.
"""

import numpy as np
import math as m
import geopandas as gpd
from shapely.geometry import Polygon, Point


def moving_average_dates(dates: np.ndarray, data: np.ndarray, v_pos: int, save_lines: bool = False) -> np.ndarray:
    """

    :param dates: an array with all the dates included in data, list
    :param data: an array where each line is (date1, date2, other elements) for which a velocity is computed
    :param v_pos: position in data of the considered variable
    :param save_lines: if True save the lines to use
    :return: moving average between consecutive dates in dates_range
    """

    ini = []
    # data[:, v_pos] = np.ma.array(data[:, v_pos])
    for i_date, date in enumerate(dates[:, 0]):
        i = 0
        moy = []

        while i < data.shape[0] and dates[i_date, 1] >= data[
            i, 0]:  # if the velocity observation is between two dates in dates_range

            if dates[i_date, 0] <= data[i, 1]:
                moy.append(data[i, v_pos])
            i += 1
        if len(moy) != 0:
            ini.append(np.nanmean(moy))
        else:
            ini.append(np.nan)

    interval_output = (dates[0, 1] - dates[0, 0]) / np.timedelta64(1, 'D')
    # ini = np.array([(np.ma.mean(ini[i:i + 2])) for i in range(len(ini) - 1)])
    dates_ini = dates[:, 1] - m.ceil(interval_output / 2)
    if save_lines:
        return np.array([[dates_ini[z], ini[z]] for z in range(len(dates_ini))])
    else:
        return np.array([[dates_ini[z], ini[z]] for z in range(len(dates_ini))])


def find_granule_by_point(input_dict, input_point):  # [lon,lat]
    """Takes an input dictionary (a geojson catalog) and a point to represent AOI.
    this returns a list of the s3 urls corresponding to zarr datacubes whose footprint covers the AOI"""

    target_granule_urls = []

    point_geom = Point(input_point[0], input_point[1])
    point_gdf = gpd.GeoDataFrame(crs='epsg:4326', geometry=[point_geom])
    for granule in input_dict['features']:

        bbox_ls = granule['geometry']['coordinates'][0]
        bbox_geom = Polygon(bbox_ls)
        bbox_gdf = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[bbox_geom])

        if bbox_gdf.contains(point_gdf).all():
            target_granule_urls.append(granule['properties']['zarr_url'])
        else:
            pass
    return target_granule_urls
