# From Jeremie Mouginot

import jdcal
import numpy as np

# make jdcal external library data-type friendly (-datatime- and -integer- instead tuples)


def mjd2date(date):
    """
    Convert the dates from Modified Julian Date to Gregorian Date
    Modified Julian Date MJD = JD-2400001.5 , 17 November 1858=0.
    :param date: Modified Julian Date, integer
    :return:  Gregoryan date, datatime type
    """

    t = jdcal.jd2gcal(2400000.5, date)
    return np.datetime64(f"{t[0]}-{f'{t[1]:02}'}-{f'{t[2]:02}'}")


def date2mjd(date):
    """Convert from Gregorian Date to Modified Julian Date.
    :param date: Gregorian Date, datatime type
    :return:  Modified Julian Date, integer"""

    try:
        JD = jdcal.gcal2jd(date.year, date.month, date.day)
    except:
        date = date.astype("M8[D]").astype("O")
        JD = jdcal.gcal2jd(date.year, date.month, date.day)
    return int(JD[1])
