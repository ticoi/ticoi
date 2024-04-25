from ticoi.inversion_functions import Construction_dates_range_np
import numpy as np

def test_Construct_Dates_range():
    dates = np.array([['2013-03-14', '2013-03-30'], ['2013-03-14', '2013-03-30'], ['2013-03-14', '2013-04-15'], ['2013-03-30', '2013-04-15'],
     ['2013-03-30', '2013-04-15'], ['2013-03-14', '2013-08-13'], ['2013-03-14', '2013-10-16'], ['2013-06-19', '2013-07-13'],
     ['2013-03-14', '2013-10-24'], ['2013-03-14', '2013-11-01']]).astype('datetime64[D]')
    assert (Construction_dates_range_np(dates) == np.array(['2013-03-14', '2013-03-30', '2013-04-15', '2013-06-19', '2013-07-13', '2013-08-13', '2013-10-16', '2013-10-24', '2013-11-01']).astype('datetime64[D]')).all()

# def test_Construction_A_LP(dates, dates_range):

