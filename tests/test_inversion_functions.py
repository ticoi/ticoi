from ticoi.inversion_functions import construction_dates_range_np, construction_a_lf
from ticoi.core import mu_regularisation
import numpy as np
import pytest

class Test_inversion:

    @pytest.fixture(autouse=True)
    def setup_method(self):
        # This method will run before each test
        self.dates = np.array([
            ['2013-03-14', '2013-03-30'], ['2013-03-14', '2013-03-30'], ['2013-03-14', '2013-04-15'],
            ['2013-03-30', '2013-04-15'], ['2013-03-30', '2013-04-15'], ['2013-03-14', '2013-08-13'],
            ['2013-03-14', '2013-10-16'], ['2013-06-19', '2013-07-13'], ['2013-03-14', '2013-10-24'],
            ['2013-03-14', '2013-11-01']
        ]).astype('datetime64[D]')

        self.dates_range = np.array([
            '2013-03-14', '2013-03-30', '2013-04-15', '2013-06-19', '2013-07-13', '2013-08-13',
            '2013-10-16', '2013-10-24', '2013-11-01'
        ]).astype('datetime64[D]')

        self.A = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ])

        # self.mu1 =

    def test_construct_dates_range(self):
        """ Test construction of Dates_range for a small subset of values"""
        expected_dates_range = self.dates_range
        result = construction_dates_range_np(self.dates)
        np.testing.assert_array_equal(result, expected_dates_range)

    def test_construction_a_lf(self):
        """ Test construction of A for a small subset of values"""

        expected = self.A
        actual = construction_a_lf(self.dates, self.dates_range)
        np.testing.assert_array_equal(actual, expected, err_msg="Construction A LP does not give the correct result")

    @pytest.mark.parametrize("regu, expected", [
        (1, np.array([[-0.0625, 0.0625, 0, 0, 0, 0, 0, 0], [0, -0.0625, 0.01538462, 0, 0, 0, 0, 0],
                             [0, 0, -0.01538462, 0.04166667, 0, 0, 0, 0], [0, 0, 0, -0.04166667, 0.03225806, 0, 0, 0],
                             [0, 0, 0, 0, -0.03225806, 0.015625, 0, 0], [0, 0, 0, 0, 0, -0.015625, 0.125, 0],
                             [0, 0, 0, 0, 0, 0, -0.125, 0.125]]).astype('float32')),  # Shortened for brevity
        (2, np.array([[ 0. ,         0. ,         0.  ,        0. ,         0.   ,       0.,   0.  ,        0.        ], [ 0.0625  ,   -0.125 ,      0.01538462 , 0.   ,       0.  ,        0.,   0. ,         0.        ], [ 0.  ,        0.0625 ,    -0.03076923 , 0.04166667 , 0.     ,     0.,   0.   ,       0.        ], [ 0.  ,        0.       ,   0.01538462 ,-0.08333333,  0.03225806 , 0.,   0.  ,        0.        ], [ 0.   ,       0.     ,     0.   ,       0.04166667 ,-0.06451613,  0.015625,   0.     ,     0.        ], [ 0.     ,     0.     ,     0.      ,    0.    ,      0.03225806, -0.03125,   0.125  ,     0.        ], [ 0.  ,        0.     ,     0.     ,     0.      ,    0.    ,      0.015625,  -0.25    ,    0.125     ], [ 0.      ,    0.   ,       0.  ,        0.      ,    0.    ,      0.,   0.   ,       0.        ]]).astype('float32')),
        ('1accelnotnull',np.array([[-0.0625 ,     0.0625  ,    0.   ,       0.    ,      0.   ,       0.,   0.   ,       0.        ], [ 0.    ,     -0.0625  ,    0.01538462 , 0.     ,     0.    ,      0.,   0.      ,    0.        ], [ 0.   ,       0.   ,      -0.01538462 , 0.04166667 , 0.      ,    0.,   0.  ,        0.        ], [ 0.     ,     0.     ,     0.   ,      -0.04166667,  0.03225806 , 0.,   0.   ,       0.        ], [ 0.   ,       0.    ,      0.    ,      0.     ,    -0.03225806 , 0.015625,   0.   ,       0.        ], [ 0.    ,      0.    ,      0.     ,     0.    ,      0.     ,    -0.015625,   0.125     ,  0.        ], [ 0.    ,      0.   ,       0.     ,     0.   ,       0.    ,      0.,  -0.125  ,     0.125     ]]).astype('float32'))])

    def test_mu_regularization(self,regu,expected):
        """ Test construction of mu for a three different regularization"""
        actual = mu_regularisation(regu, self.A, self.dates_range)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8, err_msg="mu_regularisation does not give the correct result for regu={}".format(regu))


