from ticoi.inversion_functions import construction_dates_range_np, construction_a_lf, inversion_one_component
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

        self.data = np.array([[  -0.69107729,   -8.73340321], [   2.40452456 , -13.41930866], [  -3.96273065 ,  -9.17936611], [   3.73120785 , -14.85955429], [  -2.19656491  , -9.20514107], [  10.38781738 , -28.12755966], [   3.23966694 , -17.77642059], [ 368.12982178 ,-118.80034637], [   1.13138795 , -11.47720432], [   2.95655584 , -19.49642754]])

        self.mu1accelnotnull = np.array([[-0.0625 ,     0.0625  ,    0.   ,       0.    ,      0.   ,       0.,   0.   ,       0.        ], [ 0.    ,     -0.0625  ,    0.01538462 , 0.     ,     0.    ,      0.,   0.      ,    0.        ], [ 0.   ,       0.   ,      -0.01538462 , 0.04166667 , 0.      ,    0.,   0.  ,        0.        ], [ 0.     ,     0.     ,     0.   ,      -0.04166667,  0.03225806 , 0.,   0.   ,       0.        ], [ 0.   ,       0.    ,      0.    ,      0.     ,    -0.03225806 , 0.015625,   0.   ,       0.        ], [ 0.    ,      0.    ,      0.     ,     0.    ,      0.     ,    -0.015625,   0.125     ,  0.        ], [ 0.    ,      0.   ,       0.     ,     0.   ,       0.    ,      0.,  -0.125  ,     0.125     ]]).astype('float32')

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
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8, err_msg=f"mu_regularisation does not give the correct result for regu={regu}")

    @pytest.mark.parametrize(
        "solver, expected, ini",
        [
            ('LSMR', np.array(
                [-7.55988695, -8.47012557, 53.40071386, -118.09971264, 52.38361652, 10.86749089, 5.53293348,
                 -7.3502032]).astype('float32'), None),
            ('LS', np.array(
                [-7.5791097, -8.460544, 113.54745, -118.18973, -7.585056, 10.759924, 5.560593, -7.3482523]).astype(
                'float32'), None),
            # ('LSMR_ini', np.array(
                # [[ -7.577303  ,  -8.461302  , 108.19255 ,  -118.181625   , -2.2458465,  10.769561   ,  5.558176  ,  -7.348404 ]]).astype('float32'), np.array([-7, -8., 100., -110., -7., 10., 5., -10.]).astype('float32'))
]
    )

    def test_inversion_one_component(self,solver, expected, ini):
        actual = inversion_one_component(self.A, self.dates_range, 1, self.data, solver=solver, Weight=1,
                                         mu=self.mu1accelnotnull, ini=ini)[0]
        print(actual)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4)