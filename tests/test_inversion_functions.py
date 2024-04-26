from ticoi.inversion_functions import Construction_dates_range_np, Construction_A_LP
import numpy as np

class Test_inversion:
    def setup_method(self):
        # This method will run before each test
        self.dates = np.array([
            ['2013-03-14', '2013-03-30'], ['2013-03-14', '2013-03-30'], ['2013-03-14', '2013-04-15'],
            ['2013-03-30', '2013-04-15'], ['2013-03-30', '2013-04-15'], ['2013-03-14', '2013-08-13'],
            ['2013-03-14', '2013-10-16'], ['2013-06-19', '2013-07' '-13'], ['2013-03-14', '2013-10-24'],
            ['2013-03-14', '2013-11-01']
        ]).astype('datetime64[D]')

        self.dates_range = np.array([
            '2013-03-14', '2013-03-30', '2013-04-15', '2013-06-19', '2013-07-13', '2013-08-13',
            '2013-10-16', '2013-10-24', '2013-11-01'
        ]).astype('datetime64[D]')



    def test_Construct_Dates_range(self):
        # Expected result
        expected_dates_range = self.dates_range

        # Run function under test
        result = Construction_dates_range_np(self.dates)

        # Assert the results are as expected
        np.testing.assert_array_equal(result, expected_dates_range)

    def test_Construction_A_LP(self):
        # Expected result
        expected = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ])

        # Run function under test
        actual = Construction_A_LP(self.dates, self.dates_range)

        # Assert the results are as expected
        np.testing.assert_array_equal(actual, expected, err_msg="Construct A LP does not give the correct result")

    # def test_Inversion_A_LP(self):

