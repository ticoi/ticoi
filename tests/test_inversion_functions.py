import numpy as np
import pytest
import scipy.sparse as sp

from ticoi.core import inversion_core, mu_regularisation
from ticoi.inversion_functions import (
    class_fast_linear_operator,
    class_linear_operator,
    construction_a_lf,
    construction_dates_range_np,
    find_date_obs,
    inversion_one_component,
    inversion_two_components,
    mu_regularisation_sparse_first_order,
    weight_for_inversion,
)


class Test_inversion:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        # This method will run before each test
        self.dates = np.array(
            [
                ["2013-03-14", "2013-03-30"],
                ["2013-03-14", "2013-03-30"],
                ["2013-03-14", "2013-04-15"],
                ["2013-03-30", "2013-04-15"],
                ["2013-03-30", "2013-04-15"],
                ["2013-03-14", "2013-08-13"],
                ["2013-03-14", "2013-10-16"],
                ["2013-06-19", "2013-07-13"],
                ["2013-03-14", "2013-10-24"],
                ["2013-03-14", "2013-11-01"],
            ]
        ).astype("datetime64[D]")

        self.dates_range = np.array(
            [
                "2013-03-14",
                "2013-03-30",
                "2013-04-15",
                "2013-06-19",
                "2013-07-13",
                "2013-08-13",
                "2013-10-16",
                "2013-10-24",
                "2013-11-01",
            ]
        ).astype("datetime64[D]")

        self.A = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1, 1],
            ]
        )

        self.data = np.array(
            [
                [-0.69107729, -8.73340321],
                [2.40452456, -13.41930866],
                [-3.96273065, -9.17936611],
                [3.73120785, -14.85955429],
                [-2.19656491, -9.20514107],
                [10.38781738, -28.12755966],
                [3.23966694, -17.77642059],
                [368.12982178, -118.80034637],
                [1.13138795, -11.47720432],
                [2.95655584, -19.49642754],
            ]
        )

        self.mu1accelnotnull = np.array(
            [
                [-0.0625, 0.0625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -0.0625, 0.01538462, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -0.01538462, 0.04166667, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -0.04166667, 0.03225806, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -0.03225806, 0.015625, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -0.015625, 0.125, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.125, 0.125],
            ]
        ).astype("float32")

    def test_construct_dates_range(self):
        """Test construction of Dates_range for a small subset of values"""
        expected_dates_range = self.dates_range
        result = construction_dates_range_np(self.dates)
        np.testing.assert_array_equal(result, expected_dates_range)

    def test_construction_a_lf(self):
        """Test construction of A for a small subset of values"""

        expected = self.A
        actual = construction_a_lf(self.dates, self.dates_range)
        np.testing.assert_array_equal(actual, expected, err_msg="Construction A LP does not give the correct result")
        assert actual.dtype == np.int8

    @pytest.mark.parametrize("regu", ["1", "1accelnotnull"])
    def test_first_order_regularization_matches_full_matrix_baseline_exactly(self, regu):
        n_columns = self.A.shape[1]
        expected = np.diag(np.full(n_columns, -1, dtype="float32"))
        expected[np.arange(n_columns - 1), np.arange(n_columns - 1) + 1] = 1
        expected /= np.diff(self.dates_range) / np.timedelta64(1, "D")
        expected = np.delete(expected, -1, axis=0)

        actual = mu_regularisation(regu, self.A, self.dates_range)

        np.testing.assert_array_equal(actual, expected)

    def test_sparse_first_order_regularization_matches_dense_exactly(self):
        dense = mu_regularisation("1accelnotnull", self.A, self.dates_range)
        sparse = mu_regularisation_sparse_first_order(self.A.shape[1], self.dates_range)

        assert sp.isspmatrix_csc(sparse)
        np.testing.assert_array_equal(sparse.toarray(), dense)

    def test_second_order_regularization_matches_range_baseline_exactly(self):
        delta = np.diff(self.dates_range) / np.timedelta64(1, "D")
        n_columns = self.A.shape[1]
        expected = np.zeros((n_columns, n_columns), dtype="float64")
        expected[range(1, n_columns - 1), range(n_columns - 2)] = 1 / delta[:-2]
        expected[range(1, n_columns - 1), range(1, n_columns - 1)] = -2 / delta[1:-1]
        expected[range(1, n_columns - 1), range(2, n_columns)] = 1 / delta[2:]

        actual = mu_regularisation("2", self.A, self.dates_range)

        np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize("n_ini", [2, 4])
    def test_direction_regularization_matches_loop_baseline_exactly(self, n_ini):
        n_columns = self.A.shape[1]
        x = np.linspace(1.0, 3.0, n_columns)
        y = np.linspace(2.0, 4.0, n_columns)
        ini = [x, y] if n_ini == 2 else [x, y, x + 1, y + 2]
        delta = [
            (self.dates_range[k + 1] - self.dates_range[k]) / np.timedelta64(1, "D")
            for k in range(len(self.dates_range) - 1)
        ]
        expected = np.zeros((n_columns, 2 * n_columns), dtype="float64")
        if n_ini == 2:
            vv = np.array(ini[0]) ** 2 + np.array(ini[1]) ** 2
        else:
            vv = np.sqrt(ini[0] ** 2 + ini[1] ** 2) / 365 * np.sqrt(ini[2] ** 2 + ini[3] ** 2) / delta
        for k in range(n_columns):
            scale = 1 if n_ini == 2 else 365
            expected[k, k] = ini[0][k] / scale / int(delta[k]) / vv[k]
            expected[k, k + n_columns] = ini[1][k] / scale / int(delta[k]) / vv[k]

        actual = mu_regularisation("directionxy", self.A, self.dates_range, ini=ini)

        np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize(
        "regu, expected",
        [
            (
                "1",
                np.array(
                    [
                        [-0.0625, 0.0625, 0, 0, 0, 0, 0, 0],
                        [0, -0.0625, 0.01538462, 0, 0, 0, 0, 0],
                        [0, 0, -0.01538462, 0.04166667, 0, 0, 0, 0],
                        [0, 0, 0, -0.04166667, 0.03225806, 0, 0, 0],
                        [0, 0, 0, 0, -0.03225806, 0.015625, 0, 0],
                        [0, 0, 0, 0, 0, -0.015625, 0.125, 0],
                        [0, 0, 0, 0, 0, 0, -0.125, 0.125],
                    ]
                ).astype("float32"),
            ),  # Shortened for brevity
            (
                "2",
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0625, -0.125, 0.01538462, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0625, -0.03076923, 0.04166667, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.01538462, -0.08333333, 0.03225806, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.04166667, -0.06451613, 0.015625, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.03225806, -0.03125, 0.125, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.015625, -0.25, 0.125],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ).astype("float32"),
            ),
            (
                "1accelnotnull",
                np.array(
                    [
                        [-0.0625, 0.0625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, -0.0625, 0.01538462, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, -0.01538462, 0.04166667, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, -0.04166667, 0.03225806, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, -0.03225806, 0.015625, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, -0.015625, 0.125, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.125, 0.125],
                    ]
                ).astype("float32"),
            ),
        ],
    )
    def test_mu_regularization(self, regu, expected):
        """Test construction of mu for a three different regularization"""
        actual = mu_regularisation(regu, self.A, self.dates_range)
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-6,
            atol=1,
            err_msg=f"mu_regularisation does not give the correct result for regu={regu}",
        )

    @pytest.mark.parametrize(
        "solver, expected, ini",
        [
            (
                "LSMR",
                np.array(
                    [-7.578118, -8.461816, 108.818729, -118.183104, -2.871212, 10.768036, 5.558181, -7.348526]
                ).astype("float64"),
                None,
            ),
            (
                "LS",
                np.array(
                    [-7.5791097, -8.460544, 113.54745, -118.18973, -7.585056, 10.759924, 5.560593, -7.3482523]
                ).astype("float64"),
                None,
            ),
            # (
            #     "LSMR_ini",
            #     np.array(
            #         [[  -7.576902,   -8.461586,  106.760633, -118.179535,   -0.81826 ,10.772076,    5.557501,   -7.34847 ]]
            #     ).astype("float64"),
            #     np.array([-7, -8.0, 100.0, -110.0, -7.0, 10.0, 5.0, -10.0]).astype("float64"),
            # ),
        ],
    )
    def test_inversion_one_component(self, solver, expected, ini):
        actual = inversion_one_component(
            self.A, self.dates_range, 1, self.data, solver=solver, Weight=1, mu=self.mu1accelnotnull, ini=ini
        )[0]
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-6,
            atol=1,
        )

    def test_norm_residual_matches_weighted_boundary_baseline_exactly(self):
        weight = np.array([1.0, 0.5, 0.0, 0.8, 1.0, 0.0, 0.3, 1.0, 0.6, 1.0])
        x, actual = inversion_one_component(
            self.A,
            self.dates_range,
            1,
            self.data,
            solver="LSMR",
            Weight=weight,
            mu=self.mu1accelnotnull,
            coef=100,
            result_quality=["Norm_residual"],
        )
        keep = weight != 0
        f_regu = 100 * self.mu1accelnotnull
        f = sp.csc_matrix(np.vstack([weight[keep, None] * self.A[keep], f_regu]).astype("float64"))
        d = np.hstack([weight[keep] * self.data[keep, 1], np.zeros(f_regu.shape[0])]).astype("float64")
        residual = f.dot(x) - d
        old_boundary = np.multiply(weight[keep], self.data[keep, 1]).shape[0]
        expected = [
            np.linalg.norm(residual[:old_boundary], ord=2),
            np.linalg.norm(residual[old_boundary:] / 100, ord=2),
        ]

        np.testing.assert_array_equal(actual, expected)

    def test_cached_observation_csc_matches_rebuild_exactly(self):
        weight = np.array([1.0, 0.5, 0.0, 0.8, 1.0, 0.0, 0.3, 1.0, 0.6, 1.0])
        kwargs = dict(
            solver="LSMR",
            Weight=weight,
            mu=self.mu1accelnotnull,
            coef=100,
        )
        rebuilt = inversion_one_component(self.A, self.dates_range, 1, self.data, **kwargs)[0]
        cached = inversion_one_component(
            self.A,
            self.dates_range,
            1,
            self.data,
            A_csc=sp.csc_matrix(self.A, dtype="float64"),
            **kwargs,
        )[0]
        np.testing.assert_array_equal(cached, rebuilt, strict=True)

    def test_legacy_linear_operator_remains_equivalent_to_explicit_design(self):
        operator = class_linear_operator()
        intervals = find_date_obs(self.dates, self.dates_range)
        operator.load(intervals, self.dates_range, coef=100)
        x = np.linspace(-2.0, 3.0, self.A.shape[1])
        y = np.linspace(0.25, 1.25, self.A.shape[0])

        np.testing.assert_allclose(operator.matvec(x), self.A @ x, rtol=0, atol=1e-12)
        np.testing.assert_allclose(operator.rmatvec(y), self.A.T @ y, rtol=0, atol=1e-12)

    def test_interval_linear_operator_matches_explicit_design_and_adjoint(self):
        operator = class_fast_linear_operator()
        intervals = find_date_obs(self.dates, self.dates_range)
        operator.load(intervals, self.dates_range, coef=100)
        x = np.linspace(-2.0, 3.0, self.A.shape[1])
        y = np.linspace(0.25, 1.25, self.A.shape[0])

        np.testing.assert_allclose(operator.matvec(x), self.A @ x, rtol=0, atol=1e-12)
        np.testing.assert_allclose(operator.rmatvec(y), self.A.T @ y, rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            np.dot(operator.matvec(x), y),
            np.dot(x, operator.rmatvec(y)),
            rtol=0,
            atol=1e-12,
        )

    def test_interval_linear_operator_weighted_regularization_matches_explicit(self):
        operator = class_fast_linear_operator()
        intervals = find_date_obs(self.dates, self.dates_range)
        operator.load(intervals, self.dates_range, coef=100)
        weight = np.array([1.0, 0.5, 0.0, 0.8, 1.0, 0.0, 0.3, 1.0, 0.6, 1.0])
        condition = weight != 0
        operator.update_from_weight(np.ones(len(weight)), weight)
        x = np.linspace(-2.0, 3.0, self.A.shape[1])
        y = np.linspace(0.25, 1.25, condition.sum() + self.A.shape[1] - 1)
        mu = mu_regularisation("1accelnotnull", self.A, self.dates_range)
        explicit = np.vstack([weight[condition, None] * self.A[condition], 100 * mu])

        np.testing.assert_allclose(operator.matvecregu1(x), explicit @ x, rtol=0, atol=1e-12)
        np.testing.assert_allclose(operator.rmatvecregu1(y), explicit.T @ y, rtol=0, atol=1e-12)

    def test_interval_linear_operator_second_order_matches_explicit(self):
        operator = class_fast_linear_operator()
        intervals = find_date_obs(self.dates, self.dates_range)
        operator.load(intervals, self.dates_range, coef=100)
        weight = np.linspace(0.2, 1.0, self.A.shape[0])
        weight[::4] = 0
        condition = weight != 0
        operator.update_from_weight(np.ones(len(weight)), weight)
        x = np.linspace(-2.0, 3.0, self.A.shape[1])
        y = np.linspace(0.25, 1.25, condition.sum() + self.A.shape[1])
        mu = mu_regularisation("2", self.A, self.dates_range)
        explicit = np.vstack([weight[condition, None] * self.A[condition], 100 * mu])

        np.testing.assert_allclose(operator.matvecregu2(x), explicit @ x, rtol=0, atol=1e-12)
        np.testing.assert_allclose(operator.rmatvecregu2(y), explicit.T @ y, rtol=0, atol=1e-12)

    @pytest.mark.parametrize("n_unknowns", [1, 2, 17, 64])
    @pytest.mark.parametrize("regu", ["1", "2"])
    def test_interval_operator_randomized_shapes_and_adjoint(self, n_unknowns, regu):
        rng = np.random.default_rng(1000 + n_unknowns)
        day_steps = rng.integers(1, 40, size=n_unknowns)
        dates_range = np.datetime64("2000-01-01") + np.concatenate([[0], np.cumsum(day_steps)]).astype("timedelta64[D]")
        n_observations = max(5, 4 * n_unknowns)
        starts = rng.integers(0, n_unknowns, size=n_observations)
        ends = np.array([rng.integers(start, n_unknowns) for start in starts], dtype=np.int64)
        intervals = np.column_stack([starts, ends])
        explicit_a = np.zeros((n_observations, n_unknowns))
        for row, (start, end) in enumerate(intervals):
            explicit_a[row, start : end + 1] = 1
        weight = rng.uniform(0.1, 1.0, size=n_observations)
        weight[::7] = 0
        condition = weight != 0

        operator = class_fast_linear_operator()
        operator.load(intervals, dates_range, coef=37)
        operator.update_from_weight(np.ones(n_observations), weight)
        mu = mu_regularisation(regu, explicit_a, dates_range)
        explicit = np.vstack([weight[condition, None] * explicit_a[condition], 37 * mu])
        x = rng.normal(size=n_unknowns)
        y = rng.normal(size=explicit.shape[0])
        matvec = operator.matvecregu2 if regu == "2" else operator.matvecregu1
        rmatvec = operator.rmatvecregu2 if regu == "2" else operator.rmatvecregu1

        np.testing.assert_allclose(matvec(x), explicit @ x, rtol=0, atol=1e-11)
        np.testing.assert_allclose(rmatvec(y), explicit.T @ y, rtol=0, atol=1e-11)
        np.testing.assert_allclose(np.dot(matvec(x), y), np.dot(x, rmatvec(y)), rtol=0, atol=1e-10)

    @pytest.mark.parametrize("solver", ["LSMR", "LSMR_ini"])
    @pytest.mark.parametrize("regu", ["1", "2"])
    def test_interval_linear_operator_supports_iterative_solvers(self, solver, regu):
        operator = class_fast_linear_operator()
        intervals = find_date_obs(self.dates, self.dates_range)
        operator.load(intervals, self.dates_range, coef=100)
        weight = np.linspace(0.2, 1.0, self.A.shape[0])
        weight[::4] = 0
        mu = mu_regularisation(regu, self.A, self.dates_range)
        kwargs = dict(
            coef=100,
            solver=solver,
            regu=regu,
            result_quality=["Norm_residual"],
        )
        if solver == "LSMR_ini":
            kwargs["ini"] = np.linspace(-0.5, 0.5, self.A.shape[1])

        explicit, explicit_norm = inversion_one_component(
            self.A,
            self.dates_range,
            0,
            self.data,
            weight,
            mu,
            **kwargs,
        )
        interval, interval_norm = inversion_one_component(
            self.A,
            self.dates_range,
            0,
            self.data,
            weight,
            None,
            linear_operator=operator,
            **kwargs,
        )

        # Equivalent operators can stop LSMR at slightly different iterates.  The
        # initialized second-order solve needs the looser bound; keep the tighter
        # comparison for the other solver/regularization combinations.
        solution_rtol = 1e-5 if (solver, regu) == ("LSMR_ini", "2") else 1e-6
        solution_atol = 1e-6 if (solver, regu) == ("LSMR_ini", "2") else 1e-7
        np.testing.assert_allclose(interval, explicit, rtol=solution_rtol, atol=solution_atol)
        np.testing.assert_allclose(interval_norm, explicit_norm, rtol=1e-6, atol=1e-7)

    def test_interval_linear_operator_lsmr_matches_explicit_solution(self):
        operator = class_fast_linear_operator()
        intervals = find_date_obs(self.dates, self.dates_range)
        operator.load(intervals, self.dates_range, coef=100)
        weight = np.linspace(0.2, 1.0, self.A.shape[0])
        weight[::4] = 0
        initial = np.linspace(-0.5, 0.5, self.A.shape[1])
        accel = [np.zeros(self.A.shape[1] - 1), np.zeros(self.A.shape[1] - 1)]
        mu = mu_regularisation("1accelnotnull", self.A, self.dates_range)

        explicit = inversion_one_component(
            self.A,
            self.dates_range,
            0,
            self.data,
            weight,
            mu,
            coef=100,
            solver="LSMR_ini",
            ini=initial,
            regu="1accelnotnull",
            accel=accel,
        )[0]
        interval = inversion_one_component(
            self.A,
            self.dates_range,
            0,
            self.data,
            weight,
            None,
            coef=100,
            solver="LSMR_ini",
            ini=initial,
            regu="1accelnotnull",
            accel=accel,
            linear_operator=operator,
        )[0]

        np.testing.assert_allclose(interval, explicit, rtol=0, atol=1e-8)

    @pytest.mark.parametrize(
        "solver, regu",
        [
            ("LSMR", "1"),
            ("LSMR", "1accelnotnull"),
            ("LSMR", "2"),
            ("LSMR_ini", "1"),
            ("LSMR_ini", "1accelnotnull"),
            ("LSMR_ini", "2"),
        ],
    )
    def test_fast_inversion_core_supports_quality_outputs(self, solver, regu):
        temporal_baseline = (self.dates[:, 1] - self.dates[:, 0]) / np.timedelta64(1, "D")
        data_values = np.column_stack(
            [
                self.data,
                np.full((self.data.shape[0], 2), 2.0),
                temporal_baseline,
            ]
        )
        kwargs = dict(
            dates_range=self.dates_range,
            solver=solver,
            regu=regu,
            coef=100,
            iteration=True,
            detect_temporal_decorrelation=False,
            result_quality=["X_contribution", "Norm_residual", "Error_propagation"],
        )
        if solver == "LSMR_ini" or regu == "1accelnotnull":
            kwargs["mean"] = [
                np.zeros(self.A.shape[1]),
                np.zeros(self.A.shape[1]),
            ]

        explicit = inversion_core([self.dates.copy(), data_values.copy()], 0, 0, **kwargs)[1]
        fast = inversion_core(
            [self.dates.copy(), data_values.copy()],
            0,
            0,
            linear_operator="fast",
            **kwargs,
        )[1]

        assert fast[["date1", "date2"]].equals(explicit[["date1", "date2"]])
        tolerance = 1e-6
        numeric_columns = fast.columns.difference(["date1", "date2"])
        np.testing.assert_allclose(
            fast[numeric_columns],
            explicit[numeric_columns],
            rtol=tolerance,
            atol=tolerance,
        )

    def test_fast_iteration_limit_retries_explicit_path(self, monkeypatch):
        temporal_baseline = (self.dates[:, 1] - self.dates[:, 0]) / np.timedelta64(1, "D")
        data_values = np.column_stack(
            [
                self.data,
                np.full((self.data.shape[0], 2), 2.0),
                temporal_baseline,
            ]
        )
        kwargs = dict(
            dates_range=self.dates_range,
            solver="LSMR_ini",
            regu="1accelnotnull",
            coef=100,
            iteration=True,
            detect_temporal_decorrelation=False,
            result_quality=["X_contribution"],
            mean=[
                np.zeros(self.A.shape[1]),
                np.zeros(self.A.shape[1]),
            ],
        )
        explicit = inversion_core([self.dates.copy(), data_values.copy()], 0, 0, **kwargs)[1]
        original_lsmr = sp.linalg.lsmr
        observed_maxiters = []

        def force_iteration_limit(*args, **solver_kwargs):
            observed_maxiters.append((min(args[0].shape), solver_kwargs.get("maxiter")))
            result = list(original_lsmr(*args, **solver_kwargs))
            result[1] = 7
            return tuple(result)

        monkeypatch.setattr(sp.linalg, "lsmr", force_iteration_limit)
        unchecked_diagnostics = {}
        inversion_core(
            [self.dates.copy(), data_values.copy()],
            0,
            0,
            linear_operator="fast",
            fast_fallback_on_limit=False,
            diagnostics=unchecked_diagnostics,
            **kwargs,
        )
        assert unchecked_diagnostics["lsmr_limit_hits"] > 0
        assert "fast_operator_fallbacks" not in unchecked_diagnostics

        diagnostics = {}
        fast = inversion_core(
            [self.dates.copy(), data_values.copy()],
            0,
            0,
            linear_operator="fast",
            diagnostics=diagnostics,
            **kwargs,
        )[1]

        assert fast.equals(explicit)
        assert diagnostics["fast_operator_fallbacks"] == 1
        assert diagnostics["discarded_fast_lsmr_calls"] > 0
        assert diagnostics["lsmr_limit_hits"] > 0
        fast_budgets = [(default, maximum) for default, maximum in observed_maxiters if maximum is not None]
        assert fast_budgets
        assert all(maximum == 2 * default for default, maximum in fast_budgets)

    @pytest.mark.parametrize(
        "solver, regu",
        [
            ("LS", "1"),
            ("L1", "1"),
            ("LSQR", "1"),
            ("LSMR", "directionxy"),
            ("LSMR_ini", "directionxy"),
            ("LSQR", "directionxy"),
        ],
    )
    def test_fast_inversion_core_rejects_unsupported_systems(self, solver, regu):
        data_values = np.column_stack([self.data, np.ones((self.data.shape[0], 2)), np.full(10, 16)])

        with pytest.raises(ValueError, match="linear_operator='fast' supports"):
            inversion_core(
                [self.dates.copy(), data_values],
                0,
                0,
                dates_range=self.dates_range,
                solver=solver,
                regu=regu,
                linear_operator="fast",
            )

    def test_fast_inversion_core_matches_visual_weighted_robust_path(self):
        temporal_baseline = (self.dates[:, 1] - self.dates[:, 0]) / np.timedelta64(1, "D")
        data_values = np.column_stack(
            [
                self.data,
                np.linspace(1.0, 3.0, self.data.shape[0]),
                np.linspace(1.5, 4.0, self.data.shape[0]),
                temporal_baseline,
            ]
        )
        data_str = np.column_stack([np.full(10, "S2"), np.full(10, "ITS_LIVE")])
        mean = [np.zeros(self.A.shape[1]), np.zeros(self.A.shape[1])]
        kwargs = dict(
            dates_range=self.dates_range,
            solver="LSMR_ini",
            regu="1accelnotnull",
            coef=100,
            mean=mean,
            iteration=True,
            apriori_weight=True,
            apriori_weight_in_second_iteration=True,
            detect_temporal_decorrelation=True,
            result_quality=["X_contribution", "Norm_residual", "Error_propagation"],
            visual=True,
        )

        explicit = inversion_core(
            [self.dates.copy(), data_values.copy(), data_str.copy()],
            0,
            0,
            **kwargs,
        )
        fast = inversion_core(
            [self.dates.copy(), data_values.copy(), data_str.copy()],
            0,
            0,
            linear_operator="fast",
            **kwargs,
        )

        assert fast[1][["date1", "date2"]].equals(explicit[1][["date1", "date2"]])
        result_columns = fast[1].columns.difference(["date1", "date2"])
        np.testing.assert_allclose(fast[1][result_columns], explicit[1][result_columns], rtol=1e-6, atol=1e-6)
        data_columns = [
            "vx",
            "vy",
            "errorx",
            "errory",
            "weightinix",
            "weightiniy",
            "weightlastx",
            "weightlasty",
            "residux",
            "residuy",
            "NormR",
        ]
        np.testing.assert_allclose(fast[2][data_columns], explicit[2][data_columns], rtol=1e-6, atol=1e-6)

    def test_two_component_sparse_system_matches_dense_baseline_exactly(self, monkeypatch):
        weight = np.linspace(0.2, 1.0, 2 * self.A.shape[0])
        weight[::4] = 0
        mu = np.zeros((self.A.shape[1], 2 * self.A.shape[1]), dtype="float64")
        rows = np.arange(self.A.shape[1])
        mu[rows, rows] = 0.25
        mu[rows, rows + self.A.shape[1]] = 0.75
        block_a = np.block([[self.A, np.zeros_like(self.A)], [np.zeros_like(self.A), self.A]])
        keep = weight != 0
        expected_f = sp.csc_matrix(np.vstack([weight[keep, None] * block_a[keep], 3 * mu]).astype("float64"))
        velocity = np.concatenate([self.data[:, 0], self.data[:, 1]])
        expected_d = np.hstack([weight[keep] * velocity[keep], np.ones(mu.shape[0]) * 3]).astype("float64")

        def verify_lsmr(actual_f, actual_d, **kwargs):
            np.testing.assert_array_equal(actual_f.data, expected_f.data)
            np.testing.assert_array_equal(actual_f.indices, expected_f.indices)
            np.testing.assert_array_equal(actual_f.indptr, expected_f.indptr)
            np.testing.assert_array_equal(actual_d, expected_d)
            return (np.zeros(2 * self.A.shape[1]),)

        monkeypatch.setattr(sp.linalg, "lsmr", verify_lsmr)
        direction_data = np.column_stack([self.data[:, 0], self.data[:, 1], np.zeros((len(self.data), 2))])
        inversion_two_components(
            self.A,
            self.dates_range,
            0,
            direction_data,
            weight,
            mu,
            solver="LSMR",
            coef=3,
            show_L_curve=True,
        )

    def test_error_weights_are_finite_for_constant_and_missing_quality(self):
        constant = np.column_stack([np.zeros(4), np.zeros(4), [2.0, 2.0, np.nan, 2.0]])
        np.testing.assert_array_equal(
            weight_for_inversion("error", False, constant, 2),
            [1.0, 1.0, 0.0, 1.0],
        )
        missing = constant.copy()
        missing[:, 2] = np.nan
        np.testing.assert_array_equal(weight_for_inversion("error", False, missing, 2), np.ones(4))

    def test_direction_regularisation_leaves_undefined_zero_speed_rows_unconstrained(self):
        zeros = np.zeros(self.A.shape[1])
        mu = mu_regularisation("directionxy", self.A, self.dates_range, ini=[zeros, zeros])
        assert np.isfinite(mu).all()
        assert not mu.any()

    @pytest.mark.parametrize("solver", ["LSMR", "LSMR_ini"])
    @pytest.mark.parametrize("iteration", [False, True])
    def test_directionxy_runs_through_public_core(self, solver, iteration):
        baseline = (self.dates[:, 1] - self.dates[:, 0]) / np.timedelta64(1, "D")
        values = np.column_stack([self.data, np.ones((len(self.data), 2)), baseline])
        result = inversion_core(
            [self.dates.copy(), values],
            0,
            0,
            dates_range=self.dates_range,
            solver=solver,
            regu="directionxy",
            coef=1,
            iteration=iteration,
            mean=[np.ones(self.A.shape[1]), np.ones(self.A.shape[1])],
        )[1]
        assert result is not None
        assert np.isfinite(result[["result_dx", "result_dy"]].to_numpy()).all()

    @pytest.mark.parametrize("solver", ["LS", "LSQR", "L1"])
    def test_directionxy_supports_other_public_solvers(self, solver):
        baseline = (self.dates[:, 1] - self.dates[:, 0]) / np.timedelta64(1, "D")
        values = np.column_stack([self.data, np.ones((len(self.data), 2)), baseline])
        result = inversion_core(
            [self.dates.copy(), values],
            0,
            0,
            dates_range=self.dates_range,
            solver=solver,
            regu="directionxy",
            coef=1,
            iteration=False,
            mean=[np.ones(self.A.shape[1]), np.ones(self.A.shape[1])],
        )[1]
        assert result is not None
        assert np.isfinite(result[["result_dx", "result_dy"]].to_numpy()).all()

    def test_l1_runs_through_public_core(self):
        baseline = (self.dates[:, 1] - self.dates[:, 0]) / np.timedelta64(1, "D")
        values = np.column_stack([self.data, np.ones((len(self.data), 2)), baseline])
        result = inversion_core(
            [self.dates.copy(), values],
            0,
            0,
            dates_range=self.dates_range,
            solver="L1",
            regu="1",
            coef=1,
        )[1]
        assert result is not None
        assert np.isfinite(result[["result_dx", "result_dy"]].to_numpy()).all()

    def test_legacy_linear_operator_rejects_second_order_regularisation(self):
        baseline = (self.dates[:, 1] - self.dates[:, 0]) / np.timedelta64(1, "D")
        values = np.column_stack([self.data, np.ones((len(self.data), 2)), baseline])
        with pytest.raises(ValueError, match="legacy first-order operator"):
            inversion_core(
                [self.dates.copy(), values],
                0,
                0,
                dates_range=self.dates_range,
                solver="LSMR",
                regu="2",
                linear_operator=True,
            )
