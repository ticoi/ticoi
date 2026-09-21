import numpy as np
import pandas as pd
import pytest

from ticoi.core import interpolation_core, interpolation_to_data
from ticoi.interpolation_functions import reconstruct_common_ref


def _baseline_reconstruct_common_ref(result, second_date_list=None):
    if result.empty:
        length = 1 if second_date_list is None else len(second_date_list)
        nan_list = np.full(length, np.nan)
        second_dates = [np.nan] if second_date_list is None else second_date_list
        return pd.DataFrame(
            {
                "Ref_date": nan_list,
                "Second_date": second_dates,
                "dx": nan_list,
                "dy": nan_list,
                "xcount_x": nan_list,
                "xcount_y": nan_list,
            }
        )

    data = pd.DataFrame({"Ref_date": result["date1"][0], "Second_date": result["date2"]})
    for var in result.columns.difference(["date1", "date2"]):
        if var in [
            "result_dx",
            "result_dy",
            "xcount_x",
            "xcount_y",
            "error_x",
            "error_y",
            "xcount_z",
        ]:
            data[var] = result[var].values.cumsum()
    data = data.rename(columns={"result_dx": "dx", "result_dy": "dy"})

    if second_date_list is not None:
        tmp = pd.DataFrame(
            {
                "Ref_date": pd.NaT,
                "Second_date": second_date_list,
                **{var: np.nan for var in data.columns.difference(["Ref_date", "Second_date"])},
            }
        )
        positions = np.searchsorted(second_date_list, data["Second_date"].values)
        tmp.iloc[positions] = data.values
        return tmp
    return data


def _result_fixture():
    return pd.DataFrame(
        {
            "date1": pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-06"]),
            "date2": pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-10"]),
            "xcount_y": np.array([1, 2, 4], dtype=np.int16),
            "result_dy": np.array([0.5, np.nan, 1.5], dtype=np.float32),
            "ignored": ["a", "b", "c"],
            "result_dx": np.array([1.0, 2.0, 4.0], dtype=np.float32),
            "xcount_x": np.array([3, 5, 7], dtype=np.int16),
            "error_x": np.array([0.1, 0.2, 0.3], dtype=np.float64),
        },
        index=[4, 0, 7],
    )


@pytest.mark.parametrize("with_target_dates", [False, True])
def test_reconstruct_common_ref_matches_incremental_dataframe_build(with_target_dates):
    result = _result_fixture()
    target_dates = None
    if with_target_dates:
        target_dates = np.arange(
            np.datetime64("2020-01-03"),
            np.datetime64("2020-01-11"),
            np.timedelta64(1, "D"),
        )

    expected = _baseline_reconstruct_common_ref(result, target_dates)
    actual = reconstruct_common_ref(result, target_dates)

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_reconstruct_common_ref_empty_result_unchanged():
    result = _result_fixture().iloc[:0]
    expected = _baseline_reconstruct_common_ref(result)
    actual = reconstruct_common_ref(result)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_interpolation_quality_columns_and_padding_contract():
    n = 12
    dates = pd.date_range("2020-01-01", periods=n + 1, freq="5D")
    result = pd.DataFrame(
        {
            "date1": dates[:-1],
            "date2": dates[1:],
            "result_dx": np.linspace(0.1, 1.2, n),
            "result_dy": np.linspace(-0.2, 0.9, n),
            "xcount_x": np.linspace(1, 4, n),
            "xcount_y": np.linspace(2, 5, n),
            "error_x": np.linspace(0.01, 0.1, n),
            "error_y": np.linspace(0.02, 0.2, n),
            "sigma0": np.linspace(0.3, 0.6, n),
        }
    )

    actual = interpolation_core(
        result,
        interval_output=20,
        redundancy=5,
        option_interpol="spline",
        result_quality=["X_contribution", "Error_propagation"],
        first_date_interpol=np.datetime64("2019-12-20"),
        last_date_interpol=np.datetime64("2020-03-20"),
    )

    assert actual.columns.tolist() == [
        "date1",
        "date2",
        "vx",
        "vy",
        "xcount_x",
        "xcount_y",
        "error_x",
        "error_y",
        "sigma0",
    ]
    assert actual["date1"].iloc[0] == pd.Timestamp("2019-12-20")
    # Existing redundancy-grid semantics can extend beyond the requested end.
    assert actual["date2"].iloc[-1] == pd.Timestamp("2020-03-24")
    assert actual[["vx", "vy"]].iloc[0].isna().all()


def test_interpolation_to_data_preserves_day_floor_contract():
    dates = pd.date_range("2020-01-01", periods=7, freq="D")
    result = pd.DataFrame(
        {
            "date1": dates[:-1],
            "date2": dates[1:],
            "result_dx": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "result_dy": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        }
    )
    data = pd.DataFrame(
        {
            "date1": pd.to_datetime(["2020-01-02 12:00", "2020-01-03 23:00"]),
            "date2": pd.to_datetime(["2020-01-03 12:00", "2020-01-05 01:00"]),
            "temporal_baseline": [1, 2],
        }
    )

    actual = interpolation_to_data(result, data, option_interpol="nearest", unit=1)

    expected = pd.DataFrame(
        {
            "date1": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "date2": pd.to_datetime(["2020-01-03", "2020-01-05"]),
            "vx": [2.0, 3.5],
            "vy": [4.0, 7.0],
        }
    )
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_nonoverlapping_interpolation_preserves_velocity_and_pads_boundaries():
    dates = pd.date_range("2020-01-01", periods=9, freq="30D")
    result = pd.DataFrame(
        {"date1": dates[:-1], "date2": dates[1:], "result_dx": np.full(8, 30.0), "result_dy": np.full(8, -60.0)}
    )
    actual = interpolation_core(
        result,
        interval_output=30,
        redundancy=None,
        unit=1,
        first_date_interpol=dates[0].to_datetime64(),
        last_date_interpol=dates[-1].to_datetime64(),
    )

    np.testing.assert_array_equal(actual["date1"].values, dates[:-1].values)
    np.testing.assert_array_equal(actual["date2"].values, dates[1:].values)
    np.testing.assert_allclose(actual["vx"], [np.nan, np.nan, 1, 1, 1, 1, 1, np.nan])
    np.testing.assert_allclose(actual["vy"], [np.nan, np.nan, -2, -2, -2, -2, -2, np.nan])
