import itertools
import time

import dask.array as da
import numpy as np
import xarray as xr

import ticoi.core as core
from ticoi.core import _assign_block_results, chunk_to_block, process_blocks_refine
from ticoi.cube_data_classxr import CubeDataClass
from ticoi.optimize_coefficient_functions import (
    _optimization_coordinates,
    _stable_ground_coordinates,
)


def _chunked_cube():
    shape = (20, 9, 10)
    chunks = (5, 4, 4)
    variables = {
        name: (("mid_date", "y", "x"), da.zeros(shape, chunks=chunks, dtype="float32"))
        for name in ("vx", "vy", "errorx", "errory")
    }
    ds = xr.Dataset(
        variables,
        coords={"mid_date": np.arange(shape[0]), "y": np.arange(shape[1]), "x": np.arange(shape[2])},
    )
    cube = CubeDataClass()
    cube.ds = ds
    cube.nx, cube.ny, cube.nz = shape[2], shape[1], shape[0]
    return cube


def test_chunk_to_block_counts_all_variables_and_full_time_axis():
    cube = _chunked_cube()
    ds = cube.ds.unify_chunks()
    tile_bytes = ds.isel(x=slice(0, 4), y=slice(0, 4)).nbytes
    blocks = chunk_to_block(cube, block_size=tile_bytes * 1.01 / 1024**3)

    coverage = np.zeros((cube.ny, cube.nx), dtype=np.uint8)
    for x_start, x_end, y_start, y_end in blocks:
        coverage[y_start:y_end, x_start:x_end] += 1
        assert x_end - x_start <= 4
        assert y_end - y_start <= 4

    np.testing.assert_array_equal(coverage, 1)


def test_chunk_to_block_uses_largest_spatial_chunks_for_budget():
    shape = (20, 10, 10)
    chunks = ((20,), (1, 8, 1), (1, 8, 1))
    variables = {
        name: (("mid_date", "y", "x"), da.zeros(shape, chunks=chunks, dtype="float32"))
        for name in ("vx", "vy", "errorx", "errory")
    }
    ds = xr.Dataset(
        variables,
        coords={"mid_date": np.arange(shape[0]), "y": np.arange(shape[1]), "x": np.arange(shape[2])},
    )
    cube = CubeDataClass()
    cube.ds = ds
    cube.update_dimension()
    max_tile_bytes = ds.isel(x=slice(0, 8), y=slice(0, 8)).nbytes

    blocks = chunk_to_block(cube, block_size=max_tile_bytes * 1.01 / 1024**3)
    assert blocks == [
        [0, 1, 0, 1],
        [1, 9, 0, 1],
        [9, 10, 0, 1],
        [0, 1, 1, 9],
        [1, 9, 1, 9],
        [9, 10, 1, 9],
        [0, 1, 9, 10],
        [1, 9, 9, 10],
        [9, 10, 9, 10],
    ]

    coverage = np.zeros((cube.ny, cube.nx), dtype=np.uint8)
    assembled = [None] * (cube.nx * cube.ny)
    for x_start, x_end, y_start, y_end in blocks:
        coverage[y_start:y_end, x_start:x_end] += 1
        assert ds.isel(x=slice(x_start, x_end), y=slice(y_start, y_end)).nbytes <= max_tile_bytes
        local_results = [x * cube.ny + y for x in range(x_start, x_end) for y in range(y_start, y_end)]
        _assign_block_results(
            assembled,
            local_results,
            cube.ny,
            x_start,
            y_start,
            x_end - x_start,
            y_end - y_start,
        )
    np.testing.assert_array_equal(coverage, 1)
    assert assembled == list(range(cube.nx * cube.ny))


def test_chunk_to_block_keeps_small_cube_whole():
    cube = _chunked_cube()
    assert chunk_to_block(cube, block_size=1) == [[0, cube.nx, 0, cube.ny]]


def test_assign_block_results_matches_pixel_loop_exactly():
    cube_ny = 7
    block_nx, block_ny = 3, 4
    x_start, y_start = 2, 1
    results = [object() for _ in range(block_nx * block_ny)]
    expected = [None] * (6 * cube_ny)
    for i, value in enumerate(results):
        row = i % block_ny + y_start
        col = int(np.floor(i / block_ny)) + x_start
        expected[col * cube_ny + row] = value

    actual = [None] * len(expected)
    _assign_block_results(actual, results, cube_ny, x_start, y_start, block_nx, block_ny)

    assert all(got is reference for got, reference in zip(actual, expected))


def test_stable_ground_coordinates_preserve_sel_order_for_both_dim_orders():
    x = np.array([30.0, 10.0, -5.0])
    y = np.array([8.0, 2.0, -4.0, -9.0])
    values_yx = np.array([[0, 1, 0], [2, 0, 1], [0, 0, 2], [1, 0, 0]], dtype=np.int8)
    for dims, values in ((("y", "x"), values_yx), (("x", "y"), values_yx.T)):
        flag = xr.Dataset({"flag": (dims, values)}, coords={"x": x, "y": y})
        expected = [
            (xv, yv) for xv in flag["x"].values for yv in flag["y"].values if flag.sel(x=xv, y=yv)["flag"].values == 0
        ]

        actual = _stable_ground_coordinates(flag)

        assert actual == expected


def test_nonstable_optimization_coordinates_are_lazy_and_ordered():
    cube = _chunked_cube()
    expected = list(itertools.product(cube.ds["x"].values, cube.ds["y"].values))

    actual, total = _optimization_coordinates(cube, None, "other")

    assert not isinstance(actual, list)
    assert total == cube.nx * cube.ny
    assert list(actual) == expected


def test_block_prefetch_can_be_disabled_without_changing_results(monkeypatch):
    events = []

    class FakeBlock:
        live = 0
        max_live = 0

        def __init__(self, x_start):
            type(self).live += 1
            type(self).max_live = max(type(self).max_live, type(self).live)
            self.nx = self.ny = 1
            self.ds = xr.Dataset(coords={"x": [x_start], "y": [0]})
            self.x_start = x_start

        def __del__(self):
            type(self).live -= 1

        def load_pixel(self, *args, **kwargs):
            if self.x_start == 0:
                time.sleep(0.1)
                events.append("processed_0")
            return self.x_start

    def fake_load_block(cube, x_start, x_end, y_start, y_end, flag=None):
        if x_start == 1:
            events.append("loaded_1")
        return FakeBlock(x_start), None, 0.0

    class InlineParallel:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, tasks):
            return [func(*args, **kwargs) for func, args, kwargs in tasks]

    monkeypatch.setattr(core, "chunk_to_block", lambda *args, **kwargs: [[0, 1, 0, 1], [1, 2, 0, 1]])
    monkeypatch.setattr(core, "load_block", fake_load_block)
    monkeypatch.setattr(core, "Parallel", InlineParallel)
    cube = CubeDataClass()
    cube.nx, cube.ny = 2, 1
    kwargs = {
        "proj": "EPSG:3413",
        "interpolation_load_pixel": "nearest",
        "solver": "LSMR",
        "regu": "1",
        "visual": False,
    }

    bounded = process_blocks_refine(cube, nb_cpu=1, returned="raw", inversion_kwargs=kwargs, prefetch_blocks=False)

    assert bounded == [0, 1]
    assert events.index("loaded_1") > events.index("processed_0")
    assert FakeBlock.max_live == 1


def test_uninitialized_block_solver_preserves_valid_and_masked_pixels(tmp_path):
    dates = np.datetime64("2020-01-01", "ns") + np.arange(9) * np.timedelta64(30, "D")
    values = np.full((8, 2, 2), 120.0, dtype="float32")
    values[:, 1, 1] = np.nan
    dataset = xr.Dataset(
        {
            "vx": (("mid_date", "y", "x"), values),
            "vy": (("mid_date", "y", "x"), values / 2),
            "date1": ("mid_date", dates[:-1]),
            "date2": ("mid_date", dates[1:]),
        },
        coords={"mid_date": dates[:-1] + np.timedelta64(15, "D"), "x": [500000, 500120], "y": [3100120, 3100000]},
        attrs={"proj4": "+proj=utm +zone=45 +datum=WGS84 +units=m +no_defs"},
    )
    path = tmp_path / "cube.nc"
    dataset.to_netcdf(path, engine="h5netcdf")
    cube = CubeDataClass()
    cube.load(str(path), chunks={})
    common = {"solver": "LSMR", "regu": "1", "proj": "EPSG:32645"}
    result = process_blocks_refine(
        cube,
        nb_cpu=1,
        prefetch_blocks=False,
        preData_kwargs=common,
        inversion_kwargs={
            **common,
            "path_save": str(tmp_path),
            "result_quality": None,
            "iteration": False,
            "detect_temporal_decorrelation": False,
            "redundancy": None,
        },
    )

    assert len(result) == 4
    for pixel in result[:3]:
        assert len(pixel) == 5
        np.testing.assert_allclose(pixel["vx"], 120, rtol=1e-5)
        np.testing.assert_allclose(pixel["vy"], 60, rtol=1e-5)
        assert (pixel["date2"] - pixel["date1"]).eq(np.timedelta64(30, "D")).all()
    assert result[3].empty
