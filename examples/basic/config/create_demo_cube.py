#!/usr/bin/env python3
"""Create a tiny, noiseless velocity cube for the adjacent config.json example.

Run this script, then ``ticoi run examples/basic/config/config.json`` from the
repository root. No downloads or autoRIFT installation are required.
The four pixels move at vx=[[120, 130], [140, 150]] and vy=-35 m/year.
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=Path(__file__).with_name("demo_cube.nc"))
    args = parser.parse_args()
    dates = np.datetime64("2020-01-01", "ns") + np.arange(9) * np.timedelta64(30, "D")
    # Adjacent and overlapping pairs constrain the same constant velocity.
    starts = np.concatenate((np.arange(8), np.arange(6)))
    ends = np.concatenate((np.arange(1, 9), np.arange(3, 9)))
    date1, date2 = dates[starts], dates[ends]
    shape = (len(starts), 2, 2)
    vx = np.broadcast_to(np.array([[120, 130], [140, 150]], dtype="float32"), shape)
    cube = xr.Dataset(
        {
            "vx": (("mid_date", "y", "x"), vx),
            "vy": (("mid_date", "y", "x"), np.full(shape, -35, dtype="float32")),
            "errorx": (("mid_date", "y", "x"), np.ones(shape, dtype="float32")),
            "errory": (("mid_date", "y", "x"), np.ones(shape, dtype="float32")),
            "date1": ("mid_date", date1),
            "date2": ("mid_date", date2),
        },
        coords={
            "mid_date": date1 + (date2 - date1) // 2,
            "x": [500000.0, 500120.0],
            "y": [3100120.0, 3100000.0],
        },
        attrs={
            "proj4": "+proj=utm +zone=45 +datum=WGS84 +units=m +no_defs",
            "author": "TICOI configuration example",
            "source": "Synthetic constant velocities; not scientific observations",
            "sensor": "S2",
        },
    )
    for name in ("vx", "vy", "errorx", "errory"):
        cube[name].attrs["units"] = "m/year"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cube.to_netcdf(args.output, engine="h5netcdf", mode="w")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
