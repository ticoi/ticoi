import asyncio
import itertools

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from ticoi.core import chunk_to_block, load_block
from ticoi.utils import optimize_coef


async def process_block(
    cube,
    block,
    cube_gt,
    load_pixel_kwargs,
    inversion_kwargs,
    interpolation_kwargs,
    optimization_method="stable_ground",
    regu=None,
    flag=None,
    cmin=10,
    cmax=1000,
    step=10,
    coefs=None,
    nb_cpu=8,
    preData_kwargs=None,
):
    """Optimize the coef on a given block"""

    if optimization_method == "stable_ground":  # We only compute stable ground pixels
        xy_values = list(
            filter(
                bool,
                [
                    (x, y) if flag.sel(x=x, y=y)["flag"].values == 0 else False
                    for x in flag["x"].values
                    for y in flag["y"].values
                ],
            )
        )
    else:
        xy_values = list(itertools.product(cube.ds["x"].values, cube.ds["y"].values))

    # Progression bar
    xy_values_tqdm = tqdm(xy_values, total=len(xy_values))

    # Filter cube
    obs_filt, flag_block = block.filter_cube_before_inversion(**preData_kwargs, flag=flag)

    # Optimization of the coefficient for every pixels of the block
    #    (faster using parallelization here and sequential processing in optimize_coef)
    result_block = Parallel(n_jobs=nb_cpu, verbose=0)(
        delayed(optimize_coef)(
            block,
            cube_gt,
            i,
            j,
            obs_filt,
            load_pixel_kwargs,
            inversion_kwargs,
            interpolation_kwargs,
            method=optimization_method,
            regu=regu,
            flag=flag_block,
            cmin=cmin,
            cmax=cmax,
            step=step,
            coefs=coefs,
            stats=True,
            parallel=False,
            visual=False,
        )
        for i, j in xy_values_tqdm
    )

    return result_block


async def process_blocks_main(
    cube,
    cube_gt,
    load_pixel_kwargs,
    inversion_kwargs,
    preData_kwargs,
    interpolation_kwargs,
    optimization_method="stable_ground",
    regu=None,
    flag=None,
    cmin=10,
    cmax=1000,
    step=10,
    coefs=None,
    nb_cpu=8,
    block_size=0.5,
    verbose=False,
):
    """Main function for the optimization of the coef using a block processing approach"""

    blocks = chunk_to_block(cube, block_size=block_size, verbose=True)

    dataf_list = [None] * (cube.nx * cube.ny)

    loop = asyncio.get_event_loop()

    for n in range(len(blocks)):
        print(f"Processing block {n + 1}/{len(blocks)}")

        # Load the first block and start the loop
        if n == 0:
            x_start, x_end, y_start, y_end = blocks[0]
            future = loop.run_in_executor(None, load_block, cube, x_start, x_end, y_start, y_end)

        block, block_flag, duration = await future
        if verbose:
            print(f"Block {n + 1} loaded in {duration:.2f} s")

        if n < len(blocks) - 1:
            # Load the next block while processing the current block
            x_start, x_end, y_start, y_end = blocks[n + 1]
            future = loop.run_in_executor(None, load_block, cube, x_start, x_end, y_start, y_end)

        block_result = await process_block(
            cube,
            block,
            cube_gt,
            load_pixel_kwargs,
            inversion_kwargs,
            interpolation_kwargs,
            optimization_method=optimization_method,
            regu=regu,
            flag=flag,
            cmin=cmin,
            cmax=cmax,
            step=step,
            coefs=coefs,
            nb_cpu=nb_cpu,
            preData_kwargs=preData_kwargs,
        )

        for i in range(len(block_result)):
            row = i % block.ny + blocks[n][2]
            col = np.floor(i / block.ny) + blocks[n][0]
            idx = int(col * cube.ny + row)

            dataf_list[idx] = block_result[i]

        del block_result, block

    return dataf_list


def find_good_coefs(
    coefs,
    measures,
    method="stable_ground",
    select_method="min-max relative",
    thresh=None,
    mean_disp=None,
    mean_angle=None,
    visual=False,
):
    if select_method == "curvature":
        smooth_measures = [
            measures[i - 1] / 4 + measures[i] / 2 + measures[i - 1] / 4 for i in range(1, len(measures) - 1)
        ]
        accel = np.array(
            [
                (smooth_measures[i + 1] - 2 * smooth_measures[i] + smooth_measures[i - 1])
                / ((coefs[i + 2] - coefs[i]) / 2) ** 2
                for i in range(1, len(coefs) - 3)
            ]
        )

        if visual:
            plt.plot(coefs[2:-2], accel)
            plt.ylim([-2 * 10**-6, 0.5 * 10**-6])
            plt.show()

        if method in ["ground_truth", "stable_ground"]:
            best_coef = coefs[np.argmin(measures)]
            best_measure = np.min(measures)
            good_measure = measures[np.argmin(accel) + 2]
            good_coefs = coefs[measures < good_measure]
        elif method == "vvc":
            best_coef = coefs[np.argmax(measures)]
            best_measure = np.max(measures)
            good_measure = measures[np.argmax(accel) + 2]
            good_coefs = coefs[measures > good_measure]

    elif select_method == "min-max relative":
        if method in ["ground_truth", "stable_ground"]:
            best_measure = np.min(measures)
            good_measure = best_measure + (1 - thresh / 100) * (np.max(measures) - best_measure)
            best_coef = coefs[np.argmin(measures)]
            good_coefs = coefs[measures < good_measure]
        elif method == "vvc":
            best_measure = np.max(measures)
            good_measure = best_measure - (1 - thresh / 100) * (best_measure - np.min(measures))
            best_coef = coefs[np.argmax(measures)]
            good_coefs = coefs[measures > good_measure]

    elif select_method == "max relative":
        if method in ["ground_truth", "stable_ground"]:
            best_measure = np.min(measures)
            good_measure = (1 + thresh / 100) * best_measure
            best_coef = coefs[np.argmin(measures)]
            good_coefs = coefs[measures < good_measure]
        elif method == "vvc":
            best_measure = np.max(measures)
            good_measure = (1 - thresh / 100) * best_measure
            best_coef = coefs[np.argmax(measures)]
            good_coefs = coefs[measures > good_measure]

    elif select_method == "absolute":
        if method in ["ground_truth", "stable_ground"]:
            best_measure = np.min(measures)
            good_measure = best_measure + thresh
            best_coef = coefs[np.argmin(measures)]
            good_coefs = coefs[measures < good_measure]
        elif method == "vvc":
            best_measure = np.max(measures)
            good_measure = best_measure - thresh
            best_coef = coefs[np.argmax(measures)]
            good_coefs = coefs[measures > good_measure]

    elif select_method == "vvc_angle_thresh":
        assert mean_angle is not None, (
            "Mean angle to median direction over the area must be given for 'vvc_angle_thresh' selection method"
        )

        dVVC = abs(-np.sin(mean_angle) / (2 * np.sqrt(2 * (1 + np.cos(mean_angle)))) * thresh)
        if visual:
            print(mean_angle * 360 / (2 * np.pi))
            print(dVVC)

        return find_good_coefs(coefs, measures, method="vvc", select_method="absolute", thresh=dVVC)

    elif select_method == "vvc_disp_thresh":
        assert mean_disp is not None, (
            "Mean displacements over the area must be given for 'vvc_disp_thresh' selection method"
        )

        vx, vy = mean_disp
        v0 = np.array([vx - thresh, vy + thresh])
        v1 = np.array([vx + thresh, vy - thresh])
        d_angle = np.arccos((v0[0] * v1[0] + v0[1] * v1[1]) / (np.linalg.norm(v0) * np.linalg.norm(v1)))

        if visual:
            print(mean_disp)
            print(d_angle * 360 / (2 * np.pi))

        return find_good_coefs(
            coefs,
            measures,
            method == "vvc",
            select_method="vvc_angle_thresh",
            thresh=d_angle,
            mean_angle=mean_angle,
            visual=visual,
        )

    return good_measure, good_coefs, best_measure, best_coef
