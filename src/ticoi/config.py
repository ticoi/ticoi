"""JSON configuration loading and execution for the TICOI cube workflow.

The configuration layer deliberately contains only standard-library imports.  This
keeps ``ticoi show-config`` useful on machines where the scientific dependencies
are not available, and keeps validation ahead of loading a potentially large
cube.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


class ConfigError(ValueError):
    """Raised when a TICOI JSON configuration is invalid."""


_TOP_LEVEL_KEYS = frozenset({"input", "output", "filename", "load", "preprocess", "inversion", "processing"})

# These defaults follow examples/basic/python_script/cube_ticoi_demo.py.  Values
# are kept explicit in the effective configuration so that show-config output is
# a complete, reconstructible description of a run.
_LOAD_DEFAULTS: dict[str, Any] = {
    "chunks": {},
    "conf": False,
    "subset": None,
    "buffer": None,
    "pick_date": None,
    "pick_sensor": None,
    "pick_temp_bas": None,
    "proj": "EPSG:3413",
    "mask": None,
    "reproj_coord": False,
    "reproj_vel": False,
    "verbose": False,
}

_PREPROCESS_DEFAULTS: dict[str, Any] = {
    "smooth_method": "savgol",
    "s_win": 3,
    "t_win": 90,
    "sigma": 3,
    "order": 3,
    "unit": 365,
    "delete_outliers": None,
    # None intentionally disables the small-baseline-only statistics path.  The
    # current API's callable default is True for backwards compatibility, while
    # the cube demo does not select that behavior.
    "compute_delete_outliers_stats_on_small_baselines": None,
    "flag": None,
    "regu": "1accelnotnull",
    "solver": "LSMR_ini",
    "proj": "EPSG:3413",
    "velo_or_disp": "velo",
    "select_baseline": 180,
    "random_state": None,
    "verbose": True,
}

_INVERSION_DEFAULTS: dict[str, Any] = {
    "path_save": None,  # owned by top-level output; filled during normalization
    "solver": "LSMR_ini",
    "regu": "1accelnotnull",
    "coef": 100,
    "apriori_weight": True,
    "apriori_weight_in_second_iteration": False,
    "interpolation_load_pixel": "nearest",
    "iteration": True,
    "interval_output": 30,
    "proj": "EPSG:3413",
    "threshold_it": 0.1,
    "conf": False,
    "option_interpol": "spline",
    "redundancy": 5,
    "detect_temporal_decorrelation": True,
    "unit": 365,
    "result_quality": "X_contribution",
    "nb_max_iteration": 10,
    "delete_outliers": None,
    "linear_operator": False,
    "visual": False,
    "verbose": False,
}

_PROCESSING_DEFAULTS: dict[str, Any] = {
    "nb_cpu": 1,
    "block_size": 0.5,
    "returned": "interp",
    "verbose": False,
    "prefetch_blocks": True,
}

_SECTION_DEFAULTS = {
    "load": _LOAD_DEFAULTS,
    "preprocess": _PREPROCESS_DEFAULTS,
    "inversion": _INVERSION_DEFAULTS,
    "processing": _PROCESSING_DEFAULTS,
}

_REGULARIZATIONS = frozenset({"1accelnotnull", "1", "2", "directionxy"})
_SOLVERS = frozenset({"LSMR", "LSMR_ini", "LSQR", "LS", "L1"})
_INTERPOLATION_METHODS = frozenset({"spline", "spline_smooth", "nearest"})
_PIXEL_INTERPOLATION_METHODS = frozenset({"nearest", "linear"})
_QUALITY_OPTIONS = frozenset({"Norm_residual", "X_contribution"})
_OUTLIER_OPTIONS = frozenset(
    {"error", "magnitude", "median_magnitude", "z_score", "mz_score", "iqr", "median_angle", "vvc_angle", "flow_angle"}
)


def _section(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        raise ConfigError(f"{label} must be an object, not null")
    if not isinstance(value, Mapping):
        raise ConfigError(f"{label} must be a JSON object")
    return dict(value)


def _reject_unknown(values: Mapping[str, Any], allowed: set[str] | frozenset[str], label: str) -> None:
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        names = ", ".join(repr(name) for name in unknown)
        raise ConfigError(f"unknown option(s) in {label}: {names}")


def _merge_section(values: Mapping[str, Any], defaults: Mapping[str, Any], label: str) -> dict[str, Any]:
    _reject_unknown(values, defaults.keys(), label)
    result = copy.deepcopy(dict(defaults))
    for key, value in values.items():
        # Do not use ``value or default``: JSON false and null are meaningful
        # API values (notably linear_operator, redundancy, and outlier options).
        result[key] = copy.deepcopy(value)
    return result


def _same_shared_value(values: Mapping[str, Mapping[str, Any]], key: str, default: Any) -> Any:
    supplied = [(label, section[key]) for label, section in values.items() if key in section]
    if not supplied:
        return copy.deepcopy(default)
    first_label, first_value = supplied[0]
    for label, value in supplied[1:]:
        if value != first_value:
            raise ConfigError(f"conflicting shared option {key!r}: {first_label}={first_value!r} and {label}={value!r}")
    return copy.deepcopy(first_value)


def _is_remote(value: str) -> bool:
    parsed = urlparse(value)
    return bool(parsed.scheme and ("://" in value or parsed.scheme in {"s3", "gs", "http", "https"}))


def _path_value(value: Any, base_dir: Path, label: str, *, allow_remote: bool = True) -> str:
    if isinstance(value, Path):
        value = str(value)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{label} must be a non-empty path string")
    if _is_remote(value):
        if allow_remote:
            return value
        raise ConfigError(f"{label} must be a local path, not URL {value!r}")
    return (
        str((base_dir / Path(value).expanduser()).resolve())
        if not Path(value).is_absolute()
        else str(Path(value).expanduser().resolve())
    )


def _optional_path(value: Any, base_dir: Path, label: str, *, allow_remote: bool = True) -> Any:
    if value is None:
        return None
    return _path_value(value, base_dir, label, allow_remote=allow_remote)


def _expect_bool(value: Any, label: str, *, allow_none: bool = False) -> None:
    if allow_none and value is None:
        return
    if type(value) is not bool:
        raise ConfigError(f"{label} must be a JSON boolean" + (" or null" if allow_none else ""))


def _expect_int(value: Any, label: str, *, allow_none: bool = False, minimum: int | None = None) -> None:
    if allow_none and value is None:
        return
    if type(value) is not int:
        raise ConfigError(f"{label} must be a JSON integer" + (" or null" if allow_none else ""))
    if minimum is not None and value < minimum:
        raise ConfigError(f"{label} must be >= {minimum}")


def _expect_number(value: Any, label: str, *, allow_none: bool = False, positive: bool = False) -> None:
    if allow_none and value is None:
        return
    if type(value) not in (int, float):
        raise ConfigError(f"{label} must be a JSON number" + (" or null" if allow_none else ""))
    if positive and value <= 0:
        raise ConfigError(f"{label} must be > 0")


def _expect_list_or_none(value: Any, label: str) -> None:
    if value is not None and not isinstance(value, list):
        raise ConfigError(f"{label} must be a JSON array or null")


def _validate_values(config: dict[str, Any]) -> None:
    """Validate JSON types and values without importing TICOI's heavy stack."""
    load = config["load"]
    preprocess = config["preprocess"]
    inversion = config["inversion"]
    processing = config["processing"]

    if not isinstance(load["chunks"], (dict, str, int)) or type(load["chunks"]) is bool:
        raise ConfigError("load.chunks must be a JSON object, string, or integer")
    if isinstance(load["chunks"], str) and load["chunks"] != "auto":
        raise ConfigError("load.chunks string must be 'auto'")
    _expect_bool(load["conf"], "load.conf")
    _expect_bool(load["reproj_coord"], "load.reproj_coord")
    _expect_bool(load["reproj_vel"], "load.reproj_vel")
    _expect_bool(load["verbose"], "load.verbose")
    if not isinstance(load["proj"], str) or not load["proj"]:
        raise ConfigError("load.proj must be a non-empty string")
    if load["mask"] is not None and not isinstance(load["mask"], str):
        raise ConfigError("load.mask must be a path string or null")
    for key in ("subset", "buffer", "pick_date", "pick_sensor", "pick_temp_bas"):
        _expect_list_or_none(load[key], f"load.{key}")

    if not isinstance(preprocess["smooth_method"], str) or not preprocess["smooth_method"]:
        raise ConfigError("preprocess.smooth_method must be a non-empty string")
    for key in ("s_win", "t_win", "order"):
        _expect_int(preprocess[key], f"preprocess.{key}")
    _expect_int(preprocess["unit"], "preprocess.unit", minimum=1)
    _expect_number(preprocess["sigma"], "preprocess.sigma")
    if preprocess["delete_outliers"] is not None:
        if not isinstance(preprocess["delete_outliers"], dict):
            raise ConfigError("preprocess.delete_outliers must be an object or null")
        unknown_outliers = set(preprocess["delete_outliers"]) - _OUTLIER_OPTIONS
        if unknown_outliers:
            raise ConfigError(f"unknown preprocess.delete_outliers method(s): {sorted(unknown_outliers)!r}")
    _expect_number(
        preprocess["compute_delete_outliers_stats_on_small_baselines"],
        "preprocess.compute_delete_outliers_stats_on_small_baselines",
        allow_none=True,
        positive=True,
    )
    if preprocess["flag"] is not None and not isinstance(preprocess["flag"], str):
        raise ConfigError("preprocess.flag must be a path string or null")
    if preprocess["regu"] not in _REGULARIZATIONS:
        raise ConfigError(f"preprocess.regu must be one of {sorted(_REGULARIZATIONS)!r}; aliases are not accepted")
    if preprocess["solver"] not in _SOLVERS:
        raise ConfigError(f"preprocess.solver must be one of {sorted(_SOLVERS)!r}")
    if not isinstance(preprocess["proj"], str) or not preprocess["proj"]:
        raise ConfigError("preprocess.proj must be a non-empty string")
    if preprocess["velo_or_disp"] not in {"velo", "disp"}:
        raise ConfigError("preprocess.velo_or_disp must be 'velo' or 'disp'")
    _expect_int(preprocess["select_baseline"], "preprocess.select_baseline", allow_none=True)
    _expect_int(preprocess["random_state"], "preprocess.random_state", allow_none=True)
    _expect_bool(preprocess["verbose"], "preprocess.verbose")

    if inversion["path_save"] is not None and not isinstance(inversion["path_save"], str):
        raise ConfigError("inversion.path_save must be a path string or null")
    if inversion["solver"] not in _SOLVERS:
        raise ConfigError(f"inversion.solver must be one of {sorted(_SOLVERS)!r}")
    if inversion["regu"] not in _REGULARIZATIONS:
        raise ConfigError(f"inversion.regu must be one of {sorted(_REGULARIZATIONS)!r}; aliases are not accepted")
    _expect_int(inversion["coef"], "inversion.coef")
    _expect_bool(inversion["apriori_weight"], "inversion.apriori_weight")
    _expect_bool(inversion["apriori_weight_in_second_iteration"], "inversion.apriori_weight_in_second_iteration")
    if inversion["interpolation_load_pixel"] not in _PIXEL_INTERPOLATION_METHODS:
        raise ConfigError(f"inversion.interpolation_load_pixel must be one of {sorted(_PIXEL_INTERPOLATION_METHODS)!r}")
    _expect_bool(inversion["iteration"], "inversion.iteration")
    _expect_int(inversion["interval_output"], "inversion.interval_output", minimum=1)
    if not isinstance(inversion["proj"], str) or not inversion["proj"]:
        raise ConfigError("inversion.proj must be a non-empty string")
    _expect_number(inversion["threshold_it"], "inversion.threshold_it")
    _expect_bool(inversion["conf"], "inversion.conf")
    if inversion["option_interpol"] not in _INTERPOLATION_METHODS:
        raise ConfigError(f"inversion.option_interpol must be one of {sorted(_INTERPOLATION_METHODS)!r}")
    _expect_int(inversion["redundancy"], "inversion.redundancy", allow_none=True)
    _expect_bool(inversion["detect_temporal_decorrelation"], "inversion.detect_temporal_decorrelation")
    _expect_int(inversion["unit"], "inversion.unit", minimum=1)
    quality = inversion["result_quality"]
    if quality is not None:
        quality_values = quality if isinstance(quality, list) else [quality]
        if not isinstance(quality, (str, list)) or any(value not in _QUALITY_OPTIONS for value in quality_values):
            raise ConfigError(
                f"inversion.result_quality must contain only {sorted(_QUALITY_OPTIONS)!r}, a string, or null"
            )
    if isinstance(quality, list) and not all(isinstance(value, str) for value in quality):
        raise ConfigError("inversion.result_quality array values must be strings")
    _expect_int(inversion["nb_max_iteration"], "inversion.nb_max_iteration")
    if inversion["delete_outliers"] is not None:
        if not isinstance(inversion["delete_outliers"], dict):
            raise ConfigError("inversion.delete_outliers must be an object or null")
        unknown_outliers = set(inversion["delete_outliers"]) - _OUTLIER_OPTIONS
        if unknown_outliers:
            raise ConfigError(f"unknown inversion.delete_outliers method(s): {sorted(unknown_outliers)!r}")
    linear_operator = inversion["linear_operator"]
    if linear_operator is not None and type(linear_operator) is not bool and linear_operator != "fast":
        raise ConfigError("inversion.linear_operator must be false, true, 'fast', or null; aliases are not accepted")
    _expect_bool(inversion["visual"], "inversion.visual")
    _expect_bool(inversion["verbose"], "inversion.verbose")

    _expect_int(processing["nb_cpu"], "processing.nb_cpu", minimum=1)
    _expect_number(processing["block_size"], "processing.block_size", positive=True)
    returned = processing["returned"]
    if returned != "interp":
        raise ConfigError("processing.returned must be the string 'interp' for this runner")
    _expect_bool(processing["verbose"], "processing.verbose")
    _expect_bool(processing["prefetch_blocks"], "processing.prefetch_blocks")


def load_config(
    config: str | os.PathLike[str] | Mapping[str, Any], *, base_dir: str | os.PathLike[str] | None = None
) -> dict[str, Any]:
    """Load, resolve, and validate a JSON TICOI configuration.

    ``config`` may be a JSON file path or an already parsed mapping.  Relative
    paths are resolved against the JSON file's directory (or ``base_dir`` for a
    mapping).  The returned dictionary is effective configuration: all defaults
    are explicit and all local paths are absolute, so it can be dumped and
    loaded again without changing its meaning.
    """
    if isinstance(config, Mapping):
        raw = dict(config)
        config_base = Path(base_dir or os.getcwd()).expanduser().resolve()
    else:
        config_path = Path(config).expanduser().resolve()
        config_base = config_path.parent
        try:
            with config_path.open("r", encoding="utf-8") as stream:
                raw = json.load(stream)
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"invalid JSON in {config_path}: {exc.msg} (line {exc.lineno}, column {exc.colno})"
            ) from exc
        except OSError as exc:
            raise ConfigError(f"cannot read configuration {config_path}: {exc}") from exc

    if not isinstance(raw, Mapping):
        raise ConfigError("configuration root must be a JSON object")
    _reject_unknown(raw, _TOP_LEVEL_KEYS, "configuration")
    for required in ("input", "output"):
        if required not in raw:
            raise ConfigError(f"missing required configuration option: {required}")

    input_value = raw["input"]
    if isinstance(input_value, list):
        if not input_value:
            raise ConfigError("input must contain at least one path")
        input_effective: str | list[str] = [
            _path_value(value, config_base, f"input[{index}]") for index, value in enumerate(input_value)
        ]
    else:
        input_effective = _path_value(input_value, config_base, "input")
    output_effective = _path_value(raw["output"], config_base, "output", allow_remote=False)

    filename = raw.get("filename", "Time_series")
    if not isinstance(filename, str) or not filename:
        raise ConfigError("filename must be a non-empty string")
    if Path(filename).name != filename or filename in {".", ".."}:
        raise ConfigError("filename must be a file name, not a path")

    raw_sections: dict[str, dict[str, Any]] = {}
    for section_name in _SECTION_DEFAULTS:
        raw_sections[section_name] = _section(raw.get(section_name, {}), section_name)

    # These settings are passed to several TICOI APIs.  They have one effective
    # value, and a mismatch is an error instead of silently choosing a section.
    shared = {
        "proj": _same_shared_value(
            {name: raw_sections[name] for name in ("load", "preprocess", "inversion")},
            "proj",
            _LOAD_DEFAULTS["proj"],
        ),
        "regu": _same_shared_value(
            {name: raw_sections[name] for name in ("preprocess", "inversion")},
            "regu",
            _PREPROCESS_DEFAULTS["regu"],
        ),
        "solver": _same_shared_value(
            {name: raw_sections[name] for name in ("preprocess", "inversion")},
            "solver",
            _PREPROCESS_DEFAULTS["solver"],
        ),
        "unit": _same_shared_value(
            {name: raw_sections[name] for name in ("preprocess", "inversion")},
            "unit",
            _PREPROCESS_DEFAULTS["unit"],
        ),
        "delete_outliers": _same_shared_value(
            {name: raw_sections[name] for name in ("preprocess", "inversion")},
            "delete_outliers",
            _PREPROCESS_DEFAULTS["delete_outliers"],
        ),
        "conf": _same_shared_value(
            {name: raw_sections[name] for name in ("load", "inversion")},
            "conf",
            _LOAD_DEFAULTS["conf"],
        ),
    }

    effective: dict[str, Any] = {
        "input": input_effective,
        "output": output_effective,
        "filename": filename,
        "load": _merge_section(raw_sections["load"], _LOAD_DEFAULTS, "load"),
        "preprocess": _merge_section(raw_sections["preprocess"], _PREPROCESS_DEFAULTS, "preprocess"),
        "inversion": _merge_section(raw_sections["inversion"], _INVERSION_DEFAULTS, "inversion"),
        "processing": _merge_section(raw_sections["processing"], _PROCESSING_DEFAULTS, "processing"),
    }
    for section_name in ("load", "preprocess", "inversion"):
        effective[section_name]["proj"] = copy.deepcopy(shared["proj"])
    for section_name in ("preprocess", "inversion"):
        effective[section_name]["regu"] = copy.deepcopy(shared["regu"])
        effective[section_name]["solver"] = copy.deepcopy(shared["solver"])
        effective[section_name]["unit"] = copy.deepcopy(shared["unit"])
        effective[section_name]["delete_outliers"] = copy.deepcopy(shared["delete_outliers"])
    effective["load"]["conf"] = copy.deepcopy(shared["conf"])
    effective["inversion"]["conf"] = copy.deepcopy(shared["conf"])

    # Resolve path-valued API options after section merging.  Remote input URLs
    # remain untouched; local paths become absolute relative to the config.
    effective["load"]["mask"] = _optional_path(effective["load"]["mask"], config_base, "load.mask")
    effective["preprocess"]["flag"] = _optional_path(effective["preprocess"]["flag"], config_base, "preprocess.flag")

    nested_output = effective["inversion"]["path_save"]
    if nested_output is not None:
        nested_output = _path_value(nested_output, config_base, "inversion.path_save", allow_remote=False)
        if nested_output != output_effective:
            raise ConfigError("inversion.path_save conflicts with top-level output; output owns the save directory")
    effective["inversion"]["path_save"] = output_effective

    _validate_values(effective)
    return effective


def _output_path(output_dir: Path, filename: str) -> Path:
    return output_dir / f"{filename}.nc"


def run_config(
    config: str | os.PathLike[str] | Mapping[str, Any], *, base_dir: str | os.PathLike[str] | None = None
) -> Path:
    """Execute a validated cube configuration and write interpolation output."""
    effective = load_config(config, base_dir=base_dir)
    output_dir = Path(effective["output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = _output_path(output_dir, effective["filename"])
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {output_path}")

    # Deliberately lazy: show-config/help should not import xarray, dask, or
    # rasterio merely to inspect a JSON file.
    from ticoi.core import process_blocks_refine, save_cube_parameters
    from ticoi.cube_data_classxr import CubeDataClass
    from ticoi.cube_writer import CubeResultsWriter

    cube = CubeDataClass()
    cube.load(effective["input"], **copy.deepcopy(effective["load"]))
    first_date_interpol, last_date_interpol = cube.prepare_interpolation_date()

    preprocess_kwargs = copy.deepcopy(effective["preprocess"])
    inversion_kwargs = copy.deepcopy(effective["inversion"])
    inversion_kwargs["first_date_interpol"] = first_date_interpol
    inversion_kwargs["last_date_interpol"] = last_date_interpol
    processing_kwargs = copy.deepcopy(effective["processing"])
    returned = processing_kwargs.pop("returned")

    result = process_blocks_refine(
        cube,
        preData_kwargs=preprocess_kwargs,
        inversion_kwargs=inversion_kwargs,
        returned=returned,
        **processing_kwargs,
    )
    result = result["interp"] if isinstance(result, dict) else result

    source, sensor = save_cube_parameters(
        cube, effective["load"], preprocess_kwargs, inversion_kwargs, returned="interp"
    )
    written = CubeResultsWriter(cube).write_result_ticoi(
        result,
        source,
        sensor,
        result_quality=inversion_kwargs["result_quality"],
        filename=effective["filename"],
        savepath=effective["output"],
        verbose=inversion_kwargs["verbose"],
    )
    if isinstance(written, str):
        raise RuntimeError(f"TICOI interpolation writer failed: {written}")
    return output_path


__all__ = ["ConfigError", "load_config", "run_config"]
