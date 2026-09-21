import json

import pytest

from ticoi.config import ConfigError, load_config, run_config


def test_resolved_config_survives_relocation_and_preserves_api_values(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    path = source / "task.json"
    path.write_text(
        json.dumps(
            {
                "input": "cube.nc",
                "output": "results",
                "load": {"mask": "mask.gpkg"},
                "preprocess": {"proj": "EPSG:32645", "delete_outliers": None},
                "inversion": {"linear_operator": "fast", "iteration": False, "redundancy": None},
                "processing": {"prefetch_blocks": False},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    effective = load_config(path)
    assert effective["input"] == str(source / "cube.nc")
    assert effective["load"]["mask"] == str(source / "mask.gpkg")
    assert effective["output"] == str(source / "results")
    assert effective["inversion"]["proj"] == "EPSG:32645"
    assert effective["inversion"]["linear_operator"] == "fast"
    assert effective["inversion"]["iteration"] is False
    assert effective["inversion"]["redundancy"] is None
    assert effective["processing"]["prefetch_blocks"] is False
    relocated = tmp_path / "resolved.json"
    relocated.write_text(json.dumps(effective), encoding="utf-8")
    assert load_config(relocated) == effective


def test_config_rejects_unknown_and_conflicting_settings(tmp_path):
    base = {"input": "cube.nc", "output": "results"}
    with pytest.raises(ConfigError):
        load_config({**base, "inversion": {"linear_opertor": "fast"}}, base_dir=tmp_path)
    with pytest.raises(ConfigError):
        load_config(
            {**base, "preprocess": {"solver": "LSMR"}, "inversion": {"solver": "LSMR_ini"}},
            base_dir=tmp_path,
        )
    with pytest.raises(ConfigError):
        load_config({**base, "inversion": {"interval_output": 0}}, base_dir=tmp_path)


def test_existing_result_is_not_overwritten_or_silently_renamed(tmp_path):
    result = tmp_path / "result.nc"
    result.write_bytes(b"existing scientific result")
    with pytest.raises(RuntimeError):
        run_config({"input": "missing.nc", "output": str(tmp_path), "filename": "result"})
    assert result.read_bytes() == b"existing scientific result"
    assert not (tmp_path / "result_1.nc").exists()
