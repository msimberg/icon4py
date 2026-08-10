# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Bit-exact parity test for the EXCLAIM_APE_AES experiment.

Runs the new eDSL driver and the plain-Python driver against golden outputs
recorded once from the old driver at commit f62c4e835f. All three runs use
distinct output paths and are compared bit-exact (atol=0, rtol=0).
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
from typing import Any

import gt4py.next.typing as gtx_typing
import numpy as np
import pytest

from icon4py.model.common import model_backends
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.states import prognostic_state as prognostics, tracer_states
from icon4py.model.common.utils import Pair, PredictorCorrectorPair, TimeStepPair
from icon4py.model.standalone_driver import config as driver_config, driver_utils, standalone_driver
from icon4py.model.testing import datatest_utils as dt_utils, definitions as test_defs, grid_utils

from ..fixtures import backend, download_ser_data, process_props


@pytest.fixture(params=[test_defs.Experiments.EXCLAIM_APE_AES], ids=lambda r: r.name)
def experiment_description(request: pytest.FixtureRequest) -> test_defs.ExperimentDescription:
    """Run the parity test only on the EXCLAIM_APE_AES experiment."""
    return request.param


_GOLDEN_DIR = pathlib.Path(os.environ.get("ICON4PY_TEST_DATA_PATH", "/tmp")) / "phase5_golden"


def _golden_file(name: str) -> pathlib.Path:
    return _GOLDEN_DIR / f"{name}.npz"


def _field_equality(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is right
    if hasattr(left, "ndarray") and hasattr(right, "ndarray"):
        return np.array_equal(left.ndarray, right.ndarray)
    if isinstance(left, PredictorCorrectorPair) and isinstance(right, PredictorCorrectorPair):
        return _field_equality(left.predictor, right.predictor) and _field_equality(
            left.corrector, right.corrector
        )
    if isinstance(left, Pair) and isinstance(right, Pair):
        return _field_equality(left.frozen_first, right.frozen_first) and _field_equality(
            left.frozen_second, right.frozen_second
        )
    if dataclasses.is_dataclass(left) and dataclasses.is_dataclass(right):
        left_fields = {f.name for f in dataclasses.fields(left)}
        right_fields = {f.name for f in dataclasses.fields(right)}
        if left_fields != right_fields:
            return False
        return all(
            _field_equality(getattr(left, field.name), getattr(right, field.name))
            for field in dataclasses.fields(left)
        )
    return left == right


def _assert_fields_equal(name: str, left: object, right: object) -> None:
    assert _field_equality(left, right), f"Field '{name}' differs."


def _array_equal_to_field(name: str, array: np.ndarray, field: object) -> None:
    if hasattr(field, "ndarray"):
        assert np.array_equal(array, np.asarray(field.ndarray)), f"Field '{name}' differs."
    else:
        assert np.array_equal(array, np.asarray(field)), f"Field '{name}' differs."


def _compare_dataclass_to_golden(
    name: str,
    golden: dict[str, np.ndarray],
    prefix: str,
    obj: Any,
) -> None:
    for field in dataclasses.fields(obj):
        key = f"{prefix}{field.name}"
        value = getattr(obj, field.name)
        if value is None:
            continue
        if isinstance(value, PredictorCorrectorPair):
            predictor_key = f"{key}_predictor"
            corrector_key = f"{key}_corrector"
            if predictor_key not in golden:
                pytest.fail(f"Golden file missing key '{predictor_key}' for {name}.{field.name}")
            if corrector_key not in golden:
                pytest.fail(f"Golden file missing key '{corrector_key}' for {name}.{field.name}")
            _array_equal_to_field(
                f"{name}.{field.name}.predictor", golden[predictor_key], value.predictor
            )
            _array_equal_to_field(
                f"{name}.{field.name}.corrector", golden[corrector_key], value.corrector
            )
            continue
        if key not in golden:
            pytest.fail(f"Golden file missing key '{key}' for {name}.{field.name}")
        _array_equal_to_field(f"{name}.{field.name}", golden[key], value)


def _compare_prognostic_state_to_golden(
    name: str,
    golden: dict[str, np.ndarray],
    prefix: str,
    state: prognostics.PrognosticState,
) -> None:
    _compare_dataclass_to_golden(f"{name}", golden, prefix, state)


def _compare_tracer_state_to_golden(
    name: str,
    golden: dict[str, np.ndarray],
    prefix: str,
    state: tracer_states.TracerState,
) -> None:
    for tracer in state.active_fields():
        key = f"{prefix}{tracer.name}"
        if key not in golden:
            pytest.fail(f"Golden file missing key '{key}' for {name}.{tracer.name}")
        _array_equal_to_field(f"{name}.{tracer.name}", golden[key], tracer.field)


def _compare_time_step_pair_to_golden(
    name: str,
    golden: dict[str, np.ndarray],
    pair: TimeStepPair[Any],
    saver: Any,
) -> None:
    saver(f"{name}.current", golden, "current_", pair.current)
    saver(f"{name}.next", golden, "next_", pair.next)


def _load_golden(group_name: str) -> dict[str, np.ndarray]:
    path = _golden_file(group_name)
    if not path.exists():
        pytest.skip(f"Golden output not found: {path}. Run the legacy recording script first.")
    with np.load(path) as data:
        return dict(data)


def _run_driver(
    *,
    config: driver_config.ExperimentConfig,
    grid_manager: driver_utils.gm.GridManager,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend | None,
    plain: bool,
) -> tuple[standalone_driver.driver_states.DriverStates, standalone_driver.Icon4pyDriver]:
    runner = standalone_driver.run_driver_plain if plain else standalone_driver.run_driver
    ds, driver = runner(
        config=config,
        grid_manager=grid_manager,
        process_props=process_props,
        backend=backend,
    )
    return ds, driver


def _compare_static_fields_to_golden(
    name: str,
    golden: dict[str, np.ndarray],
    static_factories: Any,
) -> None:
    for factory_name in ("geometry", "interpolation", "metrics"):
        factory = getattr(static_factories, factory_name)
        for quantity in factory._attrs:
            key = f"{factory_name}_{quantity}"
            if key not in golden:
                pytest.fail(f"Golden file missing key '{key}' for static field {key}")
            _array_equal_to_field(key, golden[key], factory.get(quantity))


def _assert_static_fields_equal(name: str, left: Any, right: Any) -> None:
    for factory_name in ("geometry", "interpolation", "metrics"):
        left_factory = getattr(left, factory_name)
        right_factory = getattr(right, factory_name)
        for quantity in left_factory._attrs:
            _assert_fields_equal(
                f"{name}.{factory_name}.{quantity}",
                left_factory.get(quantity),
                right_factory.get(quantity),
            )


@pytest.mark.datatest
@pytest.mark.level("integration")
def test_edsl_plain_and_legacy_golden_match_for_exclaim_ape_aes(
    *,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
    download_ser_data: None,
    experiment_description: test_defs.ExperimentDescription,
) -> None:
    """Run EXCLAIM_APE_AES for two steps; compare eDSL, plain, and legacy golden outputs."""
    if backend is None:
        pytest.skip(
            "Parity test requires a compiled backend; embedded backend fails during static-field setup."
        )

    grid_file_path = grid_utils._download_grid_file(experiment_description.grid)
    config_file_path = dt_utils.get_path_for_experiment(experiment_description, process_props)

    config = driver_config.read_experiment_config_from_fortran(config_file_path)
    start = config.driver.start_of_timestepping
    end = start + 2 * config.driver.dtime

    allocator = model_backends.get_allocator(backend)
    grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file_path,
        vertical_grid_config=config.vertical_grid,
        allocator=allocator,
        process_props=process_props,
    )

    config_edsl = config.with_overrides(
        driver={
            "output_path": tmp_path / "parity_output_edsl",
            "start_of_timestepping": start,
            "end_of_simulation": end,
        }
    )
    config_plain = config.with_overrides(
        driver={
            "output_path": tmp_path / "parity_output_plain",
            "start_of_timestepping": start,
            "end_of_simulation": end,
        }
    )

    ds_edsl, driver_edsl = _run_driver(
        config=config_edsl,
        grid_manager=grid_manager,
        process_props=process_props,
        backend=backend,
        plain=False,
    )
    ds_plain, driver_plain = _run_driver(
        config=config_plain,
        grid_manager=grid_manager,
        process_props=process_props,
        backend=backend,
        plain=True,
    )

    # eDSL vs plain existence proof
    _assert_fields_equal("prognostics", ds_edsl.prognostics, ds_plain.prognostics)
    _assert_fields_equal("tracers", ds_edsl.tracers, ds_plain.tracers)
    _assert_fields_equal(
        "prep_advection_prognostic",
        ds_edsl.prep_advection_prognostic,
        ds_plain.prep_advection_prognostic,
    )
    _assert_fields_equal(
        "prep_tracer_advection_prognostic",
        ds_edsl.prep_tracer_advection_prognostic,
        ds_plain.prep_tracer_advection_prognostic,
    )
    _assert_fields_equal("diagnostic", ds_edsl.diagnostic, ds_plain.diagnostic)
    if ds_edsl.solve_nonhydro_diagnostic is not None:
        _assert_fields_equal(
            "solve_nonhydro_diagnostic",
            ds_edsl.solve_nonhydro_diagnostic,
            ds_plain.solve_nonhydro_diagnostic,
        )
    if ds_edsl.diffusion_diagnostic is not None:
        _assert_fields_equal(
            "diffusion_diagnostic", ds_edsl.diffusion_diagnostic, ds_plain.diffusion_diagnostic
        )
    if ds_edsl.tracer_advection_diagnostic is not None:
        _assert_fields_equal(
            "tracer_advection_diagnostic",
            ds_edsl.tracer_advection_diagnostic,
            ds_plain.tracer_advection_diagnostic,
        )
    _assert_static_fields_equal(
        "static_fields", driver_edsl.static_field_factories, driver_plain.static_field_factories
    )
    # New drivers vs legacy golden outputs
    _compare_time_step_pair_to_golden(
        "prognostics",
        _load_golden("prognostics"),
        ds_edsl.prognostics,
        _compare_prognostic_state_to_golden,
    )
    _compare_time_step_pair_to_golden(
        "tracers",
        _load_golden("tracers"),
        ds_edsl.tracers,
        _compare_tracer_state_to_golden,
    )
    _compare_dataclass_to_golden(
        "prep_advection_prognostic",
        _load_golden("prep_advection_prognostic"),
        "",
        ds_edsl.prep_advection_prognostic,
    )
    _compare_dataclass_to_golden(
        "prep_tracer_advection_prognostic",
        _load_golden("prep_tracer_advection_prognostic"),
        "",
        ds_edsl.prep_tracer_advection_prognostic,
    )
    _compare_dataclass_to_golden(
        "diagnostic",
        _load_golden("diagnostic"),
        "",
        ds_edsl.diagnostic,
    )
    if ds_edsl.solve_nonhydro_diagnostic is not None:
        _compare_dataclass_to_golden(
            "solve_nonhydro_diagnostic",
            _load_golden("solve_nonhydro_diagnostic"),
            "",
            ds_edsl.solve_nonhydro_diagnostic,
        )
    if ds_edsl.diffusion_diagnostic is not None:
        _compare_dataclass_to_golden(
            "diffusion_diagnostic",
            _load_golden("diffusion_diagnostic"),
            "",
            ds_edsl.diffusion_diagnostic,
        )
    if ds_edsl.tracer_advection_diagnostic is not None:
        _compare_dataclass_to_golden(
            "tracer_advection_diagnostic",
            _load_golden("tracer_advection_diagnostic"),
            "",
            ds_edsl.tracer_advection_diagnostic,
        )
    _compare_static_fields_to_golden(
        "static_fields",
        _load_golden("static_fields"),
        driver_edsl.static_field_factories,
    )

    clock_golden = _load_golden("clock")
    assert driver_edsl.model_time_variables.cfl_watch_mode == bool(clock_golden["cfl_watch_mode"])
    assert driver_edsl.model_time_variables.ndyn_substeps_var == int(
        clock_golden["ndyn_substeps_var"]
    )
    assert driver_plain.model_time_variables.cfl_watch_mode == bool(clock_golden["cfl_watch_mode"])
    assert driver_plain.model_time_variables.ndyn_substeps_var == int(
        clock_golden["ndyn_substeps_var"]
    )


@pytest.mark.datatest
@pytest.mark.level("integration")
def test_driver_introspection_renders_exclaim_ape_aes_composition(
    *,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
    download_ser_data: None,
    experiment_description: test_defs.ExperimentDescription,
) -> None:
    """The parity experiment's composition tree and dataflow graph render without error."""
    if backend is None:
        pytest.skip("Introspection test requires a compiled backend.")

    grid_file_path = grid_utils._download_grid_file(experiment_description.grid)
    config_file_path = dt_utils.get_path_for_experiment(experiment_description, process_props)

    config = driver_config.read_experiment_config_from_fortran(config_file_path)
    start = config.driver.start_of_timestepping
    end = start + 2 * config.driver.dtime
    config = config.with_overrides(
        driver={
            "output_path": tmp_path / "introspection_output",
            "start_of_timestepping": start,
            "end_of_simulation": end,
        }
    )

    allocator = model_backends.get_allocator(backend)
    grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file_path,
        vertical_grid_config=config.vertical_grid,
        allocator=allocator,
        process_props=process_props,
    )

    ds, driver = standalone_driver.run_driver(
        config=config,
        grid_manager=grid_manager,
        process_props=process_props,
        backend=backend,
    )

    text_tree = driver.show()
    assert "run_time_integration_edsl" in text_tree
    assert "outer_step" in text_tree
    assert "dycore_substeps" in text_tree
    assert "physics_step" in text_tree
    assert "advect_tracers" in text_tree

    dot = driver.to_graphviz()
    assert "digraph composition {" in dot
    assert "composition tree" in dot
    assert "dataflow graph" in dot
    # Run the driver to completion so the datatest fixture is exercised end-to-end.
    del ds
