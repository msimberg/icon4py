# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Bit-exact parity test: eDSL driver vs plain-Python driver.

Uses the smallest full-driver experiment (JW) as a datatest fallback.
"""

import datetime
import pathlib

import gt4py.next.typing as gtx_typing
import numpy as np
import pytest

from icon4py.model.common import model_backends
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.states import prognostic_state as prognostics, tracer_states
from icon4py.model.common.utils import TimeStepPair
from icon4py.model.standalone_driver import config as driver_config, driver_utils, standalone_driver
from icon4py.model.testing import datatest_utils as dt_utils, definitions as test_defs, grid_utils

from ..fixtures import backend, download_ser_data, process_props


@pytest.fixture
def experiment_description() -> test_defs.ExperimentDescription:
    """Use only the JW experiment for the parity test."""
    return test_defs.Experiments.JW


_JW_START = "2008-09-01T00:00:00.000"
_JW_END_2_STEPS = "2008-09-01T00:10:00.000"


def _run_driver(
    *,
    config: driver_config.ExperimentConfig,
    grid_manager: driver_utils.gm.GridManager,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend | None,
    plain: bool,
) -> standalone_driver.driver_states.DriverStates:
    runner = standalone_driver.run_driver_plain if plain else standalone_driver.run_driver
    ds, _ = runner(
        config=config,
        grid_manager=grid_manager,
        process_props=process_props,
        backend=backend,
    )
    return ds


def _field_equality(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is right
    if hasattr(left, "ndarray") and hasattr(right, "ndarray"):
        return np.array_equal(left.ndarray, right.ndarray)
    return left == right


def _assert_fields_equal(name: str, left: object, right: object) -> None:
    assert _field_equality(left, right), f"Field '{name}' differs between eDSL and plain driver."


def _compare_prognostic_state(
    name: str,
    left: prognostics.PrognosticState,
    right: prognostics.PrognosticState,
) -> None:
    _assert_fields_equal(f"{name}.vn", left.vn, right.vn)
    _assert_fields_equal(f"{name}.w", left.w, right.w)
    _assert_fields_equal(f"{name}.exner", left.exner, right.exner)
    _assert_fields_equal(f"{name}.rho", left.rho, right.rho)
    _assert_fields_equal(f"{name}.theta_v", left.theta_v, right.theta_v)


def _compare_time_step_pair(
    name: str,
    left: TimeStepPair[prognostics.PrognosticState],
    right: TimeStepPair[prognostics.PrognosticState],
) -> None:
    _compare_prognostic_state(f"{name}.current", left.current, right.current)
    _compare_prognostic_state(f"{name}.next", left.next, right.next)


def _compare_tracer_pair(
    name: str,
    left: TimeStepPair[tracer_states.TracerState],
    right: TimeStepPair[tracer_states.TracerState],
) -> None:
    for tracer_current in left.current.active_fields():
        tracer_next_left = getattr(left.next, tracer_current.name)
        tracer_next_right = getattr(right.next, tracer_current.name)
        _assert_fields_equal(
            f"{name}.current.{tracer_current.name}", tracer_current.field, tracer_current.field
        )
        _assert_fields_equal(
            f"{name}.next.{tracer_current.name}", tracer_next_left, tracer_next_right
        )


@pytest.mark.datatest
@pytest.mark.level("integration")
def test_edsl_and_plain_driver_produce_bit_identical_results(
    *,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
    download_ser_data: None,
) -> None:
    """Run the JW experiment for two time steps with both drivers; compare all fields."""
    if backend is None:
        pytest.skip(
            "JW parity test requires a compiled backend; embedded backend fails during static-field setup."
        )

    allocator = model_backends.get_allocator(backend)

    experiment_description = test_defs.Experiments.JW
    grid_file_path = grid_utils._download_grid_file(experiment_description.grid)
    config_file_path = dt_utils.get_path_for_experiment(experiment_description, process_props)

    config = driver_config.read_experiment_config_from_fortran(config_file_path)
    config = config.with_overrides(
        driver={
            "output_path": tmp_path / "parity_output",
            "start_of_timestepping": datetime.datetime.fromisoformat(_JW_START).replace(
                tzinfo=datetime.UTC
            ),
            "end_of_simulation": datetime.datetime.fromisoformat(_JW_END_2_STEPS).replace(
                tzinfo=datetime.UTC
            ),
        }
    )

    grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file_path,
        vertical_grid_config=config.vertical_grid,
        allocator=allocator,
        process_props=process_props,
    )

    ds_edsl = _run_driver(
        config=config,
        grid_manager=grid_manager,
        process_props=process_props,
        backend=backend,
        plain=False,
    )
    ds_plain = _run_driver(
        config=config,
        grid_manager=grid_manager,
        process_props=process_props,
        backend=backend,
        plain=True,
    )

    _compare_time_step_pair("prognostics", ds_edsl.prognostics, ds_plain.prognostics)
    _compare_tracer_pair("tracers", ds_edsl.tracers, ds_plain.tracers)

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
    _assert_fields_equal(
        "diagnostic",
        ds_edsl.diagnostic,
        ds_plain.diagnostic,
    )

    if (
        ds_edsl.solve_nonhydro_diagnostic is not None
        and ds_plain.solve_nonhydro_diagnostic is not None
    ):
        _assert_fields_equal(
            "solve_nonhydro_diagnostic.max_vertical_cfl",
            ds_edsl.solve_nonhydro_diagnostic.max_vertical_cfl,
            ds_plain.solve_nonhydro_diagnostic.max_vertical_cfl,
        )

    if ds_edsl.diffusion_diagnostic is not None and ds_plain.diffusion_diagnostic is not None:
        _assert_fields_equal(
            "diffusion_diagnostic",
            ds_edsl.diffusion_diagnostic,
            ds_plain.diffusion_diagnostic,
        )

    if (
        ds_edsl.tracer_advection_diagnostic is not None
        and ds_plain.tracer_advection_diagnostic is not None
    ):
        _assert_fields_equal(
            "tracer_advection_diagnostic.airmass_now",
            ds_edsl.tracer_advection_diagnostic.airmass_now,
            ds_plain.tracer_advection_diagnostic.airmass_now,
        )
        _assert_fields_equal(
            "tracer_advection_diagnostic.airmass_new",
            ds_edsl.tracer_advection_diagnostic.airmass_new,
            ds_plain.tracer_advection_diagnostic.airmass_new,
        )
