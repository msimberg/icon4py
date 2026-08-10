# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for driver-state assembly helpers (``driver_states``).

Data-free: they use the ``simple_grid`` and need no serialized test data.
"""

import datetime

import gt4py.next as gtx
import pytest

from icon4py.model.atmosphere.dycore import dycore_states
from icon4py.model.atmosphere.tracer_advection import tracer_advection_states
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, simple
from icon4py.model.common.states import field_registry, validation
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.standalone_driver import config as driver_config, driver_states
from icon4py.model.standalone_driver.driver_loop_state import DriverLoopState, DriverServices


@pytest.fixture
def grid() -> base.Grid:
    return simple.simple_grid()


def _prep_advection(grid: base.Grid) -> dycore_states.PrepAdvection:
    def _field(*field_dims: gtx.Dimension) -> gtx.Field:
        return data_alloc.zero_field(grid, *field_dims, dtype=ta.wpfloat)

    return dycore_states.PrepAdvection(
        vn_traj=_field(dims.EdgeDim, dims.KDim),
        mass_flx_me=_field(dims.EdgeDim, dims.KDim),
        dynamical_vertical_mass_flux_at_cells_on_half_levels=_field(dims.CellDim, dims.KDim),
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels=_field(dims.CellDim, dims.KDim),
    )


def _prep_advection_registry(grid: base.Grid) -> field_registry.FieldRegistry:
    """Return a sealed registry that knows the dycore/advection handoff containers."""
    registry = field_registry.FieldRegistry(grid=grid, backend=None)
    registry.declare(dycore_states.PrepAdvection)
    registry.declare(tracer_advection_states.AdvectionPrepAdvState)
    registry.seal()
    return registry


def test_registry_build_prep_advection_and_tracer_advection_shares_buffers(
    grid: base.Grid,
) -> None:
    """The registry emits one buffer per handoff quantity; advection reads the dycore's buffers."""
    registry = _prep_advection_registry(grid)
    prep_adv = registry.build(dycore_states.PrepAdvection)
    prep_tracer_adv = registry.build(tracer_advection_states.AdvectionPrepAdvState)
    assert prep_tracer_adv is not None
    # Advection must read the very buffers the dycore accumulates into: identity, not equality.
    assert prep_tracer_adv.vn_traj is prep_adv.vn_traj
    assert prep_tracer_adv.mass_flx_me is prep_adv.mass_flx_me
    assert (
        prep_tracer_adv.mass_flx_ic is prep_adv.dynamical_vertical_mass_flux_at_cells_on_half_levels
    )


def test_registry_build_tracer_advection_alone_allocates_zero_buffers(
    grid: base.Grid,
) -> None:
    """Without a dycore the registry still allocates standalone tracer-advection buffers."""
    registry = field_registry.FieldRegistry(grid=grid, backend=None)
    registry.declare(tracer_advection_states.AdvectionPrepAdvState)
    registry.seal()
    prep_tracer_adv = registry.build(tracer_advection_states.AdvectionPrepAdvState)
    assert prep_tracer_adv is not None
    assert prep_tracer_adv.vn_traj.asnumpy().sum() == 0.0
    assert prep_tracer_adv.mass_flx_me.asnumpy().sum() == 0.0
    assert prep_tracer_adv.mass_flx_ic.asnumpy().sum() == 0.0


def test_step_info_flags() -> None:
    info = dycore_states.StepInfo(
        substep_index=2,
        at_first_substep=False,
        at_last_substep=True,
        at_initial_timestep=True,
    )
    assert info.substep_index == 2
    assert info.at_first_substep is False
    assert info.at_last_substep is True
    assert info.at_initial_timestep is True


def test_driver_loop_state_begin_time_step_sets_index() -> None:
    config = driver_config.DriverConfig.make_initial(
        experiment_name="test",
        start_of_simulation=datetime.datetime(2024, 1, 1),
        end_of_simulation=driver_config.time.NumTimeSteps(1),
        dtime=driver_config.time.RelativeTime(seconds=300),
        profiling_options=None,
    )
    mtv = driver_states.ModelTimeVariables(config=config)
    carry = DriverLoopState(
        clock=mtv,
        states=None,  # type: ignore[arg-type]
        granules=None,  # type: ignore[arg-type]
        config=None,  # type: ignore[arg-type]
        services=None,  # type: ignore[arg-type]
        wall_clock_starting_time=datetime.datetime.now(),
    )
    carry.begin_time_step(3, 10)
    assert carry.time_step_index == 3


def test_driver_loop_state_begin_substep_sets_index_and_total() -> None:
    config = driver_config.DriverConfig.make_initial(
        experiment_name="test",
        start_of_simulation=datetime.datetime(2024, 1, 1),
        end_of_simulation=driver_config.time.NumTimeSteps(1),
        dtime=driver_config.time.RelativeTime(seconds=300),
        profiling_options=None,
    )
    mtv = driver_states.ModelTimeVariables(config=config)
    carry = DriverLoopState(
        clock=mtv,
        states=None,  # type: ignore[arg-type]
        granules=None,  # type: ignore[arg-type]
        config=None,  # type: ignore[arg-type]
        services=None,  # type: ignore[arg-type]
        wall_clock_starting_time=datetime.datetime.now(),
    )
    carry.begin_substep(0, 4)
    assert carry.substep_index == 0
    assert carry.substep_total == 4
    assert dycore_states.StepInfo(
        substep_index=carry.substep_index,
        at_first_substep=carry.substep_index == 0,
        at_last_substep=carry.substep_index == carry.substep_total - 1,
        at_initial_timestep=carry.clock.is_first_step_in_simulation,
    ) == dycore_states.StepInfo(
        substep_index=0, at_first_substep=True, at_last_substep=False, at_initial_timestep=True
    )

    carry.clock.is_first_step_in_simulation = False
    carry.begin_substep(3, 4)
    assert carry.substep_index == 3
    assert carry.substep_total == 4
    assert dycore_states.StepInfo(
        substep_index=carry.substep_index,
        at_first_substep=carry.substep_index == 0,
        at_last_substep=carry.substep_index == carry.substep_total - 1,
        at_initial_timestep=carry.clock.is_first_step_in_simulation,
    ) == dycore_states.StepInfo(
        substep_index=3, at_first_substep=False, at_last_substep=True, at_initial_timestep=False
    )


def test_prep_advection_field_coverage(grid: base.Grid) -> None:
    """The standalone-driver PrepAdvection helper must pass exactly the fields the dataclass declares."""
    target_classes = (dycore_states.PrepAdvection,)
    call_sites = validation.kwargs_at_constructor_calls(_prep_advection, target_classes)
    validation.assert_field_coverage(
        dycore_states.PrepAdvection,
        {name: None for name in call_sites.get("PrepAdvection", set())},
    )
