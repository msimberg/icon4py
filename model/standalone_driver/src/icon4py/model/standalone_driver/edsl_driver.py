# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""eDSL driver: builds the time-integration composition and runs it once."""

from icon4py.model.common.composition import Step, chain, repeat, when
from icon4py.model.standalone_driver import driver_utils
from icon4py.model.standalone_driver.derived_quantities import DerivedQuantities
from icon4py.model.standalone_driver.driver_loop_state import DriverLoopState
from icon4py.model.standalone_driver.steps import (
    adjust_ndyn_step,
    advance_clock_step,
    build_advect_tracers_step,
    build_diffuse_before_time_loop_step,
    build_diffusion_step,
    build_dycore_substeps_step,
    build_physics_composition_step,
    build_update_derived_quantities_step,
    compute_airmass_new_step,
    compute_airmass_now_step,
    compute_mean_at_final_step,
    end_of_step_step,
    finalize_step,
    io_snapshot_step,
    swap_step,
    sync_step,
)


def build_time_integration_composition(
    *,
    granules: driver_utils.Granules | None = None,
    derived_quantities: DerivedQuantities | None = None,
) -> Step[DriverLoopState]:
    """Build the full time-integration composition.

    ``granules`` supplies components for introspection metadata. When ``None``,
    the leaf steps still call ``component.run(state)`` from the carry at runtime
    but expose no component metadata.

    ``derived_quantities`` is the canonical T/p/u/v component; when ``None`` the
    step is omitted from the composition.
    """
    outer_step = chain(
        advance_clock_step,
        when(
            lambda c: c.states.tracer_advection_diagnostic is not None,
            then=compute_airmass_now_step,
        ),
        when(
            lambda c: c.config.nonhydrostatic is not None,
            then=build_dycore_substeps_step(
                granules.solve_nonhydro if granules is not None else None
            ),
        ),
        when(
            lambda c: c.states.tracer_advection_diagnostic is not None,
            then=compute_airmass_new_step,
        ),
        when(
            lambda c: (
                c.granules.diffusion is not None
                and c.granules.diffusion.config.apply_to_horizontal_wind
            ),
            then=build_diffusion_step(granules.diffusion if granules is not None else None),
        ),
        when(
            lambda c: c.granules.tracer_advection is not None,
            then=build_advect_tracers_step(
                granules.tracer_advection if granules is not None else None
            ),
        ),
        when(
            lambda c: derived_quantities is not None,
            then=build_update_derived_quantities_step(derived_quantities),
        ),
        when(
            lambda c: c.granules.physics is not None,
            then=build_physics_composition_step(granules.physics if granules is not None else None),
        ),
        swap_step,
        sync_step,
        end_of_step_step,
        when(
            lambda c: c.config.nonhydrostatic is not None,
            then=adjust_ndyn_step,
        ),
        when(
            lambda c: c.services.io_monitor is not None,
            then=io_snapshot_step,
        ),
        name="outer_step",
    )

    return chain(
        when(
            lambda c: c.services.io_monitor is not None,
            then=io_snapshot_step,
        ),
        build_diffuse_before_time_loop_step(granules.diffusion if granules is not None else None),
        repeat(
            outer_step,
            times=lambda c: c.clock.n_time_steps,
            set_loop_context=DriverLoopState.begin_time_step,
            name="time_steps",
        ),
        compute_mean_at_final_step,
        finalize_step,
        name="run_time_integration_edsl",
    )


def run_time_integration_edsl(carry: DriverLoopState) -> None:
    """Run the standalone driver time loop as a composition of steps."""
    build_time_integration_composition(granules=None)(carry)
