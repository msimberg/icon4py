# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""eDSL driver: builds the time-integration composition and runs it once."""

from icon4py.model.common.composition import chain, repeat, when
from icon4py.model.standalone_driver.driver_loop_state import DriverLoopState
from icon4py.model.standalone_driver.steps import (
    adjust_ndyn_step,
    advance_clock_step,
    advect_tracers_step,
    compute_airmass_new_step,
    compute_airmass_now_step,
    compute_mean_at_final_step,
    diffuse_before_time_loop_step,
    diffusion_step,
    dycore_substeps_step,
    end_of_step_step,
    finalize_step,
    io_snapshot_step,
    physics_step,
    swap_step,
    sync_step,
)


def run_time_integration_edsl(carry: DriverLoopState) -> None:
    """Run the standalone driver time loop as a composition of steps."""
    outer_step = chain(
        advance_clock_step,
        when(
            lambda c: c.states.tracer_advection_diagnostic is not None,
            then=compute_airmass_now_step,
        ),
        when(
            lambda c: c.config.nonhydrostatic is not None,
            then=dycore_substeps_step(),
        ),
        when(
            lambda c: c.states.tracer_advection_diagnostic is not None,
            then=compute_airmass_new_step,
        ),
        when(
            lambda c: (
                c.granules.diffusion is not None
                and c.config.diffusion is not None
                and c.config.diffusion.apply_to_horizontal_wind
            ),
            then=diffusion_step,
        ),
        when(
            lambda c: c.granules.tracer_advection is not None,
            then=advect_tracers_step(),
        ),
        when(
            lambda c: c.granules.physics is not None,
            then=physics_step,
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

    run = chain(
        when(
            lambda c: c.services.io_monitor is not None,
            then=io_snapshot_step,
        ),
        diffuse_before_time_loop_step,
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

    run(carry)
