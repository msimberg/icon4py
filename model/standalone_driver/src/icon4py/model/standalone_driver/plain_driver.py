# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Plain-Python driver: same steps as the eDSL driver, using ``for``/``if``."""

from icon4py.model.standalone_driver.driver_loop_state import DriverLoopState
from icon4py.model.standalone_driver.steps import (
    _advect_tracer,
    _diffuse_before_time_loop,
    _diffusion,
    _physics,
    _solve_nh,
    _update_derived_quantities,
    adjust_ndyn_step,
    advance_clock_step,
    compute_airmass_new_step,
    compute_airmass_now_step,
    compute_mean_at_final_step,
    compute_statistics_step,
    compute_total_mass_and_energy_step,
    end_of_step_step,
    finalize_step,
    io_snapshot_step,
    swap_step,
    sync_step,
    update_time_levels_step,
)


def run_time_integration_plain(carry: DriverLoopState) -> None:  # noqa: PLR0912
    """Run the standalone driver time loop with plain Python control flow."""
    try:
        if carry.services.io_monitor is not None:
            io_snapshot_step(carry)

        _diffuse_before_time_loop(carry)

        for time_step in range(carry.clock.n_time_steps):
            carry.begin_time_step(time_step, carry.clock.n_time_steps)

            advance_clock_step(carry)

            if carry.states.tracer_advection_diagnostic is not None:
                compute_airmass_now_step(carry)

            if carry.config.nonhydrostatic is not None:
                ndyn_substeps_var = carry.clock.ndyn_substeps_var
                for dyn_substep in range(ndyn_substeps_var):
                    carry.begin_substep(dyn_substep, ndyn_substeps_var)
                    compute_statistics_step(carry)
                    update_time_levels_step(carry)
                    _solve_nh(carry)
                    if dyn_substep != ndyn_substeps_var - 1:
                        carry.states.prognostics.swap()
                compute_total_mass_and_energy_step(carry)

            if carry.states.tracer_advection_diagnostic is not None:
                compute_airmass_new_step(carry)

            if (
                carry.granules.diffusion is not None
                and carry.granules.diffusion.config.apply_to_horizontal_wind
            ):
                _diffusion(carry)

            if carry.granules.tracer_advection is not None:
                for tracer_current in carry.states.tracers.current.active_fields():
                    _advect_tracer(carry, tracer_current)

            _update_derived_quantities(carry, carry.services.derived_quantities)

            if carry.granules.physics is not None:
                _physics(carry)

            swap_step(carry)
            sync_step(carry)
            end_of_step_step(carry)

            if carry.config.nonhydrostatic is not None:
                adjust_ndyn_step(carry)

            if carry.services.io_monitor is not None:
                io_snapshot_step(carry)

        compute_mean_at_final_step(carry)
        finalize_step(carry)
    finally:
        if carry.services.io_monitor is not None:
            carry.services.io_monitor.close()
