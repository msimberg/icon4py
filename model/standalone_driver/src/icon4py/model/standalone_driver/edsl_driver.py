# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""eDSL driver: builds the time-integration composition."""

from __future__ import annotations

from icon4py.model.common.components.derived_quantities import DerivedQuantities
from icon4py.model.common.composition import Step, chain, repeat, with_index
from icon4py.model.common.io import io as common_io
from icon4py.model.standalone_driver import driver_utils
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


def opt[T](include: bool, *steps: T) -> tuple[T, ...]:
    """Splice ``steps`` into a chain when ``include`` is true, else nothing.

    The arguments are evaluated eagerly, so builders passed here must be
    pure constructors and tolerate a ``None`` component.
    """
    return steps if include else ()


def build_time_integration_composition(
    granules: driver_utils.Granules,
    derived_quantities: DerivedQuantities,
    io_monitor: common_io.IOMonitor | None,
) -> Step[DriverLoopState]:
    """Build the full time-integration composition.

    ``granules`` and ``derived_quantities`` supply the components for
    introspection metadata; leaf steps read them from the carry at runtime.

    Component inclusion is decided here, once, so statically known-absent
    parts (physics in a dry run, tracer advection when not configured)
    never enter the graph. Only genuinely dynamic branches are left to the
    eDSL: the substep swap, the physics forcing window, the ``sample``
    cadence, and the (CFL-adjusted) substep count.
    """
    has_advection = granules.tracer_advection is not None
    has_dycore = granules.solve_nonhydro is not None
    has_wind_diffusion = (
        granules.diffusion is not None and granules.diffusion.config.apply_to_horizontal_wind
    )
    writes_output = io_monitor is not None

    outer_step = chain(
        advance_clock_step,
        *opt(has_advection, compute_airmass_now_step),
        *opt(has_dycore, build_dycore_substeps_step(granules.solve_nonhydro)),
        *opt(has_advection, compute_airmass_new_step),
        *opt(has_wind_diffusion, build_diffusion_step(granules.diffusion)),
        *opt(has_advection, build_advect_tracers_step(granules.tracer_advection)),
        build_update_derived_quantities_step(derived_quantities),
        *opt(
            granules.physics is not None,
            build_physics_composition_step(granules.physics),
        ),
        swap_step,
        sync_step,
        end_of_step_step,
        *opt(has_dycore, adjust_ndyn_step),
        *opt(writes_output, io_snapshot_step),
        name="outer_step",
    )

    return chain(
        *opt(writes_output, io_snapshot_step),
        build_diffuse_before_time_loop_step(granules.diffusion),
        repeat(
            with_index(outer_step, set_index=DriverLoopState.begin_time_step),
            times=lambda c: c.clock.n_time_steps,
            name="time_steps",
        ),
        compute_mean_at_final_step,
        finalize_step,
        name="run_time_integration_edsl",
    )
