# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Leaf steps and small composites shared by the eDSL and plain drivers."""

from __future__ import annotations

import datetime
import logging
from collections.abc import Callable

from gt4py.next import config as gtx_config
from gt4py.next.instrumentation import metrics as gtx_metrics

from icon4py.model.atmosphere.dycore import dycore_states
from icon4py.model.common import type_alias as ta
from icon4py.model.common.composition import Step, SwapPolicy, chain, foreach, repeat
from icon4py.model.common.grid import geometry_attributes as geom_attr
from icon4py.model.common.interpolation import interpolation_attributes as intp_attr
from icon4py.model.common.metrics import metrics_attributes as metrics_attr
from icon4py.model.common.utils import data_allocation as data_alloc, device_utils
from icon4py.model.standalone_driver import driver_constants, driver_io, driver_states, driver_utils
from icon4py.model.standalone_driver.driver_loop_state import DriverLoopState


log = logging.getLogger(__name__)


class _NamedStep:
    """Simple wrapper giving a callable step a ``name`` attribute."""

    def __init__(self, name: str, fn: Callable[[DriverLoopState], None]) -> None:
        self.name = name
        self._fn = fn

    def __call__(self, carry: DriverLoopState) -> None:
        self._fn(carry)


def _as_step(name: str, fn: Callable[[DriverLoopState], None]) -> Step[DriverLoopState]:
    return _NamedStep(name, fn)


# --------------------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------------------


def _io_snapshot(carry: DriverLoopState) -> None:
    assert carry.services.io_monitor is not None
    metrics = carry.services.static_field_factories.metrics
    interpolation = carry.services.static_field_factories.interpolation
    prognostic_state = carry.states.prognostics.current
    state_to_store = driver_io.prognostic_state_to_dataarrays(prognostic_state)
    diagnostic_fields = carry.services.diagnostics_computer.compute(
        prognostic_state,
        ddqz_z_full=metrics.get(metrics_attr.DDQZ_Z_FULL),
        rbf_vec_coeff_c1=interpolation.get(intp_attr.RBF_VEC_COEFF_C1),
        rbf_vec_coeff_c2=interpolation.get(intp_attr.RBF_VEC_COEFF_C2),
    )
    state_to_store.update(driver_io.diagnostic_fields_to_dataarrays(diagnostic_fields))
    carry.services.io_monitor.store(state_to_store, carry.clock.simulation_current_datetime)


io_snapshot_step = _as_step("io_snapshot_step", _io_snapshot)


# --------------------------------------------------------------------------------------
# Pre-loop diffusion
# --------------------------------------------------------------------------------------


def _diffuse_before_time_loop(carry: DriverLoopState) -> None:
    if (
        not carry.config.driver.diffuse_before_time_loop
        or not carry.clock.is_first_step_in_simulation
    ):
        return
    assert carry.states.diffusion_diagnostic is not None
    assert carry.granules.diffusion is not None
    log.info("running diffusion to filter the initial state, before the time loop")
    carry.granules.diffusion.run(
        carry.states.diffusion_diagnostic,
        carry.states.prognostics.current,
        carry.clock.dtime_in_seconds,
        initial_run=True,
    )


diffuse_before_time_loop_step = _as_step("diffuse_before_time_loop_step", _diffuse_before_time_loop)


# --------------------------------------------------------------------------------------
# Clock / tendencies
# --------------------------------------------------------------------------------------


def _advance_clock(carry: DriverLoopState) -> None:
    profiling_options = carry.config.driver.profiling_options
    if profiling_options is not None:
        if not profiling_options.skip_first_timestep or carry.time_step_index > 0:
            gtx_config.COLLECT_METRICS_LEVEL = profiling_options.gt4py_metrics_level

    elapsed = datetime.datetime.now() - carry.wall_clock_starting_time
    log.info(
        f"\n"
        f"simulation date : {carry.clock.simulation_current_datetime}, at timestep : {carry.time_step_index}, Elapsed wall clock time: {elapsed.total_seconds()}"
        f"\n"
    )

    carry.clock.advance_simulation_datetime()

    if carry.services.tendencies is not None:
        assert carry.states.solve_nonhydro_diagnostic is not None
        carry.services.tendencies.update(
            diagnostic_state_nh=carry.states.solve_nonhydro_diagnostic,
            at_datetime=carry.clock.simulation_current_datetime,
        )


advance_clock_step = _as_step("advance_clock_step", _advance_clock)


# --------------------------------------------------------------------------------------
# Airmass
# --------------------------------------------------------------------------------------


def _compute_airmass_now(carry: DriverLoopState) -> None:
    assert carry.states.tracer_advection_diagnostic is not None
    carry.services.compute_airmass(
        rho_in=carry.states.prognostics.current.rho,
        airmass_out=carry.states.tracer_advection_diagnostic.airmass_now,
    )


compute_airmass_now_step = _as_step("compute_airmass_now_step", _compute_airmass_now)


def _compute_airmass_new(carry: DriverLoopState) -> None:
    assert carry.states.tracer_advection_diagnostic is not None
    rho_after_dynamics = (
        carry.states.prognostics.next.rho
        if carry.config.nonhydrostatic is not None
        else carry.states.prognostics.current.rho
    )
    carry.services.compute_airmass(
        rho_in=rho_after_dynamics,
        airmass_out=carry.states.tracer_advection_diagnostic.airmass_new,
    )


compute_airmass_new_step = _as_step("compute_airmass_new_step", _compute_airmass_new)


# --------------------------------------------------------------------------------------
# Dycore substep leaf steps
# --------------------------------------------------------------------------------------


def _compute_statistics(carry: DriverLoopState) -> None:
    assert carry.step_info is not None
    current_dyn_substep = carry.step_info.substep_index
    prognostic_state = carry.states.prognostics.current
    if carry.config.driver.enable_statistics_logging:
        rho_arg_max, max_rho = driver_utils.find_maximum_from_field(prognostic_state.rho)
        vn_arg_max, max_vn = driver_utils.find_maximum_from_field(prognostic_state.vn)
        w_arg_max, max_w = driver_utils.find_maximum_from_field(prognostic_state.w)

        def _determine_sign(input_number: float) -> str:
            return " " if input_number >= 0.0 else "-"

        rho_sign = _determine_sign(max_rho)
        vn_sign = _determine_sign(max_vn)
        w_sign = _determine_sign(max_w)

        log.info(
            f"substep / n_substeps : {current_dyn_substep:3d} / {carry.clock.ndyn_substeps_var:3d} == "
            f"MAX RHO: {rho_sign}{abs(max_rho):.5e} at lvl {rho_arg_max[1]:4d}, MAX VN: {vn_sign}{abs(max_vn):.5e} at lvl {vn_arg_max[1]:4d}, MAX W: {w_sign}{abs(max_w):.5e} at lvl {w_arg_max[1]:4d}"
        )
    else:
        log.info(
            f"substep / n_substeps : {current_dyn_substep:3d} / {carry.clock.ndyn_substeps_var:3d}"
        )


compute_statistics_step = _as_step("compute_statistics_step", _compute_statistics)


def _update_time_levels(carry: DriverLoopState) -> None:
    assert carry.step_info is not None
    assert carry.states.solve_nonhydro_diagnostic is not None
    step_info = carry.step_info
    diagnostic_state_nh = carry.states.solve_nonhydro_diagnostic
    if not (step_info.at_initial_timestep and step_info.at_first_substep):
        diagnostic_state_nh.vertical_wind_advective_tendency.swap()
    if not step_info.at_first_substep:
        diagnostic_state_nh.normal_wind_advective_tendency.swap()


update_time_levels_step = _as_step("update_time_levels_step", _update_time_levels)


def _second_order_divdamp_factor(carry: DriverLoopState) -> ta.wpfloat:
    assert carry.config.nonhydrostatic is not None
    nonhydrostatic = carry.config.nonhydrostatic
    fourth_order_divdamp_factor = nonhydrostatic.fourth_order_divdamp_factor
    if nonhydrostatic.divdamp_order != dycore_states.DivergenceDampingOrder.COMBINED:
        return fourth_order_divdamp_factor

    elapsed_time_in_seconds = carry.clock.elapsed_time_at_step_midpoint_in_seconds
    spinup_cutoff = (
        driver_constants.TRANSITION_END_PERIOD_FOR_SECOND_ORDER_DIVDAMP
        + 0.5 * carry.clock.dtime_in_seconds
    )
    if (
        not carry.config.driver.apply_extra_second_order_divdamp
        or elapsed_time_in_seconds > spinup_cutoff
    ):
        return ta.wpfloat("0.0")

    return driver_utils.spinup_second_order_divdamp_factor(
        elapsed_time_in_seconds=elapsed_time_in_seconds,
        fourth_order_divdamp_factor=fourth_order_divdamp_factor,
    )


def _solve_nh(carry: DriverLoopState) -> None:
    assert carry.step_info is not None
    assert carry.states.solve_nonhydro_diagnostic is not None
    assert carry.states.prep_advection_prognostic is not None
    assert carry.granules.solve_nonhydro is not None

    timer_name = (
        driver_states.DriverTimers.SOLVE_NH_FIRST_STEP.value
        if carry.clock.is_first_step_in_simulation
        else driver_states.DriverTimers.SOLVE_NH.value
    )
    timer_solve_nh = carry.services.timer_collection.timers[timer_name]
    with timer_solve_nh:
        carry.granules.solve_nonhydro.time_step(
            diagnostic_state_nh=carry.states.solve_nonhydro_diagnostic,
            prognostic_states=carry.states.prognostics,
            prep_adv=carry.states.prep_advection_prognostic,
            second_order_divdamp_factor=_second_order_divdamp_factor(carry),
            dtime=carry.clock.substep_timestep,
            ndyn_substeps_var=carry.clock.ndyn_substeps_var,
            at_initial_timestep=carry.step_info.at_initial_timestep,
            lprep_adv=carry.config.driver.do_prep_adv,
            at_first_substep=carry.step_info.at_first_substep,
            at_last_substep=carry.step_info.at_last_substep,
        )


solve_nh_step = _as_step("solve_nh_step", _solve_nh)


def _compute_total_mass_and_energy(carry: DriverLoopState) -> None:
    if carry.config.driver.enable_statistics_logging:
        prognostic_state = carry.states.prognostics.next
        rho_ndarray = prognostic_state.rho.ndarray
        cell_area_ndarray = carry.services.static_field_factories.geometry.get(
            geom_attr.CELL_AREA
        ).ndarray
        cell_thickness_ndarray = carry.services.static_field_factories.metrics.get(
            metrics_attr.DDQZ_Z_FULL
        ).ndarray
        local_mass = (
            rho_ndarray * cell_area_ndarray[:, carry.services.xp.newaxis] * cell_thickness_ndarray
        )
        global_total_mass = carry.services.global_reductions.sum(local_mass)
        log.info(f"GLOBAL TOTAL MASS: {global_total_mass:.15e} kg")


compute_total_mass_and_energy_step = _as_step(
    "compute_total_mass_and_energy_step", _compute_total_mass_and_energy
)


def dycore_substeps_step() -> Step[DriverLoopState]:
    """The dynamics substep loop: compute stats, update time levels, solve NH."""
    return repeat(
        chain(
            compute_statistics_step,
            update_time_levels_step,
            solve_nh_step,
        ),
        times=lambda c: c.clock.ndyn_substeps_var,
        post=compute_total_mass_and_energy_step,
        swap=SwapPolicy.EXCEPT_LAST,
        swap_target=lambda c: c.states.prognostics,
        set_loop_context=DriverLoopState.begin_substep,
        name="dycore_substeps",
    )


# --------------------------------------------------------------------------------------
# Diffusion
# --------------------------------------------------------------------------------------


def _diffusion(carry: DriverLoopState) -> None:
    assert carry.granules.diffusion is not None
    assert carry.states.diffusion_diagnostic is not None
    timer_name = (
        driver_states.DriverTimers.DIFFUSION_FIRST_STEP.value
        if carry.clock.is_first_step_in_simulation
        else driver_states.DriverTimers.DIFFUSION.value
    )
    timer_diffusion = carry.services.timer_collection.timers[timer_name]
    with timer_diffusion:
        carry.granules.diffusion.run(
            carry.states.diffusion_diagnostic,
            carry.states.prognostics.next,
            carry.clock.dtime_in_seconds,
        )


diffusion_step = _as_step("diffusion_step", _diffusion)


# --------------------------------------------------------------------------------------
# Tracer advection
# --------------------------------------------------------------------------------------


def _advect_tracer(carry: DriverLoopState) -> None:
    assert carry.current_tracer is not None
    assert carry.granules.tracer_advection is not None
    assert carry.states.tracer_advection_diagnostic is not None
    assert carry.states.prep_tracer_advection_prognostic is not None
    current_tracer = carry.current_tracer
    tracer_next_field = getattr(carry.states.tracers.next, current_tracer.name)
    assert tracer_next_field is not None, (
        f"tracer '{current_tracer.name}' active in current state but missing in next state"
    )
    carry.granules.tracer_advection.run(
        diagnostic_state=carry.states.tracer_advection_diagnostic,
        prep_adv=carry.states.prep_tracer_advection_prognostic,
        p_tracer_now=current_tracer.field,
        p_tracer_new=tracer_next_field,
        dtime=carry.clock.dtime_in_seconds,
    )


advect_tracer_step = _as_step("advect_tracer_step", _advect_tracer)


def advect_tracers_step() -> Step[DriverLoopState]:
    """Advect every active tracer once per time step."""
    return foreach(
        advect_tracer_step,
        source=lambda c: c.states.tracers.current.active_fields(),
        set_item=DriverLoopState.set_current_tracer,
        name="advect_tracers",
    )


# --------------------------------------------------------------------------------------
# Physics
# --------------------------------------------------------------------------------------


def _physics(carry: DriverLoopState) -> None:
    assert carry.granules.physics is not None
    carry.granules.physics.run(
        prognostic=carry.states.prognostics.next,
        tracers=carry.states.tracers.next,
        dtime=carry.config.driver.dtime,
        simulation_current_datetime=carry.clock.simulation_current_datetime,
    )


physics_step = _as_step("physics_step", _physics)


# --------------------------------------------------------------------------------------
# End-of-step bookkeeping
# --------------------------------------------------------------------------------------


def _swap(carry: DriverLoopState) -> None:
    carry.states.prognostics.swap()
    carry.states.tracers.swap()


swap_step = _as_step("swap_step", _swap)


def _sync(carry: DriverLoopState) -> None:
    device_utils.sync(carry.services.backend)


sync_step = _as_step("sync_step", _sync)


def _end_of_step(carry: DriverLoopState) -> None:
    carry.clock.is_first_step_in_simulation = False


end_of_step_step = _as_step("end_of_step_step", _end_of_step)


def _adjust_ndyn(carry: DriverLoopState) -> None:
    assert carry.states.solve_nonhydro_diagnostic is not None
    solve_nonhydro_diagnostic_state = carry.states.solve_nonhydro_diagnostic
    global_max_vertical_cfl = carry.services.global_reductions.max(
        carry.services.xp.asarray(
            solve_nonhydro_diagnostic_state.max_vertical_cfl[()], dtype=ta.wpfloat
        ),
    )
    if (
        global_max_vertical_cfl
        > driver_constants.CFL_ENTER_WATCHMODE_FACTOR * carry.config.driver.vertical_cfl_threshold
        and not carry.clock.cfl_watch_mode
    ):
        log.warning("High CFL number for vertical advection in dynamical core, entering watch mode")
        carry.clock.update_cfl_watch_mode(True)

    if carry.clock.cfl_watch_mode:
        substep_fraction = ta.wpfloat(
            carry.clock.ndyn_substeps_var / carry.config.driver.ndyn_substeps
        )
        if (
            global_max_vertical_cfl * substep_fraction
            > driver_constants.CFL_THRESHOLD_FACTOR * carry.config.driver.vertical_cfl_threshold
        ):
            log.warning(
                f"Maximum vertical CFL number {global_max_vertical_cfl} is close to critical threshold"
            )

        vertical_cfl_threshold_for_increment = carry.config.driver.vertical_cfl_threshold
        vertical_cfl_threshold_for_decrement = (
            driver_constants.CFL_THRESHOLD_FACTOR * carry.config.driver.vertical_cfl_threshold
        )

        if global_max_vertical_cfl > vertical_cfl_threshold_for_increment:
            if carry.services.xp.isfinite(global_max_vertical_cfl):
                ndyn_substeps_increment = max(
                    1,
                    round(
                        carry.clock.ndyn_substeps_var
                        * (global_max_vertical_cfl - vertical_cfl_threshold_for_increment)
                        / vertical_cfl_threshold_for_increment
                    ),
                )
                new_ndyn_substeps_var = min(
                    carry.clock.ndyn_substeps_var + ndyn_substeps_increment,
                    carry.clock.max_ndyn_substeps,
                )
            else:
                log.warning(
                    f"WARNING: max cfl {global_max_vertical_cfl} is not a number! Number of substeps is set to the max value! "
                )
                new_ndyn_substeps_var = carry.clock.max_ndyn_substeps
            carry.clock.update_ndyn_substeps(new_ndyn_substeps_var)
            log.warning(
                f"The number of dynamics substeps is increased to {carry.clock.ndyn_substeps_var}"
            )
        if (
            carry.clock.ndyn_substeps_var > carry.config.driver.ndyn_substeps
            and global_max_vertical_cfl
            * ta.wpfloat(carry.clock.ndyn_substeps_var / (carry.clock.ndyn_substeps_var - 1))
            < vertical_cfl_threshold_for_decrement
        ):
            carry.clock.update_ndyn_substeps(carry.clock.ndyn_substeps_var - 1)
            log.warning(
                f"The number of dynamics substeps is decreased to {carry.clock.ndyn_substeps_var}"
            )

            if (
                carry.clock.ndyn_substeps_var == carry.config.driver.ndyn_substeps
                and global_max_vertical_cfl
                < driver_constants.CFL_LEAVE_WATCHMODE_FACTOR
                * carry.config.driver.vertical_cfl_threshold
            ):
                log.warning(
                    "CFL number for vertical advection in dynamical core has decreased, leaving watch mode"
                )
                carry.clock.update_cfl_watch_mode(False)

    # reset max_vertical_cfl to zero
    solve_nonhydro_diagnostic_state.max_vertical_cfl = data_alloc.scalar_like_array(
        0.0, carry.services.allocator
    )


adjust_ndyn_step = _as_step("adjust_ndyn_step", _adjust_ndyn)


# --------------------------------------------------------------------------------------
# Finalization
# --------------------------------------------------------------------------------------


def _compute_mean_at_final(carry: DriverLoopState) -> None:
    if carry.config.driver.enable_statistics_logging:
        prognostic_state = carry.states.prognostics.current
        rho_ndarray = prognostic_state.rho.ndarray
        vn_ndarray = prognostic_state.vn.ndarray
        w_ndarray = prognostic_state.w.ndarray
        theta_v_ndarray = prognostic_state.theta_v.ndarray
        exner_ndarray = prognostic_state.exner.ndarray
        log.info("")
        log.info("Global mean of    rho         vn           w          theta_v     exner:")
        log.info(
            f"{carry.services.global_reductions.mean(rho_ndarray):.5e} "
            f"{carry.services.global_reductions.mean(vn_ndarray):.5e} "
            f"{carry.services.global_reductions.mean(w_ndarray):.5e} "
            f"{carry.services.global_reductions.mean(theta_v_ndarray):.5e} "
            f"{carry.services.global_reductions.mean(exner_ndarray):.5e} "
        )


compute_mean_at_final_step = _as_step("compute_mean_at_final_step", _compute_mean_at_final)


def _finalize(carry: DriverLoopState) -> None:
    carry.services.timer_collection.show_timer_report()
    profiling_options = carry.config.driver.profiling_options
    if (
        profiling_options is not None
        and profiling_options.gt4py_metrics_level > gtx_metrics.DISABLED
    ):
        print(gtx_metrics.dumps())
        gtx_metrics.dump_json(profiling_options.gt4py_metrics_output_file)


finalize_step = _as_step("finalize_step", _finalize)
