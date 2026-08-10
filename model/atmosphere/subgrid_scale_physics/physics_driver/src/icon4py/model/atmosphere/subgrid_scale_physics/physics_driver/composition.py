# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""eDSL composition for the physics driver."""

from __future__ import annotations

import dataclasses
import datetime
from typing import TYPE_CHECKING, Any, TypeVar

from icon4py.model.common.components.components import Component
from icon4py.model.common.components.physics_state import TypedPhysicsState
from icon4py.model.common.composition import Step, chain, named, sample, when
from icon4py.model.common.states import prognostic_state, tracer_states


if TYPE_CHECKING:
    from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver.physics_driver import (
        PhysicsProcess,
    )


@dataclasses.dataclass
class PhysicsLoopState:
    """Mutable carry object for the physics composition."""

    prognostic: prognostic_state.PrognosticState
    tracers: tracer_states.TracerState
    dtime: datetime.timedelta
    simulation_current_datetime: datetime.datetime
    sample_cache: dict[str, Any]


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


def gather_and_call_step(
    component: Component[InputT, OutputT],
    state: TypedPhysicsState[InputT, OutputT],
    process_name: str,
) -> Step[PhysicsLoopState]:
    """Gather inputs from the prognostic state and run the component."""

    def _gather_and_call(carry: PhysicsLoopState) -> None:
        inputs = state.gather_from_prognostic(carry.prognostic, carry.tracers)
        outputs = component.run(inputs)
        carry.sample_cache[process_name] = outputs

    return named(f"gather_and_call:{process_name}", _gather_and_call)


def apply_tendencies_step(
    state: TypedPhysicsState[InputT, OutputT], process_name: str
) -> Step[PhysicsLoopState]:
    """Apply the cached typed tendencies back to the prognostic state."""

    def _apply(carry: PhysicsLoopState) -> None:
        outputs = carry.sample_cache[process_name]
        state.apply_tendencies(outputs, carry.dtime)

    return named(f"apply_tendencies:{process_name}", _apply)


def store_diagnostics_step(
    state: TypedPhysicsState[InputT, OutputT], process_name: str
) -> Step[PhysicsLoopState]:
    """Store the cached diagnostic outputs without applying tendencies."""

    def _store(carry: PhysicsLoopState) -> None:
        outputs = carry.sample_cache[process_name]
        state.store_diagnostics(outputs)

    return named(f"store_diagnostics:{process_name}", _store)


def _build_process_step(
    process: PhysicsProcess[Any, Any],
) -> Step[PhysicsLoopState]:
    """Build the step chain for a single physics process."""
    if not process.time_control.enable_process:
        return named(f"{process.name}:disabled", lambda c: None)

    interval = process.time_control.interval
    if interval <= datetime.timedelta(0):
        # A non-positive interval can never fire; the driver already validates
        # intervals at construction, so this path is defensive.
        return named(f"{process.name}:never_fires", lambda c: None)

    gather_call = gather_and_call_step(process.component, process.state, process.name)

    def _in_window(carry: PhysicsLoopState) -> bool:
        return process.time_control.is_in_window(carry.simulation_current_datetime)

    def _clock(carry: PhysicsLoopState) -> datetime.timedelta:
        return carry.simulation_current_datetime - process.time_control.start_date

    if process.apply_forcing:
        post_sample_chain = apply_tendencies_step(process.state, process.name)
    else:
        post_sample_chain = store_diagnostics_step(process.state, process.name)

    return when(
        _in_window,
        then=chain(
            sample(
                gather_call,
                every=interval,
                clock=_clock,
                key=process.name,
                cache=lambda c: c.sample_cache,
            ),
            post_sample_chain,
            name=f"{process.name}:body",
        ),
        name=process.name,
    )


def build_physics_composition(
    processes: list[PhysicsProcess[Any, Any]],
) -> Step[PhysicsLoopState]:
    """Build the eDSL composition that runs each process in order."""
    return chain(*[_build_process_step(process) for process in processes], name="physics")
