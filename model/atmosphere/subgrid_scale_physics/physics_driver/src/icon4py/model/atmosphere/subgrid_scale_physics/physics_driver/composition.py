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
import enum
from typing import TYPE_CHECKING, Any, TypeVar

from icon4py.model.common.components.components import Component
from icon4py.model.common.components.physics_state import TypedPhysicsState
from icon4py.model.common.composition import Step, chain, named, sample, when
from icon4py.model.common.states import prognostic_state, tracer_states


if TYPE_CHECKING:
    from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver.physics_driver import (
        PhysicsProcess,
    )


class PhysicsCoupling(enum.Enum):
    """How physics processes exchange with the prognostic state within one step.

    SERIAL (Gauss-Seidel): each process gathers from the state left by the
    previous process's apply. ICON's default coupling, and the default here:
    it is the validated variant.

    PARALLEL (Jacobi): every process gathers from the same step-entry state,
    then all applies run after the last compute. The composition machinery
    expresses the schedule; a single accumulated apply (sum all tendencies,
    one exact-EOS write back, as in PR C2SM/icon4py#1436) needs accumulator
    buffers this branch does not have: per-process applies remain
    per-process.
    """

    SERIAL = enum.auto()
    PARALLEL = enum.auto()


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

    return named(f"gather_and_call:{process_name}", _gather_and_call, component=component)


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


def _process_parts(
    process: PhysicsProcess[Any, Any],
) -> tuple[Step[PhysicsLoopState], Step[PhysicsLoopState] | None]:
    """The windowed compute phase and (optional) write-back phase for one process.

    The compute phase runs ``component.run`` on inputs gathered from the carry
    (subject to the ``sample`` cadence) and caches the typed outputs. The
    write-back phase is present only when ``apply_forcing`` is true: it writes the
    cached tendencies/diagnostics back to the prognostic state. Diagnostic-mode
    processes (``apply_forcing=False``) instead store diagnostics and need no
    separate apply phase; their store is part of the compute phase.
    """
    if not process.time_control.enable_process:
        return named(f"{process.name}:disabled", lambda c: None), None

    interval = process.time_control.interval
    if interval <= datetime.timedelta(0):
        # A non-positive interval can never fire; the driver already validates
        # intervals at construction, so this path is defensive.
        return named(f"{process.name}:never_fires", lambda c: None), None

    gather_call = gather_and_call_step(process.component, process.state, process.name)

    def _in_window(carry: PhysicsLoopState) -> bool:
        return process.time_control.is_in_window(carry.simulation_current_datetime)

    def _clock(carry: PhysicsLoopState) -> datetime.timedelta:
        return carry.simulation_current_datetime - process.time_control.start_date

    compute = when(
        _in_window,
        then=sample(
            gather_call,
            every=interval,
            clock=_clock,
            key=process.name,
            cache=lambda c: c.sample_cache,
        ),
        name=process.name,
    )

    if not process.apply_forcing:
        store = when(
            _in_window,
            then=store_diagnostics_step(process.state, process.name),
            name=f"{process.name}:store",
        )
        return chain(compute, store, name=f"{process.name}:body"), None

    apply = when(
        _in_window,
        then=apply_tendencies_step(process.state, process.name),
        name=f"{process.name}:apply",
    )
    return compute, apply


def build_physics_composition(
    processes: list[PhysicsProcess[Any, Any]],
    coupling: PhysicsCoupling = PhysicsCoupling.SERIAL,
) -> Step[PhysicsLoopState]:
    """Build the eDSL composition that runs the registered processes.

    SERIAL (Gauss-Seidel, default): each process's compute phase is followed
    immediately by its write-back, so later processes read the state left by
    earlier applies. PARALLEL (Jacobi): all compute phases run first, every
    process gathering the same step-entry state; all write-backs run after the
    last compute.
    """
    phases = [_process_parts(process) for process in processes]
    if coupling is PhysicsCoupling.PARALLEL:
        steps = [compute for compute, _ in phases] + [
            apply for _, apply in phases if apply is not None
        ]
    else:
        steps = [
            chain(compute, apply, name=f"{name}:body") if apply is not None else compute
            for (compute, apply), name in zip(phases, (p.name for p in processes), strict=True)
        ]
    return chain(*steps, name="physics")
