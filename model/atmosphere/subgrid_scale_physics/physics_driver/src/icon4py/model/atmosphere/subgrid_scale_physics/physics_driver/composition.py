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
from collections.abc import Callable
from typing import Any, TypeVar

from icon4py.model.common.components.components import Component
from icon4py.model.common.components.physics_state import TypedPhysicsState
from icon4py.model.common.composition import Step, chain, sample, when
from icon4py.model.common.states import prognostic_state, tracer_states


class ForcingMode(enum.IntEnum):
    """Per-process apply switch -- the icon4py analogue of AES ``fc_xxx``.

    Decides whether a process's computed forcing is fed back into the prognostic
    state when the process runs:

    - APPLY:      compute and apply it (``field += tend*dt``); the process affects the run.
    - DIAGNOSTIC: compute it but do NOT apply it -- the outputs stay available for
      inspection/output while the prognostic state is left unchanged ("look, don't touch").
    """

    DIAGNOSTIC = 0
    APPLY = 1


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


class _NamedStep:
    """Simple wrapper giving a callable step a ``name`` attribute."""

    def __init__(self, name: str, fn: Callable[[PhysicsLoopState], None]) -> None:
        self.name = name
        self._fn = fn

    def __call__(self, carry: PhysicsLoopState) -> None:
        self._fn(carry)


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

    return _NamedStep(f"gather_and_call:{process_name}", _gather_and_call)


def apply_step(
    state: TypedPhysicsState[InputT, OutputT], process_name: str
) -> Step[PhysicsLoopState]:
    """Apply the cached typed outputs back to the prognostic state."""

    def _apply(carry: PhysicsLoopState) -> None:
        outputs = carry.sample_cache[process_name]
        state.scatter_to_prognostic(outputs, carry.dtime)

    return _NamedStep(f"apply:{process_name}", _apply)


def diagnose_step(process_name: str) -> Step[PhysicsLoopState]:
    """Diagnostic mode: no tendency is applied to the prognostic state.

    The component has already run and its outputs are cached; this step is a
    no-op placeholder kept so the composition tree shows the diagnose branch
    explicitly.
    """

    def _diagnose(carry: PhysicsLoopState) -> None:
        del carry

    return _NamedStep(f"diagnose:{process_name}", _diagnose)


def build_physics_composition(
    processes: list[Any],
) -> Step[PhysicsLoopState]:
    """Build the eDSL composition that runs each process in order."""

    def _process_step(process: Any) -> Step[PhysicsLoopState]:
        return chain(
            when(
                lambda c: (
                    process.time_control.enable_process
                    and process.time_control.is_in_window(c.simulation_current_datetime)
                ),
                then=chain(
                    sample(
                        gather_and_call_step(process.component, process.state, process.name),
                        every=process.time_control.interval,
                        clock=lambda c: (
                            c.simulation_current_datetime - process.time_control.start_date
                        ),
                        key=process.name,
                        cache=lambda c: c.sample_cache,
                    ),
                    when(
                        lambda c: process.forcing_mode is ForcingMode.APPLY,
                        then=apply_step(process.state, process.name),
                        else_=diagnose_step(process.name),
                    ),
                ),
                name=process.name,
            ),
        )

    return chain(*[_process_step(process) for process in processes], name="physics")
