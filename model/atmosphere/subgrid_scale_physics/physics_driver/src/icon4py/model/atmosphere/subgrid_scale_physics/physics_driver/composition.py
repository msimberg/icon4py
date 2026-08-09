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
from typing import Any

from icon4py.model.common.composition import Step, chain, sample, when


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

    prognostic: Any
    tracers: Any
    dtime: datetime.timedelta
    simulation_current_datetime: datetime.datetime
    sample_cache: dict[str, Any]


class _NamedStep:
    """Simple wrapper giving a callable step a ``name`` attribute."""

    def __init__(self, name: str, fn: Callable[[PhysicsLoopState], None]) -> None:
        self.name = name
        self._fn = fn

    def __call__(self, carry: PhysicsLoopState) -> None:
        self._fn(carry)


def gather_step(state: Any) -> Step[PhysicsLoopState]:
    def _gather(carry: PhysicsLoopState) -> None:
        state.gather_from_prognostic(carry.prognostic, carry.tracers)

    return _NamedStep("gather", _gather)


def component_call_step(process: Any) -> Step[PhysicsLoopState]:
    def _call(carry: PhysicsLoopState) -> None:
        outputs = process.component(
            process.state.as_component_input(), carry.simulation_current_datetime
        )
        carry.sample_cache[process.name] = outputs

    return _NamedStep(f"component_call:{process.name}", _call)


def apply_step(process: Any) -> Step[PhysicsLoopState]:
    def _apply(carry: PhysicsLoopState) -> None:
        process.state.scatter_to_prognostic(
            carry.prognostic,
            carry.sample_cache[process.name],
            carry.dtime,
        )

    return _NamedStep(f"apply:{process.name}", _apply)


def diagnose_step(process_name: str) -> Step[PhysicsLoopState]:
    def _diagnose(carry: PhysicsLoopState) -> None:
        del carry
        raise NotImplementedError(
            f"process '{process_name}': only ForcingMode.APPLY is implemented; "
            "DIAGNOSTIC requires splitting scatter_to_prognostic into "
            "apply-tendencies vs store-diagnostics"
        )

    return _NamedStep(f"diagnose:{process_name}", _diagnose)


def build_physics_composition(processes: list[Any]) -> Step[PhysicsLoopState]:
    """Build the eDSL composition that runs each process in order."""

    def _process_step(process: Any) -> Step[PhysicsLoopState]:
        return chain(
            gather_step(process.state),
            when(
                lambda c: (
                    process.time_control.enable_process
                    and process.time_control.is_in_window(c.simulation_current_datetime)
                ),
                then=chain(
                    sample(
                        component_call_step(process),
                        every=process.time_control.interval,
                        clock=lambda c: (
                            c.simulation_current_datetime - process.time_control.start_date
                        ),
                        key=process.name,
                        cache=lambda c: c.sample_cache,
                    ),
                    when(
                        lambda c: process.forcing_mode is ForcingMode.APPLY,
                        then=apply_step(process),
                        else_=diagnose_step(process.name),
                    ),
                ),
                name=process.name,
            ),
        )

    return chain(*[_process_step(process) for process in processes], name="physics")
