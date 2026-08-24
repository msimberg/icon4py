# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The ``PhysicsDriver`` and its process / time-control types."""

from __future__ import annotations

import dataclasses
import datetime
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver.composition import (
    PhysicsCoupling,
    PhysicsLoopState,
    build_physics_composition,
)
from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver.process_time_control import (
    ProcessTimeControl,
)
from icon4py.model.common.components.components import Component
from icon4py.model.common.components.physics_state import TypedPhysicsState
from icon4py.model.common.composition import Step


if TYPE_CHECKING:
    from icon4py.model.common.states import prognostic_state, tracer_states


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclasses.dataclass
class PhysicsProcess(Generic[InputT, OutputT]):
    """A registered physics process: a component, its state adapter, and its time control.

    The component is the per-process adapter (e.g. ``MuphysComponent``); it
    implements the typed ``Component`` protocol, which is how the driver types it.
    The state adapter is process-specific (it translates the prognostic state to/from
    *this* component's contract), so it is bundled per process rather than shared.

    ``apply_forcing`` is the per-process AES ``fc_xxx`` analogue. When true, the
    computed tendencies are fed back into the prognostic state; when false, the
    process runs in diagnostic mode and only stores its diagnostic outputs.
    """

    name: str
    component: Component[InputT, OutputT]
    state: TypedPhysicsState[InputT, OutputT]
    time_control: ProcessTimeControl
    apply_forcing: bool = True


class PhysicsDriver:
    """The physics driver: runs each registered physics process in order."""

    def __init__(
        self,
        processes: list[PhysicsProcess[Any, Any]],
        dtime: datetime.timedelta,
        coupling: PhysicsCoupling = PhysicsCoupling.SERIAL,
    ) -> None:
        self._processes = processes
        self._dtime = dtime
        self._coupling = coupling
        self.sample_cache: dict[str, Any] = {}
        self._composition: Step[PhysicsLoopState] | None = None
        self._validate_intervals()

    def _validate_intervals(self) -> None:
        for process in self._processes:
            process.time_control.validate_interval(self._dtime)

    @property
    def processes(self) -> list[PhysicsProcess[Any, Any]]:
        """The registered physics processes, in run order."""
        return self._processes

    @property
    def composition(self) -> Step[PhysicsLoopState]:
        """The physics composition (built once, then cached).

        Public so drivers can embed it in a larger composition (e.g. via
        ``nested``) for introspection, instead of only running it via ``run``.
        """
        if self._composition is None:
            self._composition = build_physics_composition(self._processes, coupling=self._coupling)
        return self._composition

    def make_carry(
        self,
        *,
        prognostic: prognostic_state.PrognosticState,
        tracers: tracer_states.TracerState,
        dtime: datetime.timedelta,
        simulation_current_datetime: datetime.datetime,
    ) -> PhysicsLoopState:
        """Build the physics loop carry, sharing this driver's sample cache."""
        return PhysicsLoopState(
            prognostic=prognostic,
            tracers=tracers,
            dtime=dtime,
            simulation_current_datetime=simulation_current_datetime,
            sample_cache=self.sample_cache,
        )

    def run(
        self,
        prognostic: prognostic_state.PrognosticState,
        tracers: tracer_states.TracerState,
        dtime: datetime.timedelta,
        simulation_current_datetime: datetime.datetime,
    ) -> None:
        carry = self.make_carry(
            prognostic=prognostic,
            tracers=tracers,
            dtime=dtime,
            simulation_current_datetime=simulation_current_datetime,
        )
        self.composition(carry)
