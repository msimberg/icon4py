# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Carry object shared by the eDSL and plain drivers."""

from __future__ import annotations

import dataclasses
import datetime
import types
from collections.abc import Callable

import gt4py.next as gtx

from icon4py.model.common import prescribed_tendencies
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.io import io as common_io
from icon4py.model.common.states import static_fields
from icon4py.model.standalone_driver import (
    config as driver_config,
    driver_io,
    driver_states,
    driver_utils,
)


@dataclasses.dataclass(frozen=True)
class DriverServices:
    """Services from ``Icon4pyDriver`` that the loop body needs."""

    exchange: decomposition_defs.ExchangeRuntime
    global_reductions: decomposition_defs.Reductions
    io_monitor: common_io.IOMonitor | None
    tendencies: prescribed_tendencies.PrescribedTendencies | None
    timer_collection: driver_states.TimerCollection
    static_field_factories: static_fields.StaticFieldFactories
    backend: gtx.typing.Backend | None
    xp: types.ModuleType
    allocator: gtx.typing.Allocator
    diagnostics_computer: driver_io.DiagnosticsComputer
    compute_airmass: Callable[..., None]


@dataclasses.dataclass
class StepInfo:
    """Per-dycore-substep context used by the substep leaf steps."""

    substep_index: int
    at_first_substep: bool
    at_last_substep: bool
    at_initial_timestep: bool


@dataclasses.dataclass
class DriverLoopState:
    """Mutable carry object for the time-integration composition."""

    clock: driver_states.ModelTimeVariables
    states: driver_states.DriverStates
    granules: driver_utils.Granules
    config: driver_config.ExperimentConfig
    services: DriverServices
    wall_clock_starting_time: datetime.datetime
    time_step_index: int = 0
    substep_index: int = 0
    substep_total: int = 0

    def begin_time_step(self, index: int, total: int) -> None:
        """Update the carry for the start of the outer time-step loop iteration."""
        del total
        self.time_step_index = index

    def begin_substep(self, index: int, total: int) -> None:
        """Store the current dycore substep index and total for leaf steps."""
        self.substep_index = index
        self.substep_total = total
