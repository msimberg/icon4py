# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Typed physics-state boundary protocol."""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Protocol, TypeVar


if TYPE_CHECKING:
    from icon4py.model.common.states import prognostic_state, tracer_states

InputT_co = TypeVar("InputT_co", covariant=True)
OutputT_contra = TypeVar("OutputT_contra", contravariant=True)


class TypedPhysicsState(Protocol[InputT_co, OutputT_contra]):
    """Typed boundary between the physics driver and a physics component.

    The adapter declares exactly the prognostic and tracer fields it reads
    (``gather_from_prognostic``) and how the component's outputs are applied
    back to the prognostic state (``scatter_to_prognostic``) by inspecting the
    ``role`` declared on each output field.
    """

    def gather_from_prognostic(
        self,
        prognostic: prognostic_state.PrognosticState,
        tracers: tracer_states.TracerState,
    ) -> InputT_co: ...

    def scatter_to_prognostic(
        self,
        outputs: OutputT_contra,
        dtime: datetime.timedelta,
    ) -> None: ...
