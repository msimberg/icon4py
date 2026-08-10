# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

from icon4py.model.atmosphere.subgrid_scale_physics.muphys import state as muphys_state
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.component import MuphysInput
from icon4py.model.common.states import validation


def test_muphys_state_gather_builds_input_with_full_field_coverage() -> None:
    """``TypedPhysicsState.gather_from_prognostic`` must pass exactly the fields ``MuphysInput`` declares."""
    source_path = Path(muphys_state.__file__)
    call_sites = validation.read_kwargs_at_constructor_calls(
        source_path, "gather_from_prognostic", (MuphysInput,)
    )
    kwargs = call_sites.get(MuphysInput.__name__, set())
    validation.assert_field_coverage(MuphysInput, {name: None for name in kwargs})
