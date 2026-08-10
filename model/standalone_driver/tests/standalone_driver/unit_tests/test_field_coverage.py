# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

from icon4py.model.atmosphere.diffusion import diffusion
from icon4py.model.atmosphere.dycore import solve_nonhydro as solve_nh
from icon4py.model.atmosphere.tracer_advection import tracer_advection
from icon4py.model.common.states import validation
from icon4py.model.standalone_driver import steps


def test_steps_build_typed_component_inputs_with_full_field_coverage() -> None:
    """Leaf steps in steps.py must pass exactly the fields each InputT dataclass declares."""
    source_path = Path(steps.__file__)
    target_classes = (
        solve_nh.SolveNonHydroInput,
        diffusion.DiffusionInput,
        tracer_advection.AdvectionInput,
    )
    call_sites: dict[str, set[str]] = {}
    for function_name in (
        "_solve_nh",
        "_diffuse_before_time_loop",
        "_diffusion",
        "_advect_tracer",
    ):
        call_sites.update(
            validation.read_kwargs_at_constructor_calls(source_path, function_name, target_classes)
        )
    for cls in target_classes:
        kwargs = call_sites.get(cls.__name__, set())
        validation.assert_field_coverage(cls, {name: None for name in kwargs})
