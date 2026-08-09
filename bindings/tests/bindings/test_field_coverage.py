# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

from icon4py.bindings import diffusion_wrapper, dycore_wrapper
from icon4py.model.atmosphere.diffusion import diffusion_states
from icon4py.model.atmosphere.dycore import dycore_states
from icon4py.model.common.states import nonhydro_states, prognostic_state as prognostics, validation


def test_dycore_wrapper_solve_nh_run_field_coverage() -> None:
    """The Fortran binding solve-nh wrapper must pass exactly the fields each dataclass declares."""
    target_classes = (
        dycore_states.PrepAdvection,
        nonhydro_states.DiagnosticStateNonHydro,
        prognostics.PrognosticState,
    )
    call_sites = validation.read_kwargs_at_constructor_calls(
        Path(dycore_wrapper.__file__), "solve_nh_run", target_classes
    )
    for cls in target_classes:
        kwargs = call_sites.get(cls.__name__, set())
        validation.assert_field_coverage(cls, {name: None for name in kwargs})


def test_diffusion_wrapper_diffusion_init_field_coverage() -> None:
    """The Fortran binding diffusion init wrapper must pass exactly the fields each dataclass declares."""
    target_classes = (
        diffusion_states.DiffusionInterpolationState,
        diffusion_states.DiffusionMetricState,
    )
    call_sites = validation.read_kwargs_at_constructor_calls(
        Path(diffusion_wrapper.__file__), "diffusion_init", target_classes
    )
    for cls in target_classes:
        kwargs = call_sites.get(cls.__name__, set())
        validation.assert_field_coverage(cls, {name: None for name in kwargs})
