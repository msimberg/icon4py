# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

from icon4py.model.atmosphere.dycore import dycore_states
from icon4py.model.common.states import nonhydro_states, validation


def test_dycore_utils_field_coverage() -> None:
    """The dycore test helpers must pass exactly the fields each dataclass declares."""
    source_path = Path(__file__).parent.parent / "utils.py"
    target_classes = (
        dycore_states.InterpolationState,
        dycore_states.MetricStateNonHydro,
        nonhydro_states.DiagnosticStateNonHydro,
    )
    call_sites: dict[str, set[str]] = {}
    for function_name in (
        "construct_interpolation_state",
        "construct_metric_state",
        "construct_diagnostics",
    ):
        call_sites.update(
            validation.read_kwargs_at_constructor_calls(source_path, function_name, target_classes)
        )
    for cls in target_classes:
        kwargs = call_sites.get(cls.__name__, set())
        validation.assert_field_coverage(cls, {name: None for name in kwargs})
