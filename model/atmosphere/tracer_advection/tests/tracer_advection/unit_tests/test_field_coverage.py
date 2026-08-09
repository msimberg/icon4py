# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

from icon4py.model.atmosphere.tracer_advection import tracer_advection_states
from icon4py.model.common.states import validation


def test_tracer_advection_utils_field_coverage() -> None:
    """The tracer-advection test helpers must pass exactly the fields each dataclass declares."""
    source_path = Path(__file__).parent.parent / "utils.py"
    target_classes = (
        tracer_advection_states.AdvectionInterpolationState,
        tracer_advection_states.AdvectionLeastSquaresState,
        tracer_advection_states.AdvectionMetricState,
        tracer_advection_states.AdvectionDiagnosticState,
        tracer_advection_states.AdvectionPrepAdvState,
    )
    call_sites: dict[str, set[str]] = {}
    for function_name in (
        "construct_interpolation_state",
        "construct_least_squares_state",
        "construct_metric_state",
        "construct_diagnostic_init_state",
        "construct_diagnostic_exit_state",
        "construct_prep_adv",
    ):
        call_sites.update(
            validation.read_kwargs_at_constructor_calls(source_path, function_name, target_classes)
        )
    for cls in target_classes:
        kwargs = call_sites.get(cls.__name__, set())
        validation.assert_field_coverage(cls, {name: None for name in kwargs})
