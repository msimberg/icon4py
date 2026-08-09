# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

from icon4py.model.atmosphere.diffusion import diffusion_states
from icon4py.model.common.states import validation


def test_diffusion_fixtures_field_coverage() -> None:
    """The diffusion test fixtures must pass exactly the fields each dataclass declares."""
    source_path = Path(__file__).parent.parent / "fixtures.py"
    target_classes = (
        diffusion_states.DiffusionInterpolationState,
        diffusion_states.DiffusionMetricState,
    )
    call_sites = validation.read_kwargs_at_constructor_calls(
        source_path, "interpolation_state", target_classes
    )
    call_sites.update(
        validation.read_kwargs_at_constructor_calls(source_path, "metric_state", target_classes)
    )
    for cls in target_classes:
        kwargs = call_sites.get(cls.__name__, set())
        validation.assert_field_coverage(cls, {name: None for name in kwargs})
