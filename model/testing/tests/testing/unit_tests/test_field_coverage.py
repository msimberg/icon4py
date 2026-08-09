# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import icon4py.model.common.grid.states as grid_states
from icon4py.model.common.states import validation
from icon4py.model.testing import serialbox as serialbox_mod


def test_serialbox_geometry_field_coverage() -> None:
    """The serialbox geometry helpers must pass exactly the fields each class declares."""
    source_path = Path(serialbox_mod.__file__)
    for function_name, target_classes in (
        ("construct_edge_geometry", (grid_states.EdgeParams,)),
        ("construct_cell_geometry", (grid_states.CellParams,)),
    ):
        call_sites = validation.read_kwargs_at_constructor_calls(
            source_path, function_name, target_classes
        )
        for cls in target_classes:
            kwargs = call_sites.get(cls.__name__, set())
            validation.assert_field_coverage(cls, {name: None for name in kwargs})
