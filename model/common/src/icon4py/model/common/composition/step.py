# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Core ``Step`` protocol and swap policy for the composition layer."""

from __future__ import annotations

import enum
from typing import Protocol, TypeVar


C_contra = TypeVar("C_contra", contravariant=True)


class Step(Protocol[C_contra]):
    """A named, state-mutating step in a driver composition.

    Steps operate on a carry object in place and return ``None``; the carry
    type is intentionally generic so that ``icon4py.model.common.composition``
    stays free of atmosphere- or driver-specific imports.
    """

    name: str

    def __call__(self, carry: C_contra) -> None: ...


class SwapPolicy(enum.Enum):
    """When ``repeat`` should swap a double-buffered target after each iteration."""

    NEVER = enum.auto()
    ALWAYS = enum.auto()
    EXCEPT_LAST = enum.auto()
