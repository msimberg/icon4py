# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Typed component protocol for model components."""

from __future__ import annotations

import dataclasses
from typing import Any, Protocol, TypeVar


InputT_contra = TypeVar("InputT_contra", contravariant=True)
OutputT_co = TypeVar("OutputT_co", covariant=True)


class Component(Protocol[InputT_contra, OutputT_co]):
    """A runnable model component with declared typed boundaries.

    Components are the building blocks of the model. They operate on a declared
    input dataclass and produce a declared output dataclass. Each field carries
    ``spec()`` metadata describing quantity, units, placement, intent, lifetime,
    and (for outputs) role.
    """

    @classmethod
    def input_type(cls) -> type[Any]:
        """Return the component's input dataclass type."""
        ...

    @classmethod
    def output_type(cls) -> type[Any]:
        """Return the component's output dataclass type."""
        ...

    @classmethod
    def declared_inputs(cls) -> tuple[str, ...]:
        """Field names of the component's input type.

        Read from ``spec()`` metadata via ``dataclasses.fields(cls.input_type())``.
        """
        return tuple(f.name for f in dataclasses.fields(cls.input_type()))

    @classmethod
    def declared_outputs(cls) -> tuple[str, ...]:
        """Field names of the component's output type.

        Read from ``spec()`` metadata via ``dataclasses.fields(cls.output_type())``.
        """
        return tuple(f.name for f in dataclasses.fields(cls.output_type()))

    def run(self, state: InputT_contra) -> OutputT_co:
        """Run the component on the input state and return the output state."""
        ...

    def __str__(self) -> str:
        return (
            f"instance of {self.__class__}(Component) uses inputs: "
            f"{self.declared_inputs()} \n "
            f"produces : {self.declared_outputs()}"
        )
