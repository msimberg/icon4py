# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Typed states and components.

A `Quantity` is a type-level tag of one quantity at one place on the grid, a `Field` a gt4py
field tagged by its quantity, a `State` a frozen dataclass of typed leaves and a `Component`
a `run(inputs, out=None) -> Output` over nested `Input` and `Output` states. Stencils read
`Field.data`; mypy and pyright see `Field[VnOnEdgeK]` and `Field[ThetaVOnCellK]` as
different types.
"""

from __future__ import annotations

import dataclasses
import functools
import types
import typing
from collections.abc import Callable, Collection, Iterator, Mapping
from typing import Any, ClassVar, Literal, dataclass_transform

import gt4py.next as gtx

from icon4py.model.common import type_alias as ta
from icon4py.model.common.utils import PredictorCorrectorPair, TimeStepPair, data_allocation


if typing.TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid


__all__ = [
    "Component",
    "Empty",
    "Field",
    "PredictorCorrectorPair",
    "Quantity",
    "State",
    "Sums",
    "Tendency",
    "TimeStepPair",
    "allocate",
    "copy",
    "file_variables",
    "zeros",
]


class Quantity:
    """
    A type-level tag, one subclass per quantity at one place on the grid, never instantiated.

    The metadata sits on the class; a keyword not given is inherited from the base (a marker
    base like `Tendency` gives none). `precision` names the floating point type the field is
    allocated with, resolved through `type_alias` when the field is allocated.
    """

    dims: ClassVar[tuple[gtx.Dimension, ...]]
    standard_name: ClassVar[str | None] = None
    units: ClassVar[str]
    long_name: ClassVar[str | None] = None
    precision: ClassVar[Literal["wp", "vp"]] = "wp"

    def __init_subclass__(
        cls,
        *,
        dims: tuple[gtx.Dimension, ...] | None = None,
        standard_name: str | None = None,
        units: str | None = None,
        long_name: str | None = None,
        precision: Literal["wp", "vp"] | None = None,
    ) -> None:
        super().__init_subclass__()
        given = {
            "dims": dims,
            "standard_name": standard_name,
            "units": units,
            "long_name": long_name,
            "precision": precision,
        }
        for name, value in given.items():
            if value is not None:
                setattr(cls, name, value)

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(f"'{cls.__name__}' is a type-level tag, not a value.")


class Tendency(Quantity):
    """Marker base of the tendencies: what the physics driver accumulates."""


class Field[Q: Quantity]:
    """
    A gt4py field tagged by its quantity.

    The tag is a phantom type parameter, invariant: `Field[VnOnEdgeK]` is not a
    `Field[Quantity]`, so a signature `(Field[Q], Field[Q])` means the same quantity twice.
    """

    __slots__ = ("data", "quantity")

    def __init__(self, quantity: type[Q], data: gtx.Field[Any, Any]) -> None:
        self.quantity = quantity
        self.data = data

    def __repr__(self) -> str:
        return f"Field({self.quantity.__name__}, {self.data.ndarray.shape})"


def _dtype(quantity: type[Quantity]) -> Any:
    return ta.vpfloat if quantity.precision == "vp" else ta.wpfloat


def zeros[Q: Quantity](
    quantity: type[Q], grid: base_grid.Grid, allocator: gtx_typing.Allocator | None
) -> Field[Q]:
    """A zero field of the quantity's dimensions and precision."""
    return Field(
        quantity,
        data_allocation.zero_field(
            grid, *quantity.dims, dtype=_dtype(quantity), allocator=allocator
        ),
    )


@dataclasses.dataclass(frozen=True)
class Decl:
    """One `Field` leaf of a `State` class: its name, its quantity and whether it may be None."""

    name: str
    quantity: type[Quantity]
    optional: bool = False


@dataclass_transform(frozen_default=True, kw_only_default=True, eq_default=False)
class State:
    """
    A frozen, keyword-only dataclass of typed leaves.

    `Field` leaves are read back from the annotations by `declarations()`; plain leaves
    (floats, ints, datetimes) are ordinary fields and are not listed. A leaf annotated
    `Field[Q] | None` is optional: `allocate` may leave it None and `leaves()` skips it.

    `declarations` and `leaves` are internal to the components package: consumers use the
    named operations (`allocate`, `copy`, `copy_from`, `fields`, `file_variables`, `Sums`).
    """

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        dataclasses.dataclass(frozen=True, eq=False, kw_only=True)(cls)

    @classmethod
    def declarations(cls) -> tuple[Decl, ...]:
        """
        The `Field` leaves of the class, in declaration order; needs no values.

        Internal to the components package: consumers use the named operations.
        """
        return _declarations(cls)

    def leaves(self) -> Iterator[tuple[Decl, Field[Any]]]:
        """
        The declarations with their fields, skipping the optional leaves that are None.

        Internal to the components package: consumers use the named operations.
        """
        for declaration in self.declarations():
            field = getattr(self, declaration.name)
            if field is not None:
                yield declaration, field

    def fields(self) -> dict[str, Field[Any]]:
        """The present leaves by name, in declaration order: the verification view."""
        return {declaration.name: field for declaration, field in self.leaves()}

    def copy_from(self, source: State) -> None:
        """
        Make every present leaf hold the source's values, in place.

        A leaf whose buffer is the source's own is skipped, so a state can be continued
        from a view of itself. The source must hold every leaf of `self`; the plain
        leaves are untouched.
        """
        for declaration, target in self.leaves():
            source_field: Field[Any] = getattr(source, declaration.name)
            if target.data is not source_field.data:
                buffer: Any = target.data.ndarray  # NDArrayObject declares no __setitem__
                buffer[...] = source_field.data.ndarray


_DECLARATIONS: dict[type[State], tuple[Decl, ...]] = {}


def _declarations(cls: type[State]) -> tuple[Decl, ...]:
    if cls not in _DECLARATIONS:
        hints = typing.get_type_hints(cls)
        found = []
        for field in dataclasses.fields(cls):  # type: ignore[arg-type]
            hint, optional = _without_none(hints[field.name])
            if typing.get_origin(hint) is Field:
                (quantity,) = typing.get_args(hint)
                found.append(Decl(field.name, quantity, optional))
        _DECLARATIONS[cls] = tuple(found)
    return _DECLARATIONS[cls]


def _without_none(hint: Any) -> tuple[Any, bool]:
    """`X | None` as `(X, True)`, any other hint as `(hint, False)`."""
    if typing.get_origin(hint) in (types.UnionType, typing.Union):
        others = [arg for arg in typing.get_args(hint) if arg is not type(None)]
        if len(others) == 1:
            return others[0], True
    return hint, False


class Empty(State):
    pass


def allocate[S: State](
    cls: type[S],
    grid: base_grid.Grid,
    allocator: gtx_typing.Allocator | None,
    *,
    fill: Callable[[str, tuple[int, ...]], Any] | None = None,
    only: Collection[str] | None = None,
) -> S:
    """
    A state with a zero field per declaration.

    `fill(name, shape)` gives the value of a leaf instead of zeros. `only` names the optional
    leaves to allocate; the other optional leaves are None. Without it every leaf is allocated.
    """
    values: dict[str, Any] = {}
    for declaration in cls.declarations():
        if declaration.optional and only is not None and declaration.name not in only:
            values[declaration.name] = None
            continue
        field = zeros(declaration.quantity, grid, allocator)
        if fill is not None:
            buffer: Any = field.data.ndarray  # NDArrayObject declares no __setitem__
            buffer[...] = fill(declaration.name, buffer.shape)
        values[declaration.name] = field
    return cls(**values)


def copy[S: State](state: S, allocator: gtx_typing.Allocator | None) -> S:
    """A new state with a copy of each present leaf on the allocator; the plain leaves are shared."""
    copies: dict[str, Any] = {
        declaration.name: Field(
            declaration.quantity, data_allocation.reallocate(field.data, allocator=allocator)
        )
        for declaration, field in state.leaves()
    }
    copied: S = dataclasses.replace(typing.cast(Any, state), **copies)
    return copied


def file_variables(cls: type[State], *, aliases: Mapping[str, str] | None = None) -> dict[str, str]:
    """
    The output file variable of every leaf, as `{variable name: leaf name}`.

    A leaf's variable is its alias when it has one, else its quantity's `standard_name`;
    a leaf with neither is an error, as are two leaves writing the same variable.
    """
    variables: dict[str, str] = {}
    for declaration in cls.declarations():
        variable = (aliases or {}).get(declaration.name, declaration.quantity.standard_name)
        if variable is None:
            raise ValueError(f"leaf '{declaration.name}' of {cls.__name__} has no variable name")
        if variable in variables:
            raise ValueError(
                f"leaves '{variables[variable]}' and '{declaration.name}' of {cls.__name__}"
                f" both write the variable '{variable}'"
            )
        variables[variable] = declaration.name
    return variables


class Sums:
    """The running elementwise sums of fields, one per leaf name, made when first seen."""

    def __init__(self, grid: base_grid.Grid, allocator: gtx_typing.Allocator | None) -> None:
        self._grid = grid
        self._allocator = allocator
        self._sums: dict[str, Field[Any]] = {}

    def add(self, state: State, *, quantities: type[Quantity] | None = None) -> None:
        """
        Add every field of `state` whose quantity is a `quantities` subclass.

        Every quantity qualifies when none is given; a sum is made when its name is
        first seen.
        """
        for declaration, field in state.leaves():
            if quantities is not None and not issubclass(declaration.quantity, quantities):
                continue
            if (total := self._sums.get(declaration.name)) is None:
                total = self._sums[declaration.name] = zeros(
                    declaration.quantity, self._grid, self._allocator
                )
            buffer: Any = total.data.ndarray  # NDArrayObject declares no __setitem__
            buffer[...] += field.data.ndarray

    def field(self, name: str) -> Field[Any] | None:
        """The sum of `name`, or None when no added field had that name."""
        return self._sums.get(name)

    def names(self) -> tuple[str, ...]:
        """The names of the sums made so far."""
        return tuple(self._sums)

    def zero(self) -> None:
        """Set every sum to zero, keeping the buffers."""
        for total in self._sums.values():
            buffer: Any = total.data.ndarray  # NDArrayObject declares no __setitem__
            buffer[...] = 0.0


class Component:
    """
    `run(inputs, out=None) -> Output`, the only method a component implements.

    `out = component.run(inputs)` writes into the component's own buffers, allocated once on
    first use; `component.run(inputs, out=view)` writes where the caller says, numpy `out=`
    style. Either way `run` returns the output. Whether `out` may alias the input is the
    component's business; nothing checks it here.

    The own buffers (`output`) hold every leaf of `Output`, the optional ones included: a
    component with optional leaves (`Advection`, `PhysicsDriver`: the tracers) would allocate
    all of them, so its composer always passes `out=`, views over the active ones only.
    """

    Input: type[State]
    Output: type[State]

    def __init__(self, grid: base_grid.Grid, allocator: gtx_typing.Allocator | None) -> None:
        self.grid = grid
        self.allocator = allocator

    @functools.cached_property
    def output(self) -> Any:
        return allocate(type(self).Output, self.grid, self.allocator)

    def buffers[S: State](self, out: S | None) -> S:
        """The caller's buffers when given, else this component's own."""
        return self.output if out is None else out

    def run(self, inputs: Any, out: Any = None) -> Any:
        raise NotImplementedError
