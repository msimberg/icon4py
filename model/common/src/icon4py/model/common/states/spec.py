# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Field-level declaration metadata for typed component boundaries."""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Sequence
from typing import Any

import gt4py.next as gtx


class Intent(enum.Enum):
    """Access intent of a field at a component boundary."""

    READ = "read"
    WRITE = "write"
    READWRITE = "readwrite"


class Lifetime(enum.Enum):
    """Lifetime class of a field."""

    STATIC = "static"
    PERSISTENT = "persistent"
    SCRATCH = "scratch"


class Role(enum.Enum):
    """Role of an output field (how the framework routes it)."""

    TENDENCY = "tendency"
    DIAGNOSTIC = "diagnostic"
    IN_PLACE = "in_place"


@dataclasses.dataclass(frozen=True)
class FieldSpec:
    """Declaration metadata attached to a field via ``spec()``."""

    #: Canonical quantity name. Unprefixed names are CF standard names;
    #: ``icon:<name>`` is used when no CF equivalent exists.
    quantity: str

    #: Physical units of the quantity.
    units: str

    #: Placement dimensions (one source of truth for full/half levels).
    dims: tuple[gtx.Dimension, ...]

    #: Access intent at the component boundary.
    intent: Intent

    #: Lifetime class.
    lifetime: Lifetime

    #: Whether the field participates in restart/checkpoint output.
    restart: bool = False

    #: Output role; meaningful only for fields declared on an ``OutputT``.
    role: Role | None = None

    #: Declarative labels for the field (e.g. "prognostic", "diagnostic",
    #: "output", "restart"). Used by queries that build output/restart sets.
    labels: tuple[str, ...] = ()


def spec(
    *,
    quantity: str,
    units: str,
    dims: tuple[gtx.Dimension, ...],
    intent: Intent,
    lifetime: Lifetime,
    restart: bool = False,
    role: Role | None = None,
    labels: Sequence[str] | None = None,
    default: Any = dataclasses.MISSING,
) -> Any:
    """Return a ``dataclasses.field`` carrying a ``FieldSpec`` in its metadata.

    The field has no default unless ``default`` is explicitly supplied. This
    matches the gt4py program-argument rule that program arguments must be
    dataclasses without defaulted fields, while still allowing optional fields
    (e.g. inactive tracers) to declare a default of ``None``.

    The return type is ``Any`` so that mypy accepts the assignment to typed
    dataclass fields (``dataclasses.field`` itself is special-cased by mypy,
    but wrappers around it are not).
    """
    field_spec = FieldSpec(
        quantity=quantity,
        units=units,
        dims=dims,
        intent=intent,
        lifetime=lifetime,
        restart=restart,
        role=role,
        labels=tuple(labels) if labels is not None else (),
    )
    metadata = {"spec": field_spec}
    if default is dataclasses.MISSING:
        return dataclasses.field(metadata=metadata)
    return dataclasses.field(default=default, metadata=metadata)


def get_field_spec(field: dataclasses.Field) -> FieldSpec | None:
    """Return the ``FieldSpec`` from a dataclass field, or ``None``."""
    return field.metadata.get("spec") if hasattr(field.metadata, "get") else None


def field_labels(field: dataclasses.Field) -> tuple[str, ...]:
    """Return the labels declared on a field, or an empty tuple."""
    spec = get_field_spec(field)
    return spec.labels if spec is not None else ()


def has_label(field: dataclasses.Field, label: str) -> bool:
    """Return whether ``field`` carries ``label`` in its ``spec()``."""
    return label in field_labels(field)


def fields_with_label(cls: type[Any], label: str) -> tuple[str, ...]:
    """Return the names of fields on ``cls`` that carry ``label``."""
    return tuple(f.name for f in dataclasses.fields(cls) if has_label(f, label))


def field_spec_from_metadata(
    field_name: str,
    metadata: dict[str, Any],
    *,
    intent: Intent = Intent.READ,
    lifetime: Lifetime = Lifetime.STATIC,
) -> FieldSpec:
    """Build a ``FieldSpec`` from the existing ``FieldMetaData`` shape.

    The canonical quantity name is taken from ``standard_name`` when present;
    otherwise ``icon:<field_name>`` is used.
    """
    standard_name = metadata.get("standard_name", f"icon:{field_name}")
    return FieldSpec(
        quantity=standard_name,
        units=metadata.get("units", ""),
        dims=metadata.get("dims", ()),
        intent=intent,
        lifetime=lifetime,
    )
