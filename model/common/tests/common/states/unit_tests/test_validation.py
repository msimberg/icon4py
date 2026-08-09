# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for ``icon4py.model.common.states.validation``."""

import dataclasses

import gt4py.next as gtx
import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.states import spec, validation


@dataclasses.dataclass(frozen=True)
class _ContainerA:
    rho: float = spec.spec(
        quantity="air_density",
        units="kg m-3",
        dims=(dims.CellDim, dims.KDim),
        intent=spec.Intent.READWRITE,
        lifetime=spec.Lifetime.PERSISTENT,
    )


@dataclasses.dataclass(frozen=True)
class _ContainerB:
    rho: float = spec.spec(
        quantity="air_density",
        units="kg m-3",
        dims=(dims.CellDim, dims.KDim),
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )


@dataclasses.dataclass(frozen=True)
class _ContainerC:
    rho: float = spec.spec(
        quantity="air_density",
        units="g m-3",
        dims=(dims.CellDim, dims.KDim),
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )


@dataclasses.dataclass(frozen=True)
class _ContainerD:
    rho: float = spec.spec(
        quantity="unknown_quantity",
        units="1",
        dims=(dims.CellDim,),
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )


@dataclasses.dataclass(frozen=True)
class _ContainerE:
    rho: float = spec.spec(
        quantity="air_density",
        units="kg m-3",
        dims=(dims.EdgeDim, dims.KDim),
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )


def _field_specs(cls: type) -> dict[str, spec.FieldSpec]:
    result: dict[str, spec.FieldSpec] = {}
    for f in dataclasses.fields(cls):
        field_spec = spec.get_field_spec(f)
        if field_spec is not None:
            result[f.name] = field_spec
    return result


def test_validate_consistent_specs_passes() -> None:
    validation.validate_consistent_specs(
        {"A": _field_specs(_ContainerA), "B": _field_specs(_ContainerB)}
    )


def test_validate_consistent_specs_raises_on_inconsistent_units() -> None:
    with pytest.raises(ValueError, match="air_density"):
        validation.validate_consistent_specs(
            {"A": _field_specs(_ContainerA), "C": _field_specs(_ContainerC)}
        )


def test_validate_consistent_specs_raises_on_unknown_quantity() -> None:
    with pytest.raises(ValueError, match="unknown_quantity"):
        validation.validate_consistent_specs(
            {"D": _field_specs(_ContainerD)},
            known_quantities={"air_density"},
        )


def test_assert_field_coverage_passes() -> None:
    validation.assert_field_coverage(_ContainerA, {"rho": 1.0})


def test_assert_field_coverage_raises_on_missing() -> None:
    with pytest.raises(ValueError, match="missing"):
        validation.assert_field_coverage(_ContainerA, {})


def test_assert_field_coverage_raises_on_extra() -> None:
    with pytest.raises(ValueError, match="extra"):
        validation.assert_field_coverage(_ContainerA, {"rho": 1.0, "extra": 2.0})


def test_validate_consistent_specs_raises_on_inconsistent_dims() -> None:
    with pytest.raises(ValueError, match="air_density"):
        validation.validate_consistent_specs(
            {"A": _field_specs(_ContainerA), "E": _field_specs(_ContainerE)}
        )
