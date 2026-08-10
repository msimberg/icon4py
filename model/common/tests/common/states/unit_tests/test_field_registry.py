# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for ``icon4py.model.common.states.field_registry``."""

import dataclasses
from typing import Any

import pytest

from icon4py.model.atmosphere.dycore import dycore_states
from icon4py.model.atmosphere.tracer_advection import tracer_advection_states
from icon4py.model.common.grid import base, simple
from icon4py.model.common.states import field_registry, spec
from icon4py.model.common.utils import data_allocation as data_alloc


@dataclasses.dataclass(frozen=True)
class _Producer:
    out: Any = spec.spec(
        quantity="icon:test_scalar",
        units="1",
        dims=(),
        intent=spec.Intent.WRITE,
        lifetime=spec.Lifetime.PERSISTENT,
        role=spec.Role.IN_PLACE,
    )


@dataclasses.dataclass(frozen=True)
class _Consumer:
    inp: Any = spec.spec(
        quantity="icon:test_scalar",
        units="1",
        dims=(),
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.PERSISTENT,
    )


@dataclasses.dataclass(frozen=True)
class _DoubleProducer:
    out1: Any = spec.spec(
        quantity="icon:test_scalar",
        units="1",
        dims=(),
        intent=spec.Intent.WRITE,
        lifetime=spec.Lifetime.PERSISTENT,
        role=spec.Role.IN_PLACE,
    )
    out2: Any = spec.spec(
        quantity="icon:test_scalar",
        units="1",
        dims=(),
        intent=spec.Intent.WRITE,
        lifetime=spec.Lifetime.PERSISTENT,
        role=spec.Role.IN_PLACE,
    )


@dataclasses.dataclass(frozen=True)
class _StaticContainer:
    field: Any = spec.spec(
        quantity="icon:static_scalar",
        units="1",
        dims=(),
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
        default=None,
    )


@pytest.fixture
def grid() -> base.Grid:
    return simple.simple_grid()


def _sealed_registry_for_handoff(grid: base.Grid) -> field_registry.FieldRegistry:
    """Return a sealed registry that knows the dycore/advection handoff containers."""
    registry = field_registry.FieldRegistry(grid=grid, backend=None)
    registry.declare(dycore_states.PrepAdvection)
    registry.declare(tracer_advection_states.AdvectionPrepAdvState)
    registry.seal()
    return registry


def test_registry_shares_handoff_buffers_between_prep_advection_containers(
    grid: base.Grid,
) -> None:
    """The E1 dycore-to-advection handoff uses one buffer per quantity (D13)."""
    registry = _sealed_registry_for_handoff(grid)
    prep_adv = registry.build(dycore_states.PrepAdvection)
    tracer_prep = registry.build(tracer_advection_states.AdvectionPrepAdvState)

    assert tracer_prep.vn_traj is prep_adv.vn_traj
    assert tracer_prep.mass_flx_me is prep_adv.mass_flx_me
    assert tracer_prep.mass_flx_ic is prep_adv.dynamical_vertical_mass_flux_at_cells_on_half_levels


def test_registry_seal_raises_when_handoff_has_no_consumer(grid: base.Grid) -> None:
    registry = field_registry.FieldRegistry(grid=grid, backend=None)
    registry.declare(_Producer)
    with pytest.raises(ValueError, match=r"icon:test_scalar.*no consumer"):
        registry.seal()


def test_registry_seal_raises_when_handoff_has_no_producer(grid: base.Grid) -> None:
    registry = field_registry.FieldRegistry(grid=grid, backend=None)
    registry.declare(_Consumer)
    with pytest.raises(ValueError, match=r"icon:test_scalar.*no producer"):
        registry.seal()


def test_registry_seal_raises_when_handoff_has_two_producers(grid: base.Grid) -> None:
    registry = field_registry.FieldRegistry(grid=grid, backend=None)
    registry.declare(_DoubleProducer)
    registry.declare(_Consumer)
    with pytest.raises(ValueError, match=r"icon:test_scalar.*multiple producers"):
        registry.seal()


def test_registry_seal_raises_when_handoff_has_two_consumers(grid: base.Grid) -> None:
    registry = field_registry.FieldRegistry(grid=grid, backend=None)
    registry.declare(_Producer)
    registry.declare(_Consumer)
    registry.declare(_Consumer)
    with pytest.raises(ValueError, match=r"icon:test_scalar.*multiple consumers"):
        registry.seal()


def test_registry_build_emits_static_recipe_buffer(grid: base.Grid) -> None:
    """A registry-emitted static container holds the very buffer the recipe computed."""
    registry = field_registry.FieldRegistry(grid=grid, backend=None)
    static_field = data_alloc.zero_field(grid, dtype=float)
    registry.recipe("icon:static_scalar", lambda: static_field)
    registry.declare(_StaticContainer)
    registry.seal()

    container = registry.build(_StaticContainer)
    assert container.field is static_field
    assert registry.buffer("icon:static_scalar") is static_field


def test_registry_stale_epoch_access_raises(grid: base.Grid) -> None:
    registry = _sealed_registry_for_handoff(grid)
    prep_adv = registry.build(dycore_states.PrepAdvection)
    registry.bump_epoch()

    with pytest.raises(ValueError, match=r"Stale epoch"):
        _ = prep_adv.vn_traj


def test_registry_generation_bump_does_not_invalidate_access(grid: base.Grid) -> None:
    registry = _sealed_registry_for_handoff(grid)
    prep_adv = registry.build(dycore_states.PrepAdvection)
    registry.bump_generation(prep_adv)

    # Access must remain valid across a generation bump.
    assert prep_adv.vn_traj is not None
