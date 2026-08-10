# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Single canonical derivation of temperature, pressure, and cell-centered winds."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.components.components import Component
from icon4py.model.common.diagnostic_calculations import pressure as pressure_diagnostics
from icon4py.model.common.diagnostic_calculations.stencils import diagnose_temperature
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.interpolation.stencils import edge_2_cell_vector_rbf_interpolation as rbf
from icon4py.model.common.states import quantities as q, spec


if TYPE_CHECKING:
    from icon4py.model.common.grid import base as grid_base


@dataclasses.dataclass(frozen=True)
class DerivedQuantitiesInput:
    """Inputs to the canonical T/p/u/v derivation.

    Output buffers appear first (no defaults) so the dataclass can mix plain
    fields with ``spec()``-declared input fields. Their producer relationship is
    declared on ``DerivedQuantitiesOutput``.
    """

    # Output buffers (written in place, declared as outputs below)
    temperature: fa.CellKField[ta.wpfloat]
    virtual_temperature: fa.CellKField[ta.wpfloat]
    pressure: fa.CellKField[ta.wpfloat]
    pressure_ifc: fa.CellKField[ta.wpfloat]
    surface_pressure: fa.CellField[ta.wpfloat]
    u: fa.CellKField[ta.wpfloat]
    v: fa.CellKField[ta.wpfloat]

    theta_v: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity=q.VIRTUAL_POTENTIAL_TEMPERATURE.name,
        units=q.VIRTUAL_POTENTIAL_TEMPERATURE.units,
        dims=q.VIRTUAL_POTENTIAL_TEMPERATURE.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.PERSISTENT,
    )
    exner: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity=q.DIMENSIONLESS_EXNER_FUNCTION.name,
        units=q.DIMENSIONLESS_EXNER_FUNCTION.units,
        dims=q.DIMENSIONLESS_EXNER_FUNCTION.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.PERSISTENT,
    )
    vn: fa.EdgeKField[ta.wpfloat] = spec.spec(
        quantity=q.NORMAL_VELOCITY.name,
        units=q.NORMAL_VELOCITY.units,
        dims=q.NORMAL_VELOCITY.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.PERSISTENT,
    )
    qv: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity=q.SPECIFIC_HUMIDITY.name,
        units=q.SPECIFIC_HUMIDITY.units,
        dims=q.SPECIFIC_HUMIDITY.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.PERSISTENT,
    )
    qc: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity=q.SPECIFIC_CLOUD_CONTENT.name,
        units=q.SPECIFIC_CLOUD_CONTENT.units,
        dims=q.SPECIFIC_CLOUD_CONTENT.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.PERSISTENT,
    )
    qi: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity=q.SPECIFIC_ICE_CONTENT.name,
        units=q.SPECIFIC_ICE_CONTENT.units,
        dims=q.SPECIFIC_ICE_CONTENT.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.PERSISTENT,
    )
    qr: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity=q.SPECIFIC_RAIN_CONTENT.name,
        units=q.SPECIFIC_RAIN_CONTENT.units,
        dims=q.SPECIFIC_RAIN_CONTENT.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.PERSISTENT,
    )
    qs: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity=q.SPECIFIC_SNOW_CONTENT.name,
        units=q.SPECIFIC_SNOW_CONTENT.units,
        dims=q.SPECIFIC_SNOW_CONTENT.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.PERSISTENT,
    )
    qg: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity=q.SPECIFIC_GRAUPEL_CONTENT.name,
        units=q.SPECIFIC_GRAUPEL_CONTENT.units,
        dims=q.SPECIFIC_GRAUPEL_CONTENT.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.PERSISTENT,
    )
    ddqz_z_full: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity=q.DDQZ_Z_HALF.name,
        units=q.DDQZ_Z_HALF.units,
        dims=q.DDQZ_Z_HALF.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    rbf_vec_coeff_c1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2EDim], ta.wpfloat] = spec.spec(
        quantity=q.RBF_VEC_COEFF_V1.name,
        units=q.RBF_VEC_COEFF_V1.units,
        dims=q.RBF_VEC_COEFF_V1.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    rbf_vec_coeff_c2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2EDim], ta.wpfloat] = spec.spec(
        quantity=q.RBF_VEC_COEFF_V2.name,
        units=q.RBF_VEC_COEFF_V2.units,
        dims=q.RBF_VEC_COEFF_V2.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )


@dataclasses.dataclass(frozen=True)
class DerivedQuantitiesOutput:
    """Buffers written by the canonical T/p/u/v derivation."""

    temperature: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity=q.AIR_TEMPERATURE.name,
        units=q.AIR_TEMPERATURE.units,
        dims=q.AIR_TEMPERATURE.dims,
        intent=spec.Intent.WRITE,
        lifetime=spec.Lifetime.PERSISTENT,
        labels=["diagnostic", "output"],
    )
    virtual_temperature: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity=q.AIR_VIRTUAL_TEMPERATURE.name,
        units=q.AIR_VIRTUAL_TEMPERATURE.units,
        dims=q.AIR_VIRTUAL_TEMPERATURE.dims,
        intent=spec.Intent.WRITE,
        lifetime=spec.Lifetime.PERSISTENT,
        labels=["diagnostic", "output"],
    )
    pressure: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity=q.AIR_PRESSURE.name,
        units=q.AIR_PRESSURE.units,
        dims=q.AIR_PRESSURE.dims,
        intent=spec.Intent.WRITE,
        lifetime=spec.Lifetime.PERSISTENT,
        labels=["diagnostic", "output"],
    )
    pressure_ifc: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity=q.AIR_PRESSURE_ON_INTERFACE_LEVELS.name,
        units=q.AIR_PRESSURE_ON_INTERFACE_LEVELS.units,
        dims=q.AIR_PRESSURE_ON_INTERFACE_LEVELS.dims,
        intent=spec.Intent.WRITE,
        lifetime=spec.Lifetime.PERSISTENT,
        labels=["diagnostic", "output", "restart"],
    )
    surface_pressure: fa.CellField[ta.wpfloat] = spec.spec(
        quantity=q.AIR_PRESSURE_AT_GROUND_LEVEL.name,
        units=q.AIR_PRESSURE_AT_GROUND_LEVEL.units,
        dims=q.AIR_PRESSURE_AT_GROUND_LEVEL.dims,
        intent=spec.Intent.WRITE,
        lifetime=spec.Lifetime.PERSISTENT,
    )
    u: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity=q.EASTWARD_WIND.name,
        units=q.EASTWARD_WIND.units,
        dims=q.EASTWARD_WIND.dims,
        intent=spec.Intent.WRITE,
        lifetime=spec.Lifetime.PERSISTENT,
        labels=["diagnostic", "output"],
    )
    v: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity=q.NORTHWARD_WIND.name,
        units=q.NORTHWARD_WIND.units,
        dims=q.NORTHWARD_WIND.dims,
        intent=spec.Intent.WRITE,
        lifetime=spec.Lifetime.PERSISTENT,
        labels=["diagnostic", "output"],
    )


class DerivedQuantities(Component[DerivedQuantitiesInput, DerivedQuantitiesOutput]):
    """Canonical in-place derivation of T, p, u, v from theta_v, exner, vn, and tracers."""

    def __init__(self, grid: grid_base.Grid, backend: gtx.typing.Backend | None) -> None:
        self._grid = grid
        self._backend = backend
        self._num_levels = grid.num_levels
        cell_domain = h_grid.domain(dims.CellDim)
        self._end_cell_end = grid.end_index(cell_domain(h_grid.Zone.END))
        self._cell_lateral_boundary_level_2 = grid.end_index(
            cell_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
        )

    @classmethod
    def input_type(cls) -> type[DerivedQuantitiesInput]:
        return DerivedQuantitiesInput

    @classmethod
    def output_type(cls) -> type[DerivedQuantitiesOutput]:
        return DerivedQuantitiesOutput

    def run(self, state: DerivedQuantitiesInput) -> DerivedQuantitiesOutput:
        """Diagnose temperature/virtual temperature, u/v, and pressure in place."""
        diagnose_temperature.diagnose_virtual_temperature_and_temperature.with_backend(
            self._backend
        )(
            qv=state.qv,
            qc=state.qc,
            qi=state.qi,
            qr=state.qr,
            qs=state.qs,
            qg=state.qg,
            theta_v=state.theta_v,
            exner=state.exner,
            virtual_temperature=state.virtual_temperature,
            temperature=state.temperature,
            horizontal_start=0,
            horizontal_end=self._end_cell_end,
            vertical_start=0,
            vertical_end=self._num_levels,
            offset_provider={},
        )

        rbf.edge_2_cell_vector_rbf_interpolation.with_backend(self._backend)(
            p_e_in=state.vn,
            ptr_coeff_1=state.rbf_vec_coeff_c1,
            ptr_coeff_2=state.rbf_vec_coeff_c2,
            p_u_out=state.u,
            p_v_out=state.v,
            horizontal_start=self._cell_lateral_boundary_level_2,
            horizontal_end=self._end_cell_end,
            vertical_start=0,
            vertical_end=self._num_levels,
            offset_provider={"C2E2C2E": self._grid.get_connectivity("C2E2C2E")},
        )

        pressure_diagnostics.diagnose_pressure_surface_to_top(
            grid=self._grid,
            backend=self._backend,
            exner=state.exner,
            virtual_temperature=state.virtual_temperature,
            ddqz_z_full=state.ddqz_z_full,
            surface_pressure=state.surface_pressure,
            pressure=state.pressure,
            pressure_on_cells_half_levels=state.pressure_ifc,
        )

        return DerivedQuantitiesOutput(
            temperature=state.temperature,
            virtual_temperature=state.virtual_temperature,
            pressure=state.pressure,
            pressure_ifc=state.pressure_ifc,
            surface_pressure=state.surface_pressure,
            u=state.u,
            v=state.v,
        )
