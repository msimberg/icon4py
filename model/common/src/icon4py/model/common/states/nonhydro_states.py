# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Diagnostic state of the nonhydrostatic dynamical core.

It is allocated by the driver, filled by the initial condition and mutated by the
dycore, so it lives here and not in the dycore package.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from icon4py.model.common import (
    dimension as dims,
    field_type_aliases as fa,
    type_alias as ta,
    utils as common_utils,
)
from icon4py.model.common.states import quantities as q, spec
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import icon as icon_grid


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import icon as icon_grid


@dataclasses.dataclass
class DiagnosticStateNonHydro:
    """Data class containing diagnostic fields that are calculated in the dynamical core (SolveNonHydro)."""

    # `max_vertical_cfl` stored as 0-d array of type ta.wpfloat (to be able to avoid cupy synchronization)
    max_vertical_cfl: data_alloc.ScalarLikeArray[ta.wpfloat] = spec.spec(  # type: ignore[type-var]
        quantity=q.MAX_VERTICAL_CFL.name,
        units=q.MAX_VERTICAL_CFL.units,
        dims=q.MAX_VERTICAL_CFL.dims,
        intent=spec.Intent.READWRITE,
        lifetime=spec.Lifetime.PERSISTENT,
    )
    """
    Declared as max_vcfl_dyn in ICON. Maximum vertical CFL number over all substeps.
    """

    tangential_wind: fa.EdgeKField[ta.vpfloat] = spec.spec(
        quantity=q.TANGENTIAL_VELOCITY.name,
        units=q.TANGENTIAL_VELOCITY.units,
        dims=q.TANGENTIAL_VELOCITY.dims,
        intent=spec.Intent.READWRITE,
        lifetime=spec.Lifetime.PERSISTENT,
    )
    """
    Declared as vt in ICON. Tangential wind at edge.
    """

    vn_on_half_levels: fa.EdgeKField[ta.vpfloat] = (
        spec.spec(  # normal wind at half levels (nproma,nlevp1,nblks_e)   [m/s]
            quantity=q.ICON_NORMAL_VELOCITY_AT_HALF_LEVELS.name,
            units=q.ICON_NORMAL_VELOCITY_AT_HALF_LEVELS.units,
            dims=q.ICON_NORMAL_VELOCITY_AT_HALF_LEVELS.dims,
            intent=spec.Intent.READWRITE,
            lifetime=spec.Lifetime.PERSISTENT,
        )
    )
    """
    Declared as vn_ie in ICON. Normal wind at edge on k-half levels.
    """

    contravariant_correction_at_cells_on_half_levels: fa.CellKField[ta.vpfloat] = (
        spec.spec(  # contravariant vert correction (nproma,nlevp1,nblks_c)[m/s]
            quantity=q.ICON_CONTRAVARIANT_CORRECTION_AT_CELLS_ON_HALF_LEVELS.name,
            units=q.ICON_CONTRAVARIANT_CORRECTION_AT_CELLS_ON_HALF_LEVELS.units,
            dims=q.ICON_CONTRAVARIANT_CORRECTION_AT_CELLS_ON_HALF_LEVELS.dims,
            intent=spec.Intent.READWRITE,
            lifetime=spec.Lifetime.PERSISTENT,
        )
    )
    """
    Declared as w_concorr_c in ICON. Contravariant correction at cell center on k-half levels. vn dz/dn + vt dz/dt, z is topography height
    """

    theta_v_at_cells_on_half_levels: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity=q.ICON_THETA_V_AT_CELLS_ON_HALF_LEVELS.name,
        units=q.ICON_THETA_V_AT_CELLS_ON_HALF_LEVELS.units,
        dims=q.ICON_THETA_V_AT_CELLS_ON_HALF_LEVELS.dims,
        intent=spec.Intent.READWRITE,
        lifetime=spec.Lifetime.PERSISTENT,
    )
    """
    Declared as theta_v_ic in ICON.
    """

    perturbed_exner_at_cells_on_model_levels: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity=q.ICON_PERTURBED_EXNER_AT_CELLS_ON_MODEL_LEVELS.name,
        units=q.ICON_PERTURBED_EXNER_AT_CELLS_ON_MODEL_LEVELS.units,
        dims=q.ICON_PERTURBED_EXNER_AT_CELLS_ON_MODEL_LEVELS.dims,
        intent=spec.Intent.READWRITE,
        lifetime=spec.Lifetime.PERSISTENT,
    )
    """
    Declared as exner_pr in ICON.
    """

    rho_at_cells_on_half_levels: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity=q.ICON_RHO_AT_CELLS_ON_HALF_LEVELS.name,
        units=q.ICON_RHO_AT_CELLS_ON_HALF_LEVELS.units,
        dims=q.ICON_RHO_AT_CELLS_ON_HALF_LEVELS.dims,
        intent=spec.Intent.READWRITE,
        lifetime=spec.Lifetime.PERSISTENT,
    )
    """
    Declared as rho_ic in ICON.
    """

    exner_tendency_due_to_slow_physics: fa.CellKField[ta.vpfloat] = spec.spec(
        quantity=q.ICON_EXNER_TENDENCY_DUE_TO_SLOW_PHYSICS.name,
        units=q.ICON_EXNER_TENDENCY_DUE_TO_SLOW_PHYSICS.units,
        dims=q.ICON_EXNER_TENDENCY_DUE_TO_SLOW_PHYSICS.dims,
        intent=spec.Intent.READWRITE,
        lifetime=spec.Lifetime.PERSISTENT,
    )
    """
    Declared as ddt_exner_phy in ICON.
    """

    grf_tend_rho: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity=q.ICON_GRF_TEND_RHO.name,
        units=q.ICON_GRF_TEND_RHO.units,
        dims=q.ICON_GRF_TEND_RHO.dims,
        intent=spec.Intent.READWRITE,
        lifetime=spec.Lifetime.PERSISTENT,
    )
    grf_tend_thv: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity=q.ICON_GRF_TEND_THV.name,
        units=q.ICON_GRF_TEND_THV.units,
        dims=q.ICON_GRF_TEND_THV.dims,
        intent=spec.Intent.READWRITE,
        lifetime=spec.Lifetime.PERSISTENT,
    )
    grf_tend_w: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity=q.ICON_GRF_TEND_W.name,
        units=q.ICON_GRF_TEND_W.units,
        dims=q.ICON_GRF_TEND_W.dims,
        intent=spec.Intent.READWRITE,
        lifetime=spec.Lifetime.PERSISTENT,
    )
    mass_flux_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat] = spec.spec(
        quantity=q.ICON_MASS_FLUX_AT_EDGES_ON_MODEL_LEVELS.name,
        units=q.ICON_MASS_FLUX_AT_EDGES_ON_MODEL_LEVELS.units,
        dims=q.ICON_MASS_FLUX_AT_EDGES_ON_MODEL_LEVELS.dims,
        intent=spec.Intent.READWRITE,
        lifetime=spec.Lifetime.PERSISTENT,
    )
    """
    Declared as mass_fl_e in ICON.
    """

    normal_wind_tendency_due_to_slow_physics_process: fa.EdgeKField[ta.vpfloat] = spec.spec(
        quantity=q.ICON_NORMAL_WIND_TENDENCY_DUE_TO_SLOW_PHYSICS_PROCESS.name,
        units=q.ICON_NORMAL_WIND_TENDENCY_DUE_TO_SLOW_PHYSICS_PROCESS.units,
        dims=q.ICON_NORMAL_WIND_TENDENCY_DUE_TO_SLOW_PHYSICS_PROCESS.dims,
        intent=spec.Intent.READWRITE,
        lifetime=spec.Lifetime.PERSISTENT,
    )
    """
    Declared as ddt_vn_phy in ICON.
    """

    grf_tend_vn: fa.EdgeKField[ta.wpfloat] = spec.spec(
        quantity=q.ICON_GRF_TEND_VN.name,
        units=q.ICON_GRF_TEND_VN.units,
        dims=q.ICON_GRF_TEND_VN.dims,
        intent=spec.Intent.READWRITE,
        lifetime=spec.Lifetime.PERSISTENT,
    )
    normal_wind_advective_tendency: common_utils.PredictorCorrectorPair[
        fa.EdgeKField[ta.wpfloat]
    ] = spec.spec(
        quantity=q.ICON_NORMAL_WIND_ADVECTIVE_TENDENCY.name,
        units=q.ICON_NORMAL_WIND_ADVECTIVE_TENDENCY.units,
        dims=q.ICON_NORMAL_WIND_ADVECTIVE_TENDENCY.dims,
        intent=spec.Intent.READWRITE,
        lifetime=spec.Lifetime.PERSISTENT,
    )
    """
    Declared as ddt_vn_apc_pc in ICON. Advective tendency of normal wind (including coriolis force).
    """

    vertical_wind_advective_tendency: common_utils.PredictorCorrectorPair[
        fa.CellKField[ta.vpfloat]
    ] = spec.spec(
        quantity=q.ICON_VERTICAL_WIND_ADVECTIVE_TENDENCY.name,
        units=q.ICON_VERTICAL_WIND_ADVECTIVE_TENDENCY.units,
        dims=q.ICON_VERTICAL_WIND_ADVECTIVE_TENDENCY.dims,
        intent=spec.Intent.READWRITE,
        lifetime=spec.Lifetime.PERSISTENT,
    )
    """
    Declared as ddt_w_adv_pc in ICON. Advective tendency of vertical wind.
    """

    # Analysis increments
    rho_iau_increment: fa.CellKField[ta.vpfloat] = spec.spec(  # moist density increment [kg/m^3].0
        quantity=q.ICON_RHO_IAU_INCREMENT.name,
        units=q.ICON_RHO_IAU_INCREMENT.units,
        dims=q.ICON_RHO_IAU_INCREMENT.dims,
        intent=spec.Intent.READWRITE,
        lifetime=spec.Lifetime.PERSISTENT,
    )
    """
    Declared as rho_incr in ICON.
    """

    normal_wind_iau_increment: fa.EdgeKField[ta.vpfloat] = (
        spec.spec(  # normal velocity increment [m/s]
            quantity=q.ICON_NORMAL_WIND_IAU_INCREMENT.name,
            units=q.ICON_NORMAL_WIND_IAU_INCREMENT.units,
            dims=q.ICON_NORMAL_WIND_IAU_INCREMENT.dims,
            intent=spec.Intent.READWRITE,
            lifetime=spec.Lifetime.PERSISTENT,
        )
    )
    """
    Declared as vn_incr in ICON.
    """

    exner_iau_increment: fa.CellKField[ta.vpfloat] = spec.spec(  # exner increment [- ]
        quantity=q.ICON_EXNER_IAU_INCREMENT.name,
        units=q.ICON_EXNER_IAU_INCREMENT.units,
        dims=q.ICON_EXNER_IAU_INCREMENT.dims,
        intent=spec.Intent.READWRITE,
        lifetime=spec.Lifetime.PERSISTENT,
    )
    """
    Declared as exner_incr in ICON.
    """

    exner_dynamical_increment: fa.CellKField[ta.vpfloat] = (
        spec.spec(  # exner function dynamics increment
            quantity=q.ICON_EXNER_DYNAMICAL_INCREMENT.name,
            units=q.ICON_EXNER_DYNAMICAL_INCREMENT.units,
            dims=q.ICON_EXNER_DYNAMICAL_INCREMENT.dims,
            intent=spec.Intent.READWRITE,
            lifetime=spec.Lifetime.PERSISTENT,
        )
    )
    """
    Declared as exner_dyn_incr in ICON.
    """

    def __post_init__(self) -> None:
        if not data_alloc.is_rank0_ndarray(self.max_vertical_cfl):
            # TODO(havogt): instead of this check, we could refactor to a special dataclass-like which promotes to 0-d array on assignment
            raise TypeError(
                "'max_vertical_cfl' must be initialized as a 0-d array for performance reasons."
            )


def initialize_solve_nonhydro_diagnostic_state(
    grid: icon_grid.IconGrid,
    allocator: gtx_typing.Allocator,
) -> DiagnosticStateNonHydro:
    """
    Allocate the diagnostic state of the dycore, with all its fields set to zero.

    The initial condition fills the fields that it owns: the perturbed exner function
    and, when restarting, the advective tendencies of the previous time step.
    """
    perturbed_exner_at_cells_on_model_levels = data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, allocator=allocator
    )
    normal_wind_advective_tendency = common_utils.PredictorCorrectorPair(
        data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, allocator=allocator, dtype=ta.vpfloat),
        data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, allocator=allocator, dtype=ta.vpfloat),
    )
    vertical_wind_advective_tendency = common_utils.PredictorCorrectorPair(
        data_alloc.zero_field(
            grid,
            dims.CellDim,
            dims.KDim,
            extend={dims.KDim: 1},
            allocator=allocator,
            dtype=ta.vpfloat,
        ),
        data_alloc.zero_field(
            grid,
            dims.CellDim,
            dims.KDim,
            extend={dims.KDim: 1},
            allocator=allocator,
            dtype=ta.vpfloat,
        ),
    )
    max_vertical_cfl = data_alloc.scalar_like_array(0.0, allocator)
    theta_v_at_cells_on_half_levels = data_alloc.zero_field(
        grid,
        dims.CellDim,
        dims.KDim,
        extend={dims.KDim: 1},
        allocator=allocator,
        dtype=ta.wpfloat,
    )
    rho_at_cells_on_half_levels = data_alloc.zero_field(
        grid,
        dims.CellDim,
        dims.KDim,
        extend={dims.KDim: 1},
        allocator=allocator,
        dtype=ta.wpfloat,
    )
    exner_tendency_due_to_slow_physics = data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, allocator=allocator, dtype=ta.vpfloat
    )
    grf_tend_rho = data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, allocator=allocator, dtype=ta.wpfloat
    )
    grf_tend_thv = data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, allocator=allocator, dtype=ta.wpfloat
    )
    grf_tend_w = data_alloc.zero_field(
        grid,
        dims.CellDim,
        dims.KDim,
        extend={dims.KDim: 1},
        allocator=allocator,
        dtype=ta.wpfloat,
    )
    mass_flux_at_edges_on_model_levels = data_alloc.zero_field(
        grid, dims.EdgeDim, dims.KDim, allocator=allocator, dtype=ta.wpfloat
    )
    normal_wind_tendency_due_to_slow_physics_process = data_alloc.zero_field(
        grid, dims.EdgeDim, dims.KDim, allocator=allocator, dtype=ta.vpfloat
    )
    grf_tend_vn = data_alloc.zero_field(
        grid, dims.EdgeDim, dims.KDim, allocator=allocator, dtype=ta.wpfloat
    )
    tangential_wind = data_alloc.zero_field(
        grid, dims.EdgeDim, dims.KDim, allocator=allocator, dtype=ta.vpfloat
    )
    vn_on_half_levels = data_alloc.zero_field(
        grid,
        dims.EdgeDim,
        dims.KDim,
        extend={dims.KDim: 1},
        allocator=allocator,
        dtype=ta.vpfloat,
    )
    contravariant_correction_at_cells_on_half_levels = data_alloc.zero_field(
        grid,
        dims.CellDim,
        dims.KDim,
        extend={dims.KDim: 1},
        allocator=allocator,
        dtype=ta.vpfloat,
    )
    rho_iau_increment = data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, allocator=allocator, dtype=ta.vpfloat
    )
    normal_wind_iau_increment = data_alloc.zero_field(
        grid, dims.EdgeDim, dims.KDim, allocator=allocator, dtype=ta.vpfloat
    )
    exner_iau_increment = data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, allocator=allocator, dtype=ta.vpfloat
    )
    exner_dynamical_increment = data_alloc.zero_field(
        grid, dims.CellDim, dims.KDim, allocator=allocator, dtype=ta.vpfloat
    )

    return DiagnosticStateNonHydro(
        max_vertical_cfl=max_vertical_cfl,
        theta_v_at_cells_on_half_levels=theta_v_at_cells_on_half_levels,
        perturbed_exner_at_cells_on_model_levels=perturbed_exner_at_cells_on_model_levels,
        rho_at_cells_on_half_levels=rho_at_cells_on_half_levels,
        exner_tendency_due_to_slow_physics=exner_tendency_due_to_slow_physics,
        grf_tend_rho=grf_tend_rho,
        grf_tend_thv=grf_tend_thv,
        grf_tend_w=grf_tend_w,
        mass_flux_at_edges_on_model_levels=mass_flux_at_edges_on_model_levels,
        normal_wind_tendency_due_to_slow_physics_process=normal_wind_tendency_due_to_slow_physics_process,
        grf_tend_vn=grf_tend_vn,
        normal_wind_advective_tendency=normal_wind_advective_tendency,
        vertical_wind_advective_tendency=vertical_wind_advective_tendency,
        tangential_wind=tangential_wind,
        vn_on_half_levels=vn_on_half_levels,
        contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
        rho_iau_increment=rho_iau_increment,
        normal_wind_iau_increment=normal_wind_iau_increment,
        exner_iau_increment=exner_iau_increment,
        exner_dynamical_increment=exner_dynamical_increment,
    )
