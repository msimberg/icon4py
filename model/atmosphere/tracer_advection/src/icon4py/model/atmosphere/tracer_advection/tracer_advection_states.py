# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.states import quantities as q, spec
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import icon as icon_grid


@dataclasses.dataclass(frozen=True)
class AdvectionDiagnosticState:
    """Represents the diagnostic fields needed in tracer_advection."""

    #: mass of air in layer at physics time step now [kg/m^2]
    airmass_now: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity=q.ICON_AIRMASS_NOW.name,
        units=q.ICON_AIRMASS_NOW.units,
        dims=q.ICON_AIRMASS_NOW.dims,
        intent=spec.Intent.READWRITE,
        lifetime=spec.Lifetime.PERSISTENT,
    )

    #: mass of air in layer at physics time step new [kg/m^2]
    airmass_new: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity=q.ICON_AIRMASS_NEW.name,
        units=q.ICON_AIRMASS_NEW.units,
        dims=q.ICON_AIRMASS_NEW.dims,
        intent=spec.Intent.READWRITE,
        lifetime=spec.Lifetime.PERSISTENT,
    )

    #: tracer tendency field for use in grid refinement [kg/kg/s]
    grf_tend_tracer: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity=q.ICON_GRF_TEND_TRACER.name,
        units=q.ICON_GRF_TEND_TRACER.units,
        dims=q.ICON_GRF_TEND_TRACER.dims,
        intent=spec.Intent.READWRITE,
        lifetime=spec.Lifetime.PERSISTENT,
    )

    #: horizontal tracer flux at edges [kg/m/s]
    hfl_tracer: fa.EdgeKField[ta.wpfloat] = spec.spec(
        quantity=q.ICON_HFL_TRACER.name,
        units=q.ICON_HFL_TRACER.units,
        dims=q.ICON_HFL_TRACER.dims,
        intent=spec.Intent.READWRITE,
        lifetime=spec.Lifetime.PERSISTENT,
    )

    #: vertical tracer flux at cells [kg/m/s]
    vfl_tracer: fa.CellKField[ta.wpfloat] = spec.spec(  # TODO(dastrm): should be KHalfDim
        quantity=q.ICON_VFL_TRACER.name,
        units=q.ICON_VFL_TRACER.units,
        dims=q.ICON_VFL_TRACER.dims,
        intent=spec.Intent.READWRITE,
        lifetime=spec.Lifetime.PERSISTENT,
    )


@dataclasses.dataclass(frozen=True)
class AdvectionPrepAdvState:
    """Represents the prepare tracer_advection state needed in tracer_advection."""

    #: horizontal velocity at edges for computation of backward trajectories averaged over dynamics substeps [m/s]
    vn_traj: fa.EdgeKField[ta.wpfloat] = spec.spec(
        quantity=q.ICON_VN_TRAJ.name,
        units=q.ICON_VN_TRAJ.units,
        dims=q.ICON_VN_TRAJ.dims,
        intent=spec.Intent.READWRITE,
        lifetime=spec.Lifetime.PERSISTENT,
    )

    #: mass flux at full level edges averaged over dynamics substeps [kg/m^2/s]
    mass_flx_me: fa.EdgeKField[ta.wpfloat] = spec.spec(
        quantity=q.ICON_MASS_FLX_ME.name,
        units=q.ICON_MASS_FLX_ME.units,
        dims=q.ICON_MASS_FLX_ME.dims,
        intent=spec.Intent.READWRITE,
        lifetime=spec.Lifetime.PERSISTENT,
    )

    #: mass flux at half level centers averaged over dynamics substeps [kg/m^2/s]
    mass_flx_ic: fa.CellKField[ta.wpfloat] = spec.spec(  # TODO(dastrm): should be KHalfDim
        quantity=q.ICON_MASS_FLX_IC.name,
        units=q.ICON_MASS_FLX_IC.units,
        dims=q.ICON_MASS_FLX_IC.dims,
        intent=spec.Intent.READWRITE,
        lifetime=spec.Lifetime.PERSISTENT,
    )


@dataclasses.dataclass(frozen=True)
class AdvectionInterpolationState:
    """Represents the interpolation state needed in tracer_advection."""

    #: factor for divergence
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat] = spec.spec(
        quantity=q.GEOFAC_DIV.name,
        units=q.GEOFAC_DIV.units,
        dims=q.GEOFAC_DIV.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )

    #: coefficients used for rbf interpolation of the tangential velocity component
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], ta.wpfloat] = spec.spec(
        quantity=q.ICON_RBF_VEC_COEFF_E.name,
        units=q.ICON_RBF_VEC_COEFF_E.units,
        dims=q.ICON_RBF_VEC_COEFF_E.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )

    #: x-components of positions of various points on local plane tangential to the edge midpoint
    pos_on_tplane_e_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat] = spec.spec(
        quantity=q.ICON_POS_ON_TPLANE_E_1.name,
        units=q.ICON_POS_ON_TPLANE_E_1.units,
        dims=q.ICON_POS_ON_TPLANE_E_1.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )

    #: y-components of positions of various points on local plane tangential to the edge midpoint
    pos_on_tplane_e_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat] = spec.spec(
        quantity=q.ICON_POS_ON_TPLANE_E_2.name,
        units=q.ICON_POS_ON_TPLANE_E_2.units,
        dims=q.ICON_POS_ON_TPLANE_E_2.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )


@dataclasses.dataclass(frozen=True)
class AdvectionLeastSquaresState:
    """Represents the least squares state needed in tracer_advection."""

    #: pseudo (or Moore-Penrose) inverse of lsq design matrix A
    lsq_pseudoinv_1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat]
    lsq_pseudoinv_2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat]


@dataclasses.dataclass(frozen=True)
class AdvectionMetricState:
    """Represents the metric fields needed in tracer_advection.

    The deep-atmosphere modification factors below are all 1 in the shallow atmosphere,
    which is the only mode icon4py supports (the dycore rejects 'deepatmos_mode', see
    'solve_nonhydro.NonHydrostaticConfig'). ICON does the same: it initialises them to 1
    in 'mo_nonhydro_state.f90' and only overwrites them inside the 'IF (ldeepatmo)'
    branch of 'mo_vertical_grid.f90'. They are kept as fields, rather than folded away,
    because the ICON stencils they feed take them unconditionally.
    """

    #: metrical modification factor for horizontal part of divergence at full levels (KDim)
    deepatmo_divh: fa.KField[ta.wpfloat]

    #: metrical modification factor for vertical part of divergence at full levels (KDim)
    deepatmo_divzl: fa.KField[ta.wpfloat]

    #: metrical modification factor for vertical part of divergence at full levels (KDim)
    deepatmo_divzu: fa.KField[ta.wpfloat]

    #: vertical grid spacing at full levels
    ddqz_z_full: fa.CellKField[ta.wpfloat]


def initialize_advection_diagnostic_state(
    grid: icon_grid.IconGrid,
    allocator: gtx_typing.Allocator,
) -> AdvectionDiagnosticState:
    return AdvectionDiagnosticState(
        airmass_now=data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, allocator=allocator, dtype=ta.wpfloat
        ),
        airmass_new=data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, allocator=allocator, dtype=ta.wpfloat
        ),
        grf_tend_tracer=data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, allocator=allocator, dtype=ta.wpfloat
        ),
        hfl_tracer=data_alloc.zero_field(
            grid, dims.EdgeDim, dims.KDim, allocator=allocator, dtype=ta.wpfloat
        ),
        # vertical flux at cell half levels: one more level than KDim
        vfl_tracer=data_alloc.zero_field(
            grid,
            dims.CellDim,
            dims.KDim,
            extend={dims.KDim: 1},
            allocator=allocator,
            dtype=ta.wpfloat,
        ),
    )
