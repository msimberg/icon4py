# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import enum
import logging
from typing import TYPE_CHECKING

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.config import config_io
from icon4py.model.common.states import quantities as q, spec
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import icon as icon_grid


log = logging.getLogger(__name__)


@config_io.register_enum
class TimeSteppingScheme(enum.IntEnum):
    """Parameter called `itime_scheme` in ICON namelist."""

    #: Contravariant vertical velocity is computed in the predictor step only, velocity tendencies are computed in the corrector step only
    MOST_EFFICIENT = 4
    #: Contravariant vertical velocity is computed in both substeps (beneficial for numerical stability in very-high resolution setups with extremely steep slopes)
    STABLE = 5
    #:  As STABLE, but velocity tendencies are also computed in both substeps (no benefit, but more expensive)
    EXPENSIVE = 6


@config_io.register_enum
class DivergenceDampingType(enum.IntEnum):
    #: divergence damping acting on 2D divergence
    TWO_DIMENSIONAL = 2
    #: divergence damping acting on 3D divergence
    THREE_DIMENSIONAL = 3
    #: combination of 3D div.damping in the troposphere with transition to 2D div. damping in the stratosphere
    COMBINED = 32


@config_io.register_enum
class DivergenceDampingOrder(gtx.int32, enum.Enum):
    #: 2nd order divergence damping
    SECOND_ORDER = 2
    #: 4th order divergence damping
    FOURTH_ORDER = 4
    #: combined 2nd and 4th orders divergence damping and enhanced vertical wind off - centering during initial spinup phase
    COMBINED = 24


@config_io.register_enum
class HorizontalPressureDiscretizationType(gtx.int32, enum.Enum):
    """Parameter called igradp_method in ICON namelist."""

    #: conventional discretization with metric correction term
    CONVENTIONAL = 1
    #: Taylor-expansion-based reconstruction of pressure
    TAYLOR = 2
    #: Similar discretization as igradp_method_taylor, but uses hydrostatic approximation for downward extrapolation over steep slopes
    TAYLOR_HYDRO = 3
    #: Cubic / quadratic polynomial interpolation for pressure reconstruction
    POLYNOMIAL = 4
    #: Same as igradp_method_polynomial, but hydrostatic approximation for downward extrapolation over steep slopes
    POLYNOMIAL_HYDRO = 5


@config_io.register_enum
class RhoThetaAdvectionType(gtx.int32, enum.Enum):
    """Parameter called iadv_rhotheta in ICON namelist."""

    #: simple 2nd order upwind-biased scheme
    SIMPLE = 1
    #: 2nd order Miura horizontal
    MIURA = 2


@dataclasses.dataclass
class InterpolationState:
    """Represents the ICON interpolation state used in the dynamical core (SolveNonhydro)."""

    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat] = spec.spec(
        quantity=q.E_BLN_C_S.name,
        units=q.E_BLN_C_S.units,
        dims=q.E_BLN_C_S.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )  # coefficent for bilinear interpolation from edge to cell ()
    rbf_coeff_1: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], ta.wpfloat] = spec.spec(
        quantity=q.RBF_VEC_COEFF_V1.name,
        units=q.RBF_VEC_COEFF_V1.units,
        dims=q.RBF_VEC_COEFF_V1.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )  # rbf_vec_coeff_v_1(nproma, rbf_vec_dim_v, nblks_v)
    rbf_coeff_2: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], ta.wpfloat] = spec.spec(
        quantity=q.RBF_VEC_COEFF_V2.name,
        units=q.RBF_VEC_COEFF_V2.units,
        dims=q.RBF_VEC_COEFF_V2.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )  # rbf_vec_coeff_v_2(nproma, rbf_vec_dim_v, nblks_v)

    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat] = spec.spec(
        quantity=q.GEOFAC_DIV.name,
        units=q.GEOFAC_DIV.units,
        dims=q.GEOFAC_DIV.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )  # factor for divergence (nproma,cell_type,nblks_c)

    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat] = spec.spec(
        quantity=q.GEOFAC_N2S.name,
        units=q.GEOFAC_N2S.units,
        dims=q.GEOFAC_N2S.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )  # factor for nabla2-scalar (nproma,cell_type+1,nblks_c)
    geofac_grg_x: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat] = spec.spec(
        quantity=q.GEOFAC_GRG_X.name,
        units=q.GEOFAC_GRG_X.units,
        dims=q.GEOFAC_GRG_X.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    geofac_grg_y: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat] = spec.spec(
        quantity=q.GEOFAC_GRG_Y.name,
        units=q.GEOFAC_GRG_Y.units,
        dims=q.GEOFAC_GRG_Y.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )  # factors for green gauss gradient (nproma,4,nblks_c,2)
    nudgecoeff_e: fa.EdgeField[ta.wpfloat] = spec.spec(
        quantity=q.NUDGECOEFFS_E.name,
        units=q.NUDGECOEFFS_E.units,
        dims=q.NUDGECOEFFS_E.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )  # Nudgeing coeffients for edges

    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat] = spec.spec(
        quantity=q.C_LIN_E.name,
        units=q.C_LIN_E.units,
        dims=q.C_LIN_E.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], ta.wpfloat] = spec.spec(
        quantity=q.GEOFAC_GRDIV.name,
        units=q.GEOFAC_GRDIV.units,
        dims=q.GEOFAC_GRDIV.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], ta.wpfloat] = spec.spec(
        quantity=q.ICON_RBF_VEC_COEFF_E.name,
        units=q.ICON_RBF_VEC_COEFF_E.units,
        dims=q.ICON_RBF_VEC_COEFF_E.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    c_intp: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2CDim], ta.wpfloat] = spec.spec(
        quantity=q.ICON_C_INTP.name,
        units=q.ICON_C_INTP.units,
        dims=q.ICON_C_INTP.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    geofac_rot: gtx.Field[gtx.Dims[dims.VertexDim, dims.V2EDim], ta.wpfloat] = spec.spec(
        quantity=q.GEOFAC_ROT.name,
        units=q.GEOFAC_ROT.units,
        dims=q.GEOFAC_ROT.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    pos_on_tplane_e_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat] = spec.spec(
        quantity=q.ICON_POS_ON_TPLANE_E_1.name,
        units=q.ICON_POS_ON_TPLANE_E_1.units,
        dims=q.ICON_POS_ON_TPLANE_E_1.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    pos_on_tplane_e_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat] = spec.spec(
        quantity=q.ICON_POS_ON_TPLANE_E_2.name,
        units=q.ICON_POS_ON_TPLANE_E_2.units,
        dims=q.ICON_POS_ON_TPLANE_E_2.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    e_flx_avg: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], ta.wpfloat] = spec.spec(
        quantity=q.E_FLX_AVG.name,
        units=q.E_FLX_AVG.units,
        dims=q.E_FLX_AVG.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )


@dataclasses.dataclass
class MetricStateNonHydro:
    """Dataclass containing metric fields needed in dynamical core (SolveNonhydro)."""

    mask_prog_halo_c: fa.CellKField[bool] = spec.spec(
        quantity=q.ICON_MASK_PROG_HALO_C.name,
        units=q.ICON_MASK_PROG_HALO_C.units,
        dims=q.ICON_MASK_PROG_HALO_C.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    rayleigh_w: fa.KField[ta.wpfloat] = spec.spec(
        quantity=q.RAYLEIGH_W.name,
        units=q.RAYLEIGH_W.units,
        dims=q.RAYLEIGH_W.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )

    wgtfac_c: fa.CellKField[ta.vpfloat] = spec.spec(
        quantity=q.WGTFAC_C.name,
        units=q.WGTFAC_C.units,
        dims=q.WGTFAC_C.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    wgtfacq_c: fa.CellKField[ta.vpfloat] = spec.spec(
        quantity=q.WGTFACQ_C.name,
        units=q.WGTFACQ_C.units,
        dims=q.WGTFACQ_C.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    wgtfac_e: fa.EdgeKField[ta.vpfloat] = spec.spec(
        quantity=q.ICON_WGTFAC_E.name,
        units=q.ICON_WGTFAC_E.units,
        dims=q.ICON_WGTFAC_E.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    wgtfacq_e: fa.EdgeKField[ta.vpfloat] = spec.spec(
        quantity=q.ICON_WGTFACQ_E.name,
        units=q.ICON_WGTFACQ_E.units,
        dims=q.ICON_WGTFACQ_E.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )

    time_extrapolation_parameter_for_exner: fa.CellKField[ta.vpfloat] = spec.spec(
        quantity=q.EXNER_EXFAC.name,
        units=q.EXNER_EXFAC.units,
        dims=q.EXNER_EXFAC.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    """
    Declared as exner_exfac in ICON.
    """
    reference_exner_at_cells_on_model_levels: fa.CellKField[ta.vpfloat] = spec.spec(
        quantity=q.EXNER_REF_MC.name,
        units=q.EXNER_REF_MC.units,
        dims=q.EXNER_REF_MC.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    """
    Declared as exner_ref_mc in ICON.
    """
    reference_rho_at_cells_on_model_levels: fa.CellKField[ta.vpfloat] = spec.spec(
        quantity=q.RHO_REF_MC.name,
        units=q.RHO_REF_MC.units,
        dims=q.RHO_REF_MC.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    """
    Declared as rho_ref_mc in ICON.
    """
    reference_theta_at_cells_on_model_levels: fa.CellKField[ta.vpfloat] = spec.spec(
        quantity=q.THETA_REF_MC.name,
        units=q.THETA_REF_MC.units,
        dims=q.THETA_REF_MC.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    """
    Declared as theta_ref_mc in ICON.
    """
    reference_rho_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat] = spec.spec(
        quantity=q.RHO_REF_ME.name,
        units=q.RHO_REF_ME.units,
        dims=q.RHO_REF_ME.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    """
    Declared as rho_ref_me in ICON.
    """
    reference_theta_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat] = spec.spec(
        quantity=q.THETA_REF_ME.name,
        units=q.THETA_REF_ME.units,
        dims=q.THETA_REF_ME.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    """
    Declared as theta_ref_me in ICON.
    """
    reference_theta_at_cells_on_half_levels: fa.CellKField[ta.vpfloat] = spec.spec(
        quantity=q.THETA_REF_IC.name,
        units=q.THETA_REF_IC.units,
        dims=q.THETA_REF_IC.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    """
    Declared as theta_ref_ic in ICON.
    """
    ddz_of_reference_exner_at_cells_on_half_levels: fa.CellKField[ta.vpfloat] = spec.spec(
        quantity=q.D_EXNER_DZ_REF_IC.name,
        units=q.D_EXNER_DZ_REF_IC.units,
        dims=q.D_EXNER_DZ_REF_IC.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    """
    Declared as d_exner_dz_ref_ic in ICON.
    """
    ddqz_z_half: fa.CellKField[ta.vpfloat] = spec.spec(  # dims.KHalfDim
        quantity=q.DDQZ_Z_HALF.name,
        units=q.DDQZ_Z_HALF.units,
        dims=q.DDQZ_Z_HALF.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    d2dexdz2_fac1_mc: fa.CellKField[ta.vpfloat] = spec.spec(
        quantity=q.D2DEXDZ2_FAC1_MC.name,
        units=q.D2DEXDZ2_FAC1_MC.units,
        dims=q.D2DEXDZ2_FAC1_MC.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    d2dexdz2_fac2_mc: fa.CellKField[ta.vpfloat] = spec.spec(
        quantity=q.D2DEXDZ2_FAC2_MC.name,
        units=q.D2DEXDZ2_FAC2_MC.units,
        dims=q.D2DEXDZ2_FAC2_MC.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    ddxn_z_full: fa.EdgeKField[ta.vpfloat] = spec.spec(
        quantity=q.ICON_DDXN_Z_FULL.name,
        units=q.ICON_DDXN_Z_FULL.units,
        dims=q.ICON_DDXN_Z_FULL.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    ddqz_z_full_e: fa.EdgeKField[ta.vpfloat] = spec.spec(
        quantity=q.ICON_DDQT_Z_FULL_E.name,
        units=q.ICON_DDQT_Z_FULL_E.units,
        dims=q.ICON_DDQT_Z_FULL_E.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    ddxt_z_full: fa.EdgeKField[ta.vpfloat] = spec.spec(
        quantity=q.ICON_DDXT_Z_FULL.name,
        units=q.ICON_DDXT_Z_FULL.units,
        dims=q.ICON_DDXT_Z_FULL.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    inv_ddqz_z_full: fa.CellKField[ta.vpfloat] = spec.spec(
        quantity=q.INV_DDQZ_Z_FULL.name,
        units=q.INV_DDQZ_Z_FULL.units,
        dims=q.INV_DDQZ_Z_FULL.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )

    vertoffset_gradp: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim, dims.KDim], gtx.int32] = (
        spec.spec(
            quantity=q.ICON_VERTOFFSET_GRADP.name,
            units=q.ICON_VERTOFFSET_GRADP.units,
            dims=q.ICON_VERTOFFSET_GRADP.dims,
            intent=spec.Intent.READ,
            lifetime=spec.Lifetime.STATIC,
        )
    )
    zdiff_gradp: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim, dims.KDim], ta.vpfloat] = spec.spec(
        quantity=q.ICON_ZDIFF_GRADP.name,
        units=q.ICON_ZDIFF_GRADP.units,
        dims=q.ICON_ZDIFF_GRADP.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    nflat_gradp: gtx.int32 = spec.spec(
        quantity=q.ICON_NFLAT_GRADP.name,
        units=q.ICON_NFLAT_GRADP.units,
        dims=q.ICON_NFLAT_GRADP.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    """
    The minimum height index at which the height of the center of an edge lies within two neighboring cells so that
    horizontal pressure gradient can be computed by first order discretization scheme.
    """

    pg_exdist: fa.EdgeKField[ta.vpfloat] = spec.spec(
        quantity=q.ICON_PG_EXDIST.name,
        units=q.ICON_PG_EXDIST.units,
        dims=q.ICON_PG_EXDIST.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    """
    Extrapolation distance needed for HorizontalPressureDiscretizationType.TAYLOR_HYDRO.
    """

    exner_w_explicit_weight_parameter: fa.CellField[ta.wpfloat] = spec.spec(
        quantity=q.EXNER_W_EXPLICIT_WEIGHT_PARAMETER.name,
        units=q.EXNER_W_EXPLICIT_WEIGHT_PARAMETER.units,
        dims=q.EXNER_W_EXPLICIT_WEIGHT_PARAMETER.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    """
    Declared as vwind_expl_wgt in ICON. The explicitness parameter for exner and w in the vertically
    implicit dycore solver.
    exner_w_explicit_weight_parameter = 1 - exner_w_implicit_weight_parameter
    """
    exner_w_implicit_weight_parameter: fa.CellField[ta.wpfloat] = spec.spec(
        quantity=q.EXNER_W_IMPLICIT_WEIGHT_PARAMETER.name,
        units=q.EXNER_W_IMPLICIT_WEIGHT_PARAMETER.units,
        dims=q.EXNER_W_IMPLICIT_WEIGHT_PARAMETER.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    """
    Declared as vwind_impl_wgt in ICON. The implicitness parameter for exner and w in the vertically
    implicit dycore solver. It is denoted as eta below eq. 3.20 in ICON tutorial 2023. However,
    it is only vwind_offctr that can be set via namelist. The actual computation of
    exner_w_implicit_weight_parameter is not shown in the tutorial.
    """

    horizontal_mask_for_3d_divdamp: fa.EdgeField[ta.wpfloat] = spec.spec(
        quantity=q.HORIZONTAL_MASK_FOR_3D_DIVDAMP.name,
        units=q.HORIZONTAL_MASK_FOR_3D_DIVDAMP.units,
        dims=q.HORIZONTAL_MASK_FOR_3D_DIVDAMP.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    """
    Declared as hmask_dd3d in ICON. A horizontal mask where 3D divergence is computed for the divergence damping.
    3D divergence is defined as divergence of horizontal wind plus vertical derivative of vertical wind (dw/dz).
    """
    scaling_factor_for_3d_divdamp: fa.KField[ta.wpfloat] = spec.spec(
        quantity=q.SCALING_FACTOR_FOR_3D_DIVDAMP.name,
        units=q.SCALING_FACTOR_FOR_3D_DIVDAMP.units,
        dims=q.SCALING_FACTOR_FOR_3D_DIVDAMP.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    """
    Declared as scalfac_dd3d in ICON. A scaling factor in vertical dimension for 3D divergence damping.
    """

    coeff1_dwdz: fa.CellKField[ta.vpfloat] = spec.spec(
        quantity=q.ICON_COEFF1_DWDZ.name,
        units=q.ICON_COEFF1_DWDZ.units,
        dims=q.ICON_COEFF1_DWDZ.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    coeff2_dwdz: fa.CellKField[ta.vpfloat] = spec.spec(
        quantity=q.ICON_COEFF2_DWDZ.name,
        units=q.ICON_COEFF2_DWDZ.units,
        dims=q.ICON_COEFF2_DWDZ.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    coeff_gradekin: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.vpfloat] = spec.spec(
        quantity=q.COEFF_GRADEKIN.name,
        units=q.COEFF_GRADEKIN.units,
        dims=q.COEFF_GRADEKIN.dims,
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )


@dataclasses.dataclass(frozen=True)
class StepInfo:
    """Per-dycore-substep context used by the substep leaf steps."""

    substep_index: int
    at_first_substep: bool
    at_last_substep: bool
    at_initial_timestep: bool


@dataclasses.dataclass(frozen=True)
class DycoreControl:
    """Dycore control flags passed to SolveNonhydro as a structured field."""

    lprep_adv: bool
    is_iau_active: bool
    iau_wgt_dyn: float


@dataclasses.dataclass
class PrepAdvection:
    """Dataclass used in SolveNonHydro that pre-calculates fields during the dynamical substepping that are later needed in tracer advection."""

    vn_traj: fa.EdgeKField[ta.wpfloat] = spec.spec(
        quantity=q.ICON_VN_TRAJ.name,
        units=q.ICON_VN_TRAJ.units,
        dims=q.ICON_VN_TRAJ.dims,
        intent=spec.Intent.READWRITE,
        lifetime=spec.Lifetime.PERSISTENT,
    )
    mass_flx_me: fa.EdgeKField[ta.wpfloat] = spec.spec(
        quantity=q.ICON_MASS_FLX_ME.name,
        units=q.ICON_MASS_FLX_ME.units,
        dims=q.ICON_MASS_FLX_ME.dims,
        intent=spec.Intent.READWRITE,
        lifetime=spec.Lifetime.PERSISTENT,
    )
    dynamical_vertical_mass_flux_at_cells_on_half_levels: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity=q.ICON_MASS_FLX_IC.name,
        units=q.ICON_MASS_FLX_IC.units,
        dims=q.ICON_MASS_FLX_IC.dims,
        intent=spec.Intent.READWRITE,
        lifetime=spec.Lifetime.PERSISTENT,
    )
    """
    Declared as mass_flx_ic in ICON.
    """
    dynamical_vertical_volumetric_flux_at_cells_on_half_levels: fa.CellKField[ta.wpfloat] = (
        spec.spec(
            quantity=q.ICON_VOL_FLX_IC.name,
            units=q.ICON_VOL_FLX_IC.units,
            dims=q.ICON_VOL_FLX_IC.dims,
            intent=spec.Intent.READWRITE,
            lifetime=spec.Lifetime.PERSISTENT,
        )
    )
    """
    Declared as vol_flx_ic in ICON.
    """


def initialize_prep_advection(
    grid: icon_grid.IconGrid, allocator: gtx_typing.Allocator
) -> PrepAdvection:
    vn_traj = data_alloc.zero_field(
        grid, dims.EdgeDim, dims.KDim, allocator=allocator, dtype=ta.wpfloat
    )
    mass_flx_me = data_alloc.zero_field(
        grid, dims.EdgeDim, dims.KDim, allocator=allocator, dtype=ta.wpfloat
    )
    dynamical_vertical_mass_flux_at_cells_on_half_levels = data_alloc.zero_field(
        grid,
        dims.CellDim,
        dims.KDim,
        extend={dims.KDim: 1},
        allocator=allocator,
        dtype=ta.wpfloat,
    )
    dynamical_vertical_volumetric_flux_at_cells_on_half_levels = data_alloc.zero_field(
        grid,
        dims.CellDim,
        dims.KDim,
        extend={dims.KDim: 1},
        allocator=allocator,
        dtype=ta.wpfloat,
    )
    return PrepAdvection(
        vn_traj=vn_traj,
        mass_flx_me=mass_flx_me,
        dynamical_vertical_mass_flux_at_cells_on_half_levels=dynamical_vertical_mass_flux_at_cells_on_half_levels,
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels=dynamical_vertical_volumetric_flux_at_cells_on_half_levels,
    )
