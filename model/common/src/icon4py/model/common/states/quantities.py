# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical quantity registry for field declarations.

Each quantity has one canonical name, one unit string, one placement (dims),
and one ICON Fortran variable name. Unprefixed names claim a CF standard name;
``icon:<name>`` is used when no CF equivalent exists.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, Final

import gt4py.next as gtx

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import geometry_attributes as geometry_attrs
from icon4py.model.common.interpolation import interpolation_attributes as interpolation_attrs
from icon4py.model.common.metrics import metrics_attributes as metrics_attrs


@dataclasses.dataclass(frozen=True)
class Quantity:
    """A canonical model quantity."""

    #: Canonical name. Unprefixed names claim a CF standard name;
    #: ``icon:<name>`` is used otherwise.
    name: str

    #: Physical units of the quantity.
    units: str

    #: Placement dimensions (one source of truth for full/half levels).
    dims: tuple[gtx.Dimension, ...]

    #: ICON Fortran variable name for the binding seam.
    icon_fortran_name: str


_REGISTRY: dict[str, Quantity] = {}


def register(
    name: str, units: str, dims: tuple[gtx.Dimension, ...], icon_fortran_name: str
) -> Quantity:
    """Register a canonical quantity."""
    if name in _REGISTRY:
        raise ValueError(f"Quantity '{name}' is already registered")
    quantity = Quantity(name=name, units=units, dims=dims, icon_fortran_name=icon_fortran_name)
    _REGISTRY[name] = quantity
    return quantity


def get(name: str) -> Quantity:
    """Return a registered quantity or raise."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown quantity '{name}'")
    return _REGISTRY[name]


def all_quantities() -> dict[str, Quantity]:
    """Return a read-only view of the registry."""
    return _REGISTRY.copy()


def _register_from_metadata(attrs: Mapping[str, Mapping[str, Any]]) -> None:
    """Populate the registry from existing ``FieldMetaData`` dictionaries."""
    for field_name, meta in attrs.items():
        standard_name = meta.get("standard_name", field_name)
        if standard_name in _REGISTRY:
            continue
        register(
            name=standard_name,
            units=meta.get("units", ""),
            dims=meta.get("dims", ()),
            icon_fortran_name=meta.get("icon_var_name", ""),
        )


_register_from_metadata(geometry_attrs.attrs)
_register_from_metadata(interpolation_attrs.attrs)
_register_from_metadata(metrics_attrs.attrs)


#: Prognostic quantities (also used in state dataclasses).
AIR_DENSITY: Final[Quantity] = register("air_density", "kg m-3", (dims.CellDim, dims.KDim), "rho")
UPWARD_AIR_VELOCITY: Final[Quantity] = register(
    "upward_air_velocity", "m s-1", (dims.CellDim, dims.KHalfDim), "w"
)
NORMAL_VELOCITY: Final[Quantity] = register(
    "normal_velocity", "m s-1", (dims.EdgeDim, dims.KDim), "vn"
)
DIMENSIONLESS_EXNER_FUNCTION: Final[Quantity] = register(
    "dimensionless_exner_function", "1", (dims.CellDim, dims.KDim), "exner"
)
VIRTUAL_POTENTIAL_TEMPERATURE: Final[Quantity] = register(
    "virtual_potential_temperature", "K", (dims.CellDim, dims.KDim), "theta_v"
)

#: Derived diagnostic quantities computed by ``update_derived_quantities``.
AIR_TEMPERATURE: Final[Quantity] = register(
    "air_temperature", "K", (dims.CellDim, dims.KDim), "temp"
)
AIR_VIRTUAL_TEMPERATURE: Final[Quantity] = register(
    "air_virtual_temperature", "K", (dims.CellDim, dims.KDim), "tempv"
)
AIR_PRESSURE: Final[Quantity] = register("air_pressure", "Pa", (dims.CellDim, dims.KDim), "pres")
AIR_PRESSURE_ON_INTERFACE_LEVELS: Final[Quantity] = register(
    "air_pressure_on_interface_levels", "Pa", (dims.CellDim, dims.KHalfDim), "pres_ifc"
)
AIR_PRESSURE_AT_GROUND_LEVEL: Final[Quantity] = register(
    "air_pressure_at_ground_level", "Pa", (dims.CellDim,), "pres_sfc"
)
EASTWARD_WIND: Final[Quantity] = register("eastward_wind", "m s-1", (dims.CellDim, dims.KDim), "u")
NORTHWARD_WIND: Final[Quantity] = register(
    "northward_wind", "m s-1", (dims.CellDim, dims.KDim), "v"
)

#: Tracer quantities.
SPECIFIC_HUMIDITY: Final[Quantity] = register(
    "specific_humidity", "1", (dims.CellDim, dims.KDim), "qv"
)
SPECIFIC_CLOUD_CONTENT: Final[Quantity] = register(
    "specific_cloud_content", "1", (dims.CellDim, dims.KDim), "qc"
)
SPECIFIC_ICE_CONTENT: Final[Quantity] = register(
    "specific_ice_content", "1", (dims.CellDim, dims.KDim), "qi"
)
SPECIFIC_RAIN_CONTENT: Final[Quantity] = register(
    "specific_rain_content", "1", (dims.CellDim, dims.KDim), "qr"
)
SPECIFIC_SNOW_CONTENT: Final[Quantity] = register(
    "specific_snow_content", "1", (dims.CellDim, dims.KDim), "qs"
)
SPECIFIC_GRAUPEL_CONTENT: Final[Quantity] = register(
    "specific_graupel_content", "1", (dims.CellDim, dims.KDim), "qg"
)

#: Diagnostic quantities that do not appear in factory metadata.
TANGENTIAL_VELOCITY: Final[Quantity] = register(
    "tangential_velocity", "m s-1", (dims.EdgeDim, dims.KDim), "vt"
)
MAX_VERTICAL_CFL: Final[Quantity] = register("icon:max_vertical_cfl", "", (), "max_vcfl_dyn")
ICON_NORMAL_VELOCITY_AT_HALF_LEVELS: Final[Quantity] = register(
    "icon:normal_velocity_at_half_levels", "m s-1", (dims.EdgeDim, dims.KHalfDim), "vn_ie"
)
ICON_CONTRAVARIANT_CORRECTION_AT_CELLS_ON_HALF_LEVELS: Final[Quantity] = register(
    "icon:contravariant_correction_at_cells_on_half_levels",
    "m s-1",
    (dims.CellDim, dims.KHalfDim),
    "w_concorr_c",
)
ICON_THETA_V_AT_CELLS_ON_HALF_LEVELS: Final[Quantity] = register(
    "icon:theta_v_at_cells_on_half_levels",
    "K",
    (dims.CellDim, dims.KHalfDim),
    "theta_v_ic",
)
ICON_PERTURBED_EXNER_AT_CELLS_ON_MODEL_LEVELS: Final[Quantity] = register(
    "icon:perturbed_exner_at_cells_on_model_levels",
    "1",
    (dims.CellDim, dims.KDim),
    "exner_pr",
)
ICON_RHO_AT_CELLS_ON_HALF_LEVELS: Final[Quantity] = register(
    "icon:rho_at_cells_on_half_levels", "kg m-3", (dims.CellDim, dims.KHalfDim), "rho_ic"
)
ICON_EXNER_TENDENCY_DUE_TO_SLOW_PHYSICS: Final[Quantity] = register(
    "icon:exner_tendency_due_to_slow_physics",
    "s-1",
    (dims.CellDim, dims.KDim),
    "ddt_exner_phy",
)
ICON_GRF_TEND_RHO: Final[Quantity] = register(
    "icon:grf_tend_rho", "kg m-3 s-1", (dims.CellDim, dims.KDim), "grf_tend_rho"
)
ICON_GRF_TEND_THV: Final[Quantity] = register(
    "icon:grf_tend_thv", "K s-1", (dims.CellDim, dims.KDim), "grf_tend_thv"
)
ICON_GRF_TEND_W: Final[Quantity] = register(
    "icon:grf_tend_w", "m s-2", (dims.CellDim, dims.KHalfDim), "grf_tend_w"
)
ICON_MASS_FLUX_AT_EDGES_ON_MODEL_LEVELS: Final[Quantity] = register(
    "icon:mass_flux_at_edges_on_model_levels",
    "kg m-1 s-1",
    (dims.EdgeDim, dims.KDim),
    "mass_fl_e",
)
ICON_NORMAL_WIND_TENDENCY_DUE_TO_SLOW_PHYSICS_PROCESS: Final[Quantity] = register(
    "icon:normal_wind_tendency_due_to_slow_physics_process",
    "m s-2",
    (dims.EdgeDim, dims.KDim),
    "ddt_vn_phy",
)
ICON_GRF_TEND_VN: Final[Quantity] = register(
    "icon:grf_tend_vn", "m s-2", (dims.EdgeDim, dims.KDim), "grf_tend_vn"
)
ICON_NORMAL_WIND_ADVECTIVE_TENDENCY: Final[Quantity] = register(
    "icon:normal_wind_advective_tendency",
    "m s-2",
    (dims.EdgeDim, dims.KDim),
    "ddt_vn_apc_pc",
)
ICON_VERTICAL_WIND_ADVECTIVE_TENDENCY: Final[Quantity] = register(
    "icon:vertical_wind_advective_tendency",
    "m s-2",
    (dims.CellDim, dims.KHalfDim),
    "ddt_w_adv_pc",
)
ICON_RHO_IAU_INCREMENT: Final[Quantity] = register(
    "icon:rho_iau_increment", "kg m-3", (dims.CellDim, dims.KDim), "rho_incr"
)
ICON_NORMAL_WIND_IAU_INCREMENT: Final[Quantity] = register(
    "icon:normal_wind_iau_increment", "m s-1", (dims.EdgeDim, dims.KDim), "vn_incr"
)
ICON_EXNER_IAU_INCREMENT: Final[Quantity] = register(
    "icon:exner_iau_increment", "1", (dims.CellDim, dims.KDim), "exner_incr"
)
ICON_EXNER_DYNAMICAL_INCREMENT: Final[Quantity] = register(
    "icon:exner_dynamical_increment", "1", (dims.CellDim, dims.KDim), "exner_dyn_incr"
)

#: Diffusion diagnostic quantities.
ICON_HDEF_IC: Final[Quantity] = register(
    "icon:hdef_ic", "s-1", (dims.CellDim, dims.KHalfDim), "hdef_ic"
)
ICON_DIV_IC: Final[Quantity] = register(
    "icon:div_ic", "s-2", (dims.CellDim, dims.KHalfDim), "div_ic"
)
ICON_DWDX: Final[Quantity] = register("icon:dwdx", "s-1", (dims.CellDim, dims.KHalfDim), "dwdx")
ICON_DWDY: Final[Quantity] = register("icon:dwdy", "s-1", (dims.CellDim, dims.KHalfDim), "dwdy")

#: Dycore-to-advection handoff quantities.
ICON_VN_TRAJ: Final[Quantity] = register(
    "icon:vn_traj", "m s-1", (dims.EdgeDim, dims.KDim), "vn_traj"
)
ICON_MASS_FLX_ME: Final[Quantity] = register(
    "icon:mass_flx_me", "kg m-1 s-1", (dims.EdgeDim, dims.KDim), "mass_flx_me"
)
ICON_MASS_FLX_IC: Final[Quantity] = register(
    "icon:mass_flx_ic", "kg m-2 s-1", (dims.CellDim, dims.KHalfDim), "mass_flx_ic"
)
ICON_VOL_FLX_IC: Final[Quantity] = register(
    "icon:vol_flx_ic", "m s-1", (dims.CellDim, dims.KHalfDim), "vol_flx_ic"
)

#: Tracer advection diagnostic quantities.
ICON_AIRMASS_NOW: Final[Quantity] = register(
    "icon:airmass_now", "kg m-2", (dims.CellDim, dims.KDim), "airmass_now"
)
ICON_AIRMASS_NEW: Final[Quantity] = register(
    "icon:airmass_new", "kg m-2", (dims.CellDim, dims.KDim), "airmass_new"
)
ICON_GRF_TEND_TRACER: Final[Quantity] = register(
    "icon:grf_tend_tracer", "kg kg-1 s-1", (dims.CellDim, dims.KDim), "grf_tend_tracer"
)
ICON_HFL_TRACER: Final[Quantity] = register(
    "icon:hfl_tracer", "kg m-1 s-1", (dims.EdgeDim, dims.KDim), "hfl_tracer"
)
ICON_VFL_TRACER: Final[Quantity] = register(
    "icon:vfl_tracer", "kg m-1 s-1", (dims.CellDim, dims.KHalfDim), "vfl_tracer"
)

#: Diffusion metric quantities not already registered from metrics_attributes.
ICON_ZD_VERTOFFSET: Final[Quantity] = register(
    "icon:zd_vertoffset", "", (dims.CellDim, dims.C2E2CDim, dims.KDim), "zd_vertoffset"
)
ICON_ZD_DIFFCOEF: Final[Quantity] = register(
    "icon:zd_diffcoef", "", (dims.CellDim, dims.KDim), "zd_diffcoef"
)
ICON_ZD_INTCOEF: Final[Quantity] = register(
    "icon:zd_intcoef", "", (dims.CellDim, dims.C2E2CDim, dims.KDim), "zd_intcoef"
)

#: Dycore interpolation quantities not already registered from interpolation_attributes.
ICON_C_INTP: Final[Quantity] = register("icon:c_intp", "", (dims.VertexDim, dims.V2CDim), "c_intp")
ICON_RBF_VEC_COEFF_E: Final[Quantity] = register(
    "icon:rbf_vec_coeff_e", "", (dims.EdgeDim, dims.E2C2EDim), "rbf_vec_coeff_e"
)
ICON_POS_ON_TPLANE_E_1: Final[Quantity] = register(
    "icon:pos_on_tplane_e_1", "", (dims.EdgeDim, dims.E2CDim), "pos_on_tplane_e_1"
)
ICON_POS_ON_TPLANE_E_2: Final[Quantity] = register(
    "icon:pos_on_tplane_e_2", "", (dims.EdgeDim, dims.E2CDim), "pos_on_tplane_e_2"
)

#: Dycore metric quantities not already registered from metrics_attributes.
ICON_WGTFAC_E: Final[Quantity] = register(
    "icon:wgtfac_e", "", (dims.EdgeDim, dims.KDim), "wgtfac_e"
)
ICON_WGTFACQ_E: Final[Quantity] = register(
    "icon:wgtfacq_e", "", (dims.EdgeDim, dims.KDim), "wgtfacq_e"
)
ICON_DDXN_Z_FULL: Final[Quantity] = register(
    "icon:ddxn_z_full", "", (dims.EdgeDim, dims.KDim), "ddxn_z_full"
)
ICON_DDQT_Z_FULL_E: Final[Quantity] = register(
    "icon:ddqz_z_full_e", "m", (dims.EdgeDim, dims.KDim), "ddqz_z_full_e"
)
ICON_DDXT_Z_FULL: Final[Quantity] = register(
    "icon:ddxt_z_full", "", (dims.EdgeDim, dims.KDim), "ddxt_z_full"
)
ICON_VERTOFFSET_GRADP: Final[Quantity] = register(
    "icon:vertoffset_gradp", "", (dims.EdgeDim, dims.E2CDim, dims.KDim), "vertoffset_gradp"
)
ICON_ZDIFF_GRADP: Final[Quantity] = register(
    "icon:zdiff_gradp", "", (dims.EdgeDim, dims.E2CDim, dims.KDim), "zdiff_gradp"
)
ICON_NFLAT_GRADP: Final[Quantity] = register("icon:nflat_gradp", "", (), "nflat_gradp")
ICON_PG_EXDIST: Final[Quantity] = register(
    "icon:pg_exdist", "", (dims.EdgeDim, dims.KDim), "pg_exdist_dsl"
)
ICON_COEFF1_DWDZ: Final[Quantity] = register(
    "icon:coeff1_dwdz", "", (dims.CellDim, dims.KDim), "coeff1_dwdz"
)
ICON_COEFF2_DWDZ: Final[Quantity] = register(
    "icon:coeff2_dwdz", "", (dims.CellDim, dims.KDim), "coeff2_dwdz"
)
ICON_MASK_PROG_HALO_C: Final[Quantity] = register(
    "icon:mask_prog_halo_c", "", (dims.CellDim, dims.KDim), "mask_prog_halo_c"
)

#: Quantities registered from factory metadata, exposed as constants for use in
#: state dataclass declarations.
C_LIN_E: Final[Quantity] = get("interpolation_coefficient_from_cell_to_edge")
COEFF_GRADEKIN: Final[Quantity] = get("coeff_gradekin")
DDQZ_Z_HALF: Final[Quantity] = get("functional_determinant_of_metrics_on_interface_levels")
D2DEXDZ2_FAC1_MC: Final[Quantity] = get("d2dexdz2_fac1_mc")
D2DEXDZ2_FAC2_MC: Final[Quantity] = get("d2dexdz2_fac2_mc")
D_EXNER_DZ_REF_IC: Final[Quantity] = get("d_exner_dz_ref_ic")
E_BLN_C_S: Final[Quantity] = get("bilinear_edge_cell_weight")
E_FLX_AVG: Final[Quantity] = get("e_flux_average")
EXNER_EXFAC: Final[Quantity] = get("exner_exfac")
EXNER_REF_MC: Final[Quantity] = get("exner_ref_mc")
EXNER_W_EXPLICIT_WEIGHT_PARAMETER: Final[Quantity] = get("exner_w_explicit_weight_parameter")
EXNER_W_IMPLICIT_WEIGHT_PARAMETER: Final[Quantity] = get("exner_w_implicit_weight_parameter")
GEOFAC_DIV: Final[Quantity] = get("geometrical_factor_for_divergence")
GEOFAC_GRDIV: Final[Quantity] = get("geometrical_factor_for_gradient_of_divergence")
GEOFAC_GRG_X: Final[Quantity] = get("geometrical_factor_for_green_gauss_gradient_x")
GEOFAC_GRG_Y: Final[Quantity] = get("geometrical_factor_for_green_gauss_gradient_y")
GEOFAC_N2S: Final[Quantity] = get("geometrical_factor_for_nabla_2_scalar")
GEOFAC_ROT: Final[Quantity] = get("geometrical_factor_for_curl")
HORIZONTAL_MASK_FOR_3D_DIVDAMP: Final[Quantity] = get("horizontal_mask_for_3d_divdamp")
INV_DDQZ_Z_FULL: Final[Quantity] = get(
    "inverse_of_functional_determinant_of_metrics_on_full_levels"
)
NUDGECOEFFS_E: Final[Quantity] = get("nudging_coefficients_for_edges")
RBF_VEC_COEFF_V1: Final[Quantity] = get("rbf_interpolation_coefficient_vertex_1")
RBF_VEC_COEFF_V2: Final[Quantity] = get("rbf_interpolation_coefficient_vertex_2")
RAYLEIGH_W: Final[Quantity] = get("rayleigh_w")
RHO_REF_MC: Final[Quantity] = get("rho_ref_mc")
RHO_REF_ME: Final[Quantity] = get("rho_ref_me")
SCALING_FACTOR_FOR_3D_DIVDAMP: Final[Quantity] = get("scaling_factor_for_3d_divergence_damping")
THETA_REF_IC: Final[Quantity] = get("theta_ref_ic")
THETA_REF_MC: Final[Quantity] = get("theta_ref_mc")
THETA_REF_ME: Final[Quantity] = get("theta_ref_me")
WGTFAC_C: Final[Quantity] = get("wgtfac_c")
WGTFACQ_C: Final[Quantity] = get("weighting_factor_for_quadratic_interpolation_to_cell_surface")
