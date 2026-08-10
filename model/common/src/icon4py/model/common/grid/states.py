# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from typing import Any

from gt4py import next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.grid import geometry_attributes as geometry_attrs
from icon4py.model.common.states import spec


def _geom_spec(attr_name: str, *, default: Any = dataclasses.MISSING) -> Any:
    """Return a ``spec()`` declaration from geometry metadata.

    Inverse geometry fields (``inverse_of_<base>``) are generated from the base
    field metadata and are not present as keys in ``geometry_attrs.attrs``.
    """
    if attr_name.startswith("inverse_of_"):
        base_name = attr_name[len("inverse_of_") :]
        meta = geometry_attrs.metadata_for_inverse(geometry_attrs.attrs[base_name])
    else:
        meta = geometry_attrs.attrs[attr_name]
    kwargs: dict[str, Any] = dict(
        quantity=meta["standard_name"],
        units=meta.get("units", ""),
        dims=tuple(meta.get("dims", ())),
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    if default is not dataclasses.MISSING:
        kwargs["default"] = default
    return spec.spec(**kwargs)


@dataclasses.dataclass(frozen=True)
class CellParams:
    #: Latitude at the cell center. The cell center is defined to be the circumcenter of a triangle.
    cell_center_lat: fa.CellField[float] = _geom_spec(geometry_attrs.CELL_LAT)
    #: Longitude at the cell center. The cell center is defined to be the circumcenter of a triangle.
    cell_center_lon: fa.CellField[float] = _geom_spec(geometry_attrs.CELL_LON)
    #: Area of a cell, defined in ICON in mo_model_domain.f90:t_grid_cells%area
    area: fa.CellField[float] = _geom_spec(geometry_attrs.CELL_AREA)
    mean_cell_area: float | None = _geom_spec(geometry_attrs.MEAN_CELL_AREA, default=None)


@dataclasses.dataclass(frozen=True)
class EdgeParams:
    tangent_orientation: fa.EdgeField[float] = _geom_spec(geometry_attrs.TANGENT_ORIENTATION)
    inverse_primal_edge_lengths: fa.EdgeField[float] = _geom_spec(
        f"inverse_of_{geometry_attrs.EDGE_LENGTH}"
    )
    inverse_dual_edge_lengths: fa.EdgeField[float] = _geom_spec(
        f"inverse_of_{geometry_attrs.DUAL_EDGE_LENGTH}"
    )
    inverse_vertex_vertex_lengths: fa.EdgeField[float] = _geom_spec(
        f"inverse_of_{geometry_attrs.VERTEX_VERTEX_LENGTH}"
    )
    primal_normal_vert_x: gtx.Field[[dims.EdgeDim, dims.E2C2VDim], float] = _geom_spec(
        geometry_attrs.EDGE_NORMAL_VERTEX_U
    )
    primal_normal_vert_y: gtx.Field[[dims.EdgeDim, dims.E2C2VDim], float] = _geom_spec(
        geometry_attrs.EDGE_NORMAL_VERTEX_V
    )
    dual_normal_vert_x: gtx.Field[[dims.EdgeDim, dims.E2C2VDim], float] = _geom_spec(
        geometry_attrs.EDGE_TANGENT_VERTEX_U
    )
    dual_normal_vert_y: gtx.Field[[dims.EdgeDim, dims.E2C2VDim], float] = _geom_spec(
        geometry_attrs.EDGE_TANGENT_VERTEX_V
    )
    primal_normal_cell_x: gtx.Field[[dims.EdgeDim, dims.E2CDim], float] = _geom_spec(
        geometry_attrs.EDGE_NORMAL_CELL_U
    )
    primal_normal_cell_y: gtx.Field[[dims.EdgeDim, dims.E2CDim], float] = _geom_spec(
        geometry_attrs.EDGE_NORMAL_CELL_V
    )
    dual_normal_cell_x: gtx.Field[[dims.EdgeDim, dims.E2CDim], float] = _geom_spec(
        geometry_attrs.EDGE_TANGENT_CELL_U
    )
    dual_normal_cell_y: gtx.Field[[dims.EdgeDim, dims.E2CDim], float] = _geom_spec(
        geometry_attrs.EDGE_TANGENT_CELL_V
    )
    edge_areas: fa.EdgeField[float] = _geom_spec(geometry_attrs.EDGE_AREA)
    coriolis_frequency: fa.EdgeField[float] = _geom_spec(geometry_attrs.CORIOLIS_PARAMETER)
    edge_center_lat: fa.EdgeField[float] = _geom_spec(geometry_attrs.EDGE_LAT)
    edge_center_lon: fa.EdgeField[float] = _geom_spec(geometry_attrs.EDGE_LON)
    primal_normal_x: gtx.Field[[dims.EdgeDim, dims.E2CDim], float] = _geom_spec(
        geometry_attrs.EDGE_NORMAL_U
    )
    primal_normal_y: gtx.Field[[dims.EdgeDim, dims.E2CDim], float] = _geom_spec(
        geometry_attrs.EDGE_NORMAL_V
    )

    @property
    def primal_normal_vert(
        self,
    ) -> tuple[
        gtx.Field[[dims.EdgeDim, dims.E2C2VDim], float],
        gtx.Field[[dims.EdgeDim, dims.E2C2VDim], float],
    ]:
        return (self.primal_normal_vert_x, self.primal_normal_vert_y)

    @property
    def dual_normal_vert(
        self,
    ) -> tuple[
        gtx.Field[[dims.EdgeDim, dims.E2C2VDim], float],
        gtx.Field[[dims.EdgeDim, dims.E2C2VDim], float],
    ]:
        return (self.dual_normal_vert_x, self.dual_normal_vert_y)

    @property
    def primal_normal_cell(
        self,
    ) -> tuple[
        gtx.Field[[dims.EdgeDim, dims.E2CDim], float],
        gtx.Field[[dims.EdgeDim, dims.E2CDim], float],
    ]:
        return (self.primal_normal_cell_x, self.primal_normal_cell_y)

    @property
    def dual_normal_cell(
        self,
    ) -> tuple[
        gtx.Field[[dims.EdgeDim, dims.E2CDim], float],
        gtx.Field[[dims.EdgeDim, dims.E2CDim], float],
    ]:
        return (self.dual_normal_cell_x, self.dual_normal_cell_y)

    @property
    def edge_center(self) -> tuple[fa.EdgeField[float], fa.EdgeField[float]]:
        return (self.edge_center_lat, self.edge_center_lon)

    @property
    def primal_normal(
        self,
    ) -> tuple[
        gtx.Field[[dims.EdgeDim, dims.E2CDim], float],
        gtx.Field[[dims.EdgeDim, dims.E2CDim], float],
    ]:
        return (self.primal_normal_x, self.primal_normal_y)
