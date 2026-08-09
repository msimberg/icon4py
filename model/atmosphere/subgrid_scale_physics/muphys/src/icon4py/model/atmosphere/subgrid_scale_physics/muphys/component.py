# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Muphys component and its typed input/output boundaries."""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import gt4py.next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.muphys import config as muphys_config
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.definitions import SPECIES, Q
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.driver.run_full_muphys import (
    setup_muphys,
)
from icon4py.model.common import (
    dimension as dims,
    field_type_aliases as fa,
    model_backends,
    model_options,
    time,
    type_alias as ta,
)
from icon4py.model.common.components.components import Component
from icon4py.model.common.diagnostic_calculations.stencils import calculate_tendency
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.math.stencils import generic_math_operations
from icon4py.model.common.states import quantities as q, spec


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import icon as icon_grid


@dataclasses.dataclass(frozen=True)
class MuphysInput:
    """Input boundary of the muphys component."""

    dz: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity="icon:layer_thickness",
        units="m",
        dims=(dims.CellDim, dims.KDim),
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.STATIC,
    )
    te: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity="icon:air_temperature",
        units="K",
        dims=(dims.CellDim, dims.KDim),
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.SCRATCH,
    )
    p: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity="icon:air_pressure",
        units="Pa",
        dims=(dims.CellDim, dims.KDim),
        intent=spec.Intent.READ,
        lifetime=spec.Lifetime.SCRATCH,
    )
    rho: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity=q.AIR_DENSITY.name,
        units=q.AIR_DENSITY.units,
        dims=q.AIR_DENSITY.dims,
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
    qi: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity=q.SPECIFIC_ICE_CONTENT.name,
        units=q.SPECIFIC_ICE_CONTENT.units,
        dims=q.SPECIFIC_ICE_CONTENT.dims,
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


@dataclasses.dataclass(frozen=True)
class MuphysOutput:
    """Output boundary of the muphys component.

    Tendency outputs are applied to the prognostic state by the physics boundary;
    precip outputs are stored as diagnostics.
    """

    tend_temperature: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity="icon:tend_temperature_due_to_muphys",
        units="K s-1",
        dims=(dims.CellDim, dims.KDim),
        intent=spec.Intent.WRITE,
        lifetime=spec.Lifetime.SCRATCH,
        role=spec.Role.TENDENCY,
    )
    tend_qv: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity="icon:tend_specific_humidity_due_to_muphys",
        units="s-1",
        dims=(dims.CellDim, dims.KDim),
        intent=spec.Intent.WRITE,
        lifetime=spec.Lifetime.SCRATCH,
        role=spec.Role.TENDENCY,
    )
    tend_qc: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity="icon:tend_specific_cloud_content_due_to_muphys",
        units="s-1",
        dims=(dims.CellDim, dims.KDim),
        intent=spec.Intent.WRITE,
        lifetime=spec.Lifetime.SCRATCH,
        role=spec.Role.TENDENCY,
    )
    tend_qr: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity="icon:tend_specific_rain_content_due_to_muphys",
        units="s-1",
        dims=(dims.CellDim, dims.KDim),
        intent=spec.Intent.WRITE,
        lifetime=spec.Lifetime.SCRATCH,
        role=spec.Role.TENDENCY,
    )
    tend_qs: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity="icon:tend_specific_snow_content_due_to_muphys",
        units="s-1",
        dims=(dims.CellDim, dims.KDim),
        intent=spec.Intent.WRITE,
        lifetime=spec.Lifetime.SCRATCH,
        role=spec.Role.TENDENCY,
    )
    tend_qi: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity="icon:tend_specific_ice_content_due_to_muphys",
        units="s-1",
        dims=(dims.CellDim, dims.KDim),
        intent=spec.Intent.WRITE,
        lifetime=spec.Lifetime.SCRATCH,
        role=spec.Role.TENDENCY,
    )
    tend_qg: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity="icon:tend_specific_graupel_content_due_to_muphys",
        units="s-1",
        dims=(dims.CellDim, dims.KDim),
        intent=spec.Intent.WRITE,
        lifetime=spec.Lifetime.SCRATCH,
        role=spec.Role.TENDENCY,
    )
    pflx: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity="icon:precipitation_flux",
        units="kg m-2 s-1",
        dims=(dims.CellDim, dims.KDim),
        intent=spec.Intent.WRITE,
        lifetime=spec.Lifetime.SCRATCH,
        role=spec.Role.DIAGNOSTIC,
    )
    pr: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity="icon:rainfall_flux",
        units="kg m-2 s-1",
        dims=(dims.CellDim, dims.KDim),
        intent=spec.Intent.WRITE,
        lifetime=spec.Lifetime.SCRATCH,
        role=spec.Role.DIAGNOSTIC,
    )
    ps: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity="icon:snowfall_flux",
        units="kg m-2 s-1",
        dims=(dims.CellDim, dims.KDim),
        intent=spec.Intent.WRITE,
        lifetime=spec.Lifetime.SCRATCH,
        role=spec.Role.DIAGNOSTIC,
    )
    pi: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity="icon:icefall_flux",
        units="kg m-2 s-1",
        dims=(dims.CellDim, dims.KDim),
        intent=spec.Intent.WRITE,
        lifetime=spec.Lifetime.SCRATCH,
        role=spec.Role.DIAGNOSTIC,
    )
    pg: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity="icon:graupelfall_flux",
        units="kg m-2 s-1",
        dims=(dims.CellDim, dims.KDim),
        intent=spec.Intent.WRITE,
        lifetime=spec.Lifetime.SCRATCH,
        role=spec.Role.DIAGNOSTIC,
    )
    pre: fa.CellKField[ta.wpfloat] = spec.spec(
        quantity="icon:precipitation_energy_flux",
        units="W m-2",
        dims=(dims.CellDim, dims.KDim),
        intent=spec.Intent.WRITE,
        lifetime=spec.Lifetime.SCRATCH,
        role=spec.Role.DIAGNOSTIC,
    )


class MuphysComponent(Component[MuphysInput, MuphysOutput]):
    """Per-process adapter wrapping the muphys microphysics program."""

    def __init__(
        self,
        grid: icon_grid.IconGrid,
        dtime: time.RelativeTime,
        qnc: float,
        backend: gtx_typing.Backend | None = None,
        *,
        scheme: muphys_config.MuphysScheme = muphys_config.MuphysScheme.KOKKOS_MUPHYS,
        step: Callable[..., Any] | None = None,
    ) -> None:
        self._ncells = grid.num_cells
        self._nlev = grid.num_levels
        self._dt_seconds = dtime.total_seconds()
        self._qnc = qnc
        self._backend = model_options.customize_backend(program=None, backend=backend)

        cell_domain = h_grid.domain(dims.CellDim)
        # ICON applies physics on the prognostic cells only:
        # grf_bdywidth_c+1 .. min_rlcell_int (NUDGING start .. LOCAL end).
        cell_start = grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        cell_end = grid.end_index(cell_domain(h_grid.Zone.LOCAL))

        # Inputs are copied over the full range: the muphys step reads whole fields.
        full_horizontal_sizes = {
            "horizontal_start": gtx.int32(0),
            "horizontal_end": gtx.int32(self._ncells),
        }
        # Tendencies only on the prognostic subdomain -- the tendency buffers stay
        # zero outside, so scattering is a no-op on boundary/halo rows, as in ICON.
        prognostic_horizontal_sizes = {
            "horizontal_start": cell_start,
            "horizontal_end": cell_end,
        }
        vertical_sizes = {"vertical_start": gtx.int32(0), "vertical_end": gtx.int32(self._nlev)}

        self._calculate_tendency = model_options.setup_program(
            program=calculate_tendency.calculate_cell_kdim_field_tendency,
            backend=self._backend,
            horizontal_sizes=prognostic_horizontal_sizes,
            vertical_sizes=vertical_sizes,
            offset_provider={},
        )
        self._copy_field = model_options.setup_program(
            program=generic_math_operations.copy_field_on_cell_k,
            backend=self._backend,
            horizontal_sizes=full_horizontal_sizes,
            vertical_sizes=vertical_sizes,
            offset_provider={},
        )

        allocator = model_backends.get_allocator(backend)

        if step is None:
            sizes = types.SimpleNamespace(ncells=self._ncells, nlev=self._nlev)
            step = setup_muphys(
                inp=sizes,  # type: ignore[arg-type]  # only .ncells/.nlev are read
                dt=self._dt_seconds,
                qnc=qnc,
                backend=backend,
                single_program=False,
                scheme=scheme,
            )
        self._step = step

        cell_k_domain = gtx.domain({dims.CellDim: self._ncells, dims.KDim: self._nlev})
        self._pflx = gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator)
        self._pr = gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator)
        self._ps = gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator)
        self._pi = gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator)
        self._pg = gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator)
        self._pre = gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator)

        self._tendencies: dict[str, fa.CellKField[ta.wpfloat]] = {
            "tend_temperature": gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator),
            "tend_qv": gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator),
            "tend_qc": gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator),
            "tend_qr": gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator),
            "tend_qs": gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator),
            "tend_qi": gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator),
            "tend_qg": gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator),
        }

        self._te_in = gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator)
        self._q_in = Q(
            v=gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator),
            c=gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator),
            r=gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator),
            s=gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator),
            i=gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator),
            g=gtx.zeros(cell_k_domain, dtype=ta.wpfloat, allocator=allocator),
        )

    @classmethod
    def input_type(cls) -> type[MuphysInput]:
        return MuphysInput

    @classmethod
    def output_type(cls) -> type[MuphysOutput]:
        return MuphysOutput

    def run(self, state: MuphysInput) -> MuphysOutput:
        """Run muphys, then convert its updated state into tendencies.

        muphys updates the state in place (t_out/q_out alias te/q_in); this boundary
        converts it to tendencies ``(new - old) / dt``. Precip outputs are diagnostics,
        passed straight through.
        """
        self._copy_field(field=state.te, output_field=self._te_in)
        for s in SPECIES:
            self._copy_field(field=getattr(state, f"q{s}"), output_field=getattr(self._q_in, s))

        # muphys must be invoked in place (the outputs alias the inputs), following
        # the same convention as the muphys driver: the dace backend compiles the
        # program with this aliasing baked in and leaves distinct output buffers
        # unwritten.
        self._step(
            dz=state.dz,
            te=self._te_in,
            p=state.p,
            rho=state.rho,
            q_in=self._q_in,
            q_out=self._q_in,
            t_out=self._te_in,
            pflx=self._pflx,
            pr=self._pr,
            ps=self._ps,
            pi=self._pi,
            pg=self._pg,
            pre=self._pre,
        )

        self._calculate_tendency(
            dtime=self._dt_seconds,
            old_field=state.te,
            new_field=self._te_in,
            tendency=self._tendencies["tend_temperature"],
        )
        for s in SPECIES:
            self._calculate_tendency(
                dtime=self._dt_seconds,
                old_field=getattr(state, f"q{s}"),
                new_field=getattr(self._q_in, s),
                tendency=self._tendencies[f"tend_q{s}"],
            )

        return MuphysOutput(
            tend_temperature=self._tendencies["tend_temperature"],
            tend_qv=self._tendencies["tend_qv"],
            tend_qc=self._tendencies["tend_qc"],
            tend_qr=self._tendencies["tend_qr"],
            tend_qs=self._tendencies["tend_qs"],
            tend_qi=self._tendencies["tend_qi"],
            tend_qg=self._tendencies["tend_qg"],
            pflx=self._pflx,
            pr=self._pr,
            ps=self._ps,
            pi=self._pi,
            pg=self._pg,
            pre=self._pre,
        )
