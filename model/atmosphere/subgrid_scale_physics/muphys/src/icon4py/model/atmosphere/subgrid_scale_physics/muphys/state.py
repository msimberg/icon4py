# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Typed physics-state boundary for the muphys component."""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import gt4py.next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.component import (
    MuphysInput,
    MuphysOutput,
)
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.definitions import (
    PRECIP_DIAGNOSTICS,
)
from icon4py.model.common import (
    dimension as dims,
    field_type_aliases as fa,
    model_options,
    type_alias as ta,
)
from icon4py.model.common.components.physics_state import TypedPhysicsState
from icon4py.model.common.diagnostic_calculations.stencils import (
    calculate_tendency,
    update_exner_and_theta_v,
)
from icon4py.model.common.math.stencils import generic_math_operations
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.common.states import (
        diagnostic_state as diagnostics,
        prognostic_state as prognostics,
        tracer_states,
    )


class State(TypedPhysicsState[MuphysInput, MuphysOutput]):
    """The muphys physics State adapter.

    Bridges the dycore's prognostic state and the typed ``MuphysComponent``
    contract. ``gather_from_prognostic`` builds a ``MuphysInput`` from the
    prognostic state and diagnosed fields; ``scatter_to_prognostic`` applies the
    returned tendencies to the prognostic state and stores the precip
    diagnostics.
    """

    def __init__(
        self,
        grid: base_grid.Grid,
        dz: fa.CellKField[ta.wpfloat],
        backend: gtx_typing.Backend | None = None,
        diagnostic: diagnostics.DiagnosticState | None = None,
    ) -> None:
        self._num_cells = grid.num_cells
        self._num_levels = grid.num_levels
        self._backend = backend
        self.diagnostic = diagnostic

        full_horizontal = {
            "horizontal_start": gtx.int32(0),
            "horizontal_end": gtx.int32(self._num_cells),
        }
        full_vertical = {
            "vertical_start": gtx.int32(0),
            "vertical_end": gtx.int32(self._num_levels),
        }

        self._apply_tendency = model_options.setup_program(
            program=generic_math_operations.compute_field_a_plus_coeff_times_field_b_on_cell_k,
            backend=self._backend,
            horizontal_sizes=full_horizontal,
            vertical_sizes=full_vertical,
            offset_provider={},
        )
        self._calculate_virtual_temperature_tendency = model_options.setup_program(
            program=calculate_tendency.calculate_virtual_temperature_tendency,
            backend=self._backend,
            horizontal_sizes=full_horizontal,
            vertical_sizes=full_vertical,
            offset_provider={},
        )
        self._update_exner_and_theta_v = model_options.setup_program(
            program=update_exner_and_theta_v.update_exner_and_theta_v,
            backend=self._backend,
            horizontal_sizes=full_horizontal,
            vertical_sizes=full_vertical,
            offset_provider={},
        )

        self.dz = dz
        self.te: fa.CellKField[ta.wpfloat] | None = None
        self.p: fa.CellKField[ta.wpfloat] | None = None
        self.tv: fa.CellKField[ta.wpfloat] | None = None
        self._prognostic: prognostics.PrognosticState | None = None
        self._tracers: tracer_states.TracerState | None = None

        # INTERNAL
        self._new_te = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)
        self._tv_tendency = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, allocator=backend)
        self._last_outputs: MuphysOutput | None = None

    def gather_from_prognostic(
        self,
        prognostic: prognostics.PrognosticState,
        tracers: tracer_states.TracerState,
    ) -> MuphysInput:
        """Build ``MuphysInput`` from the prognostic state and diagnosed fields."""
        self._prognostic = prognostic
        self.rho = prognostic.rho
        # muphys needs all six moisture species; TracerState fields are optional (a
        # tracer may be inactive per TracerConfig), so fail loudly once here rather
        # than feed None into the microphysics.
        missing = [
            f"q{s}" for s in ("v", "c", "r", "s", "i", "g") if getattr(tracers, f"q{s}") is None
        ]
        if missing:
            raise ValueError(
                f"muphys requires all moisture species active in the TracerState; missing: {missing}"
            )
        self._tracers = tracers
        qv = tracers.qv
        qc = tracers.qc
        qr = tracers.qr
        qs = tracers.qs
        qi = tracers.qi
        qg = tracers.qg
        assert qv is not None, "qv must be active for muphys"
        assert qc is not None, "qc must be active for muphys"
        assert qr is not None, "qr must be active for muphys"
        assert qs is not None, "qs must be active for muphys"
        assert qi is not None, "qi must be active for muphys"
        assert qg is not None, "qg must be active for muphys"

        # Read the canonical diagnosed fields from the diagnostic state. The canonical
        # derivation runs before physics in the time loop and uses the same stencil
        # programs as the old in-place diagnosis, so the values muphys consumes are
        # identical on the prognostic subdomain.
        assert self.diagnostic is not None, "muphys State requires a DiagnosticState"
        self.te = self.diagnostic.temperature
        self.p = self.diagnostic.pressure
        self.tv = self.diagnostic.virtual_temperature
        return MuphysInput(
            dz=self.dz,
            te=self.te,
            p=self.p,
            rho=self.rho,
            qv=qv,
            qc=qc,
            qr=qr,
            qs=qs,
            qi=qi,
            qg=qg,
        )

    def apply_tendencies(
        self,
        outputs: MuphysOutput,
        dtime: datetime.timedelta,
    ) -> None:
        """Apply muphys output tendencies back to the prognostic state.

        Moisture tendencies are applied to the tracers, and the temperature
        tendency drives an exner/theta_v update via the exact EOS.
        """
        assert self._prognostic is not None, "gather_from_prognostic must be called first"
        assert self._tracers is not None, "gather_from_prognostic must be called first"
        # convert to seconds only at the gt4py boundary (stencils take a scalar dt)
        dt_seconds = dtime.total_seconds()
        # 1. Apply moisture tendencies to the tracers (in place; tracers were bound in gather).
        for s in ("v", "c", "r", "s", "i", "g"):
            tracer = getattr(self._tracers, f"q{s}")
            self._apply_tendency(
                field_a=tracer,
                coeff=dt_seconds,
                field_b=getattr(outputs, f"tend_q{s}"),
                output_field=tracer,
            )

        # 2. tend_T -> new temperature: new_te = te + tend_T*dt
        self._apply_tendency(
            field_a=self.te,
            coeff=dt_seconds,
            field_b=outputs.tend_temperature,
            output_field=self._new_te,
        )

        # dTv/dt from the new temperature and the species just updated in step 1
        self._calculate_virtual_temperature_tendency(
            dtime=dt_seconds,
            qv=self._tracers.qv,
            qc=self._tracers.qc,
            qi=self._tracers.qi,
            qr=self._tracers.qr,
            qs=self._tracers.qs,
            qg=self._tracers.qg,
            temperature=self._new_te,
            virtual_temperature=self.tv,
            virtual_temperature_tendency=self._tv_tendency,
        )
        # Recompute exner via the exact EOS from the updated virtual temperature and
        # diagnose theta_v = Tv/exner, mirroring ICON's phy2dyn coupling
        # (mo_interface_iconam_aes.f90). The exner/rho/theta_v trio stays EOS-consistent.
        self._update_exner_and_theta_v(
            rho=self.rho,
            virtual_temperature=self.tv,
            virtual_temperature_tendency=self._tv_tendency,
            dtime=dt_seconds,
            exner=self._prognostic.exner,
            theta_v=self._prognostic.theta_v,
        )

    def store_diagnostics(
        self,
        outputs: MuphysOutput,
    ) -> None:
        """Store muphys precip diagnostics without touching the prognostic state."""
        self._last_outputs = outputs

    @property
    def precip_diagnostics(self) -> dict[str, fa.CellKField[ta.wpfloat]]:
        """muphys precip diagnostics keyed by name, ready for IO / plotting."""
        if self._last_outputs is None:
            raise RuntimeError(
                "precip_diagnostics accessed before apply_tendencies or store_diagnostics"
            )
        return {name: getattr(self._last_outputs, name) for name in PRECIP_DIAGNOSTICS}
