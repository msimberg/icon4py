# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""The ``PhysicsDriver`` and its processes."""

from __future__ import annotations

import dataclasses
import datetime
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import gt4py.next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver import (
    physics_state,
    static_fields,
)
from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver.process_time_control import (
    ProcessTimeControl,
)
from icon4py.model.common import dimension as dims, model_backends, model_options
from icon4py.model.common.components import framework as fw, quantities as qty, states
from icon4py.model.common.interpolation.stencils.compute_vn_from_uv import compute_vn_from_uv
from icon4py.model.common.interpolation.stencils.edge_2_cell_vector_rbf_interpolation import (
    edge_2_cell_vector_rbf_interpolation,
)
from icon4py.model.common.math.stencils import generic_math_operations
from icon4py.model.common.physics.thermodynamics import (
    compute_pressure,
    compute_temperature,
    compute_tendencies,
)
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    from icon4py.model.common.grid import base as base_grid


__all__ = ["PhysicsDriver", "PhysicsProcess", "ProcessTimeControl", "Step", "bind"]

type Step = Callable[[physics_state.EntryState], fw.State]


def bind[I: fw.State, O: fw.State](
    run: Callable[[I], O], collect_input: Callable[[physics_state.EntryState], I]
) -> Step:
    """
    A process step: a component's `run` on what its `collect_input` picks from the entry state.

    Checks once that the two belong together (another component's `collect_input` is a type
    error); the driver then sees the step only as a `Step`.
    """
    return lambda entry: run(collect_input(entry))


@dataclasses.dataclass(frozen=True)
class PhysicsProcess:
    """One physics process: its step and its time control, which belongs to the composer."""

    name: str
    step: Step
    time_control: ProcessTimeControl


def _present[Q: fw.Quantity](field: fw.Field[Q] | None) -> fw.Field[Q]:
    """A tracer leaf the physics requires."""
    if field is None:
        raise ValueError("physics requires all moisture species active in the TracerState")
    return field


class TendencySums(fw.State):
    """
    The summed tendencies of one step, one optional leaf per tendency the driver applies.

    A leaf is None until some process reports it; `zero` keeps the presence across steps.
    """

    tend_qv: fw.Field[qty.TendencyOfQvOnCellK] | None = None
    tend_qc: fw.Field[qty.TendencyOfQcOnCellK] | None = None
    tend_qi: fw.Field[qty.TendencyOfQiOnCellK] | None = None
    tend_qr: fw.Field[qty.TendencyOfQrOnCellK] | None = None
    tend_qs: fw.Field[qty.TendencyOfQsOnCellK] | None = None
    tend_qg: fw.Field[qty.TendencyOfQgOnCellK] | None = None
    tend_temperature: fw.Field[qty.TendencyOfTemperatureOnCellK] | None = None
    tend_u: fw.Field[qty.TendencyOfUOnCellK] | None = None
    tend_v: fw.Field[qty.TendencyOfVOnCellK] | None = None
    tend_w: fw.Field[qty.TendencyOfWOnCellKHalf] | None = None


class PhysicsDriver(fw.Component):
    """
    Runs the physics processes under parallel coupling (ICON AES's aes_phy_main).

    In one step the entry state is diagnosed once from the prognostics, every process active on
    the step reads that same entry state, the tendencies of their outputs are summed and the sum
    is applied to the prognostics once, at the end. The processes never write the prognostics.

    Not implemented: ICON AES couples microphysics and turbulence sequentially (each process
    sees the previous one's provisional update), which would need per-process state advances.

    In place like the diffusion: the composer passes the same buffers as input and output, `run`
    copies the input into `out` first where they differ. `rho` is read only.
    """

    class Input(fw.State):
        vn: fw.Field[qty.VnOnEdgeK]
        w: fw.Field[qty.WOnCellKHalf]
        exner: fw.Field[qty.ExnerOnCellK]
        theta_v: fw.Field[qty.ThetaVOnCellK]
        rho: fw.Field[qty.RhoOnCellK]
        qv: fw.Field[qty.QvOnCellK] | None = None
        qc: fw.Field[qty.QcOnCellK] | None = None
        qi: fw.Field[qty.QiOnCellK] | None = None
        qr: fw.Field[qty.QrOnCellK] | None = None
        qs: fw.Field[qty.QsOnCellK] | None = None
        qg: fw.Field[qty.QgOnCellK] | None = None
        dtime: datetime.timedelta
        # the end of the step being integrated
        simulation_current_datetime: datetime.datetime

    class Output(fw.State):
        vn: fw.Field[qty.VnOnEdgeK]
        w: fw.Field[qty.WOnCellKHalf]
        exner: fw.Field[qty.ExnerOnCellK]
        theta_v: fw.Field[qty.ThetaVOnCellK]
        qv: fw.Field[qty.QvOnCellK] | None = None
        qc: fw.Field[qty.QcOnCellK] | None = None
        qi: fw.Field[qty.QiOnCellK] | None = None
        qr: fw.Field[qty.QrOnCellK] | None = None
        qs: fw.Field[qty.QsOnCellK] | None = None
        qg: fw.Field[qty.QgOnCellK] | None = None

    def __init__(
        self,
        processes: Sequence[PhysicsProcess],
        *,
        grid: base_grid.Grid,
        static_state: static_fields.PhysicsDriverStaticState,
        backend: model_backends.BackendLike = None,
    ) -> None:
        super().__init__(grid, model_backends.get_allocator(backend))
        self.processes = tuple(processes)
        # the diagnostics the processes read, derived here each step
        self.diagnostics = fw.allocate(states.Diagnostics, grid, self.allocator)
        # the summed tendencies of the step; a leaf appears when a process first reports it
        self.tendency_sums = fw.allocate(TendencySums, grid, self.allocator, only=())
        # the last output of each process, reused on the steps it is not active
        # TODO(jcanton): part 02 fix this. Key by the process's component (identity), once the
        # process holds it, and drop `PhysicsProcess.name`, which exists only for this key.
        self.outputs: dict[str, fw.State] = {}

        cells = {"horizontal_start": gtx.int32(0), "horizontal_end": gtx.int32(grid.num_cells)}
        edges = {"horizontal_start": gtx.int32(0), "horizontal_end": gtx.int32(grid.num_edges)}
        levels = {"vertical_start": gtx.int32(0), "vertical_end": gtx.int32(grid.num_levels)}
        half_levels = {
            "vertical_start": gtx.int32(0),
            "vertical_end": gtx.int32(grid.num_levels + 1),
        }

        self._ddqz_z_full = static_state.ddqz_z_full
        self._diagnose_temperature = model_options.setup_program(
            program=compute_temperature.compute_virtual_temperature_and_temperature,
            backend=backend,
            horizontal_sizes=cells,
            vertical_sizes=levels,
            offset_provider={},
        )
        self._compute_surface_and_hydrostatic_pressure = model_options.setup_program(
            program=compute_pressure.compute_surface_and_hydrostatic_pressure,
            backend=backend,
            horizontal_sizes=cells,
            vertical_sizes=levels,
            offset_provider={},
        )
        # RBF reconstruction of the cell-centre wind from the edge-normal vn
        self._rbf_interpolation = model_options.setup_program(
            program=edge_2_cell_vector_rbf_interpolation,
            backend=backend,
            constant_args={
                "ptr_coeff_1": static_state.rbf_vec_coeff_c1,
                "ptr_coeff_2": static_state.rbf_vec_coeff_c2,
            },
            horizontal_sizes=cells,
            vertical_sizes=levels,
            offset_provider=grid.connectivities,
        )
        self._apply_tendency = model_options.setup_program(
            program=generic_math_operations.compute_field_a_plus_coeff_times_field_b_on_cell_k,
            backend=backend,
            horizontal_sizes=cells,
            vertical_sizes=levels,
            offset_provider={},
        )
        self._apply_tendency_w = model_options.setup_program(
            program=generic_math_operations.compute_field_a_plus_coeff_times_field_b_on_cell_khalf,
            backend=backend,
            horizontal_sizes=cells,
            vertical_sizes=half_levels,
            offset_provider={},
        )
        self._apply_tendency_vn = model_options.setup_program(
            program=generic_math_operations.compute_field_a_plus_coeff_times_field_b_on_edge_k,
            backend=backend,
            horizontal_sizes=edges,
            vertical_sizes=levels,
            offset_provider={},
        )
        self._compute_virtual_temperature_tendency = model_options.setup_program(
            program=compute_tendencies.compute_virtual_temperature_tendency,
            backend=backend,
            horizontal_sizes=cells,
            vertical_sizes=levels,
            offset_provider={},
        )
        self._update_exner_and_theta_v = model_options.setup_program(
            program=compute_temperature.update_exner_and_theta_v,
            backend=backend,
            horizontal_sizes=cells,
            vertical_sizes=levels,
            offset_provider={},
        )
        self._compute_vn_from_uv = model_options.setup_program(
            program=compute_vn_from_uv,
            backend=backend,
            constant_args={
                "primal_normal_cell_x": static_state.primal_normal_cell_x,
                "primal_normal_cell_y": static_state.primal_normal_cell_y,
                "c_lin_e": static_state.c_lin_e,
            },
            horizontal_sizes=edges,
            vertical_sizes=levels,
            offset_provider=grid.connectivities,
        )

        # A scan's range is deduced from its single output domain, so the half-level pressure
        # lands on model levels first and compute_surface_and_hydrostatic_pressure copies it up.
        self._pressure_ifc_on_model_levels = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, allocator=self.allocator
        )
        self._new_temperature = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, allocator=self.allocator
        )
        self._virtual_temperature_tendency = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, allocator=self.allocator
        )
        self._vn_tendency = data_alloc.zero_field(
            grid, dims.EdgeDim, dims.KDim, allocator=self.allocator
        )

    def run(self, inputs: Input, out: Output | None = None) -> Output:
        out = self.buffers(out)
        entry = self._diagnose(inputs)
        fw.zero(self.tendency_sums)
        # the time control reads the start of the step being integrated
        step_start = inputs.simulation_current_datetime - inputs.dtime
        for process in self.processes:
            time_control = process.time_control
            time_control.validate_interval(inputs.dtime)
            # outside its window a process contributes nothing
            if not time_control.is_in_window(step_start):
                continue
            # a process computes when it fires, and on its first step in the window when there
            # is no output to reuse yet; otherwise its last output is accumulated again
            if time_control.is_active(step_start) or process.name not in self.outputs:
                self.outputs[process.name] = process.step(entry)
            fw.sum_tendencies(self.tendency_sums, self.outputs[process.name], self.allocator)
        self._apply(entry, inputs.dtime.total_seconds(), out)
        return out

    def _diagnose(self, inputs: Input) -> physics_state.EntryState:
        """The entry state: the inputs as given and the diagnostics derived from them (dyn2phy)."""
        # The tracers are optional leaves (a tracer may be inactive in the TracerConfig): fail
        # here rather than feed None to the physics.
        if missing := [name for name in states.TRACERS if getattr(inputs, name) is None]:
            raise ValueError(
                f"physics requires all moisture species active in the TracerState; missing: {missing}"
            )
        diagnostics = self.diagnostics
        entry = physics_state.EntryState(
            vn=inputs.vn,
            w=inputs.w,
            exner=inputs.exner,
            theta_v=inputs.theta_v,
            rho=inputs.rho,
            qv=_present(inputs.qv),
            qc=_present(inputs.qc),
            qi=_present(inputs.qi),
            qr=_present(inputs.qr),
            qs=_present(inputs.qs),
            qg=_present(inputs.qg),
            temperature=diagnostics.temperature,
            virtual_temperature=diagnostics.virtual_temperature,
            pressure=diagnostics.pressure,
            pressure_ifc=diagnostics.pressure_ifc,
            u=diagnostics.u,
            v=diagnostics.v,
        )
        # 1. virtual temperature and temperature
        self._diagnose_temperature(
            qv=entry.qv.data,
            qc=entry.qc.data,
            qi=entry.qi.data,
            qr=entry.qr.data,
            qs=entry.qs.data,
            qg=entry.qg.data,
            theta_v=entry.theta_v.data,
            exner=entry.exner.data,
            virtual_temperature=entry.virtual_temperature.data,
            temperature=entry.temperature.data,
        )
        # 2. surface pressure at the bottom interface, then the full pressure column
        self._compute_surface_and_hydrostatic_pressure(
            exner=entry.exner.data,
            virtual_temperature=entry.virtual_temperature.data,
            ddqz_z_full=self._ddqz_z_full,
            pressure=entry.pressure.data,
            pressure_ifc_on_model_levels=self._pressure_ifc_on_model_levels,
            pressure_ifc=entry.pressure_ifc.data,
        )
        # 3. cell-centre (u, v) from the edge-normal vn
        self._rbf_interpolation(p_e_in=entry.vn.data, p_u_out=entry.u.data, p_v_out=entry.v.data)
        return entry

    def _apply(self, entry: physics_state.EntryState, dt_seconds: float, out: Output) -> None:
        """
        The summed tendencies into `out`, once (ICON's phy2dyn).

        The order matters: the tracers first, because the exner/theta_v update uses the final
        moisture, then the temperature, then the winds. A tendency no process produced is not
        applied.
        """
        fw.copy_into(out, entry)
        sums = self.tendency_sums

        # 1. tracers: q += dt * sum of the tendencies (mo_interface_iconam_aes:513)
        for name in states.TRACERS:
            if (tendency := sums.get(f"tend_{name}")) is not None:
                tracer = _present(getattr(out, name)).data
                self._apply_tendency(
                    field_a=tracer, coeff=dt_seconds, field_b=tendency.data, output_field=tracer
                )

        # 2. temperature -> exner/theta_v: one exact-EOS update from the entry temperature plus
        #    the summed tendency, with the final (post step 1) moisture
        if (tend_temperature := sums.tend_temperature) is not None:
            self._apply_tendency(
                field_a=entry.temperature.data,
                coeff=dt_seconds,
                field_b=tend_temperature.data,
                output_field=self._new_temperature,
            )
            self._compute_virtual_temperature_tendency(
                dtime=dt_seconds,
                qv=_present(out.qv).data,
                qc=_present(out.qc).data,
                qi=_present(out.qi).data,
                qr=_present(out.qr).data,
                qs=_present(out.qs).data,
                qg=_present(out.qg).data,
                temperature=self._new_temperature,
                virtual_temperature=entry.virtual_temperature.data,
                virtual_temperature_tendency=self._virtual_temperature_tendency,
            )
            self._update_exner_and_theta_v(
                rho=entry.rho.data,
                virtual_temperature=entry.virtual_temperature.data,
                virtual_temperature_tendency=self._virtual_temperature_tendency,
                dtime=dt_seconds,
                exner=out.exner.data,
                theta_v=out.theta_v.data,
            )

        # 3. winds: one projection of the summed (u, v) tendencies onto the edge normals
        tend_u, tend_v = sums.tend_u, sums.tend_v
        if (tend_u is None) != (tend_v is None):
            raise ValueError(
                "the horizontal wind tendencies are applied as a pair; got only"
                f" {'tend_u' if tend_u is not None else 'tend_v'}"
            )
        if tend_u is not None and tend_v is not None:
            self._compute_vn_from_uv(u=tend_u.data, v=tend_v.data, vn=self._vn_tendency)
            self._apply_tendency_vn(
                field_a=out.vn.data,
                coeff=dt_seconds,
                field_b=self._vn_tendency,
                output_field=out.vn.data,
            )

        # 4. w, on half levels
        if (tend_w := sums.tend_w) is not None:
            self._apply_tendency_w(
                field_a=out.w.data,
                coeff=dt_seconds,
                field_b=tend_w.data,
                output_field=out.w.data,
            )
