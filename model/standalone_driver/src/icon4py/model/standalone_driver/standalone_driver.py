# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import datetime
import functools
import logging
import pathlib
import types
from collections.abc import Callable

import gt4py.next as gtx

from icon4py.model.atmosphere.dycore.stencils import compute_airmass
from icon4py.model.atmosphere.subgrid_scale_physics.muphys import state as muphys_state
from icon4py.model.common import (
    dimension as dims,
    initial_condition,
    model_backends,
    model_options,
    prescribed_tendencies,
    topography,
)
from icon4py.model.common.components.derived_quantities import DerivedQuantities
from icon4py.model.common.composition import show, to_graphviz
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.grid import grid_manager as gm, vertical as v_grid
from icon4py.model.common.grid.icon import IconGrid
from icon4py.model.common.io import io as common_io
from icon4py.model.common.metrics import metrics_attributes as metrics_attr
from icon4py.model.common.states import (
    diagnostic_state as diagnostics,
    nonhydro_states,
    prognostic_state as prognostics,
    static_fields,
    tracer_states,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.standalone_driver import (
    config as driver_config,
    driver_io,
    driver_states,
    driver_utils,
    edsl_driver,
    plain_driver,
)
from icon4py.model.standalone_driver.driver_loop_state import DriverLoopState, DriverServices


log = logging.getLogger(__name__)


class Icon4pyDriver:
    def __init__(
        self,
        *,
        config: driver_config.ExperimentConfig,
        backend: gtx.typing.Backend | None,
        grid: IconGrid,
        decomposition_info: decomposition_defs.DecompositionInfo,
        static_field_factories: static_fields.StaticFieldFactories,
        granules: driver_utils.Granules,
        model_time_variables: driver_states.ModelTimeVariables,
        vertical_grid_config: v_grid.VerticalGridConfig,
        vertical_grid: v_grid.VerticalGrid,
        exchange: decomposition_defs.ExchangeRuntime,
        global_reductions: decomposition_defs.Reductions,
        io_monitor: common_io.IOMonitor | None = None,
        tendencies: prescribed_tendencies.PrescribedTendencies | None = None,
    ):
        self.config = config
        self.io_monitor = io_monitor
        self.backend = backend
        self.grid = grid
        self.decomposition_info = decomposition_info
        self.static_field_factories = static_field_factories
        self.granules = granules
        self.vertical_grid_config = vertical_grid_config
        self.vertical_grid = vertical_grid
        self.model_time_variables = model_time_variables
        self.timer_collection = driver_states.TimerCollection(
            [timer.value for timer in driver_states.DriverTimers]
        )
        self.exchange = exchange
        self.global_reductions = global_reductions
        self.tendencies = tendencies

        driver_utils.log_driver_setup(
            config=self.config.driver,
            model_time_variables=self.model_time_variables,
            vertical_params=self.vertical_grid,
        )

    @functools.cached_property
    def _allocator(self) -> gtx.typing.Backend:
        return model_backends.get_allocator(self.backend)

    @functools.cached_property
    def _xp(self) -> types.ModuleType:
        return data_alloc.import_array_ns(self._allocator)

    @functools.cached_property
    def _derived_quantities(self) -> DerivedQuantities:
        """Canonical T/p/u/v derivation (allocated once, run every time step).

        Dry experiments pass zero-filled hydrometeor buffers, so the step is always
        part of the composition and the diagnostic state is always available for IO
        and physics.
        """
        return DerivedQuantities(grid=self.grid, backend=self.backend)

    @functools.cached_property
    def _compute_airmass(self) -> Callable[..., None]:
        """Airmass program (``rho * ddqz_z_full * deepatmo_t1mc``) with its static inputs bound.

        ``deepatmo_t1mc`` (ICON's ``deepatmo_vol_mc``) is 1 in the shallow atmosphere, which
        is the only mode the dycore supports; see the matching factors that tracer advection
        gets in ``driver_utils.initialize_granules``.
        """
        assert self.granules.registry is not None
        return model_options.setup_program(
            program=compute_airmass.compute_airmass,
            backend=self.backend,
            constant_args={
                "ddqz_z_full_in": self.granules.registry.buffer(metrics_attr.DDQZ_Z_FULL),
                "deepatmo_t1mc_in": data_alloc.constant_field(
                    self.grid, 1.0, dims.KDim, allocator=self._allocator
                ),
            },
            horizontal_sizes={
                "horizontal_start": gtx.int32(0),
                "horizontal_end": gtx.int32(self.grid.num_cells),
            },
            vertical_sizes={
                "vertical_start": gtx.int32(0),
                "vertical_end": gtx.int32(self.grid.num_levels),
            },
            offset_provider={},
        )

    def _build_carry(self, ds: driver_states.DriverStates) -> DriverLoopState:
        return DriverLoopState(
            clock=self.model_time_variables,
            states=ds,
            granules=self.granules,
            config=self.config,
            services=DriverServices(
                exchange=self.exchange,
                global_reductions=self.global_reductions,
                io_monitor=self.io_monitor,
                tendencies=self.tendencies,
                timer_collection=self.timer_collection,
                backend=self.backend,
                xp=self._xp,
                allocator=self._allocator,
                derived_quantities=self._derived_quantities,
                compute_airmass=self._compute_airmass,
            ),
            wall_clock_starting_time=datetime.datetime.now(),
        )

    def time_integration(
        self,
        ds: driver_states.DriverStates,
    ) -> None:
        log.debug(
            f"starting time loop for dtime = {self.model_time_variables.dtime_in_seconds} s, substep_timestep = {self.model_time_variables.substep_timestep} s, n_timesteps = {self.model_time_variables.n_time_steps}"
        )

        carry = self._build_carry(ds)

        try:  # fail gracefully and close `io_monitor` if something goes wrong
            edsl_driver.build_time_integration_composition(
                granules=self.granules,
                derived_quantities=self._derived_quantities,
            )(carry)
        finally:
            if self.io_monitor is not None:
                self.io_monitor.close()

    def show(self) -> str:
        """Return a text tree of the time-integration composition."""
        composition = edsl_driver.build_time_integration_composition(
            granules=self.granules,
            derived_quantities=self._derived_quantities,
        )
        return show(composition)

    def to_graphviz(self) -> str:
        """Return a graphviz dot string of the composition tree and dataflow graph."""
        composition = edsl_driver.build_time_integration_composition(
            granules=self.granules,
            derived_quantities=self._derived_quantities,
        )
        return to_graphviz(composition)


def initialize_driver(
    *,
    config: driver_config.ExperimentConfig,
    grid_manager: gm.GridManager,
    process_props: decomposition_defs.ProcessProperties,
    backend: gtx.typing.Backend | None,
) -> Icon4pyDriver:
    output_path = driver_config.prepare_output_directory(
        config_output_path=config.driver.output_path,
        cli_output_path=None,
        process_props=process_props,
    )
    config = dataclasses.replace(
        config, driver=dataclasses.replace(config.driver, output_path=output_path)
    )

    allocator = model_backends.get_allocator(backend)

    decomposition_info = grid_manager.decomposition_info
    exchange = decomposition_defs.create_exchange(process_props, decomposition_info)
    global_reductions = decomposition_defs.create_reduction(process_props, decomposition_info)

    log.info("initializing the vertical grid")
    vertical_grid = driver_utils.create_vertical_grid(
        vertical_grid_config=config.vertical_grid,
        allocator=allocator,
    )

    log.info("initializing the topography")
    cell_topography = topography.create(
        config=config.topography,
        grid_manager=grid_manager,
        backend=backend,
        exchange=exchange,
    )

    log.info("initializing the static-field factories")
    static_field_factories = driver_utils.create_static_field_factories(
        grid_manager=grid_manager,
        decomposition_info=decomposition_info,
        vertical_grid=vertical_grid,
        cell_topography=gtx.as_field((dims.CellDim,), data=cell_topography, allocator=allocator),  # type: ignore[arg-type] # due to array_ns opacity
        backend=backend,
        process_props=process_props,
        exchange=exchange,
        global_reductions=global_reductions,
        geometry_config=config.geometry,
        interpolation_config=config.interpolation,
        metrics_config=config.metrics,
    )

    model_time_variables = driver_states.ModelTimeVariables(config=config.driver)

    log.info("initializing granules")
    granules = driver_utils.initialize_granules(
        config=config,
        grid=grid_manager.grid,
        vertical_grid=vertical_grid,
        static_field_factories=static_field_factories,
        model_time_variables=model_time_variables,
        exchange=exchange,
        owner_mask=gtx.as_field(
            (dims.CellDim,),
            decomposition_info.owner_mask(dims.CellDim),  # type: ignore[arg-type]  # due to array_ns opacity
            allocator=allocator,
        ),
        backend=backend,
    )
    io_monitor = None
    if config.driver.enable_output:
        if process_props.comm_size > 1:
            # IO is single-node only for now: under MPI every rank would construct its own
            # monitor and write overlapping files. Disable until IO becomes distributed.
            log.warning("output is not supported in distributed (MPI) runs yet: disabling IO")
        else:
            log.info("Initializing single-node IO monitor")
            io_monitor = driver_io.create_io_monitor(
                output_path=config.driver.output_path,
                grid_file_path=pathlib.Path(grid_manager.file_path),
                grid=grid_manager.grid,
                vertical_grid=vertical_grid,
                dtime=config.driver.dtime,
                process_props=process_props,
            )

    icon4py_driver = Icon4pyDriver(
        config=config,
        backend=backend,
        grid=grid_manager.grid,
        decomposition_info=decomposition_info,
        static_field_factories=static_field_factories,
        granules=granules,
        model_time_variables=model_time_variables,
        vertical_grid_config=config.vertical_grid,
        vertical_grid=vertical_grid,
        exchange=exchange,
        global_reductions=global_reductions,
        tendencies=(
            prescribed_tendencies.PrescribedTendencies(
                config=config.prescribed_tendencies,
                grid=grid_manager.grid,
                backend=backend,
                rank=exchange.my_rank(),
            )
            if config.prescribed_tendencies.data_path is not None
            else None
        ),
        io_monitor=io_monitor,
    )

    return icon4py_driver


def _assemble_run(
    *,
    config: driver_config.ExperimentConfig,
    grid_manager: gm.GridManager,
    process_props: decomposition_defs.ProcessProperties,
    backend: gtx.typing.Backend | None,
) -> tuple[driver_states.DriverStates, Icon4pyDriver]:
    icon4py_driver = initialize_driver(
        config=config,
        grid_manager=grid_manager,
        process_props=process_props,
        backend=backend,
    )
    allocator = model_backends.get_allocator(backend)
    prognostic_state_now = prognostics.initialize_prognostic_state(
        grid=icon4py_driver.grid,
        allocator=allocator,
    )
    tracer_state_now = tracer_states.initialize_tracer_state(
        grid=icon4py_driver.grid,
        allocator=allocator,
        tracer_config=icon4py_driver.config.tracer_config,
    )
    solve_nonhydro_diagnostic_state = (
        nonhydro_states.initialize_solve_nonhydro_diagnostic_state(
            grid=icon4py_driver.grid, allocator=allocator
        )
        if icon4py_driver.config.nonhydrostatic is not None
        else None
    )
    initial_condition.create(
        config=icon4py_driver.config.initial_condition,
        vertical_config=icon4py_driver.config.vertical_grid,
        grid=icon4py_driver.grid,
        static_fields=icon4py_driver.static_field_factories,
        prognostic_state_now=prognostic_state_now,
        tracer_state_now=tracer_state_now,
        solve_nonhydro_diagnostic_state=solve_nonhydro_diagnostic_state,
        backend=icon4py_driver.backend,
        exchange=icon4py_driver.exchange,
        global_reductions=icon4py_driver.global_reductions,
    )
    diagnostic_state = diagnostics.initialize_diagnostic_state(
        grid=icon4py_driver.grid, allocator=allocator
    )
    if icon4py_driver.granules.physics is not None:
        for process in icon4py_driver.granules.physics._processes:
            if isinstance(process.state, muphys_state.State):
                process.state.diagnostic = diagnostic_state
    assert icon4py_driver.granules.registry is not None
    ds = driver_states.assemble_driver_states(
        grid=icon4py_driver.grid,
        allocator=allocator,
        backend=icon4py_driver.backend,
        exchange=icon4py_driver.exchange,
        registry=icon4py_driver.granules.registry,
        prognostic_state_now=prognostic_state_now,
        tracer_state_now=tracer_state_now,
        diagnostic_state=diagnostic_state,
        experiment_config=icon4py_driver.config,
        solve_nonhydro_diagnostic_state=solve_nonhydro_diagnostic_state,
    )
    driver_utils.validate_granule_state_consistency(
        config=icon4py_driver.config,
        granules=icon4py_driver.granules,
        states=ds,
    )
    return ds, icon4py_driver


def run_driver(
    *,
    config: driver_config.ExperimentConfig,
    grid_manager: gm.GridManager,
    process_props: decomposition_defs.ProcessProperties,
    backend: gtx.typing.Backend | None,
) -> tuple[driver_states.DriverStates, Icon4pyDriver]:
    ds, icon4py_driver = _assemble_run(
        config=config,
        grid_manager=grid_manager,
        process_props=process_props,
        backend=backend,
    )
    icon4py_driver.time_integration(ds)
    return ds, icon4py_driver


def run_driver_plain(
    *,
    config: driver_config.ExperimentConfig,
    grid_manager: gm.GridManager,
    process_props: decomposition_defs.ProcessProperties,
    backend: gtx.typing.Backend | None,
) -> tuple[driver_states.DriverStates, Icon4pyDriver]:
    """Test-only plain-Python driver entry point (no eDSL combinators)."""
    ds, icon4py_driver = _assemble_run(
        config=config,
        grid_manager=grid_manager,
        process_props=process_props,
        backend=backend,
    )
    carry = icon4py_driver._build_carry(ds)
    plain_driver.run_time_integration_plain(carry)
    return ds, icon4py_driver
