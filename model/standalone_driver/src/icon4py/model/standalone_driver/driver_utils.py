# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import logging
import os
import pathlib
import sys
from collections.abc import Callable
from typing import Any, Literal

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing

from icon4py.model.atmosphere.diffusion import diffusion, diffusion_states
from icon4py.model.atmosphere.dycore import dycore_states, solve_nonhydro as solve_nh
from icon4py.model.atmosphere.subgrid_scale_physics.muphys import (
    component as muphys_component,
    state as muphys_state,
)
from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver import physics_driver
from icon4py.model.atmosphere.tracer_advection import tracer_advection, tracer_advection_states
from icon4py.model.common import (
    constants,
    dimension as dims,
    field_type_aliases as fa,
    model_backends,
    time,
    type_alias as ta,
)
from icon4py.model.common.decomposition import (
    decomposer as decomp,
    definitions as decomposition_defs,
)
from icon4py.model.common.grid import (
    geometry as grid_geometry,
    geometry_attributes as geometry_meta,
    geometry_config as geometry_configuration,
    grid_manager as gm,
    gridfile,
    icon as icon_grid,
    states as grid_states,
    vertical as v_grid,
)
from icon4py.model.common.interpolation import interpolation_attributes, interpolation_factory
from icon4py.model.common.metrics import metrics_attributes, metrics_factory
from icon4py.model.common.states import (
    factory,
    field_registry,
    model as states_model,
    quantities,
    static_fields,
    tracer_states,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.standalone_driver import config as driver_config, driver_constants, driver_states


log = logging.getLogger(__name__)

DRIVER_LOGGING_LEVEL: str = os.environ.get("ICON4PY_DRIVER_LOGGING_LEVEL", "debug")

_LOGGING_LEVELS: dict[str, int] = {
    "notset": logging.NOTSET,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


@dataclasses.dataclass
class Granules:
    diffusion: diffusion.Diffusion | None = None
    solve_nonhydro: solve_nh.SolveNonhydro | None = None
    tracer_advection: tracer_advection.Advection | None = None
    physics: physics_driver.PhysicsDriver | None = None
    registry: field_registry.FieldRegistry | None = None

    def __post_init__(self) -> None:
        if self.registry is None:
            raise ValueError("Granules.registry must be set.")


def validate_granule_state_consistency(
    config: driver_config.ExperimentConfig,
    granules: Granules,
    states: driver_states.DriverStates,
) -> None:
    """
    Validate that enabled granules have their required states allocated.
    The graupel granule is currently not checked as it will be moved to the
    physics driver.

    Raises:
        ValueError: if a granule is enabled but a state it requires is None.
    """

    if config.diffusion is not None and granules.diffusion is None:
        raise ValueError("diffusion is enabled but granules.diffusion is None.")
    if config.nonhydrostatic is not None and granules.solve_nonhydro is None:
        raise ValueError("solve_nonhydro is enabled but granules.solve_nonhydro is None.")
    if config.tracer_advection is not None and granules.tracer_advection is None:
        raise ValueError("tracer_advection is enabled but granules.tracer_advection is None.")

    if granules.diffusion is not None and states.diffusion_diagnostic is None:
        raise ValueError("diffusion granule is present but diffusion_diagnostic state is None.")
    if granules.solve_nonhydro is not None:
        if states.solve_nonhydro_diagnostic is None:
            raise ValueError(
                "solve_nonhydro granule is present but solve_nonhydro_diagnostic state is None."
            )
        if states.prep_advection_prognostic is None:
            raise ValueError(
                "solve_nonhydro granule is present but prep_advection_prognostic state is None."
            )
    if granules.tracer_advection is not None:
        if states.tracer_advection_diagnostic is None:
            raise ValueError(
                "tracer_advection granule is present but tracer_advection_diagnostic state is None."
            )
        if states.prep_tracer_advection_prognostic is None:
            raise ValueError(
                "tracer_advection granule is present but prep_tracer_advection_prognostic state is None."
            )


def create_grid_manager(
    grid_file_path: pathlib.Path,
    vertical_grid_config: v_grid.VerticalGridConfig,
    allocator: gtx_typing.Allocator,
    process_props: decomposition_defs.ProcessProperties,
) -> gm.GridManager:
    decomposer = (
        decomp.SingleNodeDecomposer()
        if process_props.is_single_rank()
        else decomp.MetisDecomposer()
    )
    grid_manager = gm.GridManager(
        grid_file=grid_file_path,
        config=vertical_grid_config,
        offset_transformation=gridfile.ToZeroBasedIndexTransformation(),
    )
    grid_manager(
        allocator=allocator,
        keep_skip_values=True,
        process_props=process_props,
        decomposer=decomposer,
    )

    return grid_manager


def create_vertical_grid(
    vertical_grid_config: v_grid.VerticalGridConfig,
    allocator: gtx_typing.Allocator,
) -> v_grid.VerticalGrid:
    vct_a, vct_b = v_grid.get_vct_a_and_vct_b(
        vertical_config=vertical_grid_config, allocator=allocator
    )

    vertical_grid = v_grid.VerticalGrid(
        config=vertical_grid_config,
        vct_a=vct_a,
        vct_b=vct_b,
    )
    return vertical_grid


def create_static_field_factories(
    *,
    grid_manager: gm.GridManager,
    decomposition_info: decomposition_defs.DecompositionInfo,
    vertical_grid: v_grid.VerticalGrid,
    cell_topography: fa.CellField[ta.wpfloat],
    backend: gtx_typing.Backend | None,
    process_props: decomposition_defs.ProcessProperties,
    exchange: decomposition_defs.ExchangeRuntime,
    global_reductions: decomposition_defs.Reductions,
    geometry_config: geometry_configuration.GeometryConfig,
    interpolation_config: interpolation_factory.InterpolationConfig,
    metrics_config: metrics_factory.MetricsConfig,
) -> static_fields.StaticFieldFactories:
    geometry_field_source = grid_geometry.GridGeometry(
        grid=grid_manager.grid,
        decomposition_info=decomposition_info,
        backend=backend,
        coordinates=grid_manager.coordinates,
        extra_fields=grid_manager.geometry_fields,
        metadata=geometry_meta.attrs,
        config=geometry_config,
        process_props=process_props,
        exchange=exchange,
        global_reductions=global_reductions,
    )

    interpolation_field_source = interpolation_factory.InterpolationFieldsFactory(
        config=interpolation_config,
        grid=grid_manager.grid,
        decomposition_info=decomposition_info,
        geometry_source=geometry_field_source,
        backend=backend,
        metadata=interpolation_attributes.attrs,
        exchange=exchange,
    )

    metrics_field_source = metrics_factory.MetricsFieldsFactory(
        grid=grid_manager.grid,
        vertical_grid=vertical_grid,
        decomposition_info=decomposition_info,
        geometry_source=geometry_field_source,
        topography=cell_topography,
        interpolation_source=interpolation_field_source,
        backend=backend,
        metadata=metrics_attributes.attrs,
        config=metrics_config,
        exchange=exchange,
        global_reductions=global_reductions,
    )

    return static_fields.StaticFieldFactories(
        geometry_field_source, interpolation_field_source, metrics_field_source
    )


def _register_static_field_recipes(
    registry: field_registry.FieldRegistry,
    static_field_factories: static_fields.StaticFieldFactories,
) -> None:
    """Register one recipe per static-field quantity, delegating to the factories.

    Recipes are keyed by the canonical quantity name used in ``spec()``
    declarations. The factory metadata keys are usually the canonical name, but
    for some ICON-specific quantities the canonical name carries an ``icon:``
    prefix while the factory key does not; in that case the recipe is registered
    under the canonical name.
    """

    def _recipe(source: factory.FieldSource, name: str) -> Callable[[], Any]:
        return lambda: source.get(name)

    def _canonical_quantity_name(metadata: states_model.FieldMetaData) -> str | None:
        """Return the canonical quantity name for a factory metadata entry."""
        standard_name = metadata.get("standard_name", "")
        icon_fortran_name = metadata.get("icon_var_name", "")

        # Prefer matching by ICON Fortran name when present; this resolves
        # ICON-specific quantities whose canonical name carries an ``icon:``
        # prefix (e.g. ``icon:zd_vertoffset``).
        if icon_fortran_name:
            candidates = [
                quantity
                for quantity in quantities.all_quantities().values()
                if quantity.icon_fortran_name == icon_fortran_name
            ]
            if candidates:
                icon_candidates = [q for q in candidates if q.name.startswith("icon:")]
                if icon_candidates:
                    return icon_candidates[0].name
                matching = [q for q in candidates if q.name == standard_name]
                if matching:
                    return matching[0].name
                return candidates[0].name

        # Otherwise fall back to the standard name.
        if standard_name and standard_name in quantities.all_quantities():
            return standard_name

        return standard_name if standard_name else None

    geometry_field_source = static_field_factories.geometry
    interpolation_field_source = static_field_factories.interpolation
    metrics_field_source = static_field_factories.metrics

    for name, meta in geometry_field_source.metadata.items():
        canonical = _canonical_quantity_name(meta) or name
        registry.recipe(canonical, _recipe(geometry_field_source, name))
    for name, meta in interpolation_field_source.metadata.items():
        canonical = _canonical_quantity_name(meta) or name
        registry.recipe(canonical, _recipe(interpolation_field_source, name))
    for name, meta in metrics_field_source.metadata.items():
        canonical = _canonical_quantity_name(meta) or name
        registry.recipe(canonical, _recipe(metrics_field_source, name))


def _declare_static_and_handoff_containers(registry: field_registry.FieldRegistry) -> None:
    """Declare the static-field containers and the dycore/advection boundaries."""
    registry.declare(grid_states.CellParams)
    registry.declare(grid_states.EdgeParams)
    registry.declare(diffusion_states.DiffusionInterpolationState)
    registry.declare(diffusion_states.DiffusionMetricState)
    registry.declare(dycore_states.InterpolationState)
    registry.declare(dycore_states.MetricStateNonHydro)
    registry.declare(tracer_advection_states.AdvectionInterpolationState)
    registry.declare(dycore_states.PrepAdvection)
    registry.declare(tracer_advection_states.AdvectionPrepAdvState)
    registry.declare(solve_nh.SolveNonHydroInput)
    registry.declare(solve_nh.SolveNonHydroOutput)
    registry.declare(tracer_advection.AdvectionInput)
    registry.declare(tracer_advection.AdvectionOutput)


def initialize_granules(
    *,
    config: driver_config.ExperimentConfig,
    grid: icon_grid.IconGrid,
    vertical_grid: v_grid.VerticalGrid,
    static_field_factories: static_fields.StaticFieldFactories,
    model_time_variables: driver_states.ModelTimeVariables,
    exchange: decomposition_defs.ExchangeRuntime,
    owner_mask: fa.CellField[bool],
    backend: gtx_typing.Backend | None,
) -> Granules:
    log.info("creating field registry")
    registry = field_registry.FieldRegistry(grid=grid, backend=backend)
    _register_static_field_recipes(registry, static_field_factories)
    _declare_static_and_handoff_containers(registry)
    registry.seal()

    log.info("creating cell geometry")
    cell_geometry = registry.build(grid_states.CellParams)

    log.info("creating edge geometry")
    edge_geometry = registry.build(grid_states.EdgeParams)

    log.info("creating diffusion interpolation state")
    diffusion_interpolation_state = registry.build(diffusion_states.DiffusionInterpolationState)

    log.info("creating diffusion metric state")
    diffusion_metric_state = registry.build(diffusion_states.DiffusionMetricState)

    log.info("creating solve nonhydro interpolation state")
    solve_nonhydro_interpolation_state = registry.build(dycore_states.InterpolationState)

    log.info("creating solve nonhydro metric state")
    solve_nonhydro_metric_state = registry.build(dycore_states.MetricStateNonHydro)

    solve_nonhydro_granule: solve_nh.SolveNonhydro | None = None
    if config.nonhydrostatic is not None:
        nonhydro_params = solve_nh.NonHydrostaticParams(config.nonhydrostatic)
        solve_nonhydro_granule = solve_nh.SolveNonhydro(
            grid=grid,
            backend=backend,
            config=config.nonhydrostatic,
            params=nonhydro_params,
            metric_state_nonhydro=solve_nonhydro_metric_state,
            interpolation_state=solve_nonhydro_interpolation_state,
            vertical_params=vertical_grid,
            edge_geometry=edge_geometry,
            cell_geometry=cell_geometry,
            owner_mask=owner_mask,
            exchange=exchange,
        )

    diffusion_granule: diffusion.Diffusion | None = None
    if config.diffusion is not None:
        diffusion_params = diffusion.DiffusionParams(config.diffusion)
        diffusion_granule = diffusion.Diffusion(
            grid=grid,
            config=config.diffusion,
            params=diffusion_params,
            vertical_grid=vertical_grid,
            metric_state=diffusion_metric_state,
            interpolation_state=diffusion_interpolation_state,
            edge_params=edge_geometry,
            cell_params=cell_geometry,
            backend=backend,
            exchange=exchange,
        )

    tracer_advection_interpolation_state: (
        tracer_advection_states.AdvectionInterpolationState | None
    ) = None
    tracer_advection_least_squares_state: (
        tracer_advection_states.AdvectionLeastSquaresState | None
    ) = None
    tracer_advection_metric_state: tracer_advection_states.AdvectionMetricState | None = None
    tracer_advection_granule: tracer_advection.Advection | None = None
    if config.tracer_advection is not None:
        deepatmo_shallow_factor = data_alloc.constant_field(
            grid, 1.0, dims.KDim, allocator=model_backends.get_allocator(backend)
        )
        tracer_advection_interpolation_state = registry.build(
            tracer_advection_states.AdvectionInterpolationState
        )
        tracer_advection_least_squares_state = tracer_advection_states.AdvectionLeastSquaresState(
            lsq_pseudoinv_1=static_field_factories.interpolation.get(
                interpolation_attributes.LSQ_PSEUDOINV
            )[:, 0, :],
            lsq_pseudoinv_2=static_field_factories.interpolation.get(
                interpolation_attributes.LSQ_PSEUDOINV
            )[:, 1, :],
        )
        tracer_advection_metric_state = tracer_advection_states.AdvectionMetricState(
            # Shallow atmosphere: the deep-atmosphere modification factors are 1, as
            # in ICON with 'ldeepatmo = .FALSE.' (mo_nonhydro_state.f90 initialises
            # them to 1 and only mo_vertical_grid.f90 overwrites them, guarded by
            # 'ldeepatmo'). Using the factory's deep-atmosphere values here would be
            # inconsistent with the dycore, which has no deep-atmosphere mode, and
            # with the airmass (rho * ddqz_z_full * deepatmo_vol) that tracer
            # advection divides by.
            deepatmo_divh=deepatmo_shallow_factor,
            deepatmo_divzl=deepatmo_shallow_factor,
            deepatmo_divzu=deepatmo_shallow_factor,
            ddqz_z_full=registry.buffer(metrics_attributes.DDQZ_Z_FULL),
        )
        tracer_advection_granule = tracer_advection.convert_config_to_advection(
            grid=grid,
            backend=backend,
            config=config.tracer_advection,
            interpolation_state=tracer_advection_interpolation_state,
            least_squares_state=tracer_advection_least_squares_state,
            metric_state=tracer_advection_metric_state,
            edge_params=edge_geometry,
            cell_params=cell_geometry,
            exchange=exchange,
        )

    physics_granule: physics_driver.PhysicsDriver | None = None
    if config.muphys is not None:
        muphys_process = physics_driver.PhysicsProcess(
            name="muphys",
            component=muphys_component.MuphysComponent(
                grid=grid,
                dtime=config.driver.dtime,
                qnc=config.muphys.qnc,
                backend=backend,
                scheme=config.muphys.scheme,
            ),
            state=muphys_state.State(
                grid=grid, metrics=static_field_factories.metrics, backend=backend
            ),
            time_control=physics_driver.ProcessTimeControl(
                interval=config.driver.dtime,
                start_date=config.driver.start_of_simulation,
                end_date=model_time_variables.simulation_end_datetime,
                enable_process=True,
            ),
        )
        physics_granule = physics_driver.PhysicsDriver([muphys_process], config.driver.dtime)

    return Granules(
        solve_nonhydro=solve_nonhydro_granule,
        diffusion=diffusion_granule,
        tracer_advection=tracer_advection_granule,
        physics=physics_granule,
        registry=registry,
    )


def spinup_second_order_divdamp_factor(
    *,
    elapsed_time_in_seconds: ta.wpfloat,
    fourth_order_divdamp_factor: ta.wpfloat,
) -> ta.wpfloat:
    """
    Second order divergence damping factor (divdamp_fac_o2) during the spin-up phase.

    update_spinup_damping in mo_nh_stepping.f90: the damping is enhanced during
    the first half hour of integration and then decreases linearly to zero.
    """
    initial_period = driver_constants.INITIAL_PERIOD_FOR_SECOND_ORDER_DIVDAMP
    transition_end_period = driver_constants.TRANSITION_END_PERIOD_FOR_SECOND_ORDER_DIVDAMP
    enhanced_factor = (
        driver_constants.ADJUST_FACTOR_FOR_SECOND_ORDER_DIVDAMP * fourth_order_divdamp_factor
    )
    if elapsed_time_in_seconds <= initial_period:
        return enhanced_factor
    if elapsed_time_in_seconds <= transition_end_period:
        return (
            enhanced_factor
            * (transition_end_period - elapsed_time_in_seconds)
            / (transition_end_period - initial_period)
        )
    return ta.wpfloat("0.0")


def find_maximum_from_field(
    input_field: gtx.Field,
) -> tuple[tuple[int, ...], float]:
    array_ns = data_alloc.array_namespace(input_field.ndarray)
    max_indices = array_ns.unravel_index(
        array_ns.abs(input_field.ndarray).argmax(),
        input_field.ndarray.shape,
    )
    return max_indices, input_field.ndarray[max_indices]  # type: ignore[return-value] ## this is congruent with observed numpy behavior


def display_icon4py_logo_in_log_file() -> None:
    r"""
    Print out icon4py signature and some important information of the initial setup to the log file.

                                                               ___
          -------                                    //      ||   \
            | |                                     //       ||    |
            | |       __      _ _        _ _       //  ||    ||___/
            | |     //       /   \     |/   \     //_ _||_   ||        \\      //
            | |    ||       |     |    |     |    --------   ||         \\    //
            | |     \\__     \_ _/     |     |         ||    ||          \\  //
          -------                                                           //
                                                                           //
                                                              = = = = = = //
    """
    boundary_line = ["*" * 91]
    icon4py_signature = []
    icon4py_signature += boundary_line
    empty_line = ["*" + 89 * " " + "*"]
    for _ in range(3):
        icon4py_signature += empty_line

    icon4py_signature += [
        "*                                                                ___                      *"
    ]
    icon4py_signature += [
        r"*            -------                                    //      ||   \                    *"
    ]
    icon4py_signature += [
        "*              | |                                     //       ||    |                   *"
    ]
    icon4py_signature += [
        "*              | |       __      _ _        _ _       //  ||    ||___/                    *"
    ]
    icon4py_signature += [
        r"*              | |     //       /   \     |/   \     //_ _||_   ||        \\      //      *"
    ]
    icon4py_signature += [
        r"*              | |    ||       |     |    |     |    --------   ||         \\    //       *"
    ]
    icon4py_signature += [
        r"*              | |     \\__     \_ _/     |     |         ||    ||          \\  //        *"
    ]
    icon4py_signature += [
        "*            -------                                                           //         *"
    ]
    icon4py_signature += [
        "*                                                                             //          *"
    ]
    icon4py_signature += [
        "*                                                                = = = = = = //           *"
    ]

    for _ in range(3):
        icon4py_signature += empty_line
    icon4py_signature += boundary_line
    icon4py_signature_str = "\n".join(icon4py_signature)
    log.info(icon4py_signature_str)


def display_driver_setup_in_log_file(
    config: driver_config.DriverConfig,
    model_time_variables: driver_states.ModelTimeVariables,
    vertical_params: v_grid.VerticalGrid,
    tracer_config: tracer_states.TracerConfig | None = None,
) -> None:
    if tracer_config is None:
        tracer_config = tracer_states.TracerConfig.none()
    log.info("===== ICON4Py Driver Configuration =====")
    log.info(f"Experiment name        : {config.experiment_name}")
    log.info(f"Time step              : {config.dtime.total_seconds()} s")
    log.info(f"Start of simulation    : {model_time_variables.simulation_start_datetime}")
    log.info(f"End of simulation      : {model_time_variables.simulation_end_datetime}")
    log.info(f"Number of timesteps    : {model_time_variables.n_time_steps}")
    match config.end_of_simulation:
        case time.NumTimeSteps():
            log.info("Running mode           : num_timesteps")
        case time.RelativeTime():
            log.info("Running mode           : relative_time")
        case time.AbsoluteTime():
            log.info("Running mode           : absolute_time")
    log.info(f"Initial ndyn_substeps  : {config.ndyn_substeps}")
    log.info(f"Vertical CFL threshold : {config.vertical_cfl_threshold}")
    log.info(f"Second-order divdamp   : {config.apply_extra_second_order_divdamp}")
    log.info(f"Prepare advection      : {config.do_prep_adv}")
    log.info(f"Initial diffusion      : {config.diffuse_before_time_loop}")
    log.info(f"Statistics enabled     : {config.enable_statistics_logging}")
    log.info(f"Active tracers         : {tracer_config}")
    log.info("")

    log.info("==== Vertical Grid Parameters ====")
    log.info(vertical_params)
    log.info("==== Physical Constants ====")
    for name, value in constants.PhysicsConstants.__class__.__dict__.items():
        if name.startswith("_") or callable(value):
            continue
        log.info(f"{name:30s}: {value}")


@dataclasses.dataclass
class _InfoFormatter(logging.Formatter):
    style: Literal["%", "{", "$"]
    default_fmt: str
    info_fmt: str
    defaults: dict[str, Any] | None

    _info_formatter: logging.Formatter = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        super().__init__(fmt=self.default_fmt, style=self.style, defaults=self.defaults)
        self._info_formatter = logging.Formatter(
            fmt=self.info_fmt,
            style=self.style,
        )

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno == logging.INFO:
            return self._info_formatter.format(record)
        return super().format(record)


def make_handler(
    logging_level: int | None,
    log_filter: logging.Filter | None,
    formatter: str | logging.Formatter | None,
    file_name: str | None,
) -> logging.Handler:
    handler = (
        logging.StreamHandler(stream=sys.stdout)
        if file_name is None
        else logging.FileHandler(filename=file_name)
    )
    if log_filter is not None:
        handler.addFilter(log_filter)
    if formatter is not None:
        if isinstance(formatter, str):
            formatter = logging.Formatter(fmt=formatter, style="{")
        handler.setFormatter(formatter)
    if logging_level is not None:
        handler.setLevel(logging_level)
    return handler


def configure_logging(
    logging_level: str,
    print_distributed_debug_msg: bool,
    process_props: decomposition_defs.ProcessProperties | None = None,
) -> None:
    """
    Configure logging.

    Log output with user-defined logging level across the entire icon4py, except
    for the driver whose logging level is fixed at debug, is sent to console
    (stdout) and the error message is sent to stderr.

    Args:
        logging_level: log level
        process_props: ProcessProperties

    """
    if logging_level.lower() not in _LOGGING_LEVELS:
        raise ValueError(
            f"Invalid logging level {logging_level}, please make sure that the logging level matches either {' / '.join([*_LOGGING_LEVELS.keys()])}"
        )

    logging.Formatter.converter = time.localtime  # set to local time instead of utc

    log_filter = decomposition_defs.ParallelLogger(
        process_props, print_distributed_debug_msg=print_distributed_debug_msg
    )
    formatter = _InfoFormatter(
        style="{",
        default_fmt="{rank_info_str} {asctime} - {filename}: {funcName:<20}: {levelname:<7} {message}",
        info_fmt="{message}",
        defaults={"rank_info_str": ""},
    )
    handler = make_handler(
        logging_level=logging.DEBUG,
        log_filter=log_filter,
        formatter=formatter,
        file_name=None,
    )
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[
            handler,
        ],
    )
    driver_module_name = __name__[: __name__.rindex(".")]
    logging.getLogger("icon4py.model").setLevel(_LOGGING_LEVELS[logging_level])
    logging.getLogger(driver_module_name).setLevel(
        _LOGGING_LEVELS.get(DRIVER_LOGGING_LEVEL, logging.DEBUG)
    )
    logging.getLogger("filelock").setLevel(logging.WARNING)
    logging.getLogger("factory.generate").setLevel(logging.WARNING)
    logging.getLogger("blib2to3").setLevel(logging.WARNING)

    display_icon4py_logo_in_log_file()


def get_backend_from_name(
    backend_name: str | model_backends.BackendLike,
) -> model_backends.BackendLike:
    if not isinstance(backend_name, str):
        return backend_name
    if backend_name not in model_backends.BACKENDS:
        raise ValueError(
            f"Invalid driver backend: {backend_name}. \n"
            f"Available backends are {', '.join([*model_backends.BACKENDS.keys()])}"
        )
    backend = model_backends.BACKENDS[backend_name]
    log.info(f"Backend name used for the model: {backend_name}")
    log.info(f"BackendLike derived from the backend name: {backend}")
    return backend
