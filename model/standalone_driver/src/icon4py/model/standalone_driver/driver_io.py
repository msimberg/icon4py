# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Driver-side glue between the model state and the ``icon4py.model.common.io`` module.

- assembles the prognostic model state into the ``dict[str, xarray.DataArray]`` consumed
  by ``IOMonitor.store`` (:func:`prognostic_state_to_dataarrays`),
- assembles the canonical diagnostic state for output (:func:`diagnostic_state_to_fields`
  and :func:`diagnostic_fields_to_dataarrays`),
- builds an ``IOMonitor`` that writes all requested fields to one file
  (:func:`create_io_monitor`).
"""

import dataclasses
import datetime
import pathlib
import uuid
from typing import Any, Final

import gt4py.next as gtx
import xarray as xr

from icon4py.model.common import time
from icon4py.model.common.decomposition import definitions as decomposition_defs
from icon4py.model.common.grid import base as grid_base, vertical as v_grid
from icon4py.model.common.io import io as common_io, utils as io_utils
from icon4py.model.common.states import (
    data as state_data,
    diagnostic_state as diagnostics,
    prognostic_state as prognostics,
    quantities,
    spec as field_spec,
    tracer_states as tracers,
)


#: File-name stub for the output file (a counter + ``.nc`` is appended).
DEFAULT_OUTPUT_FILENAME: Final[str] = "icon4py_output"


# --------------------------------------------------------------------------------------
# Prognostic fields
# --------------------------------------------------------------------------------------


def _quantities_with_label(cls: type[Any], label: str) -> list[str]:
    """Return canonical quantity names of fields on ``cls`` carrying ``label``."""
    return [
        field_spec.quantity
        for field_spec in (field_spec.get_field_spec(f) for f in dataclasses.fields(cls))
        if field_spec is not None and label in field_spec.labels
    ]


def _cf_keys(quantity_names: list[str]) -> list[str]:
    """Map canonical quantity names to their CF output keys."""
    return sorted(
        (quantity.cf_key if quantity.cf_key is not None else quantity.name)
        for quantity in (quantities.get(q) for q in quantity_names)
    )


#: Default prognostic output variables: the CF names of ``PrognosticState`` fields
#: labeled ``output``. The metadata, the state attribute (``icon_var_name``) and the
#: vertical placement (``is_on_half_levels``) come from the
#: ``states.data.PROGNOSTIC_CF_ATTRIBUTES`` catalog; this list only selects which
#: entries to emit.
PROGNOSTIC_VARIABLES: Final[list[str]] = _cf_keys(
    _quantities_with_label(prognostics.PrognosticState, "output")
)


def prognostic_state_to_dataarrays(
    prognostic_state: prognostics.PrognosticState,
    variables: list[str] | None = None,
) -> dict[str, xr.DataArray]:
    """Assemble a CF/UGRID-annotated model-state dict from a ``PrognosticState``."""
    selected = PROGNOSTIC_VARIABLES if variables is None else variables

    state: dict[str, xr.DataArray] = {}
    for name in selected:
        try:
            metadata = state_data.PROGNOSTIC_CF_ATTRIBUTES[name]
        except KeyError as err:
            raise ValueError(
                f"Unknown prognostic output variable '{name}'. "
                f"Known variables are: {PROGNOSTIC_VARIABLES}."
            ) from err
        field = getattr(prognostic_state, metadata["icon_var_name"])
        state[name] = io_utils.to_data_array(
            field,
            metadata,
            is_on_half_levels=metadata.get("is_on_half_levels", False),
            to_host=True,
        )
    return state


# --------------------------------------------------------------------------------------
# Diagnostic fields
# --------------------------------------------------------------------------------------


#: Diagnostic output variables: the CF names of ``DiagnosticState`` fields labeled
#: ``output``.
DIAGNOSTIC_VARIABLES: Final[list[str]] = _cf_keys(
    _quantities_with_label(diagnostics.DiagnosticState, "output")
)


def output_variables() -> list[str]:
    """Return the CF variable names of all fields labeled ``output``.

    A query over the declarations; ``PROGNOSTIC_VARIABLES`` and
    ``DIAGNOSTIC_VARIABLES`` are its two per-class components.
    """
    return sorted([*PROGNOSTIC_VARIABLES, *DIAGNOSTIC_VARIABLES])


#: All output variables (prognostic + diagnostic), written together into the same file.
DEFAULT_OUTPUT_VARIABLES: Final[list[str]] = output_variables()


def restart_variables() -> list[str]:
    """Return the canonical quantity names of all fields labeled ``restart``.

    The checkpoint workstream consumes these declarations. The list mixes
    prognostic fields, tracer fields, and the half-level pressure field.
    """
    variable_quantities = [
        *_quantities_with_label(prognostics.PrognosticState, "restart"),
        *_quantities_with_label(tracers.TracerState, "restart"),
        *_quantities_with_label(diagnostics.DiagnosticState, "restart"),
    ]
    return sorted(variable_quantities)


def diagnostic_state_to_fields(
    diagnostic_state: diagnostics.DiagnosticState,
) -> dict[str, gtx.Field]:
    """Map the canonical diagnostic state to the field dict consumed by IO.

    The keys match ``DIAGNOSTIC_CF_ATTRIBUTES`` so that
    :func:`diagnostic_fields_to_dataarrays` can annotate them for output.
    """
    return {
        "eastward_wind": diagnostic_state.u,
        "northward_wind": diagnostic_state.v,
        "temperature": diagnostic_state.temperature,
        "virtual_temperature": diagnostic_state.virtual_temperature,
        "pressure": diagnostic_state.pressure,
        "surface_pressure": diagnostic_state.surface_pressure,
    }


def diagnostic_fields_to_dataarrays(
    diagnostic_fields: dict[str, gtx.Field],
) -> dict[str, xr.DataArray]:
    """Assemble CF/UGRID-annotated DataArrays from computed diagnostic fields."""
    state: dict[str, xr.DataArray] = {}
    for name, field in diagnostic_fields.items():
        metadata = state_data.DIAGNOSTIC_CF_ATTRIBUTES[name]
        state[name] = io_utils.to_data_array(
            field,
            metadata,
            is_on_half_levels=metadata.get("is_on_half_levels", False),
            to_host=True,
        )
    return state


# --------------------------------------------------------------------------------------
# Monitor factory
# --------------------------------------------------------------------------------------


def create_io_monitor(
    *,
    output_path: pathlib.Path,
    grid_file_path: pathlib.Path,
    grid: grid_base.Grid,
    vertical_grid: v_grid.VerticalGrid,
    dtime: datetime.timedelta,
    variables: list[str] | None = None,
    output_interval: common_io.OutputInterval = time.NumTimeSteps(1),
    process_props: decomposition_defs.ProcessProperties | None = None,
) -> common_io.IOMonitor:
    """Build a single-node ``IOMonitor`` with one field group holding all output fields.

    ``output_interval`` is either a number of model steps or a simulation-time delta
    (normalized to steps using ``dtime``); it defaults to every step.

    ``process_props`` is currently unused: IO is single-node only. It is kept on the
    signature so the distributed path (per-rank IO setup) can be wired in without a
    signature change.
    """
    del process_props  # reserved for the distributed IO path; unused while single-node
    monitor_variables = output_variables() if variables is None else variables
    field_groups = [
        common_io.FieldGroupIOConfig(
            output_interval=output_interval,
            filename=DEFAULT_OUTPUT_FILENAME,
            variables=monitor_variables,
            nc_title="ICON4Py output",
            nc_comment="Fields computed by ICON4Py.",
        )
    ]

    config = common_io.IOConfig(output_path=str(output_path), field_groups=field_groups)
    return common_io.IOMonitor(
        config=config,
        vertical_size=vertical_grid,
        horizontal_size=grid.config.horizontal_config,
        grid_file_name=grid_file_path,
        # Grid.id holds the file's `uuidOfHGrid` as a string; the IO layer wants a UUID.
        grid_id=uuid.UUID(grid.id),
        dtime=dtime,
    )
