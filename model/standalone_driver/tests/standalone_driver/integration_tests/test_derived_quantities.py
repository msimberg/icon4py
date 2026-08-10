# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase-6 datatest: validate the canonical T/p/u/v derivation against ICON savepoints."""

from __future__ import annotations

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
import numpy as np
import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.components.derived_quantities import (
    DerivedQuantities,
    DerivedQuantitiesInput,
)
from icon4py.model.common.grid import base as base_grid
from icon4py.model.common.states.data import QC, QG, QI, QR, QS, QV
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions as test_defs, serialbox as sb, test_utils

from ..fixtures import (
    backend,
    data_provider,
    download_ser_data,
    experiment,
    experiment_description,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    metrics_savepoint,
    process_props,
)


# Tolerances for the EXCLAIM_APE_AES initial-state validation. The moist path
# exercises all six hydrometeors; the dominant residual is in the surface/lower
# levels where the moist EOS and the reference differ most.
_EXCLAIM_APE_AES_TOLERANCE: dict[str, tuple[float, float]] = {
    "temperature": (5.0e-6, 0.0),
    "virtual_temperature": (5.0e-6, 0.0),
    "pressure": (2.0e-1, 0.0),
    "pressure_ifc": (2.0e-1, 0.0),
    "u": (1.0e-6, 0.0),
    "v": (1.0e-6, 0.0),
}


@pytest.mark.datatest
@pytest.mark.level("integration")
@pytest.mark.parametrize(
    "experiment_description",
    [test_defs.Experiments.EXCLAIM_APE_AES],
    ids=["EXCLAIM_APE_AES"],
)
def test_derived_quantities_match_icon_savepoints(  # noqa: PLR0917
    data_provider: sb.IconSerialDataProvider,
    icon_grid: base_grid.Grid,
    backend: gtx_typing.Backend,
    interpolation_savepoint: sb.InterpolationSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    experiment_description: test_defs.ExperimentDescription,
) -> None:
    """The canonical T/p/u/v component agrees with ICON's diagnostic savepoint.

    This is the phase-6 validation: the single derivation used by the driver
    (``DerivedQuantities``) is checked against the reference diagnostics for a
    moist experiment. The maximum observed deviation is recorded below and in
    the test output.
    """
    if backend is None:
        pytest.skip("Derived-quantity validation requires a compiled backend.")

    prognostic_savepoint = data_provider.from_savepoint_prognostics_initial()
    diagnostic_savepoint = data_provider.from_savepoint_diagnostics_initial()

    def _zero_cell_k(extend: dict[gtx.Dimension, int] | None = None) -> gtx.Field:
        return data_alloc.zero_field(
            icon_grid, dims.CellDim, dims.KDim, dtype=float, extend=extend, allocator=backend
        )

    temperature = _zero_cell_k()
    virtual_temperature = _zero_cell_k()
    pressure = _zero_cell_k()
    pressure_ifc = _zero_cell_k(extend={dims.KDim: 1})
    surface_pressure = data_alloc.zero_field(
        icon_grid, dims.CellDim, dtype=float, allocator=backend
    )
    u = _zero_cell_k()
    v = _zero_cell_k()

    component = DerivedQuantities(grid=icon_grid, backend=backend)
    component.run(
        DerivedQuantitiesInput(
            theta_v=prognostic_savepoint.theta_v_now(),
            exner=prognostic_savepoint.exner_now(),
            vn=prognostic_savepoint.vn_now(),
            qv=prognostic_savepoint.tracer_now(QV),
            qc=prognostic_savepoint.tracer_now(QC),
            qi=prognostic_savepoint.tracer_now(QI),
            qr=prognostic_savepoint.tracer_now(QR),
            qs=prognostic_savepoint.tracer_now(QS),
            qg=prognostic_savepoint.tracer_now(QG),
            ddqz_z_full=metrics_savepoint.ddqz_z_full(),
            rbf_vec_coeff_c1=interpolation_savepoint.rbf_vec_coeff_c1(),
            rbf_vec_coeff_c2=interpolation_savepoint.rbf_vec_coeff_c2(),
            temperature=temperature,
            virtual_temperature=virtual_temperature,
            pressure=pressure,
            pressure_ifc=pressure_ifc,
            surface_pressure=surface_pressure,
            u=u,
            v=v,
        )
    )

    computed = {
        "temperature": temperature,
        "virtual_temperature": virtual_temperature,
        "pressure": pressure,
        "pressure_ifc": pressure_ifc,
        "u": u,
        "v": v,
    }
    references = {
        "temperature": diagnostic_savepoint.temperature(),
        "virtual_temperature": diagnostic_savepoint.virtual_temperature(),
        "pressure": diagnostic_savepoint.pressure(),
        "pressure_ifc": diagnostic_savepoint.pressure_ifc(),
        "u": diagnostic_savepoint.zonal_wind(),
        "v": diagnostic_savepoint.meridional_wind(),
    }

    max_deviations: dict[str, float] = {}
    for name, reference in references.items():
        ref = reference.asnumpy()
        comp = computed[name].asnumpy()
        atol, rtol = _EXCLAIM_APE_AES_TOLERANCE[name]
        max_deviations[name] = float(np.max(np.abs(comp - ref)))
        test_utils.assert_dallclose(
            comp,
            ref,
            atol=atol,
            rtol=rtol,
            err_msg=f"{name} max_deviation={max_deviations[name]:.3e}",
        )

    # Recorded maxima are printed in the test log for the phase-6 report.
    print(f"DerivedQuantities max deviations: {max_deviations}")
