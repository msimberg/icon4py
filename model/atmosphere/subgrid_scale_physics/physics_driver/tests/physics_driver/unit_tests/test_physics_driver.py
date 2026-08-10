# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the physics driver and its eDSL composition."""

from __future__ import annotations

import dataclasses
import datetime

import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver.composition import (
    PhysicsLoopState,
    build_physics_composition,
)
from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver.physics_driver import (
    PhysicsDriver,
    PhysicsProcess,
)
from icon4py.model.atmosphere.subgrid_scale_physics.physics_driver.process_time_control import (
    ProcessTimeControl,
)
from icon4py.model.common.components.physics_state import TypedPhysicsState


_T0 = datetime.datetime(2024, 1, 1, 0, 0, 0)
_DT = datetime.timedelta(seconds=300)  # 5-min physics interval


def _tc(
    interval: datetime.timedelta = _DT,
    start: datetime.datetime = _T0,
    end: datetime.datetime = _T0 + datetime.timedelta(days=1),
    enable_process: bool = True,
) -> ProcessTimeControl:
    return ProcessTimeControl(
        interval=interval,
        start_date=start,
        end_date=end,
        enable_process=enable_process,
    )


class TestProcessTimeControl:
    def test_enable_process_defaults_true(self) -> None:
        assert _tc().enable_process is True

    def test_is_active_false_when_disabled(self) -> None:
        assert _tc(enable_process=False).is_active(_T0) is False

    def test_is_active_false_when_interval_zero(self) -> None:
        assert _tc(interval=datetime.timedelta(0)).is_active(_T0) is False

    def test_is_in_window_at_start_is_true(self) -> None:
        assert _tc().is_in_window(_T0) is True

    def test_is_in_window_at_end_is_false(self) -> None:
        end = _T0 + datetime.timedelta(hours=1)
        assert _tc(end=end).is_in_window(end) is False

    def test_is_in_window_before_start_is_false(self) -> None:
        assert _tc().is_in_window(_T0 - datetime.timedelta(seconds=1)) is False

    def test_is_in_window_inside_is_true(self) -> None:
        assert _tc().is_in_window(_T0 + datetime.timedelta(hours=12)) is True

    def test_is_active_at_start_is_true(self) -> None:
        assert _tc().is_active(_T0) is True

    def test_is_active_at_one_interval_is_true(self) -> None:
        assert _tc().is_active(_T0 + _DT) is True

    def test_is_active_at_half_interval_is_false(self) -> None:
        assert _tc().is_active(_T0 + _DT / 2) is False

    def test_is_active_before_start_is_false(self) -> None:
        assert _tc().is_active(_T0 - datetime.timedelta(seconds=1)) is False

    def test_is_active_requires_exact_interval_multiple(self) -> None:
        # Fires only at an exact integer multiple of the interval.
        assert _tc().is_active(_T0 + 2 * _DT) is True
        # 1 microsecond off the boundary does not fire (no tolerance).
        jitter = datetime.timedelta(microseconds=1)
        assert _tc().is_active(_T0 + 2 * _DT + jitter) is False

    def test_frozen_dataclass(self) -> None:
        tc = _tc()
        with pytest.raises(dataclasses.FrozenInstanceError):
            tc.interval = datetime.timedelta(seconds=1)  # type: ignore[misc]

    def test_validate_interval_accepts_integer_multiple(self) -> None:
        _tc(interval=2 * _DT).validate_interval(_DT)

    def test_validate_interval_rejects_non_multiple(self) -> None:
        with pytest.raises(ValueError, match="integer multiple"):
            _tc(interval=1.5 * _DT).validate_interval(_DT)

    def test_validate_interval_rejects_zero_interval_when_enabled(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            _tc(interval=datetime.timedelta(0)).validate_interval(_DT)

    def test_validate_interval_rejects_disabled_process_with_non_positive_interval(self) -> None:
        # M1: interval validation now covers disabled processes too.
        with pytest.raises(ValueError, match="positive"):
            _tc(interval=datetime.timedelta(0), enable_process=False).validate_interval(_DT)


@dataclasses.dataclass(frozen=True)
class RecordingInput:
    """Typed input placeholder for the recording component."""

    payload: object


@dataclasses.dataclass(frozen=True)
class RecordingOutput:
    """Typed output placeholder for the recording component."""

    payload: dict[str, object]


@dataclasses.dataclass
class RecordingComponent:
    """Stub Component: records calls, returns configured outputs."""

    outputs: dict[str, object]
    call_count: int = 0
    last_input: RecordingInput | None = dataclasses.field(default=None, repr=False)

    @classmethod
    def input_type(cls) -> type[RecordingInput]:
        return RecordingInput

    @classmethod
    def output_type(cls) -> type[RecordingOutput]:
        return RecordingOutput

    def run(self, state: RecordingInput) -> RecordingOutput:
        self.call_count += 1
        self.last_input = state
        return RecordingOutput(payload=dict(self.outputs))


@dataclasses.dataclass
class RecordingPhysicsState(TypedPhysicsState[RecordingInput, RecordingOutput]):
    """Stub TypedPhysicsState: records gather / apply / store calls."""

    gather_calls: list = dataclasses.field(default_factory=list)
    apply_calls: list = dataclasses.field(default_factory=list)
    store_calls: list = dataclasses.field(default_factory=list)

    def gather_from_prognostic(self, prognostic: object, tracers: object) -> RecordingInput:
        self.gather_calls.append((prognostic, tracers))
        return RecordingInput(payload=(prognostic, tracers))

    def apply_tendencies(self, outputs: RecordingOutput, dtime: datetime.timedelta) -> None:
        self.apply_calls.append((outputs, dtime))

    def store_diagnostics(self, outputs: RecordingOutput) -> None:
        self.store_calls.append(outputs)


def _make_process(
    name: str,
    outputs: dict[str, object],
    time_control: ProcessTimeControl | None = None,
    apply_forcing: bool = True,
) -> PhysicsProcess:
    component = RecordingComponent(outputs=outputs)
    return PhysicsProcess(
        name=name,
        component=component,  # type: ignore[arg-type]
        state=RecordingPhysicsState(),
        time_control=time_control if time_control is not None else _tc(),
        apply_forcing=apply_forcing,
    )


def _make_carry(
    sample_cache: dict[str, object] | None = None,
    simulation_current_datetime: datetime.datetime = _T0,
) -> PhysicsLoopState:
    return PhysicsLoopState(
        prognostic="prog",  # type: ignore[arg-type]
        tracers="tracers",  # type: ignore[arg-type]
        dtime=_DT,
        simulation_current_datetime=simulation_current_datetime,
        sample_cache=sample_cache if sample_cache is not None else {},
    )


def test_composition_invokes_components_in_order() -> None:
    proc_a = _make_process("A", {"tend_temperature": "A"})
    proc_b = _make_process("B", {"tend_temperature": "B"})
    composition = build_physics_composition([proc_a, proc_b])

    composition(_make_carry())

    assert proc_a.component.call_count == 1
    assert proc_b.component.call_count == 1
    # B's apply must follow A's (operator-splitting ordering)
    assert proc_a.state.apply_calls[0][0].payload == {"tend_temperature": "A"}
    assert proc_b.state.apply_calls[0][0].payload == {"tend_temperature": "B"}


def test_driver_construction_raises_for_non_multiple_interval() -> None:
    proc = _make_process("X", {"tend_temperature": "X"}, time_control=_tc(interval=1.5 * _DT))
    with pytest.raises(ValueError, match="integer multiple"):
        PhysicsDriver([proc], dtime=_DT)
    assert proc.component.call_count == 0


def test_driver_construction_validates_disabled_process_interval() -> None:
    # M1: a disabled process with a non-positive interval fails at construction.
    proc = _make_process(
        "disabled_bad_interval",
        {"tend_temperature": "X"},
        time_control=_tc(interval=datetime.timedelta(0), enable_process=False),
    )
    with pytest.raises(ValueError, match="positive"):
        PhysicsDriver([proc], dtime=_DT)


def test_composition_skips_disabled_process() -> None:
    proc = _make_process(
        "disabled",
        {"tend_temperature": "X"},
        time_control=_tc(enable_process=False),
    )
    composition = build_physics_composition([proc])

    composition(_make_carry())

    assert proc.component.call_count == 0
    assert proc.state.apply_calls == []
    assert proc.state.store_calls == []


def test_composition_out_of_window_process_does_nothing() -> None:
    future = _T0 + datetime.timedelta(days=1)
    proc = _make_process(
        "future",
        {"tend_temperature": "X"},
        time_control=_tc(start=future, end=future + datetime.timedelta(hours=1)),
    )
    composition = build_physics_composition([proc])

    composition(_make_carry(simulation_current_datetime=_T0))

    assert proc.component.call_count == 0
    assert proc.state.apply_calls == []
    assert proc.state.store_calls == []


def test_composition_active_call_caches_outputs_and_applies_them() -> None:
    proc = _make_process("p", {"tend_temperature": "FRESH"})
    composition = build_physics_composition([proc])
    cache: dict[str, object] = {}

    composition(_make_carry(sample_cache=cache))

    cached = cache["p"]
    assert isinstance(cached, RecordingOutput)
    assert proc.component.call_count == 1
    assert cached.payload == {"tend_temperature": "FRESH"}
    assert proc.state.apply_calls[0][0].payload == {"tend_temperature": "FRESH"}
    assert proc.state.apply_calls[0][1] == _DT
    assert proc.state.store_calls == []


def test_composition_inactive_in_window_recycles_cached_outputs() -> None:
    # Component returns "FRESH" the first time, would return "STALE" the second
    # if called — but on the recycle step it MUST NOT be called.
    proc = _make_process("p", {"tend_temperature": "FRESH"}, time_control=_tc(interval=2 * _DT))
    composition = build_physics_composition([proc])
    cache: dict[str, object] = {}

    # Step 1: active (elapsed == 0), compute + cache.
    composition(_make_carry(sample_cache=cache))
    # Step 2: in window, but not active (elapsed == _DT, not a multiple of 2*_DT).
    composition(_make_carry(sample_cache=cache, simulation_current_datetime=_T0 + _DT))

    # Component invoked once total (compute step only).
    assert proc.component.call_count == 1
    # But apply happened twice — once with the fresh tendency, once recycled.
    assert len(proc.state.apply_calls) == 2
    assert proc.state.apply_calls[0][0].payload == {"tend_temperature": "FRESH"}
    assert proc.state.apply_calls[1][0].payload == {"tend_temperature": "FRESH"}  # recycled


def test_composition_first_in_window_step_inactive_computes_without_keyerror() -> None:
    # Regression: a process whose first-ever in-window step is NOT active
    # (interval = 2*dt, first step lands at start + dt) used to KeyError on the empty recycle
    # cache. With nothing cached to recycle yet, it must compute instead.
    proc = _make_process("p", {"tend_temperature": "FRESH"}, time_control=_tc(interval=2 * _DT))
    composition = build_physics_composition([proc])
    cache: dict[str, object] = {}

    # First call lands in-window but off the firing tick (elapsed == _DT, interval == 2*_DT).
    composition(_make_carry(sample_cache=cache, simulation_current_datetime=_T0 + _DT))

    assert proc.component.call_count == 1
    assert len(proc.state.apply_calls) == 1
    assert proc.state.apply_calls[0][0].payload == {"tend_temperature": "FRESH"}


def test_composition_apply_mode_selects_apply_chain() -> None:
    proc = _make_process("p", {"tend_temperature": "FRESH"}, apply_forcing=True)
    composition = build_physics_composition([proc])

    composition(_make_carry())

    assert len(proc.state.apply_calls) == 1
    assert proc.state.apply_calls[0][0].payload == {"tend_temperature": "FRESH"}
    assert proc.state.store_calls == []


def test_composition_diagnostic_mode_selects_store_diagnostics_step() -> None:
    proc = _make_process(
        "p",
        {"tend_temperature": "FRESH"},
        apply_forcing=False,
    )
    composition = build_physics_composition([proc])

    composition(_make_carry())

    assert proc.component.call_count == 1
    assert proc.state.apply_calls == []  # diagnostic branch does not apply tendencies
    assert len(proc.state.store_calls) == 1
    assert proc.state.store_calls[0].payload == {"tend_temperature": "FRESH"}


def test_driver_run_matches_composition() -> None:
    proc = _make_process("p", {"tend_temperature": "FRESH"}, time_control=_tc(interval=2 * _DT))
    driver = PhysicsDriver([proc], dtime=_DT)

    driver.run(
        prognostic="prog",  # type: ignore[arg-type]
        tracers="tracers",  # type: ignore[arg-type]
        dtime=_DT,
        simulation_current_datetime=_T0,
    )
    driver.run(
        prognostic="prog",  # type: ignore[arg-type]
        tracers="tracers",  # type: ignore[arg-type]
        dtime=_DT,
        simulation_current_datetime=_T0 + _DT,
    )

    assert proc.component.call_count == 1
    assert len(proc.state.apply_calls) == 2
