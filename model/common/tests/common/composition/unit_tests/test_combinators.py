# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Data-free unit tests for the generic composition combinators."""

from __future__ import annotations

import dataclasses
import datetime
from typing import Any

import pytest

from icon4py.model.common.composition import (
    Step,
    chain,
    foreach,
    named,
    repeat,
    sample,
    swap,
    when,
    with_index,
)


@dataclasses.dataclass
class _Carry:
    log: list[str] = dataclasses.field(default_factory=list)
    counter: int = 0
    items: list[Any] = dataclasses.field(default_factory=list)
    cache: dict[str, Any] = dataclasses.field(default_factory=dict)
    swap_calls: int = 0
    indices: list[int] = dataclasses.field(default_factory=list)
    totals: list[int] = dataclasses.field(default_factory=list)


class _SwapTarget:
    def __init__(self, carry: Any) -> None:
        self._carry = carry

    def swap(self) -> None:
        self._carry.swap_calls += 1


def _swap_target(carry: _Carry) -> _SwapTarget:
    return _SwapTarget(carry)


def _append(token: str) -> Step[_Carry]:
    return _AppendStep(token)


class _AppendStep:
    def __init__(self, token: str, *, name: str | None = None) -> None:
        self.name = name or token
        self._token = token

    def __call__(self, carry: _Carry, item: Any = None) -> None:
        del item
        carry.log.append(self._token)


def test_chain_runs_steps_in_order() -> None:
    carry = _Carry()
    chain(_append("a"), _append("b"), _append("c"))(carry)
    assert carry.log == ["a", "b", "c"]


def test_chain_name_can_be_overridden() -> None:
    step = chain(_append("a"), name="my_chain")
    assert step.name == "my_chain"


def test_repeat_runs_exactly_times() -> None:
    carry = _Carry()
    repeat(_append("x"), times=3)(carry)
    assert carry.log == ["x", "x", "x"]


def test_repeat_re_reads_callable_times() -> None:
    carry = _Carry()
    repeat(_append("x"), times=lambda c: c.counter)(carry)
    assert carry.log == []

    carry.counter = 2
    repeat(_append("x"), times=lambda c: c.counter)(carry)
    assert carry.log == ["x", "x"]


def test_repeat_pre_runs_once_before_loop_and_post_after() -> None:
    carry = _Carry()
    repeat(
        _append("body"),
        times=2,
        pre=_append("pre"),
        post=_append("post"),
    )(carry)
    assert carry.log == ["pre", "body", "body", "post"]


def test_repeat_passes_index_and_total_to_child() -> None:
    recorded: list[tuple[int, int]] = []

    def _record(carry: _Carry, item: tuple[int, int]) -> None:
        del carry
        recorded.append(item)

    record_step: Step[_Carry] = named("record", _record, pass_item=True)
    repeat(record_step, times=3)(_Carry())

    assert recorded == [(0, 3), (1, 3), (2, 3)]


def test_swap_calls_swap_on_target() -> None:
    carry = _Carry()
    swap(_swap_target)(carry)
    assert carry.swap_calls == 1


def test_with_index_sets_index_and_runs_body() -> None:
    carry = _Carry()

    def _set_index(carry: _Carry, index: int, total: int) -> None:
        carry.indices.append(index)
        carry.totals.append(total)

    def _body(carry: _Carry) -> None:
        carry.log.append("body")

    with_index(named("body", _body), set_index=_set_index)(carry, (2, 5))
    assert carry.indices == [2]
    assert carry.totals == [5]
    assert carry.log == ["body"]


def test_with_index_requires_a_two_tuple_item() -> None:
    carry = _Carry()

    def _set_index(carry: _Carry, index: int, total: int) -> None:
        del carry, index, total

    step = with_index(named("body", lambda c: None), set_index=_set_index)
    with pytest.raises(TypeError, match="2-tuple"):
        step(carry)


def test_repeat_chain_with_index_and_conditional_swap_on_synthetic_carry() -> None:
    """A non-dycore loop proves ``repeat`` is generic: it forwards the loop index
    through ``chain`` to ``with_index``, and the conditional swap step is an
    ordinary child of the same chain.
    """

    def _set_index(carry: _Carry, index: int, total: int) -> None:
        carry.indices.append(index)
        carry.totals.append(total)

    def _body(carry: _Carry) -> None:
        carry.log.append("body")

    step = repeat(
        chain(
            with_index(named("body", _body), set_index=_set_index),
            when(
                lambda c: c.indices[-1] != c.totals[-1] - 1,
                then=swap(_swap_target),
            ),
        ),
        times=3,
    )

    carry = _Carry()
    step(carry)
    assert carry.indices == [0, 1, 2]
    assert carry.totals == [3, 3, 3]
    assert carry.log == ["body", "body", "body"]
    assert carry.swap_calls == 2


def test_when_runs_then_when_predicate_is_true() -> None:
    carry = _Carry()
    when(lambda c: True, then=_append("yes"), else_=_append("no"))(carry)
    assert carry.log == ["yes"]


def test_when_runs_else_when_predicate_is_false() -> None:
    carry = _Carry()
    when(lambda c: False, then=_append("yes"), else_=_append("no"))(carry)
    assert carry.log == ["no"]


def test_when_without_else_does_nothing_when_false() -> None:
    carry = _Carry()
    when(lambda c: False, then=_append("yes"))(carry)
    assert carry.log == []


def test_foreach_iterates_source_and_passes_item() -> None:
    carry = _Carry()

    def _source(c: _Carry) -> list[int]:
        return [1, 2, 3]

    def _body(c: _Carry, item: int) -> None:
        c.items.append(item)

    foreach(_append("body"), _body, source=_source)(carry)

    assert carry.items == [1, 2, 3]
    assert carry.log == ["body", "body", "body"]


def test_foreach_without_item_step_runs_steps_per_item() -> None:
    carry = _Carry()
    foreach(_append("x"), source=lambda c: ["a", "b"])(carry)
    assert carry.log == ["x", "x"]


class _ComputeStep(Step[_Carry]):
    def __init__(self) -> None:
        self.name: str = "compute"

    def __call__(self, c: _Carry, item: Any = None) -> None:
        del item
        c.cache["p"] = c.cache.get("p", 0) + 1


class _StaleStep(Step[_Carry]):
    def __init__(self) -> None:
        self.name: str = "compute"

    def __call__(self, c: _Carry, item: Any = None) -> None:
        del item
        c.cache["p"] = "STALE"


class _FreshStep(Step[_Carry]):
    def __init__(self) -> None:
        self.name: str = "compute"

    def __call__(self, c: _Carry, item: Any = None) -> None:
        del item
        c.cache["p"] = "FRESH"


def test_sample_runs_on_firing_step_and_caches() -> None:
    carry = _Carry()

    step = sample(
        _ComputeStep(),
        every=datetime.timedelta(seconds=300),
        clock=lambda c: datetime.timedelta(seconds=300),
        key="p",
        cache=lambda c: c.cache,
    )
    step(carry)
    assert carry.cache["p"] == 1


def test_sample_recycles_cached_outputs_on_inactive_step() -> None:
    carry = _Carry()
    carry.cache["p"] = "FRESH"

    step = sample(
        _StaleStep(),
        every=datetime.timedelta(seconds=300),
        clock=lambda c: datetime.timedelta(seconds=150),
        key="p",
        cache=lambda c: c.cache,
    )
    step(carry)
    assert carry.cache["p"] == "FRESH"


def test_sample_computes_when_first_in_window_inactive_and_cache_empty() -> None:
    carry = _Carry()

    step = sample(
        _FreshStep(),
        every=datetime.timedelta(seconds=300),
        clock=lambda c: datetime.timedelta(seconds=150),
        key="p",
        cache=lambda c: c.cache,
    )
    step(carry)
    assert carry.cache["p"] == "FRESH"


def test_sample_every_positive_is_required_for_modulo() -> None:
    """The recycle semantics assume ``every > 0`` so that modulo is well-defined."""
    with pytest.raises(ValueError, match=r"every.*positive"):
        sample(
            _append("x"),
            every=datetime.timedelta(0),
            clock=lambda c: datetime.timedelta(seconds=0),
            key="p",
            cache=lambda c: c.cache,
        )


def test_sample_every_rejects_non_timedelta() -> None:
    """An integer ``every`` is a type error, not a silently accepted value."""
    with pytest.raises(TypeError, match=r"every.*timedelta"):
        sample(
            _append("x"),
            every=300,  # type: ignore[arg-type]
            clock=lambda c: datetime.timedelta(seconds=0),
            key="p",
            cache=lambda c: c.cache,
        )


def test_sample_every_rejects_non_positive_timedelta() -> None:
    """A non-positive ``every`` raises ``ValueError`` so modulo is well-defined."""
    with pytest.raises(ValueError, match=r"every.*positive"):
        sample(
            _append("x"),
            every=datetime.timedelta(seconds=-1),
            clock=lambda c: datetime.timedelta(seconds=0),
            key="p",
            cache=lambda c: c.cache,
        )
