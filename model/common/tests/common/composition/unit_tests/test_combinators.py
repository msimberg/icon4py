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

from icon4py.model.common.composition import Step, SwapPolicy, chain, foreach, repeat, sample, when


@dataclasses.dataclass
class _Carry:
    log: list[str] = dataclasses.field(default_factory=list)
    counter: int = 0
    items: list[Any] = dataclasses.field(default_factory=list)
    cache: dict[str, Any] = dataclasses.field(default_factory=dict)
    swap_calls: int = 0


class _AppendStep:
    def __init__(self, token: str, *, name: str | None = None) -> None:
        self.name = name or token
        self._token = token

    def __call__(self, carry: _Carry) -> None:
        carry.log.append(self._token)


class _SwapTarget:
    def __init__(self, carry: _Carry) -> None:
        self._carry = carry

    def swap(self) -> None:
        self._carry.swap_calls += 1


def _append(token: str) -> Step[_Carry]:
    return _AppendStep(token)


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


def test_repeat_set_loop_context_receives_index_and_total() -> None:
    recorded: list[tuple[int, int]] = []

    def _ctx(carry: _Carry, index: int, total: int) -> None:
        recorded.append((index, total))

    repeat(_append("x"), times=3, set_loop_context=_ctx)(_Carry())
    assert recorded == [(0, 3), (1, 3), (2, 3)]


def test_repeat_swap_never_does_not_swap() -> None:
    carry = _Carry()
    repeat(
        _append("x"),
        times=2,
        swap=SwapPolicy.NEVER,
        swap_target=_SwapTarget,
    )(carry)
    assert carry.swap_calls == 0


def test_repeat_swap_always_swaps_each_iteration() -> None:
    carry = _Carry()
    repeat(
        _append("x"),
        times=3,
        swap=SwapPolicy.ALWAYS,
        swap_target=_SwapTarget,
    )(carry)
    assert carry.swap_calls == 3


def test_repeat_swap_except_last_skips_final_swap() -> None:
    carry = _Carry()
    repeat(
        _append("x"),
        times=3,
        swap=SwapPolicy.EXCEPT_LAST,
        swap_target=_SwapTarget,
    )(carry)
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


def test_foreach_iterates_source_and_sets_item() -> None:
    carry = _Carry()

    def _source(c: _Carry) -> list[int]:
        return [1, 2, 3]

    def _set_item(c: _Carry, item: int) -> None:
        c.items.append(item)

    foreach(_append("body"), source=_source, set_item=_set_item)(carry)
    assert carry.items == [1, 2, 3]
    assert carry.log == ["body", "body", "body"]


def test_foreach_without_set_item_runs_steps_per_item() -> None:
    carry = _Carry()
    foreach(_append("x"), source=lambda c: ["a", "b"])(carry)
    assert carry.log == ["x", "x"]


def test_sample_runs_on_firing_step_and_caches() -> None:
    carry = _Carry()

    class _ComputeStep:
        name = "compute"

        def __call__(self, c: _Carry) -> None:
            c.cache["p"] = c.cache.get("p", 0) + 1

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

    class _ComputeStep:
        name = "compute"

        def __call__(self, c: _Carry) -> None:
            c.cache["p"] = "STALE"

    step = sample(
        _ComputeStep(),
        every=datetime.timedelta(seconds=300),
        clock=lambda c: datetime.timedelta(seconds=150),
        key="p",
        cache=lambda c: c.cache,
    )
    step(carry)
    assert carry.cache["p"] == "FRESH"


def test_sample_computes_when_first_in_window_inactive_and_cache_empty() -> None:
    carry = _Carry()

    class _ComputeStep:
        name = "compute"

        def __call__(self, c: _Carry) -> None:
            c.cache["p"] = "FRESH"

    step = sample(
        _ComputeStep(),
        every=datetime.timedelta(seconds=300),
        clock=lambda c: datetime.timedelta(seconds=150),
        key="p",
        cache=lambda c: c.cache,
    )
    step(carry)
    assert carry.cache["p"] == "FRESH"


def test_sample_every_positive_is_required_for_modulo() -> None:
    """The recycle semantics assume ``every > 0`` so that modulo is well-defined."""
    with pytest.raises(ZeroDivisionError):
        sample(
            _append("x"),
            every=datetime.timedelta(0),
            clock=lambda c: datetime.timedelta(seconds=0),
            key="p",
            cache=lambda c: c.cache,
        )(_Carry())
