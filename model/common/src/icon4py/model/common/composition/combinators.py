# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Generic combinators that build new ``Step`` instances from simpler ones."""

from __future__ import annotations

import datetime
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from icon4py.model.common.components.components import Component
from icon4py.model.common.composition.step import Step


C = TypeVar("C")
InnerC = TypeVar("InnerC")
OuterC = TypeVar("OuterC")


def _default_name(stem: str, name: str | None) -> str:
    return stem if name is None else name


class _NamedStep(Step[C]):
    """Simple wrapper giving a callable step a ``name`` attribute."""

    name: str
    component: Component[Any, Any] | None
    _fn: Callable[..., None]
    _pass_item: bool

    def __init__(
        self,
        name: str,
        fn: Callable[..., None],
        component: Component[Any, Any] | None = None,
        pass_item: bool = False,
    ) -> None:
        self.name = name
        self._fn = fn
        self.component = component
        self._pass_item = pass_item

    def __call__(self, carry: C, item: Any = None) -> None:
        if self._pass_item:
            self._fn(carry, item)
        else:
            del item
            self._fn(carry)


def named[C](
    name: str,
    fn: Callable[..., None],
    component: Component[Any, Any] | None = None,
    pass_item: bool = False,
) -> Step[C]:
    """Wrap ``fn`` as a ``Step`` with the given ``name``.

    ``component`` is optional metadata used by introspection to derive the
    step's declared inputs and outputs.

    ``pass_item`` is used when the step runs inside ``foreach`` or ``repeat``;
    it forwards the iteration item to ``fn``.
    """
    return _NamedStep(name, fn, component=component, pass_item=pass_item)


class _Chain(Step[C]):
    def __init__(self, *steps: Step[C], name: str | None = None) -> None:
        self.name = _default_name("chain", name)
        self._steps = steps

    def __call__(self, carry: C, item: Any = None) -> None:
        for step in self._steps:
            step(carry, item)


def chain[C](*steps: Step[C], name: str | None = None) -> Step[C]:
    """Run ``steps`` in order, each mutating the same carry."""
    return _Chain(*steps, name=name)


class _Repeat(Step[C]):
    def __init__(
        self,
        step: Step[C],
        *,
        times: int | Callable[[C], int],
        pre: Step[C] | None,
        post: Step[C] | None,
        name: str | None,
    ) -> None:
        self.name = _default_name("repeat", name)
        self._step = step
        self._times = times
        self._pre = pre
        self._post = post

    def __call__(self, carry: C, item: Any = None) -> None:
        del item
        total = self._times(carry) if callable(self._times) else self._times
        if self._pre is not None:
            self._pre(carry)
        for index in range(total):
            self._step(carry, (index, total))
        if self._post is not None:
            self._post(carry)


def repeat[C](
    step: Step[C],
    *,
    times: int | Callable[[C], int],
    pre: Step[C] | None = None,
    post: Step[C] | None = None,
    name: str | None = None,
) -> Step[C]:
    """Run ``step`` ``times`` in a row, with optional pre/post hooks.

    ``times`` is re-evaluated each time the returned step is invoked when it is
    a callable. The child step receives the current ``(index, total)`` as its
    ``item`` on each iteration.
    """
    return _Repeat(step, times=times, pre=pre, post=post, name=name)


class _When(Step[C]):
    def __init__(
        self,
        predicate: Callable[[C], bool],
        *,
        then: Step[C],
        else_: Step[C] | None,
        name: str | None,
    ) -> None:
        self.name = _default_name("when", name)
        self._predicate = predicate
        self._then = then
        self._else = else_

    def __call__(self, carry: C, item: Any = None) -> None:
        if self._predicate(carry):
            self._then(carry, item)
        elif self._else is not None:
            self._else(carry, item)


def when[C](
    predicate: Callable[[C], bool],
    *,
    then: Step[C],
    else_: Step[C] | None = None,
    name: str | None = None,
) -> Step[C]:
    """Run ``then`` when ``predicate(carry)`` is true, otherwise ``else_`` if given."""
    return _When(predicate, then=then, else_=else_, name=name)


class _WithIndex(Step[C]):
    """Set the loop index from an incoming ``(index, total)`` item, then run a body."""

    def __init__(
        self,
        body: Step[C],
        *,
        set_index: Callable[[C, int, int], None],
        name: str | None = None,
    ) -> None:
        self.name = _default_name("with_index", name)
        self._body = body
        self._set_index = set_index

    def __call__(self, carry: C, item: Any = None) -> None:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(f"with_index expected a 2-tuple (index, total), got {item!r}")
        index, total = item
        self._set_index(carry, index, total)
        self._body(carry)


def with_index[C](
    body: Step[C],
    *,
    set_index: Callable[[C, int, int], None],
    name: str | None = None,
) -> Step[C]:
    """Run ``body`` after writing the current ``(index, total)`` into the carry."""
    return _WithIndex(body, set_index=set_index, name=name)


class _Swap(Step[C]):
    """Swap a double-buffered target selected from the carry."""

    def __init__(self, swapper: Callable[[C], Any], *, name: str | None = None) -> None:
        self.name = _default_name("swap", name)
        self._swapper = swapper

    def __call__(self, carry: C, item: Any = None) -> None:
        del item
        self._swapper(carry).swap()


def swap[C](swapper: Callable[[C], Any], *, name: str | None = None) -> Step[C]:
    """Call ``swapper(carry).swap()`` as a standalone step."""
    return _Swap(swapper, name=name)


class _Foreach(Step[C]):
    def __init__(
        self,
        *steps: Callable[[C, Any], None],
        source: Callable[[C], Iterable[Any]],
        name: str | None,
    ) -> None:
        self.name = _default_name("foreach", name)
        self._steps = steps
        self._source = source

    def __call__(self, carry: C, _item: Any = None) -> None:
        for item in self._source(carry):
            for step in self._steps:
                step(carry, item)


def foreach[C](
    *steps: Callable[[C, Any], None],
    source: Callable[[C], Iterable[Any]],
    name: str | None = None,
) -> Step[C]:
    """Iterate over ``source(carry)`` and run ``steps`` for each item.

    Each ``step`` is called as ``step(carry, item)``; the item is passed
    explicitly so steps do not rely on mutable carry state.
    """
    return _Foreach(*steps, source=source, name=name)


class _Sample(Step[C]):
    def __init__(
        self,
        step: Step[C],
        *,
        every: datetime.timedelta,
        clock: Callable[[C], datetime.timedelta],
        key: str,
        cache: Callable[[C], dict[str, Any]],
        name: str | None,
    ) -> None:
        self.name = _default_name("sample", name)
        self._step = step
        self._every = every
        self._clock = clock
        self._key = key
        self._cache = cache
        if not isinstance(every, datetime.timedelta):
            raise TypeError("sample 'every' must be a datetime.timedelta")
        if every <= datetime.timedelta(0):
            raise ValueError("sample 'every' must be positive")

    def __call__(self, carry: C, item: Any = None) -> None:
        cache = self._cache(carry)
        clock_value = self._clock(carry)
        firing = clock_value % self._every == datetime.timedelta(0)
        if firing or self._key not in cache:
            self._step(carry, item)


def sample[C](
    step: Step[C],
    *,
    every: datetime.timedelta,
    clock: Callable[[C], datetime.timedelta],
    key: str,
    cache: Callable[[C], dict[str, Any]],
    name: str | None = None,
) -> Step[C]:
    """Run ``step`` only when the clock fires or the cache entry is missing.

    ``firing`` is ``clock(carry) % every == 0``. On a firing step (or the first
    in-window step with no cached value) ``step`` is expected to write a fresh
    value into ``cache(carry)[key]``; otherwise the cached value is left in
    place for downstream readers.
    """
    return _Sample(step, every=every, clock=clock, key=key, cache=cache, name=name)


class _Nested(Step[OuterC]):
    """Embed a composition over ``InnerC`` into a composition over ``OuterC``."""

    def __init__(
        self,
        step: Step[InnerC],
        *,
        enter: Callable[[OuterC], InnerC],
        name: str | None,
    ) -> None:
        self.name = _default_name("nested", name)
        self._step = step
        self._enter = enter

    def __call__(self, carry: OuterC, item: Any = None) -> None:
        self._step(self._enter(carry), item)


def nested[OuterC, InnerC](
    step: Step[InnerC],
    *,
    enter: Callable[[OuterC], InnerC],
    name: str | None = None,
) -> Step[OuterC]:
    """Run ``step`` on an inner carry derived from the outer carry.

    This lets a sub-composition with a different carry type (e.g. the physics
    loop over ``PhysicsLoopState``) appear as a single step in the outer
    composition while remaining introspectable as a tree.
    """
    return _Nested(step, enter=enter, name=name)
