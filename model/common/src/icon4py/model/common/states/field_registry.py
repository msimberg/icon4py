# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Field registry for declared model state and static-field containers.

The registry collects recipes for static fields, adopted external buffers, and
container schemas declared through ``spec()`` metadata. After ``seal()`` resolves
the recipe DAG and validates the declared boundaries, ``build(ContainerClass)``
emits frozen dataclass instances that share one buffer per quantity and time
level.
"""

from __future__ import annotations

import dataclasses
import typing
from collections.abc import Callable, Sequence
from typing import Any

import gt4py.next as gtx

from icon4py.model.common import dimension as dims_module, model_backends, type_alias as ta
from icon4py.model.common.grid import base as base_grid
from icon4py.model.common.states import spec
from icon4py.model.common.states.spec import get_field_spec
from icon4py.model.common.utils import TimeStepPair, data_allocation as data_alloc


@dataclasses.dataclass
class _Recipe:
    name: str
    compute_fn: Callable[[], Any]
    depends_on: tuple[str, ...]


@dataclasses.dataclass
class _Slot:
    """A shared slot for one quantity (and optional time level)."""

    quantity: str
    time_level: str | None
    buffer: Any = None


@dataclasses.dataclass
class _Occurrence:
    """One declared appearance of a quantity inside a container."""

    container: str
    field: str
    quantity: str
    spec: spec.FieldSpec
    slot: _Slot
    producer: bool = False
    consumer: bool = False
    static: bool = False
    diagnostic: bool = False
    time_level: str | None = None
    is_container: bool = False


@dataclasses.dataclass
class _Context:
    producer: bool = False
    consumer: bool = False
    static: bool = False
    diagnostic: bool = False
    time_level: str | None = None

    def evolve(self, **changes: Any) -> _Context:
        return dataclasses.replace(self, **changes)


def _is_container(cls: type) -> bool:
    """Return whether ``cls`` is a dataclass that carries ``spec()`` metadata."""
    if not dataclasses.is_dataclass(cls):
        return False
    return any(get_field_spec(f) is not None for f in dataclasses.fields(cls))


def _is_diagnostic_container(cls: Any) -> bool:
    """Return whether ``cls`` names a diagnostic container (e.g. DiagnosticState*)."""
    origin = typing.get_origin(cls)
    if origin is not None:
        cls = origin
    try:
        return "Diagnostic" in cls.__name__
    except AttributeError:
        return False


def _is_time_step_pair(cls: Any) -> bool:
    """Return whether ``cls`` is (or parametrizes) ``TimeStepPair``."""
    origin = typing.get_origin(cls)
    if origin is not None:
        return origin is TimeStepPair
    return cls is TimeStepPair


def _resolve_type(container_cls: type, field_name: str) -> Any:
    """Return the resolved annotation for ``field_name`` on ``container_cls``."""
    try:
        hints = typing.get_type_hints(container_cls)
        return hints.get(field_name, Any)
    except Exception:
        # Fall back to the raw annotation string if resolution fails.
        for field in dataclasses.fields(container_cls):
            if field.name == field_name:
                return field.type
        return Any


def _make_guarded_class(container_cls: type, registry: FieldRegistry) -> type:
    """Return a subclass of ``container_cls`` that checks epoch on field access."""
    field_names = {f.name for f in dataclasses.fields(container_cls)}

    class _Guarded(container_cls):
        def __getattribute__(self, name: str) -> Any:
            if name in field_names:
                cls = object.__getattribute__(self, "__class__")
                reg = cls._registry
                epoch = object.__getattribute__(self, "_field_registry_epoch")
                if epoch != reg._epoch:
                    raise ValueError(
                        f"Stale epoch for field {name!r} of {container_cls.__name__}: "
                        f"container was built at epoch {epoch}, registry is at epoch {reg._epoch}."
                    )
            return object.__getattribute__(self, name)

    _Guarded._registry = registry
    return _Guarded


class FieldRegistry:
    """Registry of static-field recipes and declared container schemas."""

    def __init__(
        self,
        grid: base_grid.Grid,
        *,
        backend: gtx.typing.Backend | None = None,
        allocator: gtx.typing.Allocator | None = None,
    ) -> None:
        self._grid = grid
        self._backend = backend
        self._allocator = (
            allocator if allocator is not None else model_backends.get_allocator(backend)
        )
        self._recipes: dict[str, _Recipe] = {}
        self._adopted: dict[str, Any] = {}
        self._declared_classes: list[type] = []
        self._occurrences: list[_Occurrence] = []
        self._slots: dict[tuple[str, str | None], _Slot] = {}
        self._buffers: dict[str, Any] = {}
        self._persistent_allocated: set[tuple[str, str | None]] = set()
        self._sealed = False
        self._epoch = 0
        self._guard_classes: dict[type, type] = {}

    @property
    def epoch(self) -> int:
        """Current epoch; bumps when a buffer identity changes."""
        return self._epoch

    def recipe(
        self,
        name: str,
        compute_fn: Callable[[], Any],
        depends_on: Sequence[str] = (),
    ) -> None:
        """Register how a static field is computed."""
        if self._sealed:
            raise RuntimeError("Cannot register a recipe after seal().")
        if name in self._recipes or name in self._adopted:
            raise ValueError(f"Duplicate recipe or adoption for quantity '{name}'.")
        self._recipes[name] = _Recipe(name, compute_fn, tuple(depends_on))

    def adopt(self, name: str, field: Any) -> None:
        """Adopt an externally owned buffer as a declared leaf."""
        if self._sealed:
            raise RuntimeError("Cannot adopt a field after seal().")
        if name in self._recipes or name in self._adopted:
            raise ValueError(f"Duplicate recipe or adoption for quantity '{name}'.")
        self._adopted[name] = field

    def declare(self, container_class: type) -> None:
        """Collect the schema of a frozen dataclass with ``spec()`` metadata."""
        if self._sealed:
            raise RuntimeError("Cannot declare a container after seal().")
        if not dataclasses.is_dataclass(container_class):
            raise TypeError(f"Declared container {container_class.__name__} must be a dataclass.")
        self._declared_classes.append(container_class)
        diagnostic = "Diagnostic" in container_class.__name__
        self._walk_fields(
            container_class,
            _Context(diagnostic=diagnostic),
            (container_class.__name__,),
        )

    def _walk_fields(
        self,
        container_cls: type,
        context: _Context,
        path: tuple[str, ...],
    ) -> None:
        for field in dataclasses.fields(container_cls):
            field_spec = get_field_spec(field)
            field_type = _resolve_type(container_cls, field.name)

            if field_spec is not None:
                time_level = context.time_level
                if _is_time_step_pair(field_type):
                    time_level = field.name
                slot = self._get_slot(field_spec.quantity, time_level)
                is_diagnostic = context.diagnostic or field_spec.role == spec.Role.DIAGNOSTIC
                is_static = context.static or field_spec.lifetime == spec.Lifetime.STATIC
                is_producer = not is_diagnostic and (
                    context.producer
                    or (field_spec.role is not None and field_spec.role != spec.Role.DIAGNOSTIC)
                )
                is_consumer = not is_diagnostic and (
                    context.consumer
                    or (field_spec.role is None and field_spec.intent == spec.Intent.READ)
                )
                self._occurrences.append(
                    _Occurrence(
                        container=path[0],
                        field=".".join((*path[1:], field.name)),
                        quantity=field_spec.quantity,
                        spec=field_spec,
                        slot=slot,
                        producer=is_producer,
                        consumer=is_consumer,
                        static=is_static,
                        diagnostic=is_diagnostic,
                        time_level=time_level,
                        is_container=_is_container(field_type),
                    )
                )

            child_context = context
            if field_spec is not None:
                child_producer = context.producer or (
                    field_spec.role is not None and field_spec.role != spec.Role.DIAGNOSTIC
                )
                child_consumer = context.consumer or (
                    field_spec.role is None and field_spec.intent == spec.Intent.READ
                )
                child_diagnostic = (
                    context.diagnostic
                    or field_spec.role == spec.Role.DIAGNOSTIC
                    or _is_diagnostic_container(field_type)
                )
                child_static = context.static or field_spec.lifetime == spec.Lifetime.STATIC
                child_time_level = context.time_level
                if _is_time_step_pair(field_type):
                    child_time_level = field.name
                child_context = _Context(
                    producer=child_producer,
                    consumer=child_consumer,
                    static=child_static,
                    diagnostic=child_diagnostic,
                    time_level=child_time_level,
                )

            if _is_container(field_type):
                self._walk_fields(field_type, child_context, (*path, field.name))

    def _get_slot(self, quantity: str, time_level: str | None) -> _Slot:
        key = (quantity, time_level)
        if key not in self._slots:
            self._slots[key] = _Slot(quantity=quantity, time_level=time_level)
        return self._slots[key]

    def seal(self) -> None:
        """Validate declarations, resolve recipes, and check handoff contracts."""
        if self._sealed:
            raise RuntimeError("seal() has already been called.")

        self._validate_specs()
        self._resolve_recipes()
        self._check_handoffs()
        self._sealed = True

    def _validate_specs(self) -> None:
        """Ensure one quantity has one consistent (dims, units) and statics are known."""
        by_quantity: dict[str, dict[tuple[tuple[Any, ...], str], set[str]]] = {}
        for occurrence in self._occurrences:
            if occurrence.quantity not in by_quantity:
                by_quantity[occurrence.quantity] = {}
            key = (occurrence.spec.dims, occurrence.spec.units)
            by_quantity[occurrence.quantity].setdefault(key, set()).add(occurrence.container)

        for quantity, declarations in by_quantity.items():
            if len(declarations) > 1:
                details = "; ".join(
                    f"dims={dims}, units={units!r} in {', '.join(sorted(containers))}"
                    for (dims, units), containers in sorted(
                        declarations.items(), key=lambda item: str(item[0])
                    )
                )
                raise ValueError(f"Quantity '{quantity}' has inconsistent declarations: {details}")

        for occurrence in self._occurrences:
            if occurrence.static and occurrence.quantity not in self._recipes:
                if occurrence.quantity not in self._adopted:
                    raise ValueError(
                        f"Static quantity '{occurrence.quantity}' (declared in "
                        f"{occurrence.container}.{occurrence.field}) has no recipe or adopted field."
                    )

    def _resolve_recipes(self) -> None:
        """Topologically sort static recipes and compute each once."""
        recipe_names = list(self._recipes.keys())
        # Validate that all declared recipe dependencies exist.
        for name in recipe_names:
            recipe = self._recipes[name]
            for dep in recipe.depends_on:
                if dep not in self._recipes and dep not in self._adopted:
                    raise ValueError(f"Recipe for '{name}' is missing dependency '{dep}'.")

        # Kahn's algorithm for the declared recipe graph.
        in_degree: dict[str, int] = {}
        dependents: dict[str, list[str]] = {name: [] for name in recipe_names}
        for name in recipe_names:
            deps = [d for d in self._recipes[name].depends_on if d in self._recipes]
            in_degree[name] = len(deps)
            for dep in deps:
                dependents.setdefault(dep, []).append(name)

        ready = [name for name in recipe_names if in_degree.get(name, 0) == 0]
        ordered: list[str] = []
        while ready:
            name = ready.pop()
            ordered.append(name)
            for dependent in dependents.get(name, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    ready.append(dependent)

        unresolved = set(recipe_names) - set(ordered)
        if unresolved:
            # Report the first quantity in a cycle deterministically.
            raise ValueError(
                f"Recipe cycle or unresolved dependency involving quantities: "
                f"{', '.join(sorted(unresolved))}."
            )

        for name in ordered:
            self._buffers[name] = self._recipes[name].compute_fn()

        for name, field in self._adopted.items():
            self._buffers[name] = field

    def _check_handoffs(self) -> None:
        """Verify exactly-one-producer / exactly-one-consumer for handoff quantities."""
        by_quantity: dict[str, list[_Occurrence]] = {}
        for occurrence in self._occurrences:
            by_quantity.setdefault(occurrence.quantity, []).append(occurrence)

        for quantity, occurrences in by_quantity.items():
            relevant = [
                o
                for o in occurrences
                if not o.static
                and not o.diagnostic
                and o.spec.lifetime != spec.Lifetime.SCRATCH
                and o.time_level is None
                and not o.is_container
            ]
            if not relevant:
                continue
            producers = [o for o in relevant if o.producer]
            consumers = [o for o in relevant if o.consumer]

            # A quantity that is declared as a handoff must have exactly one
            # producer and exactly one consumer, both using the same buffer.
            # If neither side is declared, it is not a handoff quantity.
            if not producers and not consumers:
                continue

            if len(producers) == 0:
                raise ValueError(
                    f"Handoff quantity '{quantity}' has no producer (declared consumer in "
                    f"{consumers[0].container}.{consumers[0].field})."
                )
            if len(producers) > 1:
                sites = ", ".join(sorted(f"{o.container}.{o.field}" for o in producers))
                raise ValueError(f"Handoff quantity '{quantity}' has multiple producers: {sites}.")
            if len(consumers) == 0:
                raise ValueError(
                    f"Handoff quantity '{quantity}' has no consumer (declared producer in "
                    f"{producers[0].container}.{producers[0].field})."
                )
            if len(consumers) > 1:
                sites = ", ".join(sorted(f"{o.container}.{o.field}" for o in consumers))
                raise ValueError(f"Handoff quantity '{quantity}' has multiple consumers: {sites}.")
            if producers[0].slot is not consumers[0].slot:
                raise ValueError(
                    f"Handoff quantity '{quantity}' is produced and consumed from different buffers "
                    f"({producers[0].container}.{producers[0].field} vs "
                    f"{consumers[0].container}.{consumers[0].field})."
                )

    def build(self, container_class: type, config: Any | None = None) -> Any:
        """Emit a frozen dataclass instance with registry-managed buffers."""
        if not self._sealed:
            raise RuntimeError("build() cannot be called before seal().")

        guarded_cls = self._guard_classes.get(container_class)
        if guarded_cls is None:
            guarded_cls = _make_guarded_class(container_class, self)
            self._guard_classes[container_class] = guarded_cls

        kwargs = self._build_kwargs(container_class, config=config)
        instance = guarded_cls(**kwargs)
        object.__setattr__(instance, "_field_registry_epoch", self._epoch)
        object.__setattr__(instance, "_field_registry_generation", 0)
        return instance

    def _build_kwargs(
        self,
        container_class: type,
        config: Any | None,
        time_level: str | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        for field in dataclasses.fields(container_class):
            field_spec = get_field_spec(field)
            field_type = _resolve_type(container_class, field.name)

            if field_spec is not None:
                slot = self._get_slot(field_spec.quantity, time_level)
                if field_spec.lifetime == spec.Lifetime.STATIC:
                    if field_spec.quantity in self._buffers:
                        kwargs[field.name] = self._buffers[field_spec.quantity]
                        continue
                    # Optional static fields may stay at their declared default.
                    if field.default is not dataclasses.MISSING:
                        kwargs[field.name] = field.default
                        continue
                    raise ValueError(
                        f"Static field '{field.name}' of {container_class.__name__} "
                        f"(quantity '{field_spec.quantity}') has no computed buffer."
                    )

                if field_spec.lifetime == spec.Lifetime.PERSISTENT:
                    if self._is_inactive_tracer(field, config):
                        kwargs[field.name] = None
                        continue
                    if slot.buffer is None:
                        slot.buffer = self._allocate_persistent(field_spec, time_level)
                        self._persistent_allocated.add((field_spec.quantity, time_level))
                    kwargs[field.name] = slot.buffer
                    continue

                # Scratch fields are not part of state containers.
                raise ValueError(
                    f"Field '{field.name}' of {container_class.__name__} has unsupported "
                    f"lifetime '{field_spec.lifetime.value}' for build()."
                )

            if _is_container(field_type):
                child_time_level = time_level
                if _is_time_step_pair(field_type):
                    child_time_level = field.name
                kwargs[field.name] = self._build_nested(
                    field_type, config=config, time_level=child_time_level
                )
            elif field.default is not dataclasses.MISSING:
                kwargs[field.name] = field.default
            else:
                kwargs[field.name] = None

        return kwargs

    def _build_nested(
        self,
        container_class: type,
        config: Any | None,
        time_level: str | None,
    ) -> Any:
        """Build a nested container and wrap it with the epoch guard."""
        guarded_cls = self._guard_classes.get(container_class)
        if guarded_cls is None:
            guarded_cls = _make_guarded_class(container_class, self)
            self._guard_classes[container_class] = guarded_cls
        kwargs = self._build_kwargs(container_class, config=config, time_level=time_level)
        instance = guarded_cls(**kwargs)
        object.__setattr__(instance, "_field_registry_epoch", self._epoch)
        object.__setattr__(instance, "_field_registry_generation", 0)
        return instance

    def _is_inactive_tracer(self, field: dataclasses.Field, config: Any | None) -> bool:
        """Return True for optional tracer fields that are disabled by the config."""
        if config is None:
            return False
        # TracerConfig is defined in icon4py.model.common.states.tracer_states.
        # Avoid importing it here to keep the registry generic; check by duck typing.
        if hasattr(config, "active_names") and hasattr(config, "__contains__"):
            if field.default is None and field.name not in config.active_names:
                return True
        return False

    def _allocate_persistent(
        self,
        field_spec: spec.FieldSpec,
        time_level: str | None,
    ) -> Any:
        """Allocate a zero-filled persistent field from the declared dims."""
        dims = list(field_spec.dims)
        extend: dict[gtx.Dimension, int] | None = None
        if dims and dims[-1] == dims_module.KHalfDim:
            dims[-1] = dims_module.KDim
            extend = {dims[-1]: 1}

        # Use the default working-precision float for persistent state fields.
        dtype = ta.wpfloat
        if not dims:
            raise ValueError(f"Cannot allocate persistent scalar quantity '{field_spec.quantity}'.")

        return data_alloc.zero_field(
            self._grid,
            *dims,
            allocator=self._allocator,
            dtype=dtype,
            extend=extend,
        )

    def buffer(self, name: str) -> Any:
        """Return the computed buffer for a static quantity after ``seal()``."""
        if not self._sealed:
            raise RuntimeError("buffer() cannot be called before seal().")
        if name not in self._buffers:
            raise ValueError(f"Quantity '{name}' has no computed or adopted buffer.")
        return self._buffers[name]

    def bump_epoch(self) -> None:
        """Bump the epoch; existing registry-emitted containers become stale."""
        self._epoch += 1

    def bump_generation(self, container: Any) -> None:
        """Bump the generation counter on a container; does not invalidate access."""
        generation = object.__getattribute__(container, "_field_registry_generation")
        object.__setattr__(container, "_field_registry_generation", generation + 1)

    def check_epoch(self, container: Any) -> None:
        """Raise if the container was emitted before the current epoch."""
        epoch = object.__getattribute__(container, "_field_registry_epoch")
        if epoch != self._epoch:
            raise ValueError(
                f"Container {type(container).__name__} was built at epoch {epoch}, "
                f"but the registry is at epoch {self._epoch}."
            )
