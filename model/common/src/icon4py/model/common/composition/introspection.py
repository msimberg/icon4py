# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Introspection utilities for composition trees and dataflow graphs."""

from __future__ import annotations

import dataclasses
import datetime
import typing
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from icon4py.model.common.components.components import Component
from icon4py.model.common.composition.combinators import (
    _Chain,
    _Foreach,
    _NamedStep,
    _Nested,
    _Repeat,
    _Sample,
    _When,
    _WithIndex,
)
from icon4py.model.common.composition.step import Step
from icon4py.model.common.states.spec import get_field_spec


def _is_combinator(step: Step[Any]) -> bool:
    return isinstance(
        step,
        _Chain | _Repeat | _When | _Foreach | _Sample | _Nested | _WithIndex,
    )


def _children(step: Step[Any]) -> Iterable[Step[Any]]:
    """Return immediate child steps that are themselves introspectable.

    ``_Foreach`` holds arbitrary callables; only those that are ``Step``
    instances carry the metadata needed to derive the dataflow graph.
    """
    children: list[Step[Any]] = []
    if isinstance(step, _Chain):
        children.extend(step._steps)
    elif isinstance(step, _Repeat):
        if step._pre is not None:
            children.append(step._pre)
        children.append(step._step)
        if step._post is not None:
            children.append(step._post)
    elif isinstance(step, _When):
        children.append(step._then)
        if step._else is not None:
            children.append(step._else)
    elif isinstance(step, _Foreach):
        children.extend(s for s in step._steps if isinstance(s, _NamedStep))
    elif isinstance(step, _Sample | _Nested):
        children.append(step._step)
    elif isinstance(step, _WithIndex):
        children.append(step._body)
    return children


def _cadence_label(step: Step[Any]) -> str:
    if isinstance(step, _Repeat):
        times = step._times
        times_text = f"{times} times" if isinstance(times, int) else "dynamic times"
        return f" [repeat {times_text}]"
    if isinstance(step, _Sample):
        every = step._every
        if isinstance(every, datetime.timedelta):
            return f" [sample every {every.total_seconds()}s]"
        return f" [sample every {every}]"
    return ""


def show(step: Step[Any], *, indent: str = "") -> str:
    """Return a text tree of the composition rooted at ``step``."""
    lines = [f"{indent}{step.name}{_cadence_label(step)}"]
    if isinstance(step, _NamedStep) and step.component is not None:
        component = step.component
        lines[-1] += f"  ({component.__class__.__name__})"
    for child in _children(step):
        lines.extend(show(child, indent=indent + "  ").split("\n"))
    return "\n".join(lines)


def _component_of(step: Step[Any]) -> Component[Any, Any] | None:
    if isinstance(step, _NamedStep):
        return step.component
    return None


def _collect_component_steps(step: Step[Any]) -> list[Step[Any]]:
    """Collect leaves that carry component metadata."""
    results: list[Step[Any]] = []
    if _component_of(step) is not None:
        results.append(step)
    for child in _children(step):
        results.extend(_collect_component_steps(child))
    return results


def _quantity_fields(cls: type[Any]) -> dict[str, str]:
    """Map leaf field name to canonical quantity name for a declared dataclass.

    Nested dataclass fields (e.g. ``prep_adv``) are flattened so their per-field
    quantities appear as individual dataflow nodes instead of container-level
    quantities.
    """
    result: dict[str, str] = {}
    try:
        hints = typing.get_type_hints(cls)
    except Exception:
        hints = {}
    for field in dataclasses.fields(cls):
        spec = get_field_spec(field)
        field_type = hints.get(field.name, field.type)
        if isinstance(field_type, type) and dataclasses.is_dataclass(field_type):
            result.update(_quantity_fields(field_type))
        elif spec is not None:
            result[field.name] = spec.quantity
    return result


def _component_label(step: Step[Any]) -> str:
    component = _component_of(step)
    assert component is not None
    return component.__class__.__name__


def _node_id(name: str) -> str:
    """Sanitize a name for use as a graphviz node ID."""
    return "n_" + "".join(c if c.isalnum() else "_" for c in name)


def _component_node_id(step: Step[Any]) -> str:
    return _node_id(f"comp_{step.name}_{id(step)}")


def _quantity_node_id(quantity: str) -> str:
    return _node_id(f"qty_{quantity}")


def to_graphviz(step: Step[Any]) -> str:
    """Return a graphviz dot string of the composition tree and dataflow graph."""
    lines = ["digraph composition {"]
    lines.append("  rankdir=LR;")
    lines.append("  node [shape=box];")

    # Composition tree subgraph
    lines.append("  subgraph cluster_composition {")
    lines.append('    label="composition tree";')
    lines.append("    color=blue;")
    lines.extend(_composition_tree_dot(step))
    lines.append("  }")

    # Dataflow graph subgraph
    lines.append("  subgraph cluster_dataflow {")
    lines.append('    label="dataflow graph";')
    lines.append("    color=green;")
    lines.extend(_dataflow_dot(step))
    lines.append("  }")

    lines.append("}")
    return "\n".join(lines)


def _composition_tree_dot(
    step: Step[Any],
    parent_id: str | None = None,
    edges: list[str] | None = None,
    nodes: dict[str, str] | None = None,
) -> list[str]:
    if edges is None:
        edges = []
    if nodes is None:
        nodes = {}
    node_id = _node_id(f"tree_{step.name}_{id(step)}")
    label = step.name + _cadence_label(step)
    if isinstance(step, _NamedStep) and step.component is not None:
        label += f"\\n({_component_label(step)})"
    nodes[node_id] = label
    if parent_id is not None:
        edges.append(f"    {parent_id} -> {node_id};")
    for child in _children(step):
        _composition_tree_dot(child, parent_id=node_id, edges=edges, nodes=nodes)
    if parent_id is not None:
        return []
    return [f'    {node_id} [label="{label}"];' for node_id, label in nodes.items()] + edges


def _dataflow_dot(step: Step[Any]) -> list[str]:
    component_steps = _collect_component_steps(step)

    reads: dict[str, list[Step[Any]]] = defaultdict(list)
    writes: dict[str, list[Step[Any]]] = defaultdict(list)

    for comp_step in component_steps:
        component = _component_of(comp_step)
        assert component is not None
        input_type = component.input_type()
        output_type = component.output_type()
        for quantity in _quantity_fields(input_type).values():
            reads[quantity].append(comp_step)
        for quantity in _quantity_fields(output_type).values():
            writes[quantity].append(comp_step)

    quantities = sorted(set(reads.keys()) | set(writes.keys()))

    lines: list[str] = []
    node_ids: set[str] = set()

    for comp_step in component_steps:
        node_id = _component_node_id(comp_step)
        if node_id not in node_ids:
            node_ids.add(node_id)
            lines.append(f'    {node_id} [label="{_component_label(comp_step)}"];')

    for quantity in quantities:
        qty_id = _quantity_node_id(quantity)
        if qty_id not in node_ids:
            node_ids.add(qty_id)
            lines.append(f'    {qty_id} [label="{quantity}", shape=ellipse];')

    for quantity in quantities:
        qty_id = _quantity_node_id(quantity)
        for producer in writes.get(quantity, []):
            prod_id = _component_node_id(producer)
            lines.append(f"    {prod_id} -> {qty_id};")
        for consumer in reads.get(quantity, []):
            cons_id = _component_node_id(consumer)
            lines.append(f"    {qty_id} -> {cons_id};")

    return lines
