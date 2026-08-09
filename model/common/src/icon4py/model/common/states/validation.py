# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Setup-time validation of declared field metadata."""

from __future__ import annotations

import ast
import dataclasses
import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from icon4py.model.common.states.spec import FieldSpec, get_field_spec


def validate_consistent_specs(
    field_specs_by_container: Mapping[str, Mapping[str, FieldSpec]],
    *,
    known_quantities: set[str] | None = None,
) -> None:
    """Validate that each quantity has one consistent ``(dims, units)`` pair.

    Args:
        field_specs_by_container: mapping from container name to a mapping from
            field name to the declared ``FieldSpec``.
        known_quantities: optional set of canonical quantity names. If provided,
            any quantity not in this set raises.

    Raises:
        ValueError: if a quantity is unknown or has inconsistent ``dims``/``units``.
    """
    by_quantity: dict[str, dict[tuple[tuple[Any, ...], str], set[str]]] = {}
    for container_name, fields in field_specs_by_container.items():
        for field_name, field_spec in fields.items():
            if known_quantities is not None and field_spec.quantity not in known_quantities:
                raise ValueError(
                    f"Unknown quantity '{field_spec.quantity}' in {container_name}.{field_name}"
                )
            key = (field_spec.dims, field_spec.units)
            by_quantity.setdefault(field_spec.quantity, {}).setdefault(key, set()).add(
                container_name
            )

    for quantity, declarations in by_quantity.items():
        if len(declarations) > 1:
            details = "; ".join(
                f"dims={dims}, units={units!r} in {', '.join(sorted(containers))}"
                for (dims, units), containers in sorted(
                    declarations.items(), key=lambda item: str(item[0])
                )
            )
            raise ValueError(f"Quantity '{quantity}' has inconsistent declarations: {details}")


def assert_field_coverage(target_class: type, kwargs: Mapping[str, Any]) -> None:
    """Assert that the keyword names exactly match the target class's constructor.

    Supports dataclasses (via ``dataclasses.fields``) and regular classes with a
    plain ``__init__`` (via ``inspect.signature``).

    Raises:
        ValueError: if a field is missing or an unexpected keyword is present.
    """
    if dataclasses.is_dataclass(target_class):
        expected = {f.name for f in dataclasses.fields(target_class)}
    else:
        sig = inspect.signature(target_class)
        expected = {
            p.name
            for p in sig.parameters.values()
            if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        }
    actual = set(kwargs.keys())
    if actual != expected:
        missing = expected - actual
        extra = actual - expected
        raise ValueError(
            f"Field coverage mismatch for {target_class.__name__}: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )


def field_specs_from_container(container: Any) -> dict[str, FieldSpec]:
    """Return the declared ``FieldSpec`` for each field of a container."""
    result: dict[str, FieldSpec] = {}
    for field in dataclasses.fields(container):
        field_spec = get_field_spec(field)
        if field_spec is not None:
            result[field.name] = field_spec
    return result


def _name_of(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def kwargs_at_constructor_calls(
    func: Any,
    target_classes: tuple[type, ...],
) -> dict[str, set[str]]:
    """Return keyword names passed to each target class constructor inside ``func``.

    The result maps target class name to the set of keyword argument names used
    at each call site (combined into one set).
    """
    source = inspect.getsource(func)
    tree = ast.parse(source)
    target_names = {cls.__name__ for cls in target_classes}
    result: dict[str, set[str]] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _name_of(node.func)
            if name in target_names:
                result.setdefault(name, set()).update(
                    kw.arg for kw in node.keywords if kw.arg is not None
                )

    return result


def read_kwargs_at_constructor_calls(
    source_path: Path,
    function_name: str,
    target_classes: tuple[type, ...],
) -> dict[str, set[str]]:
    """Like ``kwargs_at_constructor_calls`` but reads the function from a file."""
    source = Path(source_path).read_text()
    tree = ast.parse(source)
    target_names = {cls.__name__ for cls in target_classes}
    result: dict[str, set[str]] = {}

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    name = _name_of(child.func)
                    if name in target_names:
                        result.setdefault(name, set()).update(
                            kw.arg for kw in child.keywords if kw.arg is not None
                        )

    return result
