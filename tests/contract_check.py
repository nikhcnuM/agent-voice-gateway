"""Test helper that loads canonical contracts from agent-bus/contracts.

Each downstream repo keeps its own test helper instead of importing the
``agent_bus`` package: the workspace is a multi-repo, not a monorepo, so we
prefer to read the canonical JSON files (schemas + fixtures) directly.

The validator implements only the JSON Schema subset shipped in the
``agent-bus/contracts/schemas`` directory: ``type`` (string or list, including
``"null"``), ``required``, ``properties``, ``additionalProperties``,
``items``, ``enum`` and ``minLength``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONTRACTS_DIR = Path(__file__).resolve().parents[2] / "agent-bus" / "contracts"
REGISTRY_PATH = CONTRACTS_DIR / "event-types.json"


class ContractError(AssertionError):
    def __init__(self, reason: str, *, field: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.field = field


def _registry() -> dict[str, Any]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def known_types() -> dict[str, dict[str, Any]]:
    return _registry()["types"]


def load_fixture(type_name: str) -> dict[str, Any]:
    return json.loads((CONTRACTS_DIR / "fixtures" / f"{type_name}.json").read_text(encoding="utf-8"))


def load_schema(type_name: str) -> dict[str, Any]:
    return json.loads((CONTRACTS_DIR / "schemas" / f"{type_name}.json").read_text(encoding="utf-8"))


def validate_envelope(envelope: dict[str, Any]) -> None:
    registry = _registry()
    for required in registry.get("envelope_required", []):
        if required not in envelope or (envelope[required] is None and required != "payload"):
            raise ContractError("envelope field missing", field=required)
    type_name = envelope.get("type")
    if not isinstance(type_name, str) or not type_name:
        raise ContractError("envelope type must be a non-empty string", field="type")
    types = registry["types"]
    if type_name not in types:
        raise ContractError(f"unknown type: {type_name}", field="type")
    info = types[type_name]
    if info.get("target_required") and not isinstance(envelope.get("target"), str):
        raise ContractError(
            f"target is required for command type {type_name}",
            field="target",
        )
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise ContractError("payload must be an object", field="payload")
    schema = load_schema(type_name)
    _validate(payload, schema, path="payload")


_BASIC_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
    "null": type(None),
}


def _validate(value: Any, schema: dict[str, Any], *, path: str) -> None:
    expected = schema.get("type")
    if expected is not None and not _matches(value, expected):
        raise ContractError(f"{path} must be of type {expected}", field=path)
    if "enum" in schema and value not in schema["enum"]:
        raise ContractError(f"{path} must be one of {schema['enum']}", field=path)
    if isinstance(value, str):
        min_length = schema.get("minLength")
        if isinstance(min_length, int) and len(value) < min_length:
            raise ContractError(f"{path} must have minLength {min_length}", field=path)
    if isinstance(value, dict):
        for required in schema.get("required") or []:
            if required not in value or value[required] is None:
                raise ContractError(f"{path}.{required} is required", field=f"{path}.{required}")
        for key, sub in (schema.get("properties") or {}).items():
            if key in value:
                _validate(value[key], sub, path=f"{path}.{key}")
        if schema.get("additionalProperties") is False:
            allowed = set((schema.get("properties") or {}).keys())
            for key in value:
                if key not in allowed:
                    raise ContractError(f"{path}.{key} is not allowed", field=f"{path}.{key}")
    if isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(value):
                _validate(item, items, path=f"{path}[{index}]")


def _matches(value: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return any(_matches(value, item) for item in expected)
    py = _BASIC_TYPES.get(expected)
    if py is None:
        return True
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, py)
