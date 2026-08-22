"""Normalize a toolkit catalog and emit its dependency graph."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class FieldDef:
    name: str
    path: str
    type: str
    description: str
    entity: str


@dataclass(frozen=True)
class ToolDef:
    id: str
    description: str
    inputs: tuple[FieldDef, ...]
    required_inputs: frozenset[str]
    outputs: tuple[FieldDef, ...]
    tags: frozenset[str]
    deprecated: bool
    service: str | None


def load_catalog(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        tools = data
    elif isinstance(data, dict):
        tools = data.get("tools", data.get("items"))
    else:
        tools = None
    if not isinstance(tools, list):
        raise ValueError("catalog must be an array or an object containing a tools/items array")
    if not all(isinstance(tool, dict) for tool in tools):
        raise ValueError("every catalog tool must be an object")
    return tools


def _resolve_ref(schema: dict[str, Any], root: dict[str, Any]) -> tuple[dict[str, Any], str]:
    ref = schema.get("$ref")
    prefix = "#/$defs/"
    if not isinstance(ref, str) or not ref.startswith(prefix):
        return schema, ""
    name = ref.removeprefix(prefix)
    target = root.get("$defs", {}).get(name)
    return (target, name) if isinstance(target, dict) else (schema, name)


def _field_type(schema: dict[str, Any], root: dict[str, Any]) -> str:
    resolved, _ = _resolve_ref(schema, root)
    value = schema.get("type", resolved.get("type", "unknown"))
    return value if isinstance(value, str) else "unknown"


def _field_description(schema: dict[str, Any], root: dict[str, Any]) -> str:
    resolved, _ = _resolve_ref(schema, root)
    value = schema.get("description", resolved.get("description", ""))
    return value if isinstance(value, str) else ""


def _schema_fields(
    schema: dict[str, Any],
    root: dict[str, Any],
    path: tuple[str, ...] = (),
    entity: str = "",
    seen_refs: frozenset[str] = frozenset(),
) -> list[FieldDef]:
    resolved, ref_entity = _resolve_ref(schema, root)
    ref = schema.get("$ref")
    if ref_entity:
        if ref in seen_refs:
            return []
        return _schema_fields(resolved, root, path, ref_entity, seen_refs | {str(ref)})

    fields: list[FieldDef] = []
    for keyword in ("allOf", "anyOf", "oneOf"):
        variants = schema.get(keyword, [])
        if isinstance(variants, list):
            for variant in variants:
                if isinstance(variant, dict):
                    fields.extend(_schema_fields(variant, root, path, entity, seen_refs))

    properties = schema.get("properties", {})
    if isinstance(properties, dict):
        for name, child in properties.items():
            if not isinstance(child, dict):
                continue
            child_path = path + (name,)
            _, child_entity = _resolve_ref(child, root)
            fields.append(
                FieldDef(
                    name=name,
                    path=".".join(child_path),
                    type=_field_type(child, root),
                    description=_field_description(child, root),
                    entity=child_entity or entity,
                )
            )
            fields.extend(_schema_fields(child, root, child_path, child_entity or entity, seen_refs))

    items = schema.get("items")
    if isinstance(items, dict):
        array_path = path[:-1] + (f"{path[-1]}[]",) if path else ("[]",)
        fields.extend(_schema_fields(items, root, array_path, entity, seen_refs))

    return list(dict.fromkeys(fields))


def _tool_id(tool: dict[str, Any]) -> str:
    function = tool.get("function")
    candidates = (
        tool.get("slug"),
        tool.get("name"),
        function.get("name") if isinstance(function, dict) else None,
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    raise ValueError("every tool must have a non-empty tool id")


def _service(tags: frozenset[str]) -> str | None:
    metadata = {"graphql", "important", "mcpignore"}
    domain_tags = [
        tag for tag in tags if not tag.lower().endswith("hint") and tag.lower() not in metadata
    ]
    return domain_tags[0] if len(domain_tags) == 1 else None


def normalize_catalog(raw_tools: list[dict[str, Any]]) -> list[ToolDef]:
    tools: list[ToolDef] = []
    ids: set[str] = set()
    for raw in raw_tools:
        tool_id = _tool_id(raw)
        if tool_id in ids:
            raise ValueError(f"duplicate tool id: {tool_id}")
        ids.add(tool_id)

        input_schema = raw.get("inputParameters", {})
        output_schema = raw.get("outputParameters", {})
        if not isinstance(input_schema, dict) or not isinstance(output_schema, dict):
            raise ValueError(f"tool {tool_id} must have object input/output schemas")
        required = input_schema.get("required", [])
        if not isinstance(required, list) or not all(isinstance(name, str) for name in required):
            raise ValueError(f"tool {tool_id} has an invalid required-input list")
        tags = frozenset(tag for tag in raw.get("tags", []) if isinstance(tag, str))
        description = raw.get("description", "")
        tools.append(
            ToolDef(
                id=tool_id,
                description=description if isinstance(description, str) else "",
                inputs=tuple(_schema_fields(input_schema, input_schema)),
                required_inputs=frozenset(required),
                outputs=tuple(_schema_fields(output_schema, output_schema)),
                tags=tags,
                deprecated=raw.get("isDeprecated") is True,
                service=_service(tags),
            )
        )
    return tools


def build_graph(
    tools: list[ToolDef],
    classify: Callable[..., Any] | None = None,
) -> dict[str, list[dict[str, str]]]:
    nodes = []
    for tool in sorted(tools, key=lambda item: item.id):
        node = {"id": tool.id}
        if tool.service:
            node["service"] = tool.service
        nodes.append(node)
    return {"nodes": nodes, "edges": []}


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv if argv is None else argv)
    if len(args) < 2:
        raise ValueError("pass the toolkit catalog path as the final argument")
    graph = build_graph(normalize_catalog(load_catalog(Path(args[-1]))))
    output = Path("dependency_graph.json")
    output.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {len(graph['nodes'])} nodes, {len(graph['edges'])} edges to {output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error
