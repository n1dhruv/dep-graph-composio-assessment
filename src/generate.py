"""Normalize a toolkit catalog and emit its dependency graph."""

from __future__ import annotations

import json
import re
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


@dataclass(frozen=True)
class Candidate:
    producer: str
    consumer: str
    label: str
    output: FieldDef
    score: float
    reason: str


def canonical_tokens(value: str) -> tuple[str, ...]:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    tokens = re.findall(r"[A-Za-z0-9]+", value.lower())
    normalized = []
    aliases = {"identifier": "id", "identifiers": "id", "num": "number"}
    for token in tokens:
        token = aliases.get(token, token)
        if token.endswith("ies") and len(token) > 3:
            token = f"{token[:-3]}y"
        elif token.endswith("s") and len(token) > 3 and not token.endswith(("ss", "us", "is")):
            token = token[:-1]
        normalized.append(token)
    return tuple(normalized)


_USER_CONTEXT_FIELDS = {
    "body",
    "content",
    "description",
    "message",
    "name",
    "org",
    "owner",
    "repo",
    "title",
}
_WRAPPER_FIELDS = {"data", "error", "successful"}


def _compatible_types(required: FieldDef, output: FieldDef) -> bool:
    if "unknown" in (required.type, output.type) or required.type == output.type:
        return True
    return {required.type, output.type} <= {"integer", "number"}


def _candidate_score(
    required: FieldDef,
    output: FieldDef,
    producer: ToolDef,
    consumer: ToolDef,
) -> tuple[float, str] | None:
    if output.name.lower() in _WRAPPER_FIELDS or not _compatible_types(required, output):
        return None

    required_tokens = canonical_tokens(required.name)
    output_tokens = canonical_tokens(output.name)
    context_tokens = set(
        canonical_tokens(output.path)
        + canonical_tokens(output.entity)
        + canonical_tokens(producer.id)
    )
    if required_tokens == output_tokens:
        score, reason = 0.98, "exact normalized field name"
    elif output_tokens and set(required_tokens) <= context_tokens | set(output_tokens):
        score, reason = 0.88, "output field plus entity/path context"
    else:
        return None

    if producer.service and producer.service == consumer.service:
        score = min(score + 0.01, 1.0)
        reason += "; same service"
    return score, reason


def find_candidates(
    tools: list[ToolDef], limit: int = 8
) -> dict[tuple[str, str], list[Candidate]]:
    if limit < 1:
        raise ValueError("candidate limit must be positive")
    output_index: dict[tuple[str, ...], list[tuple[ToolDef, FieldDef]]] = {}
    for producer in tools:
        if producer.deprecated:
            continue
        for output in producer.outputs:
            tokens = canonical_tokens(output.name)
            if not tokens or output.name.lower() in _WRAPPER_FIELDS:
                continue
            for key in {tokens, (tokens[-1],)}:
                output_index.setdefault(key, []).append((producer, output))

    groups: dict[tuple[str, str], list[Candidate]] = {}
    for consumer in tools:
        inputs = {field.name: field for field in consumer.inputs}
        for label in sorted(consumer.required_inputs):
            required_tokens = canonical_tokens(label)
            if "_".join(required_tokens) in _USER_CONTEXT_FIELDS:
                continue
            required = inputs.get(label, FieldDef(label, label, "unknown", "", ""))
            best_by_producer: dict[str, Candidate] = {}
            possible_outputs = []
            for key in {required_tokens, (required_tokens[-1],)} if required_tokens else set():
                possible_outputs.extend(output_index.get(key, ()))
            for producer, output in possible_outputs:
                if producer.id == consumer.id:
                    continue
                if required_tokens in {
                    canonical_tokens(producer_input) for producer_input in producer.required_inputs
                }:
                    continue
                match = _candidate_score(required, output, producer, consumer)
                if not match:
                    continue
                score, reason = match
                candidate = Candidate(producer.id, consumer.id, label, output, score, reason)
                previous = best_by_producer.get(producer.id)
                if previous is None or candidate.score > previous.score:
                    best_by_producer[producer.id] = candidate
            ranked = sorted(best_by_producer.values(), key=lambda item: (-item.score, item.producer))
            if ranked:
                groups[(consumer.id, label)] = ranked[:limit]
    return groups


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
    tools = normalize_catalog(load_catalog(Path(args[-1])))
    if "--inspect-candidates" in args:
        option = args.index("--inspect-candidates")
        if option + 1 >= len(args) - 1 or ":" not in args[option + 1]:
            raise ValueError("--inspect-candidates requires CONSUMER_SLUG:FIELD")
        consumer, label = args[option + 1].rsplit(":", 1)
        selected = next((tool for tool in tools if tool.id == consumer), None)
        if selected is None or label not in selected.required_inputs:
            raise ValueError(f"unknown required input: {consumer}:{label}")
        candidates = find_candidates(tools).get((consumer, label), [])
        print(
            json.dumps(
                {
                    "consumer": consumer,
                    "label": label,
                    "candidates": [
                        {
                            "producer": candidate.producer,
                            "score": round(candidate.score, 3),
                            "reason": candidate.reason,
                            "output": {
                                "name": candidate.output.name,
                                "path": candidate.output.path,
                                "type": candidate.output.type,
                                "entity": candidate.output.entity,
                                "description": candidate.output.description,
                            },
                        }
                        for candidate in candidates
                    ],
                },
                indent=2,
            )
        )
        return 0

    graph = build_graph(tools)
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
