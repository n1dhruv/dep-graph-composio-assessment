"""Validate generated dependency-graph artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__:
    from .generate import load_catalog, normalize_catalog
else:
    from generate import load_catalog, normalize_catalog


def check(catalog: Path, graph_path: Path, html_path: Path) -> list[str]:
    errors: list[str] = []
    try:
        tools = normalize_catalog(load_catalog(catalog))
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return [f"cannot load inputs: {error}"]
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list) or not isinstance(graph.get("edges"), list):
        return ["graph must contain nodes and edges arrays"]

    valid_ids = {tool.id for tool in tools}
    required = {tool.id: tool.required_inputs for tool in tools}
    node_ids = [node.get("id") for node in graph["nodes"] if isinstance(node, dict)]
    if len(node_ids) != len(graph["nodes"]) or not all(isinstance(node_id, str) for node_id in node_ids):
        errors.append("every node must have a string id")
    if not node_ids:
        errors.append("graph must contain at least one node")
    if len(node_ids) != len(set(node_ids)):
        errors.append("node ids must be unique")
    if node_ids and sum(node_id in valid_ids for node_id in node_ids) / len(node_ids) < 0.8:
        errors.append("node provenance is below 0.8")
    if not graph["edges"]:
        errors.append("graph must contain at least one edge")
    for edge in graph["edges"]:
        if not isinstance(edge, dict) or not all(isinstance(edge.get(key), str) and edge[key] for key in ("from", "to", "label")):
            errors.append("every edge must have non-empty from, to, and label strings")
            continue
        if edge["from"] not in valid_ids or edge["to"] not in valid_ids:
            errors.append("edge endpoint is not present in the catalog")
        elif edge["from"] not in node_ids or edge["to"] not in node_ids:
            errors.append("edge endpoint is not present in graph nodes")
        elif edge["label"] not in required[edge["to"]]:
            errors.append("edge label is not a required consumer input")
    if not html_path.is_file():
        errors.append("dependency_graph.html is missing")
    return sorted(set(errors))


def main(
    catalog: Path | None = None,
    output: Path | None = None,
    html: Path | None = None,
) -> int:
    catalog = (catalog or Path("github_catalog.json")).resolve()
    output = (output or Path("dependency_graph.json")).resolve()
    html = (html or output.with_name("dependency_graph.html")).resolve()
    failures = check(catalog, output, html)
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1

    slugs = {tool.id for tool in normalize_catalog(load_catalog(catalog))}
    graph = json.loads(output.read_text(encoding="utf-8"))
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    in_catalog = sum(node.get("id") in slugs for node in nodes)
    provenance = in_catalog / len(nodes) if nodes else 0.0
    print(
        json.dumps(
            {
                "nodes": len(nodes),
                "edges": len(edges),
                "provenance_ratio": round(provenance, 3),
                "labeled_edges": sum(bool(edge.get("label")) for edge in edges),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
