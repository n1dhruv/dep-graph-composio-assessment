"""Run the generator and report graph shape and provenance metrics."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

if __package__:
    from .generate import load_catalog, normalize_catalog
else:
    from generate import load_catalog, normalize_catalog


def main(catalog: Path | None = None, output: Path | None = None) -> int:
    catalog = (catalog or Path("github_catalog.json")).resolve()
    output = (output or Path("dependency_graph.json")).resolve()
    subprocess.run(
        [sys.executable, str(Path(__file__).with_name("generate.py")), str(catalog)],
        cwd=output.parent,
        check=True,
    )

    slugs = {tool.id.upper() for tool in normalize_catalog(load_catalog(catalog))}
    graph = json.loads(output.read_text(encoding="utf-8"))
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    in_catalog = sum(str(node.get("id", "")).upper() in slugs for node in nodes)
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
    if provenance < 0.8:
        print("WARNING: provenance < 0.8; node ids must come from the catalog", file=sys.stderr)
    if not edges:
        print("WARNING: 0 edges; dependency inference is added in Phase 2", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
