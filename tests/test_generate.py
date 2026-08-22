import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from src.generate import build_graph, load_catalog, main, normalize_catalog
from src.selfcheck import main as selfcheck


ISSUE_LIST_TOOL = {
    "slug": "EXAMPLE_LIST_ISSUES",
    "description": "Lists issues.",
    "inputParameters": {
        "type": "object",
        "properties": {
            "owner": {"type": "string", "description": "Repository owner."},
            "state": {"type": "string", "description": "Optional issue state."},
        },
        "required": ["owner"],
    },
    "outputParameters": {
        "type": "object",
        "properties": {"data": {"$ref": "#/$defs/ListIssuesResponse"}},
        "$defs": {
            "ListIssuesResponse": {
                "type": "object",
                "properties": {
                    "issues": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/Issue"},
                    }
                },
            },
            "Issue": {
                "type": "object",
                "properties": {
                    "number": {
                        "type": "integer",
                        "description": "The issue number.",
                    },
                    "parent": {"$ref": "#/$defs/Issue"},
                },
            },
        },
    },
    "tags": ["issues", "readOnlyHint"],
    "isDeprecated": False,
}


class CatalogTests(unittest.TestCase):
    def test_load_catalog_accepts_supported_root_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename, payload in (
                ("list.json", [ISSUE_LIST_TOOL]),
                ("tools.json", {"tools": [ISSUE_LIST_TOOL]}),
                ("items.json", {"items": [ISSUE_LIST_TOOL]}),
            ):
                path = root / filename
                path.write_text(json.dumps(payload), encoding="utf-8")
                self.assertEqual(load_catalog(path), [ISSUE_LIST_TOOL])

    def test_load_catalog_rejects_an_unsupported_root(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            path.write_text('{"unexpected": []}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "array or an object containing"):
                load_catalog(path)

    def test_normalize_catalog_preserves_schema_context(self):
        tool = normalize_catalog([ISSUE_LIST_TOOL])[0]
        number = next(field for field in tool.outputs if field.path == "data.issues[].number")

        self.assertEqual((number.name, number.type, number.entity), ("number", "integer", "Issue"))
        self.assertEqual(tool.required_inputs, frozenset({"owner"}))
        self.assertEqual({field.name for field in tool.inputs}, {"owner", "state"})
        self.assertEqual(tool.service, "issues")
        self.assertLess(len(tool.outputs), 10)  # the self-reference must terminate

    def test_normalize_catalog_walks_composed_schemas(self):
        tool = json.loads(json.dumps(ISSUE_LIST_TOOL))
        tool["slug"] = "EXAMPLE_COMPOSED_OUTPUT"
        tool["outputParameters"] = {
            "type": "object",
            "allOf": [
                {"properties": {"first": {"type": "string"}}},
                {
                    "anyOf": [
                        {"properties": {"second": {"type": "integer"}}},
                        {"properties": {"third": {"type": "boolean"}}},
                    ]
                },
            ],
        }

        paths = {field.path for field in normalize_catalog([tool])[0].outputs}

        self.assertTrue({"first", "second", "third"}.issubset(paths))

    def test_normalize_catalog_rejects_missing_and_duplicate_ids(self):
        missing = json.loads(json.dumps(ISSUE_LIST_TOOL))
        missing.pop("slug")
        missing.pop("name", None)
        with self.assertRaisesRegex(ValueError, "tool id"):
            normalize_catalog([missing])

        with self.assertRaisesRegex(ValueError, "duplicate tool id"):
            normalize_catalog([ISSUE_LIST_TOOL, ISSUE_LIST_TOOL])

    def test_build_graph_emits_sorted_catalog_nodes_without_edges(self):
        second = json.loads(json.dumps(ISSUE_LIST_TOOL))
        second["slug"] = "EXAMPLE_ALPHA"
        second["tags"] = ["readOnlyHint", "updateHint"]

        graph = build_graph(normalize_catalog([ISSUE_LIST_TOOL, second]))

        self.assertEqual(
            graph,
            {
                "nodes": [
                    {"id": "EXAMPLE_ALPHA"},
                    {"id": "EXAMPLE_LIST_ISSUES", "service": "issues"},
                ],
                "edges": [],
            },
        )

    def test_main_writes_dependency_graph_in_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps([ISSUE_LIST_TOOL]), encoding="utf-8")
            previous = Path.cwd()
            try:
                os.chdir(root)
                status = main(["generate.py", str(catalog)])
            finally:
                os.chdir(previous)

            graph = json.loads((root / "dependency_graph.json").read_text(encoding="utf-8"))
            self.assertEqual(status, 0)
            self.assertEqual(graph["nodes"], [{"id": "EXAMPLE_LIST_ISSUES", "service": "issues"}])
            self.assertEqual(graph["edges"], [])

    def test_selfcheck_runs_generator_and_reports_baseline_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.json"
            output = root / "dependency_graph.json"
            catalog.write_text(json.dumps([ISSUE_LIST_TOOL]), encoding="utf-8")

            stdout = StringIO()
            with redirect_stdout(stdout):
                status = selfcheck(catalog, output)

            self.assertEqual(status, 0)
            self.assertTrue(output.exists())
            self.assertEqual(
                json.loads(stdout.getvalue()),
                {"nodes": 1, "edges": 0, "provenance_ratio": 1.0, "labeled_edges": 0},
            )

    def test_selfcheck_uses_the_same_fallback_ids_as_the_generator(self):
        fallback = json.loads(json.dumps(ISSUE_LIST_TOOL))
        fallback.pop("slug")
        fallback["function"] = {"name": "EXAMPLE_FUNCTION_TOOL"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps([fallback]), encoding="utf-8")

            stdout = StringIO()
            with redirect_stdout(stdout):
                selfcheck(catalog, root / "dependency_graph.json")

            self.assertEqual(json.loads(stdout.getvalue())["provenance_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
