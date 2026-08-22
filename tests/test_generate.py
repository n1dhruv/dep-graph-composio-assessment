import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from src.generate import (
    FieldDef,
    ToolDef,
    build_graph,
    canonical_tokens,
    find_candidates,
    load_catalog,
    main,
    normalize_catalog,
)
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


def field(name, *, path=None, entity="", description="", type="string"):
    return FieldDef(name, path or name, type, description, entity)


def tool(tool_id, *, required=(), inputs=(), outputs=(), deprecated=False, service="issues"):
    return ToolDef(
        id=tool_id,
        description=tool_id.replace("_", " ").lower(),
        inputs=tuple(inputs),
        required_inputs=frozenset(required),
        outputs=tuple(outputs),
        tags=frozenset({service}),
        deprecated=deprecated,
        service=service,
    )


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


class CandidateTests(unittest.TestCase):
    def test_canonical_tokens_normalize_case_separators_and_simple_plurals(self):
        self.assertEqual(canonical_tokens("migrationId"), ("migration", "id"))
        self.assertEqual(canonical_tokens("migration_id"), ("migration", "id"))
        self.assertEqual(canonical_tokens("Issues"), ("issue",))

    def test_exact_identifier_match_ranks_above_bare_id(self):
        exact = tool(
            "EXAMPLE_LIST_COMMENTS",
            outputs=[field("comment_id", path="data.comments[].comment_id", entity="Comment")],
        )
        bare = tool(
            "EXAMPLE_GET_COMMENT",
            outputs=[field("id", path="data.comment.id", entity="Comment")],
        )
        consumer = tool(
            "EXAMPLE_UPDATE_COMMENT",
            required=["comment_id"],
            inputs=[field("comment_id", description="The comment identifier.")],
        )

        candidates = find_candidates([bare, consumer, exact])[(consumer.id, "comment_id")]

        self.assertEqual(candidates[0].producer, exact.id)
        self.assertGreater(candidates[0].score, candidates[1].score)

    def test_entity_context_distinguishes_issue_and_pull_numbers(self):
        issue_producer = tool(
            "EXAMPLE_LIST_ISSUES",
            outputs=[field("number", path="data.issues[].number", entity="Issue")],
        )
        pull_producer = tool(
            "EXAMPLE_LIST_PULL_REQUESTS",
            outputs=[field("number", path="data.pull_requests[].number", entity="PullRequest")],
            service="pull-requests",
        )
        issue_consumer = tool(
            "EXAMPLE_COMMENT_ON_ISSUE",
            required=["issue_number"],
            inputs=[field("issue_number", description="The issue number.")],
        )
        pull_consumer = tool(
            "EXAMPLE_MERGE_PULL_REQUEST",
            required=["pull_number"],
            inputs=[field("pull_number", description="The pull request number.")],
            service="pull-requests",
        )

        groups = find_candidates([issue_producer, pull_producer, issue_consumer, pull_consumer])

        self.assertEqual(
            [candidate.producer for candidate in groups[(issue_consumer.id, "issue_number")]],
            [issue_producer.id],
        )
        self.assertEqual(
            [candidate.producer for candidate in groups[(pull_consumer.id, "pull_number")]],
            [pull_producer.id],
        )

    def test_user_context_fields_do_not_create_candidates(self):
        generic = ("owner", "repo", "org", "body", "title", "name", "description", "message", "content")
        producer = tool("EXAMPLE_GENERIC_OUTPUT", outputs=[field(name) for name in generic])
        consumer = tool(
            "EXAMPLE_GENERIC_INPUT",
            required=generic,
            inputs=[field(name) for name in generic],
        )

        self.assertEqual(find_candidates([producer, consumer]), {})

    def test_self_edges_and_deprecated_producers_are_excluded(self):
        self_matching = tool(
            "EXAMPLE_SELF_MATCH",
            required=["issue_number"],
            inputs=[field("issue_number")],
            outputs=[field("issue_number", entity="Issue")],
        )
        deprecated = tool(
            "EXAMPLE_OLD_ISSUES",
            outputs=[field("issue_number", entity="Issue")],
            deprecated=True,
        )

        self.assertEqual(find_candidates([self_matching, deprecated]), {})

    def test_producer_cannot_require_the_value_it_claims_to_produce(self):
        source = tool(
            "EXAMPLE_LIST_ISSUES",
            outputs=[field("number", path="data.issues[].number", entity="Issue")],
        )
        circular = tool(
            "EXAMPLE_CLOSE_ISSUE",
            required=["issue_number"],
            inputs=[field("issue_number")],
            outputs=[field("number", path="data.issue.number", entity="Issue")],
        )
        consumer = tool(
            "EXAMPLE_COMMENT_ON_ISSUE",
            required=["issue_number"],
            inputs=[field("issue_number")],
        )

        candidates = find_candidates([source, circular, consumer])[(consumer.id, "issue_number")]

        self.assertEqual([candidate.producer for candidate in candidates], [source.id])

    def test_candidate_shortlists_are_capped(self):
        producers = [
            tool(
                f"EXAMPLE_PRODUCER_{index}",
                outputs=[field("comment_id", entity="Comment")],
            )
            for index in range(10)
        ]
        consumer = tool(
            "EXAMPLE_CONSUMER",
            required=["comment_id"],
            inputs=[field("comment_id")],
        )

        candidates = find_candidates([*producers, consumer])[(consumer.id, "comment_id")]

        self.assertEqual(len(candidates), 8)

    def test_inspect_candidates_prints_a_compact_shortlist(self):
        consumer = {
            "slug": "EXAMPLE_COMMENT_ON_ISSUE",
            "description": "Comments on an issue.",
            "inputParameters": {
                "type": "object",
                "properties": {
                    "issue_number": {
                        "type": "integer",
                        "description": "The issue number.",
                    }
                },
                "required": ["issue_number"],
            },
            "outputParameters": {"type": "object", "properties": {}},
            "tags": ["issues"],
            "isDeprecated": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.json"
            catalog.write_text(json.dumps([ISSUE_LIST_TOOL, consumer]), encoding="utf-8")
            stdout = StringIO()

            with redirect_stdout(stdout):
                status = main(
                    [
                        "generate.py",
                        "--inspect-candidates",
                        "EXAMPLE_COMMENT_ON_ISSUE:issue_number",
                        str(catalog),
                    ]
                )

        report = json.loads(stdout.getvalue())
        self.assertEqual(status, 0)
        self.assertEqual((report["consumer"], report["label"]), (consumer["slug"], "issue_number"))
        self.assertEqual(report["candidates"][0]["producer"], ISSUE_LIST_TOOL["slug"])
        self.assertEqual(report["candidates"][0]["output"]["path"], "data.issues[].number")
        self.assertEqual(report["candidates"][0]["output"]["entity"], "Issue")


if __name__ == "__main__":
    unittest.main()
