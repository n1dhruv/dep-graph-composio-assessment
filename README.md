# Tool Dependency Graph Generator

This project turns a toolkit catalog into a directed graph that helps an agent
decide whether a required tool input should come from the user or from another
tool. An edge `producer -> consumer` means the producer can supply the required
input named by the edge label.

## Run

Python 3.10+ is the only local dependency. The semantic adjudication endpoint
is OpenAI-compatible:

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://your-openai-compatible-endpoint/v1"
python3 -m compileall -q src
python3 src/generate.py path/to/catalog.json
python3 src/selfcheck.py
```

The generator writes `dependency_graph.json` and a visualization at
`dependency_graph.html`. `generator.json` declares the grader commands. Never
commit the API key.

Open the HTML file directly in a browser. Search focuses matching tool IDs;
dragging pans and scrolling zooms. Graph data is embedded, so local `file://`
viewing works. The vis-network renderer uses a CDN and needs network access;
the page shows a diagnostic if it cannot load.

## How matching works

1. The loader accepts a catalog array or an object containing `tools` or
   `items`. It extracts IDs from `slug`, `name`, or `function.name`.
2. JSON Schema-like inputs and outputs are flattened while retaining field
   path, type, description, and containing entity. `$ref`, arrays, and composed
   schemas are handled with cycle protection.
3. Candidate retrieval normalizes snake/camel case, aliases, and simple
   plurals. Exact matches rank above contextual matches such as output
   `issues[].number` satisfying required input `issue_number`.
4. Unambiguous exact matches with entity evidence are accepted
   deterministically. Remaining shortlists are sent to `openai/gpt-4o` in
   batches. Model output is restricted to supplied candidates; invented and
   duplicate rows cannot enter the graph.
5. Finalization keeps scores at or above `0.75`, validates endpoint provenance
   and required-input labels, keeps the highest-confidence duplicate, preserves
   distinct valid producers, and sorts output deterministically.

## Judgment calls

- `owner`, `repo`, `org`, `body`, `title`, `name`, `description`, `message`,
  and `content` are user/context inputs. Connecting them would create dense but
  operationally meaningless graphs.
- A producer that itself requires the proposed value is rejected, preventing a
  circular dependency that does not help obtain the value.
- Multiple genuine producers are preserved. Same-service context improves
  ranking but does not erase alternatives.
- When no producer survives, no edge is forced; the input remains user-supplied
  or unavailable from the catalog.
- Descriptions are untrusted prompt data. Responses are structurally validated,
  and rate-limit/network failures use bounded retries.

## Audit findings and limitations

The GitHub audit found eight candidate producers for each required example:
`issue_number` into `GITHUB_CREATE_AN_ISSUE_COMMENT`, and `pull_number` into
`GITHUB_MERGE_A_PULL_REQUEST`. Generic context labels were absent, every node
came from the catalog, and every edge label was required by its consumer.

The committed graph is a conservative offline identifier snapshot because the
assessment API sweep was stopped to limit token spend. It contains 893 nodes
and 2,074 edges. Running the official generator with credentials replaces it
with the fully LLM-adjudicated graph.

Known false-positive risk: shared identifiers such as `repository_id` can have
several structurally valid producers although only some are useful in a given
workflow. Known false-negative risk: missing output schemas or descriptions
provide too little evidence, so the generator intentionally avoids speculative
edges. Free-form values are excluded even when a tool technically echoes them.

## Generalization boundary

No GitHub slug or relationship is hardcoded. The implementation relies on
catalog IDs, descriptions, tags, and JSON Schema-like input/output metadata.
Catalogs without explicit outputs need richer descriptions or an upstream
schema-inference step; this generator will not invent undocumented fields.

## Verification

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall -q src tests
python3 src/selfcheck.py
python3 -m json.tool dependency_graph.json >/dev/null
```

The self-check validates shape, unique/provenanced nodes, non-empty labeled
edges, catalog endpoints, required consumer labels, and visualization presence
without making an API call.
