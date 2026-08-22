# Tool Dependency Graph Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the TypeScript skeleton with a general Python program that reads a Composio-style toolkit catalog, discovers defensible producer-to-consumer dependencies, writes `dependency_graph.json`, and provides a viewable graph.

**Architecture:** A single Python generator owns catalog normalization, deterministic candidate retrieval, batched LLM adjudication, edge validation, JSON emission, and HTML generation. The standard library is sufficient; tests inject an LLM response function so verification does not consume API credits. Candidate retrieval remains deterministic and broad enough to preserve alternatives, while only validated semantic matches become graph edges.

**Tech Stack:** Python 3.10+, standard library (`dataclasses`, `json`, `pathlib`, `re`, `urllib.request`, `unittest`), OpenAI-compatible Chat Completions API, embedded HTML/CSS/JavaScript visualization.

**Spec:** [`README.md`](../README.md) and [`NOTES.md`](../NOTES.md)

## Global Constraints

- The catalog path is the final CLI argument and may point to a top-level array or an object containing `tools` or `items`.
- Node IDs and edge endpoints must be slugs present in the supplied catalog; no GitHub-specific relation may be hardcoded.
- Each edge is directed `producer -> consumer`, and its label is a required consumer input supplied by the producer.
- Preserve multiple genuinely valid producers; do not force an edge when the input is user/context supplied or no producer is credible.
- Use deterministic rules to retrieve candidates and the LLM only to adjudicate ambiguous shortlists.
- Never send the complete catalog to the LLM. Batch compact questions with at most eight producer candidates per required input.
- Read `OPENAI_API_KEY` and `OPENAI_BASE_URL` from the environment; default `OPENAI_MODEL` to `openai/gpt-4o`.
- Fail clearly on missing credentials, malformed catalog data, API failure, or invalid model output rather than writing a misleading partial graph.
- Keep confidence and reasoning internal; the submitted JSON contains only the required node and edge fields.
- Use only the Python standard library unless a measured limitation requires reconsideration.
- Commit after each phase using the commit message stated below.

---

## Phase 0: Inspect and document the catalog — complete

**Files:**

- Created: `NOTES.md`

**Outcome:** Compact `jq` queries established the catalog shape without loading its 7.7 MB contents into the conversation. The notes record tool counts, schema availability, nested output structure, naming drift, description quality, generic context fields, tags, and deprecated tools.

**Verification performed:** Quantitative claims were recalculated directly from `github_catalog.json`, and `git diff --check` passed.

**Commit:** `0f3f455` (`Add catalog inspection notes to document tool metadata and schemas`)

---

## Phase 1: Python catalog parser and normalized model — complete

**Files:**

- Create: `src/generate.py`
- Create: `tests/test_generate.py`
- Create: `src/selfcheck.py`
- Modify: `generator.json`
- Delete: `src/generate.ts`
- Delete: `src/selfcheck.ts`
- Delete: `package.json`
- Delete: `package-lock.json`

**Interfaces:**

- Produces `FieldDef(name: str, path: str, type: str, description: str, entity: str)`.
- Produces `ToolDef(id: str, description: str, inputs: tuple[FieldDef, ...], required_inputs: frozenset[str], outputs: tuple[FieldDef, ...], tags: frozenset[str], deprecated: bool, service: str | None)`.
- Produces `load_catalog(path: Path) -> list[dict[str, Any]]`.
- Produces `normalize_catalog(raw_tools: list[dict[str, Any]]) -> list[ToolDef]`.
- Produces `build_graph(tools: list[ToolDef], classify: Callable | None = None) -> dict[str, list[dict[str, str]]]`; in this phase it emits all valid nodes and no edges.
- Produces `main(argv: Sequence[str] | None = None) -> int`, writing `dependency_graph.json` in the working directory.

- [x] **Step 1: Write parser tests using a tiny synthetic catalog**

  Cover a top-level list, `{ "tools": [...] }`, required versus optional inputs, nested arrays, local `$ref` resolution, output paths, containing entities, duplicate slugs, and missing slugs. The core assertion should prove that `data.issues[].number` normalizes to a field whose name is `number` and entity is `Issue`.

  ```python
  def test_normalize_catalog_preserves_output_context():
      tools = normalize_catalog([ISSUE_LIST_TOOL])
      number = next(field for field in tools[0].outputs if field.path.endswith("issues[].number"))
      assert (number.name, number.type, number.entity) == ("number", "integer", "Issue")
  ```

- [x] **Step 2: Run the focused test and confirm the missing implementation failure**

  Run: `python3 -m unittest tests.test_generate.CatalogTests -v`

  Expected: FAIL because `src.generate` or its parser interfaces do not exist.

- [x] **Step 3: Implement the normalized model and loader**

  Use frozen dataclasses. Accept only dictionaries with a non-empty `slug`/`name`/`function.name`, reject duplicate IDs, and recursively walk properties, arrays, `allOf`, `anyOf`, `oneOf`, and local `#/$defs/...` references with cycle protection. Preserve the full dotted path and the nearest referenced definition name as `entity`.

  Treat `service` as optional: select the first non-behavioral tag only when it is unambiguous; otherwise store `None`. Do not derive it from the toolkit prefix.

- [x] **Step 4: Implement the Python CLI baseline**

  Parse the final argument, load and normalize the catalog, write every tool as a node, and write an empty edge list. Serialize stable output with `indent=2` and a trailing newline. Print node/edge counts to stderr.

- [x] **Step 5: Replace Node entrypoints**

  Set `generator.json` to:

  ```json
  {
    "build": "python3 -m compileall -q src",
    "run": "python3 src/generate.py"
  }
  ```

  Add a Python self-check that invokes the generator with `github_catalog.json`, reloads the output, calculates node provenance, edge count, and labeled-edge count, and prints the same metrics as the old check. Remove the obsolete TypeScript and npm files.

- [x] **Step 6: Run Phase 1 verification**

  Run:

  ```bash
  python3 -m unittest tests.test_generate.CatalogTests -v
  python3 -m compileall -q src tests
  python3 src/selfcheck.py
  ```

  Expected: parser tests PASS; compilation exits 0; self-check reports 893 catalog-backed nodes, zero edges, and the expected baseline warning.

- [x] **Step 7: Commit Phase 1**

  ```bash
  git add generator.json src tests package.json package-lock.json
  git commit -m "feat: catalog parser and normalized tool model"
  ```

---

## Phase 2A: Deterministic edge candidate retrieval — complete

**Files:**

- Modify: `src/generate.py`
- Modify: `tests/test_generate.py`

**Interfaces:**

- Consumes `ToolDef` and `FieldDef` from Phase 1.
- Produces `Candidate(producer: str, consumer: str, label: str, output: FieldDef, score: float, reason: str)` so Phase 2B receives the exact schema evidence behind each shortlist entry.
- Produces `canonical_tokens(value: str) -> tuple[str, ...]`, normalizing camelCase, snake_case, case, and singular/plural variants.
- Produces `find_candidates(tools: list[ToolDef], limit: int = 8) -> dict[tuple[str, str], list[Candidate]]` keyed by `(consumer_slug, required_input_name)`.

- [x] **Step 1: Write candidate-retrieval tests**

  Use synthetic issue and pull-request tools to prove:

  - exact non-generic identifiers rank highly;
  - `migrationId` and `migration_id` canonicalize identically;
  - an issue output `data.issues[].number` is a candidate for `issue_number` but not `pull_number`;
  - `owner`, `repo`, `org`, free-form `body`, `title`, `name`, `description`, `message`, and `content` create no candidates;
  - self-edges and deprecated producers are excluded;
  - no shortlist exceeds eight producers.

- [x] **Step 2: Run the tests and confirm retrieval is absent**

  Run: `python3 -m unittest tests.test_generate.CandidateTests -v`

  Expected: FAIL because `canonical_tokens` and `find_candidates` do not exist.

- [x] **Step 3: Implement canonicalization and scoring**

  Split camelCase and punctuation into lowercase tokens, normalize simple plurals, and compare the required input with the output leaf, schema path, entity name, output description, and producer name. Use these deterministic signals:

  - exact canonical field match: strongest lexical signal;
  - identifier suffix match such as `issue + number` against entity `Issue` plus leaf `number`: strong contextual signal;
  - meaningful description/entity token overlap: supporting signal;
  - same domain tag: tie-breaker only;
  - wrapper leaves `data`, `successful`, and `error`: excluded.

  Deterministic scores retrieve candidates but do not yet authorize final ambiguous edges.

- [x] **Step 4: Audit compact shortlists against the real catalog**

  Add a diagnostic CLI option `--inspect-candidates CONSUMER_SLUG:FIELD` that prints only the selected consumer input and its compact shortlist. Inspect at least:

  ```text
  GITHUB_CREATE_AN_ISSUE_COMMENT:issue_number
  GITHUB_MERGE_A_PULL_REQUEST:pull_number
  GITHUB_ACCEPT_REPOSITORY_INVITATION:invitation_id
  ```

  Confirm that issue and pull-request candidates remain entity-specific and that generic repository context is absent.

- [x] **Step 5: Run Phase 2A verification and commit**

  Run: `python3 -m unittest tests.test_generate.CandidateTests -v`

  Expected: all candidate tests PASS.

  ```bash
  git add src/generate.py tests/test_generate.py
  git commit -m "feat: edge candidate matcher"
  ```

**Recorded outcome:** The GitHub catalog produced 777 non-empty consumer-input groups and 5,052 candidates, with no shortlist exceeding eight. The audits included `GITHUB_LIST_REPOSITORY_ISSUES`, `GITHUB_LIST_PULL_REQUESTS`, and `GITHUB_LIST_REPO_INVITATIONS_FOR_AUTH_USER` for the three planned examples; circular tools that already require the target value were excluded.

---

## Phase 2B: Batched LLM semantic adjudication — implementation complete; live smoke pending

**Files:**

- Modify: `src/generate.py`
- Modify: `tests/test_generate.py`

**Interfaces:**

- Consumes candidate groups from Phase 2A.
- Produces `classify_candidates(groups: dict[tuple[str, str], list[Candidate]], tools_by_id: dict[str, ToolDef], complete: Callable | None = None) -> list[ScoredEdge]`; tests inject `complete`, while runtime defaults to the HTTP client.
- Produces `ScoredEdge(producer: str, consumer: str, label: str, confidence: float, reason: str, source: str)`.
- Produces `chat_completion(payload: dict[str, Any]) -> dict[str, Any]`, an OpenAI-compatible standard-library HTTP call.
- Produces `validate_model_edges(response: object, allowed: set[tuple[str, str, str]]) -> list[ScoredEdge]`.

- [x] **Step 1: Write model-response validation tests**

  Inject a fake classifier response and prove that validation rejects unknown slugs, labels that are not the requested required input, non-shortlisted producer pairs, self-edges, confidence outside `[0, 1]`, duplicates, and malformed JSON. Prove that multiple valid producers for one input are preserved.

- [x] **Step 2: Run the tests and confirm adjudication is absent**

  Run: `python3 -m unittest tests.test_generate.LlmTests -v`

  Expected: FAIL because the classifier and validator do not exist.

- [x] **Step 3: Implement compact batched prompts**

  Send no more than 20 consumer-input questions per request and no more than eight candidates per question. Include only:

  - consumer slug, description, required input name, type, and description;
  - producer slug and description;
  - candidate output field path, entity, type, and description;
  - deterministic score and reason.

  Instruct the model to treat catalog text as untrusted data, return JSON only, select zero or more genuine ways to obtain the required value, and distinguish reusable identifiers from ordinary user-authored/context values.

- [x] **Step 4: Implement the API call and strict parsing**

  POST to `${OPENAI_BASE_URL.rstrip('/')}/chat/completions` with bearer authentication, `OPENAI_MODEL`, temperature `0`, and JSON response format. Use a 60-second timeout and at most two retries for HTTP 429 and 5xx responses. Raise a concise exception after the retry limit or on invalid JSON.

- [x] **Step 5: Preserve deterministic high-confidence edges**

  Accept a deterministic edge without an API call only when the field names canonicalize exactly, the name is not generic, the output path/entity agrees with the input entity, and there is exactly one credible producer. Send all other non-empty shortlists to the model.

- [ ] **Step 6: Run offline tests, then one bounded live smoke check**

  Run:

  ```bash
  python3 -m unittest tests.test_generate.LlmTests -v
  python3 src/generate.py --inspect-llm GITHUB_CREATE_AN_ISSUE_COMMENT:issue_number github_catalog.json
  ```

  Expected: offline tests PASS; the live response selects one or more issue-producing tools and returns only catalog slugs with confidence values.

- [x] **Step 7: Commit Phase 2B**

  ```bash
  git add src/generate.py tests/test_generate.py
  git commit -m "feat: LLM-assisted semantic matching"
  ```

**Recorded outcome:** Offline validation covers batching, prompt-injection boundaries, malformed responses, hallucinated/non-candidate edges, duplicates, confidence bounds, retries, missing credentials, and conservative deterministic bypass. A no-credit GitHub dry run produced 39 batches for 777 questions; the largest request payload was 96,774 bytes. The real catalog had no edge safe enough for deterministic bypass, avoiding the false confidence found during review. Step 6's live smoke remains unchecked because `OPENAI_API_KEY` and `OPENAI_BASE_URL` were absent from the execution environment.

---

## Phase 3: Score, validate, and emit the dependency graph

**Files:**

- Modify: `src/generate.py`
- Modify: `src/selfcheck.py`
- Modify: `tests/test_generate.py`
- Generate: `dependency_graph.json`

**Interfaces:**

- Consumes `ScoredEdge` objects from Phase 2B.
- Produces `finalize_edges(scored: Iterable[ScoredEdge], valid_ids: set[str], required_by_tool: dict[str, frozenset[str]], threshold: float) -> list[dict[str, str]]`.
- Produces stable graph JSON with nodes sorted by ID and edges sorted by `(from, to, label)`.

- [ ] **Step 1: Write finalization tests**

  Prove that finalization removes edges below the configured threshold, keeps the highest-confidence duplicate, preserves distinct valid alternative producers, rejects invalid provenance, rejects labels not required by the consumer, and sorts output deterministically.

- [ ] **Step 2: Run the tests and confirm finalization is absent**

  Run: `python3 -m unittest tests.test_generate.GraphTests -v`

  Expected: FAIL because `finalize_edges` does not exist.

- [ ] **Step 3: Implement final validation and emission**

  Start with a confidence threshold of `0.75`. Keep it as a named constant, not an environment option, until evidence shows a second operational use case. Write the output atomically through a temporary file in the working directory followed by `Path.replace`.

- [ ] **Step 4: Generate and audit the GitHub graph**

  Run the full generator with the assessment credentials. Confirm these positive examples appear through a valid producer path:

  - an issue-list/search/get tool to `GITHUB_CREATE_AN_ISSUE_COMMENT` labeled `issue_number`;
  - `GITHUB_LIST_PULL_REQUESTS` or another valid pull-request producer to `GITHUB_MERGE_A_PULL_REQUEST` labeled `pull_number`.

  Also inspect at least 20 evenly spaced edges and search for `owner`, `repo`, `org`, and free-form-body labels. If obvious false positives remain, adjust the threshold or candidate rule, rerun the same audit, and record the final threshold in the README during Phase 5.

- [ ] **Step 5: Run Phase 3 verification**

  Run:

  ```bash
  python3 -m unittest discover -s tests -v
  python3 src/selfcheck.py
  python3 -m json.tool dependency_graph.json >/dev/null
  ```

  Expected: tests PASS; provenance is `1.0`; edges are non-zero and labeled; JSON validation exits 0.

- [ ] **Step 6: Commit Phase 3**

  ```bash
  git add src/generate.py src/selfcheck.py tests/test_generate.py dependency_graph.json
  git commit -m "feat: emit validated dependency graph"
  ```

---

## Phase 4: Static interactive visualization

**Files:**

- Modify: `src/generate.py`
- Modify: `tests/test_generate.py`
- Generate: `dependency_graph.html`

**Interfaces:**

- Consumes the finalized graph dictionary.
- Produces `write_visualization(graph: dict[str, Any], path: Path) -> None`.
- Produces a self-contained HTML file with graph data embedded as JSON; it must not fetch `dependency_graph.json` through `file://`.

- [ ] **Step 1: Write visualization tests**

  Generate HTML from a two-node, one-edge graph and assert that both node IDs, the edge label, and embedded JSON appear. Assert that the document does not contain `fetch("dependency_graph.json")`.

- [ ] **Step 2: Run the test and confirm visualization is absent**

  Run: `python3 -m unittest tests.test_generate.VisualizationTests -v`

  Expected: FAIL because `write_visualization` does not exist.

- [ ] **Step 3: Implement the smallest useful viewer**

  Embed graph data directly into one HTML file. Use a CDN-hosted graph renderer only for layout/rendering, show node IDs and directed labeled edges, and add search plus basic zoom/pan. Display a clear message if the CDN cannot load; do not add a frontend build system.

- [ ] **Step 4: Generate and inspect the committed artifact**

  Run: `python3 src/generate.py github_catalog.json`

  Open `dependency_graph.html`, search for the two README consumer examples, and confirm their incoming edges and labels are visible.

- [ ] **Step 5: Run Phase 4 verification and commit**

  Run: `python3 -m unittest tests.test_generate.VisualizationTests -v`

  Expected: visualization tests PASS.

  ```bash
  git add src/generate.py tests/test_generate.py dependency_graph.html
  git commit -m "feat: static dependency graph visualization"
  ```

---

## Phase 5: Documentation, complete self-check, and submission readiness

**Files:**

- Modify: `README.md`
- Modify: `src/selfcheck.py`
- Modify: `tests/test_generate.py`
- Verify: `generator.json`
- Verify: `dependency_graph.json`
- Verify: `dependency_graph.html`

**Interfaces:**

- Documents the final parser, candidate retrieval, LLM adjudication, confidence threshold, known limitations, and non-GitHub assumptions.
- Self-check exits non-zero for invalid graph shape, provenance below `0.8`, zero edges, unknown edge endpoints, labels absent from consumer required inputs, or missing visualization.

- [ ] **Step 1: Add self-check failure tests**

  Use temporary files to prove each invalid condition produces a non-zero status and a concise diagnostic. Keep the self-check callable as `check(catalog_path: Path, graph_path: Path, html_path: Path) -> list[str]` so tests do not spawn subprocesses.

- [ ] **Step 2: Run tests and confirm stricter checks are absent**

  Run: `python3 -m unittest tests.test_generate.SelfCheckTests -v`

  Expected: FAIL until the new validation rules are implemented.

- [ ] **Step 3: Complete the self-check**

  Validate graph shape, unique nodes, endpoint provenance, required-input labels, non-zero labeled edges, and visualization presence. Print metrics on success and all failures on stderr before returning status 1.

- [ ] **Step 4: Update the README**

  Add:

  - Python build/run/self-check commands and required environment variables;
  - normalized schema model and preservation of output path/entity context;
  - generic context-field exclusion rule;
  - deterministic shortlist and batched LLM decision flow;
  - multiple-producer policy, no-producer policy, final confidence threshold, and audit method;
  - actual false-positive/false-negative examples found during Phase 3;
  - generalization boundary: relies on JSON Schema-like inputs/outputs and catalog descriptions, never GitHub slugs or relations;
  - visualization opening instructions and its CDN limitation.

- [ ] **Step 5: Run complete verification from a clean generated state**

  Remove only the two generated outputs, rerun the declared build/run commands from `generator.json`, then run:

  ```bash
  python3 -m unittest discover -s tests -v
  python3 -m compileall -q src tests
  python3 src/selfcheck.py
  python3 -m json.tool dependency_graph.json >/dev/null
  git diff --check
  ```

  Expected: all commands exit 0, provenance is `1.0`, the graph has non-zero labeled edges, and the two README examples have credible incoming dependencies.

- [ ] **Step 6: Review the final diff and commit Phase 5**

  Confirm no credentials, caches, temporary files, or catalog-specific hardcoded edges are tracked.

  ```bash
  git add README.md src/selfcheck.py tests/test_generate.py generator.json dependency_graph.json dependency_graph.html
  git commit -m "docs: explain generator decisions and verification"
  ```

- [ ] **Step 7: Submit only after user approval**

  Show the final verification metrics and commit list. Run `litmus submit` only after the user explicitly authorizes submission because it is an external, irreversible handoff.

---

## Phase checkpoints

Stop after each phase and report exactly:

- files changed;
- tests and checks actually run;
- representative output or metrics;
- remaining risk for the next phase;
- commit hash.

Do not begin the next phase until the user approves it.
