# Catalog inspection notes

- `github_catalog.json` is a top-level array of 893 tools. Every entry has the same core fields, including `slug`, `description`, `inputParameters`, `outputParameters`, `tags`, `toolkit`, and `isDeprecated`; 22 tools are marked deprecated.
- Tool slugs such as `GITHUB_LIST_REPOSITORY_ISSUES` are the stable graph node IDs. The embedded `toolkit.slug` is `github`, but the generator must take IDs and toolkit metadata from whichever catalog it receives.
- All 893 tools have object-shaped input and output JSON Schemas, and every output schema has nested `$defs`. Outputs therefore do not need to be invented from tool names or inferred by an LLM.
- Output schemas commonly wrap the real result under `data` alongside `successful` and `error`, with `$ref`, arrays, and nested objects beneath it. Normalization must preserve field paths and containing entity names rather than keeping only leaf names.
- There are 3,606 input properties, and 3,605 have descriptions. Descriptions often contain direct dependency clues; for example, `invitation_id` says to obtain it by listing pending invitations.
- Required inputs exist on 828 tools, averaging 2.27 required fields per tool. Optional inputs should not create prerequisite edges because the stated graph models information required before execution.
- Naming is mostly snake_case but is not uniform: the catalog contains 1,352 underscored input names and 36 camelCase names. The same concept can appear as `migrationId` and `migration_id`, so matching needs canonicalized names.
- Entity meaning is often carried by context rather than the leaf field: issue and pull-request producers expose fields such as `number`, while consumers request `issue_number` or `pull_number`. A bare `number == number` comparison is insufficient.
- `owner` and `repo` are required by 442 and 441 tools respectively. These are normally user/session context, so treating every matching output as a producer would create a dense graph of meaningless edges.
- Tags mix domain hints (`repos`, `actions`) with behavioral metadata (`readOnlyHint`, `updateHint`, `mcpIgnore`). They can help rank candidates, but are not reliable enough to derive a mandatory `service` value on their own.
