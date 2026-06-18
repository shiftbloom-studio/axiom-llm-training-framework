# relation_extraction_v0.1

Return JSON only.

Extract candidate relations between claim candidates. Relations are construction
signals, not evaluation verdicts.

Rules:

- Do not output binary labels.
- Use schema-compatible relations such as `supports`, `contradicts`, `extends`,
  `refines`, `supersedes`, `mentions`, or `related`.
- Preserve evidence spans when available.
- Include hard-negative candidates when the source supports near misses.
- Do not invent missing source material.
- Keep temporal order separate from lateral context.

Expected top-level JSON:

```json
{
  "relations": [],
  "candidate_hard_negatives": [],
  "warnings": []
}
```
