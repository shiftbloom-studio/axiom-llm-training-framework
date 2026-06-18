# view_generation_v0.1

Return JSON only.

Generate structured text projections from source-grounded claim material. These
views are construction artifacts and later comparison/projection material, not
the native substrate by themselves.

Required views:

- `neutral_summary`
- `technical_summary`
- `teaching_note`
- `faq`
- `counterargument`
- `limitations`
- `historical_update_or_temporal_note`

Rules:

- Do not output binary labels.
- Do not invent sources.
- Use `null` where the source does not support a view.
- Preserve source-span references when available.
- Separate temporal updates from lateral context descriptions.

Expected top-level JSON:

```json
{
  "views": {
    "neutral_summary": null,
    "technical_summary": null,
    "teaching_note": null,
    "faq": null,
    "counterargument": null,
    "limitations": null,
    "historical_update_or_temporal_note": null
  },
  "source_spans_used": [],
  "warnings": []
}
```
