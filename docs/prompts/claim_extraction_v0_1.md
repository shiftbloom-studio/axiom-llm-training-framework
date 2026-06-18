# claim_extraction_v0.1

Return JSON only.

Extract source-grounded claim candidates for Axiom claim-field corpus
construction. Use schema-compatible claim types when possible:
`scientific_claim`, `causal_claim`, `measurement_claim`, `method_claim`,
`definitional_claim`, `historical_claim`, or `other`.

Rules:

- Do not output binary labels.
- Output claim-state candidates and uncertainty only.
- Preserve source-span grounding with start/end offsets when available.
- Distinguish temporal facts from lateral context.
- Do not invent sources or citations.
- Use `null` for unsupported or unknown fields.

Expected top-level JSON:

```json
{
  "claims": [
    {
      "text": "...",
      "claim_type": "scientific_claim",
      "confidence": 0.0,
      "source_spans": []
    }
  ],
  "warnings": []
}
```
