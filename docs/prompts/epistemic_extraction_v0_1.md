# epistemic_extraction_v0.1

Return JSON only.

Estimate conservative construction proxies for claim-state modeling. Proxies are
not verdicts and must include method and confidence.

Required proxy families:

- `ontology_compatibility`
- `evidential_anchoring`
- `transformation_pressure`
- `uncertainty`
- `independent_redundancy`

Rules:

- Do not output binary labels.
- Redundancy must not be raw popularity.
- Use `null` for unsupported proxy values.
- Preserve source count, relation degree, method diversity, and provider
  disagreement as separate construction metadata.
- Do not use future-facing information in predictor-side fields.

Expected top-level JSON:

```json
{
  "ontology_compatibility": {"value": null, "method": "unknown", "confidence": 0.0},
  "evidential_anchoring": {"value": null, "method": "unknown", "confidence": 0.0},
  "transformation_pressure": {"value": null, "method": "unknown", "confidence": 0.0},
  "uncertainty": {"value": null, "method": "unknown", "confidence": 0.0},
  "independent_redundancy": {"value": null, "method": "unknown", "confidence": 0.0},
  "warnings": []
}
```
