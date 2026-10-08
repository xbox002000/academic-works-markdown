# DOI / PMID / arXiv → Academic Work Markdown

Batch-resolve **DOIs, PubMed PMIDs and arXiv ids** (plus optional keyword `queries`) into scholarly-work **Markdown RAG cards** and JSON.

**Sources (public, keyless, HTTPS only):** Crossref REST, OpenAlex, NCBI PubMed E-utilities, arXiv export API. No paywalled PDFs, no LLMs, no scraping.

## Input
- `ids`: DOIs / doi.org URLs, PMIDs, arXiv ids / URLs. `queries`: optional keyword searches.
- `provider`: `auto` (default: DOI→Crossref, PMID→PubMed, arXiv→arXiv, enriched with an exact-id OpenAlex lookup), or force one source.
- Optional **your own** `openAlexApiKey` / `ncbiApiKey` (secret) for higher quotas. `contactEmail` for polite pools.

## Output
- Dataset: one card per work (`overview`, `failed` views).
- KV: `REPORT.md`, `CHUNKS.jsonl`, `SUMMARY`, `OUTPUT`.
- Card sections: **Identity · Authors · Abstract · Bibliometrics · Links · Source**. Only metadata returned by the APIs is shown; citations/counts are never invented.

## Pricing (pay per event)
- Actor start $0.001, **work-card $0.004** per successful card. Invalid, not-found and failed rows are **free**.

License: AGPL-3.0.

## Source code

Open source (AGPL-3.0): https://github.com/xbox002000/academic-works-markdown
