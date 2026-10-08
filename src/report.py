"""Markdown RAG cards, REPORT.md and CHUNKS.jsonl for scholarly works."""
from __future__ import annotations

import json
from typing import Any

from client import SOURCE_LABELS, SOURCE_URLS

PRIVACY_NOTE = (
    "Author names only, exactly as returned by the source APIs — no affiliations, e-mails or contact data. "
    "Citation counts are the sources' own figures; nothing is estimated or invented. PDFs are never downloaded."
)


def esc(v: Any, limit: int = 300) -> str:
    if v is None or v == "" or v == []:
        return "—"
    s = str(v).replace("|", "\\|").replace("\n", " ").strip()
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _yn(v: Any) -> str:
    return "yes" if v is True else "no" if v is False else "—"


def card_sections(r: dict) -> list[tuple[str, list[str]]]:
    identity = [
        "| Field | Value |", "|---|---|",
        f"| Title | {esc(r.get('title'), 500)} |",
        f"| Input | `{esc(r.get('inputRaw'), 120)}` ({esc(r.get('idType'))}) |",
        f"| DOI | {('`' + r['doi'] + '`') if r.get('doi') else '—'} |",
        f"| PMID | {('`' + r['pmid'] + '`') if r.get('pmid') else '—'} |",
        f"| PMCID | {('`' + r['pmcid'] + '`') if r.get('pmcid') else '—'} |",
        f"| arXiv | {('`' + r['arxivId'] + (r.get('arxivVersion') or '') + '`') if r.get('arxivId') else '—'} |",
        f"| OpenAlex | {('`' + r['openAlexId'] + '`') if r.get('openAlexId') else '—'} |",
        f"| Type | {esc(r.get('workType'))} |",
        f"| Year | {esc(r.get('year'))} |",
        f"| Published | {esc(r.get('publicationDate'))} |",
        f"| Venue | {esc(r.get('venue'))} |",
        f"| Publisher | {esc(r.get('publisher'))} |",
        f"| Volume / issue / pages | {esc(r.get('volume'))} / {esc(r.get('issue'))} / {esc(r.get('pages'))} |",
    ]
    if r.get("arxivPrimaryCategory"):
        identity.append(f"| arXiv category | {esc(r.get('arxivPrimaryCategory'))} |")
    if r.get("keywords"):
        identity.append(f"| Keywords / subjects | {esc(', '.join(r['keywords']), 400)} |")
    if r.get("isRetracted"):
        identity.append("| ⚠ Retracted | **yes (per OpenAlex)** |")

    authors = r.get("authors") or []
    if authors:
        shown = authors[:50]
        auth = [", ".join(shown)]
        more = (r.get("authorCount") or len(authors)) - len(shown)
        if more > 0:
            auth.append(f"_…and {more} more (total {r.get('authorCount')})_")
        else:
            auth.append(f"_{len(authors)} author(s)_")
    else:
        auth = ["_No authors listed by the source._"]

    if r.get("abstract"):
        ab = [r["abstract"] + (" _(truncated)_" if r.get("abstractTruncated") else "")]
    else:
        ab = ["_No abstract available from the queried sources (or includeAbstract=false)._"]

    biblio = [
        "| Metric | Value |", "|---|---|",
        f"| Cited by | {esc(r.get('citedBy'))}" + (f" ({SOURCE_LABELS.get(r['citedBySource'], r['citedBySource'])})" if r.get("citedBySource") else "") + " |",
        f"| Cited by (Crossref) | {esc(r.get('citedByCrossref'))} |",
        f"| Cited by (OpenAlex) | {esc(r.get('citedByOpenAlex'))} |",
        f"| References | {esc(r.get('referencesCount'))} |",
        f"| Open access | {_yn(r.get('isOA'))}" + (f" ({r['oaStatus']})" if r.get("oaStatus") else "") + " |",
        f"| License | {esc(r.get('license'))} |",
    ]

    links = []
    for label, k in (("Landing page", "landingUrl"), ("DOI", "doiUrl"), ("PubMed", "pubmedUrl"), ("PMC", "pmcUrl"),
                     ("arXiv", "arxivUrl"), ("OpenAlex", "openAlexUrl"), ("Open-access copy", "oaUrl"),
                     ("PDF (open access)", "pdfUrl")):
        if r.get(k):
            links.append(f"- {label}: <{r[k]}>")
    if r.get("journalRef"):
        links.append(f"- Journal ref (arXiv): {esc(r['journalRef'])}")
    if not links:
        links = ["_No links._"]

    src = [f"- Source(s): {esc(r.get('sourceLabel') or r.get('source'))}"]
    for s in r.get("sources") or []:
        if s in SOURCE_URLS:
            src.append(f"  - {SOURCE_LABELS[s]}: <{SOURCE_URLS[s]}>")
    if r.get("query"):
        src.append(f"- Search query: `{esc(r['query'], 200)}`")
    if r.get("enrichmentErrors"):
        src.append(f"- Enrichment skipped: {esc(', '.join(r['enrichmentErrors']))}")
    if r.get("checkedAt"):
        src.append(f"- Fetched: {r['checkedAt']}")
    src.append(f"- {PRIVACY_NOTE}")
    if r.get("bibtex"):
        src += ["", "**BibTeX** (generated from the fields above only):", "", "```bibtex", r["bibtex"], "```"]

    return [
        ("Identity", identity),
        ("Authors", auth),
        ("Abstract", ab),
        ("Bibliometrics", biblio),
        ("Links", links),
        ("Source", src),
    ]


def card_markdown(r: dict) -> str:
    title = r.get("title") or r.get("doi") or r.get("inputRaw") or "Scholarly work"
    lead = []
    if r.get("authors"):
        a = r["authors"]
        lead.append(a[0] + (" et al." if (r.get("authorCount") or len(a)) > 1 else ""))
    if r.get("year"):
        lead.append(f"({r['year']})")
    if r.get("venue"):
        lead.append(f"· _{esc(r['venue'], 160)}_")
    L = [f"## {esc(title, 300)}", ""]
    if lead:
        L += [" ".join(lead), ""]
    for name, lines in card_sections(r):
        L += [f"### {name}", "", *lines, ""]
    return "\n".join(L).rstrip() + "\n"


def build_markdown(rows: list[dict], summary: dict) -> str:
    ok = [r for r in rows if r.get("status") == "ok"]
    bad = [r for r in rows if r.get("status") != "ok"]
    L = [
        "# Scholarly works — Markdown cards", "",
        f"- Generated: {summary.get('finishedAt')}",
        f"- Cards OK: **{len(ok)}** · failed (not charged): **{len(bad)}**",
        f"- Provider: `{summary.get('provider')}` · sources used: `{summary.get('bySource') or {}}`",
        f"- {PRIVACY_NOTE}", "",
    ]
    for r in ok:
        L += [r.get("markdown") or card_markdown(r), "---", ""]
    if bad:
        L += ["## Failed (not charged)", "", "| Input | Type | Error | HTTP | Hint |", "|---|---|---|---|---|"]
        for r in bad:
            L.append(f"| `{esc(r.get('inputRaw'), 60)}` | {esc(r.get('idType'))} | {esc(r.get('errorClass'))} | "
                     f"{esc(r.get('httpStatus'))} | {esc(r.get('hint') or r.get('errorMessage'), 200)} |")
        L.append("")
    return "\n".join(L).rstrip() + "\n"


def rag_chunks(r: dict, max_chars: int = 1500) -> list[dict]:
    if r.get("status") != "ok":
        return []
    ident = r.get("doi") or (f"pmid:{r['pmid']}" if r.get("pmid") else None) or (
        f"arxiv:{r['arxivId']}" if r.get("arxivId") else None) or r.get("inputRaw")
    head = f"{r.get('title') or ident} ({r.get('year') or 'n.d.'})"
    out: list[dict] = []
    for title, lines in card_sections(r):
        text = "\n".join(lines).strip()
        parts: list[str] = []
        if len(text) <= max_chars:
            parts = [text]
        else:  # split long sections (abstracts) on sentence-ish boundaries
            buf = ""
            for piece in text.replace("\n", " \n").split(" "):
                if len(buf) + len(piece) + 1 > max_chars and buf:
                    parts.append(buf.strip())
                    buf = ""
                buf += piece + " "
            if buf.strip():
                parts.append(buf.strip())
        for i, body in enumerate(parts, 1):
            sec = title if len(parts) == 1 else f"{title} (part {i}/{len(parts)})"
            out.append({
                "id": f"{ident}#{len(out) + 1}",
                "doi": r.get("doi"), "pmid": r.get("pmid"), "arxivId": r.get("arxivId"),
                "title": r.get("title"), "year": r.get("year"),
                "section": sec, "source": r.get("source"), "landingUrl": r.get("landingUrl"),
                "text": f"# {head} — {sec}\n\n{body}",
            })
    for i, c in enumerate(out, 1):
        c["chunkIndex"], c["chunkCount"] = i, len(out)
    return out


def build_chunks_jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(c, ensure_ascii=False) + "\n" for r in rows for c in rag_chunks(r))
