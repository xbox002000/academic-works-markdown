"""Detect and normalize scholarly identifiers (DOI / PMID / arXiv) and keyword queries.

Sources: Actor input `ids` / `queries`, another run's dataset, or a key-value record.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import unquote

ID_FIELDS = ("id", "doi", "DOI", "pmid", "PMID", "arxivId", "arxiv", "arxiv_id", "url", "identifier")
QUERY_FIELDS = ("query", "q", "keywords", "search")

# DOI core: 10.<registrant>/<suffix>  (Crossref recommended pattern, slightly relaxed)
_DOI_CORE = re.compile(r"(10\.\d{4,9}/[^\s\"<>]+)", re.I)
_DOI_URL = re.compile(r"^(?:https?://)?(?:dx\.)?doi\.org/(10\.\d{4,9}/\S+)$", re.I)
_DOI_PREFIXED = re.compile(r"^(?:doi\s*:\s*)(10\.\d{4,9}/\S+)$", re.I)
_ARXIV_DOI = re.compile(r"^10\.48550/arxiv\.(.+)$", re.I)

# arXiv ids: new style YYMM.NNNN(N)(vN) ; old style archive(.SC)/YYMMNNN(vN)
_ARXIV_NEW = r"\d{4}\.\d{4,5}(?:v\d+)?"
_ARXIV_OLD = r"[a-z][a-z\-]*(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?"
_ARXIV_BARE = re.compile(rf"^(?:arxiv\s*:\s*)?({_ARXIV_NEW}|{_ARXIV_OLD})$", re.I)
_ARXIV_URL = re.compile(
    rf"^(?:https?://)?(?:www\.|export\.)?arxiv\.org/(?:abs|pdf|html)/({_ARXIV_NEW}|{_ARXIV_OLD})(?:\.pdf)?/?(?:[?#].*)?$",
    re.I,
)

# PMID: 1–9 digits, optionally "PMID:" prefixed or a PubMed URL
_PMID_BARE = re.compile(r"^(?:pmid\s*:?\s*)?(\d{1,9})$", re.I)
_PMID_URL = re.compile(
    r"^(?:https?://)?(?:www\.)?(?:pubmed\.ncbi\.nlm\.nih\.gov|ncbi\.nlm\.nih\.gov/pubmed)/(\d{1,9})/?(?:[?#].*)?$",
    re.I,
)


@dataclass(frozen=True)
class Target:
    """One unit of work."""
    kind: str        # doi | pmid | arxiv | query
    value: str       # normalized id or query text
    raw: str = ""


def clean_doi(doi: str) -> str:
    d = unquote(doi.strip()).rstrip(".,;)")
    # strip trailing ">" or quotes that sneak in from copy/paste
    d = d.strip("\"'<> ")
    return d.lower()


def strip_arxiv_version(aid: str) -> str:
    return re.sub(r"v\d+$", "", aid)


def detect_id(raw: Any) -> Target | None:
    """Classify a single identifier token. Returns None if it is not a DOI / PMID / arXiv id."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        n = str(int(raw))
        return Target("pmid", n, n) if 0 < len(n) <= 9 and int(raw) > 0 else None
    s = str(raw).strip()
    if not s:
        return None
    m = _ARXIV_URL.match(s) or _ARXIV_BARE.match(s)
    if m:
        return Target("arxiv", m.group(1), s)
    m = _PMID_URL.match(s) or _PMID_BARE.match(s)
    if m and int(m.group(1)) > 0:
        return Target("pmid", str(int(m.group(1))), s)
    m = _DOI_URL.match(s) or _DOI_PREFIXED.match(s)
    doi = m.group(1) if m else None
    if not doi:
        m2 = _DOI_CORE.search(s)
        # accept embedded DOIs only if the token is URL-ish or the DOI is the whole token
        if m2 and (m2.group(1) == s or s.lower().startswith(("http://", "https://", "doi"))):
            doi = m2.group(1)
    if doi:
        # arXiv DataCite DOIs (10.48550/arXiv.<id>) stay kind=doi; the client routes them to arXiv.
        return Target("doi", clean_doi(doi), s)
    return None


def arxiv_from_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    m = _ARXIV_DOI.match(doi)
    return m.group(1) if m else None


def arxiv_doi(arxiv_id: str) -> str:
    return f"10.48550/arxiv.{strip_arxiv_version(arxiv_id)}".lower()


def value_from_item(item: dict, field: str | None, fields: tuple[str, ...]) -> Any:
    for f in ((field,) if field else ()) + fields:
        if f and item.get(f) not in (None, ""):
            return item.get(f)
    return None


def _as_list(values: Iterable[Any] | str | None) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, str):
        return [v for v in re.split(r"[\r\n]+", values) if v.strip()]
    if isinstance(values, (int, float)) and not isinstance(values, bool):
        return [values]
    return list(values)


def _split_id_tokens(v: Any) -> list[Any]:
    """'a, b; c' → ['a','b','c'] for id strings (queries are NOT split on commas)."""
    if isinstance(v, str) and re.search(r"[,;\s]", v.strip()):
        parts = [p for p in re.split(r"[,;\s]+", v.strip()) if p]
        if len(parts) > 1 and all(detect_id(p) for p in parts):
            return parts
    return [v]


def collect_targets(
    *,
    ids: Iterable[Any] | str | None = None,
    queries: Iterable[Any] | str | None = None,
    field: str | None = None,
) -> tuple[list[Target], list[str]]:
    """Return (unique targets, invalid raw strings). Order preserved: ids first, then queries."""
    out: list[Target] = []
    bad: list[str] = []
    seen: set[tuple[str, str]] = set()

    def add(t: Target) -> None:
        key = (t.kind, t.value.lower() if t.kind != "query" else t.value)
        if key in seen:
            return
        seen.add(key)
        out.append(t)

    for v in _as_list(ids):
        if isinstance(v, dict):
            q = value_from_item(v, None, QUERY_FIELDS)
            idv = value_from_item(v, field, ID_FIELDS)
            if idv is None and q is not None:
                add(Target("query", str(q).strip(), str(q).strip()))
                continue
            v = idv
        if v is None or str(v).strip() == "":
            continue
        for tok in _split_id_tokens(v):
            t = detect_id(tok)
            if t:
                add(t)
            else:
                bad.append(str(tok).strip()[:160])

    for v in _as_list(queries):
        if isinstance(v, dict):
            v = value_from_item(v, field, QUERY_FIELDS)
        s = str(v or "").strip()
        if s:
            add(Target("query", s[:500], s))
    return out, bad


def split_payload(payload: Any, field: str | None = None) -> tuple[list[Target], list[str]]:
    """KV record (JSON object/array or text lines) → (targets, invalid)."""
    if payload is None:
        return [], []
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8", "replace")
    if isinstance(payload, str):
        s = payload.strip()
        if s.startswith(("[", "{")):
            try:
                return split_payload(json.loads(s), field)
            except ValueError:
                pass
        return collect_targets(ids=s, field=field)
    if isinstance(payload, dict):
        if any(k in payload for k in ("ids", "queries", "dois", "pmids", "arxivIds", "items")):
            ids: list[Any] = []
            for k in ("ids", "dois", "pmids", "arxivIds", "items"):
                ids += _as_list(payload.get(k))
            return collect_targets(ids=ids, queries=payload.get("queries"), field=field)
        return collect_targets(ids=[payload], field=field)
    if isinstance(payload, list):
        return collect_targets(ids=payload, field=field)
    return collect_targets(ids=[payload], field=field)
