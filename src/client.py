"""Public scholarly APIs → normalized work records.

Providers (all keyless; optional buyer keys only raise quotas):
- Crossref REST     https://api.crossref.org/works/{doi}  and /works?query.bibliographic=
- OpenAlex          https://api.openalex.org/works/https://doi.org/{doi} | /works/pmid:{pmid} | /works?search=
- PubMed E-utilities https://eutils.ncbi.nlm.nih.gov/entrez/eutils/ esearch / efetch
- arXiv export API  https://export.arxiv.org/api/query?id_list= | search_query=

Never downloads PDFs. pdfUrl is only reported for open-access copies (arXiv, OpenAlex best OA location).
Author output is names only — affiliations (which can contain e-mail addresses) are never emitted.
"""
from __future__ import annotations

import asyncio
import random
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from html import unescape
from typing import Any
from urllib.parse import quote

import httpx

from inputs import Target, arxiv_doi, arxiv_from_doi, strip_arxiv_version

VERSION = "0.1"
DEFAULT_MAILTO = "apify-academic-works@users.noreply.github.com"
NCBI_TOOL = "apify-academic-works-markdown"

CROSSREF = "https://api.crossref.org"
OPENALEX = "https://api.openalex.org"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ARXIV = "https://export.arxiv.org/api/query"

PROVIDERS = ("auto", "crossref", "openalex", "pubmed", "arxiv")

# Minimum seconds between requests per host (politeness). Tests set these to 0.
INTERVALS: dict[str, float] = {
    "api.crossref.org": 0.2,
    "api.openalex.org": 0.12,
    "eutils.ncbi.nlm.nih.gov": 0.35,
    "export.arxiv.org": 3.0,
}
NCBI_KEYED_INTERVAL = 0.11
BACKOFF_BASE = 1.0  # seconds; tests set 0

SOURCE_LABELS = {
    "crossref": "Crossref REST API",
    "openalex": "OpenAlex API",
    "pubmed": "PubMed E-utilities (NCBI)",
    "arxiv": "arXiv export API",
}
SOURCE_URLS = {
    "crossref": "https://api.crossref.org/works",
    "openalex": "https://api.openalex.org/works",
    "pubmed": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
    "arxiv": "https://export.arxiv.org/api/query",
}

HINTS = {
    "invalid_input": "Not a DOI (10.xxxx/…), PMID (digits) or arXiv id (e.g. 1706.03762, hep-th/9901001, arxiv.org/abs/… URL). "
                     "Put free-text searches in `queries`.",
    "not_found": "No record for this identifier at the queried source(s). Check the id, or try provider=auto.",
    "unsupported_for_provider": "This identifier type is not supported by the selected provider. Use provider=auto.",
    "rate_limited": "Upstream rate limit after retries. Lower maxConcurrency, or add your own OpenAlex / NCBI API key.",
    "server_error": "Upstream server error after retries. Try again later.",
    "timeout": "Upstream timed out after retries. Raise requestTimeoutSecs or retry later.",
    "network": "Network error reaching the upstream API.",
    "bad_response": "Upstream returned an unexpected payload.",
    "bad_request": "Upstream rejected the request (HTTP 400). Check the identifier / query syntax.",
    "unauthorized": "Upstream rejected the API key you supplied (HTTP 401). Remove or fix the key.",
    "forbidden": "Upstream refused the request (HTTP 403). Check your API key / quota.",
}


# ---------------------------------------------------------------- helpers
def redact(text: str | None, secrets: list[str | None] | None = None) -> str:
    if not text:
        return text or ""
    for s in secrets or []:
        if s:
            text = text.replace(s, "***")
    return re.sub(r"(?i)(api_key=|apikey=|api-key=)[^\s&\"']+", r"\1***", text)


def _s(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, list):
        v = next((x for x in v if x), None)
        if v is None:
            return None
    s = re.sub(r"\s+", " ", str(v)).strip()
    return s or None


_TAG = re.compile(r"<[^>]+>")


def strip_markup(text: str | None) -> str | None:
    """JATS / HTML abstract → plain text."""
    if not text:
        return None
    t = re.sub(r"</(?:jats:)?(?:p|sec|title)>", "\n\n", text)
    t = _TAG.sub(" ", t)
    t = unescape(t)
    t = re.sub(r"[ \t\r\f\v]+", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    t = re.sub(r"^(?:abstract|summary)\s*[:.]?\s*\n+", "", t, flags=re.I)
    return t or None


def inverted_index_to_text(inv: dict | None) -> str | None:
    if not inv or not isinstance(inv, dict):
        return None
    pos: dict[int, str] = {}
    for word, idxs in inv.items():
        for i in idxs or []:
            if isinstance(i, int):
                pos[i] = word
    if not pos:
        return None
    return " ".join(pos[i] for i in sorted(pos))


def truncate(text: str | None, max_chars: int) -> tuple[str | None, bool]:
    if not text:
        return None, False
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rstrip() + "…", True
    return text, False


def _date_parts(obj: dict | None) -> tuple[int | None, str | None]:
    try:
        parts = (obj or {}).get("date-parts")[0]
    except (TypeError, IndexError, AttributeError):
        return None, None
    if not parts or parts[0] is None:
        return None, None
    y = int(parts[0])
    s = "-".join(f"{int(p):02d}" if i else str(int(p)) for i, p in enumerate(parts) if p is not None)
    return y, s


def norm_doi(v: str | None) -> str | None:
    if not v:
        return None
    v = re.sub(r"^(?:https?://)?(?:dx\.)?doi\.org/", "", str(v).strip(), flags=re.I)
    v = re.sub(r"^doi:\s*", "", v, flags=re.I)
    return v.lower() or None


def classify_http(status: int) -> str:
    return {400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found",
            410: "not_found", 429: "rate_limited"}.get(status, "server_error" if status >= 500 else f"http_{status}")


# ---------------------------------------------------------------- config / http
@dataclass
class Config:
    provider: str = "auto"
    mailto: str = DEFAULT_MAILTO
    openalex_key: str | None = None
    ncbi_key: str | None = None
    include_abstract: bool = True
    include_bibtex: bool = True
    enrich: bool = True
    abstract_max: int = 4000
    max_authors: int = 100
    search_limit: int = 5
    max_retries: int = 3
    timeout: float = 30.0

    @property
    def secrets(self) -> list[str | None]:
        return [self.openalex_key, self.ncbi_key]

    @property
    def user_agent(self) -> str:
        return f"AcademicWorksMarkdown/{VERSION} (+https://apify.com; mailto:{self.mailto})"


class Throttle:
    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._last: dict[str, float] = {}

    async def wait(self, host: str, interval: float) -> None:
        if interval <= 0:
            return
        lock = self._locks.setdefault(host, asyncio.Lock())
        async with lock:
            now = time.monotonic()
            delay = self._last.get(host, 0.0) + interval - now
            if delay > 0:
                await asyncio.sleep(delay)
            self._last[host] = time.monotonic()


THROTTLE = Throttle()


@dataclass
class Http:
    ok: bool
    status: int | None = None
    data: Any = None
    text: str | None = None
    error_class: str | None = None
    message: str | None = None
    attempts: int = 0


async def http_get(client: httpx.AsyncClient, url: str, cfg: Config, params: dict | None = None,
                   want: str = "json") -> Http:
    host = httpx.URL(url).host
    interval = INTERVALS.get(host, 0.0)
    if host == "eutils.ncbi.nlm.nih.gov" and cfg.ncbi_key and interval:
        interval = min(interval, NCBI_KEYED_INTERVAL)
    headers = {"User-Agent": cfg.user_agent,
               "Accept": "application/json" if want == "json" else "application/atom+xml, application/xml, text/xml"}
    attempts = 0
    last_cls, last_msg, last_status = "network", "", None
    while attempts <= cfg.max_retries:
        attempts += 1
        await THROTTLE.wait(host, interval)
        retry_after = None
        try:
            r = await client.get(url, params=params, headers=headers, timeout=cfg.timeout)
        except httpx.TimeoutException as e:
            last_cls, last_msg, last_status = "timeout", f"timeout: {type(e).__name__}", None
        except httpx.HTTPError as e:
            last_cls, last_msg, last_status = "network", f"{type(e).__name__}: {e}", None
        else:
            last_status = r.status_code
            if 200 <= r.status_code < 300:
                if want == "json":
                    try:
                        return Http(True, r.status_code, data=r.json(), attempts=attempts)
                    except ValueError:
                        return Http(False, r.status_code, error_class="bad_response",
                                    message="Response was not JSON", attempts=attempts)
                return Http(True, r.status_code, text=r.text, attempts=attempts)
            last_cls = classify_http(r.status_code)
            last_msg = f"HTTP {r.status_code}: {r.text[:200]}"
            if last_cls not in ("rate_limited", "server_error"):
                break  # 400/401/403/404 are final
            ra = r.headers.get("Retry-After")
            if ra and ra.strip().isdigit():
                retry_after = min(30.0, float(ra))
        if attempts > cfg.max_retries:
            break
        wait = retry_after if retry_after is not None else min(30.0, BACKOFF_BASE * (2 ** (attempts - 1)))
        if wait > 0:
            await asyncio.sleep(wait + (random.random() * 0.25 if BACKOFF_BASE else 0))
    return Http(False, last_status, error_class=last_cls, message=redact(last_msg, cfg.secrets), attempts=attempts)


# ---------------------------------------------------------------- parsers
def _person_crossref(a: dict) -> str | None:
    if a.get("name"):
        return _s(a["name"])
    n = " ".join(x for x in (_s(a.get("given")), _s(a.get("family"))) if x)
    return n or None


def parse_crossref(m: dict) -> dict:
    year, date = None, None
    for k in ("issued", "published", "published-print", "published-online", "created"):
        year, date = _date_parts(m.get(k))
        if year:
            break
    authors = [n for n in (_person_crossref(a) for a in (m.get("author") or []) if isinstance(a, dict)) if n]
    if not authors:
        authors = [n for n in (_person_crossref(a) for a in (m.get("editor") or []) if isinstance(a, dict)) if n]
    lic = None
    for L in m.get("license") or []:
        if isinstance(L, dict) and L.get("content-version") in ("vor", "am", "unspecified"):
            lic = L.get("URL")
            break
    title = _s(m.get("title"))
    sub = _s(m.get("subtitle"))
    if title and sub and sub.lower() not in title.lower():
        title = f"{title}: {sub}"
    doi = norm_doi(m.get("DOI"))
    return {
        "doi": doi,
        "title": title,
        "authors": authors,
        "year": year,
        "publicationDate": date,
        "venue": _s(m.get("container-title")) or _s(m.get("short-container-title")),
        "publisher": _s(m.get("publisher")),
        "workType": _s(m.get("type")),
        "volume": _s(m.get("volume")),
        "issue": _s(m.get("issue")),
        "pages": _s(m.get("page")),
        "issn": (m.get("ISSN") or [None])[0],
        "abstract": strip_markup(m.get("abstract")),
        "citedByCrossref": m.get("is-referenced-by-count") if isinstance(m.get("is-referenced-by-count"), int) else None,
        "referencesCount": m.get("references-count") if isinstance(m.get("references-count"), int) else None,
        "license": lic,
        "landingUrl": f"https://doi.org/{doi}" if doi else _s(m.get("URL")),
        "keywords": [s for s in (m.get("subject") or []) if isinstance(s, str)][:10],
    }


def parse_openalex(w: dict) -> dict:
    ids = w.get("ids") or {}
    pmid = None
    if ids.get("pmid"):
        mm = re.search(r"(\d+)\s*$", str(ids["pmid"]))
        pmid = mm.group(1) if mm else None
    pmcid = None
    if ids.get("pmcid"):
        mm = re.search(r"(PMC\d+)", str(ids["pmcid"]), re.I)
        pmcid = mm.group(1).upper() if mm else None
    oa = w.get("open_access") or {}
    best = w.get("best_oa_location") or {}
    prim = w.get("primary_location") or {}
    src = prim.get("source") or {}
    biblio = w.get("biblio") or {}
    fp, lp = _s(biblio.get("first_page")), _s(biblio.get("last_page"))
    pages = f"{fp}-{lp}" if fp and lp and fp != lp else fp
    is_oa = oa.get("is_oa") if isinstance(oa.get("is_oa"), bool) else None
    authors = []
    for a in w.get("authorships") or []:
        n = _s(((a or {}).get("author") or {}).get("display_name")) or _s((a or {}).get("raw_author_name"))
        if n:
            authors.append(n)
    pdf = _s(best.get("pdf_url")) if is_oa else None
    kws = [_s(k.get("display_name")) for k in (w.get("keywords") or []) if isinstance(k, dict)]
    lic = _s(best.get("license")) if is_oa else None
    return {
        "doi": norm_doi(w.get("doi")),
        "pmid": pmid,
        "pmcid": pmcid,
        "openAlexId": (_s(w.get("id")) or "").rsplit("/", 1)[-1] or None,
        "title": _s(w.get("title")) or _s(w.get("display_name")),
        "authors": authors,
        "year": w.get("publication_year") if isinstance(w.get("publication_year"), int) else None,
        "publicationDate": _s(w.get("publication_date")),
        "venue": _s(src.get("display_name")),
        "publisher": _s(src.get("host_organization_name")),
        "workType": _s(w.get("type")),
        "volume": _s(biblio.get("volume")),
        "issue": _s(biblio.get("issue")),
        "pages": pages,
        "abstract": inverted_index_to_text(w.get("abstract_inverted_index")),
        "citedByOpenAlex": w.get("cited_by_count") if isinstance(w.get("cited_by_count"), int) else None,
        "referencesCount": w.get("referenced_works_count") if isinstance(w.get("referenced_works_count"), int) else None,
        "isOA": is_oa,
        "oaStatus": _s(oa.get("oa_status")),
        "oaUrl": _s(oa.get("oa_url")) if is_oa else None,
        "pdfUrl": pdf,
        "license": lic,
        "landingUrl": _s(prim.get("landing_page_url")),
        "isRetracted": w.get("is_retracted") if isinstance(w.get("is_retracted"), bool) else None,
        "keywords": [k for k in kws if k][:10],
    }


def _itxt(el: ET.Element | None) -> str | None:
    if el is None:
        return None
    return _s("".join(el.itertext()))


def parse_pubmed_xml(xml_text: str) -> list[dict]:
    try:
        root = ET.fromstring(xml_text.encode("utf-8") if isinstance(xml_text, str) else xml_text)
    except ET.ParseError:
        raise ValueError("PubMed returned invalid XML")
    out = []
    for art in root.findall("PubmedArticle"):
        mc = art.find("MedlineCitation")
        if mc is None:
            continue
        a = mc.find("Article")
        if a is None:
            continue
        pmid = _itxt(mc.find("PMID"))
        j = a.find("Journal")
        ji = j.find("JournalIssue") if j is not None else None
        pd = ji.find("PubDate") if ji is not None else None
        year = None
        date = None
        if pd is not None:
            y = _itxt(pd.find("Year"))
            md = _itxt(pd.find("MedlineDate"))
            ys = y or (re.search(r"(\d{4})", md).group(1) if md and re.search(r"\d{4}", md) else None)
            year = int(ys) if ys else None
            mon, day = _itxt(pd.find("Month")), _itxt(pd.find("Day"))
            date = " ".join(x for x in (y or md, mon, day) if x) or None
        # abstract (structured sections keep their labels)
        parts = []
        ab = a.find("Abstract")
        if ab is not None:
            for t in ab.findall("AbstractText"):
                txt = _itxt(t)
                if not txt:
                    continue
                lab = t.get("Label")
                parts.append(f"{lab}: {txt}" if lab else txt)
        authors = []
        al = a.find("AuthorList")
        if al is not None:
            for au in al.findall("Author"):
                coll = _itxt(au.find("CollectiveName"))
                if coll:
                    authors.append(coll)
                    continue
                fn, ln = _itxt(au.find("ForeName")), _itxt(au.find("LastName"))
                suf = _itxt(au.find("Suffix"))
                n = " ".join(x for x in (fn, ln, suf) if x)
                if n:
                    authors.append(n)
        doi = pmcid = None
        pdata = art.find("PubmedData")
        if pdata is not None:
            for aid in pdata.findall("ArticleIdList/ArticleId"):  # article's OWN ids (not ReferenceList)
                t = aid.get("IdType")
                if t == "doi" and not doi:
                    doi = norm_doi(_itxt(aid))
                elif t == "pmc" and not pmcid:
                    pmcid = (_itxt(aid) or "").upper() or None
        if not doi:
            for el in a.findall("ELocationID"):
                if el.get("EIdType") == "doi":
                    doi = norm_doi(_itxt(el))
                    break
        mesh = [_itxt(d) for d in mc.findall("MeshHeadingList/MeshHeading/DescriptorName")]
        kw = [_itxt(k) for k in mc.findall("KeywordList/Keyword")]
        ptypes = [_itxt(p) for p in a.findall("PublicationTypeList/PublicationType")]
        out.append({
            "pmid": pmid,
            "doi": doi,
            "pmcid": pmcid,
            "title": _itxt(a.find("ArticleTitle")) or _itxt(a.find("VernacularTitle")),
            "authors": authors,
            "year": year,
            "publicationDate": date,
            "venue": _itxt(j.find("Title")) if j is not None else None,
            "venueAbbrev": _itxt(j.find("ISOAbbreviation")) if j is not None else None,
            "workType": next((p for p in ptypes if p), None),
            "volume": _itxt(ji.find("Volume")) if ji is not None else None,
            "issue": _itxt(ji.find("Issue")) if ji is not None else None,
            "pages": _itxt(a.find("Pagination/MedlinePgn")),
            "issn": _itxt(j.find("ISSN")) if j is not None else None,
            "abstract": "\n\n".join(parts) or None,
            "keywords": [k for k in (mesh + kw) if k][:12],
            "landingUrl": f"https://doi.org/{doi}" if doi else (f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None),
        })
    return out


ATOM = "{http://www.w3.org/2005/Atom}"
ARX = "{http://arxiv.org/schemas/atom}"
OS = "{http://a9.com/-/spec/opensearch/1.1/}"


def parse_arxiv_atom(xml_text: str) -> list[dict]:
    try:
        root = ET.fromstring(xml_text.encode("utf-8") if isinstance(xml_text, str) else xml_text)
    except ET.ParseError:
        raise ValueError("arXiv returned invalid XML")
    out = []
    for e in root.findall(f"{ATOM}entry"):
        eid = _itxt(e.find(f"{ATOM}id")) or ""
        title = _itxt(e.find(f"{ATOM}title"))
        if "/api/errors" in eid or (title or "").strip().lower() == "error":
            continue
        m = re.search(r"arxiv\.org/abs/(.+)$", eid)
        full = m.group(1) if m else None
        if not full:
            continue
        base = strip_arxiv_version(full)
        ver = full[len(base):] or None
        published = _itxt(e.find(f"{ATOM}published"))
        year = int(published[:4]) if published and published[:4].isdigit() else None
        authors = [n for n in (_itxt(a.find(f"{ATOM}name")) for a in e.findall(f"{ATOM}author")) if n]
        cats = [c.get("term") for c in e.findall(f"{ATOM}category") if c.get("term")]
        prim = e.find(f"{ARX}primary_category")
        jdoi = norm_doi(_itxt(e.find(f"{ARX}doi")))
        out.append({
            "arxivId": base,
            "arxivVersion": ver,
            "title": title,
            "authors": authors,
            "year": year,
            "publicationDate": published[:10] if published else None,
            "updated": (_itxt(e.find(f"{ATOM}updated")) or "")[:10] or None,
            "abstract": _itxt(e.find(f"{ATOM}summary")),
            "journalRef": _itxt(e.find(f"{ARX}journal_ref")),
            "journalDoi": jdoi,
            "comment": _itxt(e.find(f"{ARX}comment")),
            "arxivPrimaryCategory": prim.get("term") if prim is not None else (cats[0] if cats else None),
            "keywords": cats[:10],
            "venue": _itxt(e.find(f"{ARX}journal_ref")) or "arXiv",
            "workType": "preprint",
            "isOA": True,
            "oaStatus": "green",
            "landingUrl": f"https://arxiv.org/abs/{base}",
            "pdfUrl": f"https://arxiv.org/pdf/{base}",  # arXiv is free to read; link only, never downloaded
        })
    return out


# ---------------------------------------------------------------- source fetchers
@dataclass
class Fetch:
    ok: bool
    records: list[dict] = field(default_factory=list)
    error_class: str | None = None
    http_status: int | None = None
    message: str | None = None
    attempts: int = 0
    source: str = ""
    url: str | None = None


def _fail(source: str, h: Http, url: str | None = None) -> Fetch:
    return Fetch(False, error_class=h.error_class, http_status=h.status, message=h.message,
                 attempts=h.attempts, source=source, url=url)


def _oa_params(cfg: Config, extra: dict | None = None) -> dict:
    p = dict(extra or {})
    if cfg.mailto:
        p["mailto"] = cfg.mailto
    if cfg.openalex_key:
        p["api_key"] = cfg.openalex_key
    return p


def _ncbi_params(cfg: Config, extra: dict) -> dict:
    p = dict(extra)
    p["tool"] = NCBI_TOOL
    if cfg.mailto:
        p["email"] = cfg.mailto
    if cfg.ncbi_key:
        p["api_key"] = cfg.ncbi_key
    return p


async def crossref_doi(client, doi: str, cfg: Config) -> Fetch:
    url = f"{CROSSREF}/works/{quote(doi, safe='/:;()._-')}"
    h = await http_get(client, url, cfg, params={"mailto": cfg.mailto} if cfg.mailto else None)
    if not h.ok:
        return _fail("crossref", h, url)
    msg = (h.data or {}).get("message") if isinstance(h.data, dict) else None
    if not isinstance(msg, dict):
        return Fetch(False, error_class="bad_response", http_status=h.status, message="No message object",
                     attempts=h.attempts, source="crossref", url=url)
    return Fetch(True, [parse_crossref(msg)], attempts=h.attempts, source="crossref", url=url)


async def crossref_search(client, q: str, n: int, cfg: Config) -> Fetch:
    url = f"{CROSSREF}/works"
    params = {"query.bibliographic": q, "rows": min(100, max(n + 3, n)), "mailto": cfg.mailto,
              "select": "DOI,title,subtitle,author,editor,issued,published,container-title,publisher,type,volume,"
                        "issue,page,ISSN,abstract,is-referenced-by-count,references-count,license,URL,subject"}
    h = await http_get(client, url, cfg, params=params)
    if not h.ok:
        return _fail("crossref", h, url)
    items = (((h.data or {}).get("message") or {}).get("items")) or []
    recs = [parse_crossref(i) for i in items if isinstance(i, dict) and i.get("type") not in ("component",)]
    return Fetch(True, recs[:n], attempts=h.attempts, source="crossref", url=url)


async def openalex_get(client, key: str, cfg: Config) -> Fetch:
    """key: 'https://doi.org/<doi>' | 'pmid:<n>' | 'W123'."""
    url = f"{OPENALEX}/works/{key}"
    h = await http_get(client, url, cfg, params=_oa_params(cfg))
    if not h.ok:
        return _fail("openalex", h, url)
    if not isinstance(h.data, dict) or not h.data.get("id"):
        return Fetch(False, error_class="bad_response", http_status=h.status, message="No work object",
                     attempts=h.attempts, source="openalex", url=url)
    return Fetch(True, [parse_openalex(h.data)], attempts=h.attempts, source="openalex", url=url)


async def openalex_search(client, q: str, n: int, cfg: Config) -> Fetch:
    url = f"{OPENALEX}/works"
    h = await http_get(client, url, cfg, params=_oa_params(cfg, {"search": q, "per-page": min(200, n)}))
    if not h.ok:
        return _fail("openalex", h, url)
    res = (h.data or {}).get("results") if isinstance(h.data, dict) else None
    if not isinstance(res, list):
        return Fetch(False, error_class="bad_response", http_status=h.status, message="No results list",
                     attempts=h.attempts, source="openalex", url=url)
    return Fetch(True, [parse_openalex(w) for w in res[:n] if isinstance(w, dict)],
                 attempts=h.attempts, source="openalex", url=url)


async def pubmed_fetch(client, pmids: list[str], cfg: Config) -> Fetch:
    url = f"{EUTILS}/efetch.fcgi"
    h = await http_get(client, url, cfg, params=_ncbi_params(cfg, {"db": "pubmed", "id": ",".join(pmids),
                                                                     "retmode": "xml"}), want="text")
    if not h.ok:
        return _fail("pubmed", h, url)
    try:
        recs = parse_pubmed_xml(h.text or "")
    except ValueError as e:
        return Fetch(False, error_class="bad_response", http_status=h.status, message=str(e),
                     attempts=h.attempts, source="pubmed", url=url)
    return Fetch(True, recs, attempts=h.attempts, source="pubmed", url=url)


async def pubmed_search(client, term: str, n: int, cfg: Config) -> Fetch:
    url = f"{EUTILS}/esearch.fcgi"
    h = await http_get(client, url, cfg, params=_ncbi_params(cfg, {"db": "pubmed", "term": term,
                                                                     "retmode": "json", "retmax": n}))
    if not h.ok:
        return _fail("pubmed", h, url)
    ids = (((h.data or {}).get("esearchresult") or {}).get("idlist")) if isinstance(h.data, dict) else None
    if ids is None:
        return Fetch(False, error_class="bad_response", http_status=h.status, message="No esearchresult",
                     attempts=h.attempts, source="pubmed", url=url)
    if not ids:
        return Fetch(True, [], attempts=h.attempts, source="pubmed", url=url)
    f = await pubmed_fetch(client, [str(i) for i in ids[:n]], cfg)
    f.attempts += h.attempts
    return f


async def arxiv_query(client, cfg: Config, *, id_list: str | None = None, search: str | None = None,
                      n: int = 10) -> Fetch:
    params: dict[str, Any] = {"max_results": n}
    if id_list:
        params["id_list"] = id_list
    if search:
        words = [w for w in re.split(r"\s+", search.strip()) if w]
        params["search_query"] = " AND ".join(f"all:{w}" for w in words)
        params["sortBy"] = "relevance"
    h = await http_get(client, ARXIV, cfg, params=params, want="text")
    if not h.ok:
        return _fail("arxiv", h, ARXIV)
    try:
        recs = parse_arxiv_atom(h.text or "")
    except ValueError as e:
        return Fetch(False, error_class="bad_response", http_status=h.status, message=str(e),
                     attempts=h.attempts, source="arxiv", url=ARXIV)
    return Fetch(True, recs, attempts=h.attempts, source="arxiv", url=ARXIV)


# ---------------------------------------------------------------- merge / finalize
MERGE_KEYS = ("doi", "pmid", "pmcid", "arxivId", "openAlexId", "title", "authors", "year", "publicationDate",
              "venue", "publisher", "workType", "volume", "issue", "pages", "issn", "abstract", "license",
              "landingUrl", "keywords", "referencesCount")


def merge(primary: dict, extra: dict) -> dict:
    """Fill gaps in `primary` from `extra`. Never overwrites a present primary value (except OA facts)."""
    out = dict(primary)
    for k in MERGE_KEYS:
        if out.get(k) in (None, "", []) and extra.get(k) not in (None, "", []):
            out[k] = extra[k]
    for k in ("citedByCrossref", "citedByOpenAlex", "isRetracted"):
        if extra.get(k) is not None and out.get(k) is None:
            out[k] = extra[k]
    # OA facts come from OpenAlex (Crossref / PubMed don't assert OA). arXiv already says OA.
    if extra.get("isOA") is not None and out.get("isOA") is None:
        out["isOA"] = extra["isOA"]
        out["oaStatus"] = extra.get("oaStatus")
        out["oaUrl"] = extra.get("oaUrl")
    if not out.get("pdfUrl") and extra.get("pdfUrl") and (extra.get("isOA") or out.get("isOA")):
        out["pdfUrl"] = extra["pdfUrl"]
    return out


def _bib_escape(s: str) -> str:
    return re.sub(r"([&%$#_])", r"\\\1", s).replace("{", "\\{").replace("}", "\\}")


def _family(name: str) -> str:
    parts = name.split()
    if len(parts) > 1 and parts[-1].rstrip(".") in ("Jr", "Sr", "II", "III", "IV"):
        parts = parts[:-1]
    return parts[-1] if parts else name


def bibtex(r: dict) -> str | None:
    """BibTeX built ONLY from fields present in the record — nothing is invented."""
    title = r.get("title")
    if not title:
        return None
    authors = r.get("authors") or []
    first = re.sub(r"[^A-Za-z0-9]", "", _family(authors[0])) if authors else "anon"
    word = next((re.sub(r"[^A-Za-z0-9]", "", w) for w in title.split() if len(re.sub(r"[^A-Za-z0-9]", "", w)) > 3), "work")
    key = f"{first.lower() or 'anon'}{r.get('year') or ''}{word.lower()}"
    wt = (r.get("workType") or "").lower()
    if r.get("arxivId") and (wt == "preprint" or not r.get("pmid") and r.get("venue") in (None, "arXiv")):
        etype = "misc"
    elif wt in ("proceedings-article", "conference-paper"):
        etype = "inproceedings"
    elif wt in ("book-chapter", "book-section"):
        etype = "incollection"
    elif wt in ("book", "monograph", "edited-book"):
        etype = "book"
    elif wt in ("dissertation",):
        etype = "phdthesis"
    elif wt in ("posted-content", "preprint", "dataset", "report"):
        etype = "misc"
    else:
        etype = "article"
    f: list[tuple[str, str]] = [("title", "{" + _bib_escape(title) + "}")]
    if authors:
        f.append(("author", " and ".join(_bib_escape(a) for a in authors[:100]) + (" and others" if len(authors) > 100 else "")))
    if r.get("year"):
        f.append(("year", str(r["year"])))
    venue = r.get("venue")
    if venue and etype == "article":
        f.append(("journal", _bib_escape(venue)))
    elif venue and etype in ("inproceedings", "incollection"):
        f.append(("booktitle", _bib_escape(venue)))
    for k in ("volume", "pages"):
        if r.get(k):
            f.append((k, _bib_escape(str(r[k])).replace("-", "--") if k == "pages" else _bib_escape(str(r[k]))))
    if r.get("issue"):
        f.append(("number", _bib_escape(str(r["issue"]))))
    if r.get("publisher") and etype in ("book", "incollection", "inproceedings"):
        f.append(("publisher", _bib_escape(r["publisher"])))
    if r.get("doi"):
        f.append(("doi", r["doi"]))
    if r.get("arxivId"):
        f += [("eprint", r["arxivId"]), ("archivePrefix", "arXiv")]
        if r.get("arxivPrimaryCategory"):
            f.append(("primaryClass", r["arxivPrimaryCategory"]))
    if r.get("pmid"):
        f.append(("pmid", r["pmid"]))
    if r.get("landingUrl"):
        f.append(("url", r["landingUrl"]))
    body = ",\n".join(f"  {k} = {{{v}}}" if not v.startswith("{") else f"  {k} = {{{v}}}" for k, v in f)
    return f"@{etype}{{{key},\n{body}\n}}"


def finalize(rec: dict, *, target: Target, sources: list[str], cfg: Config, kind: str = "work",
             query: str | None = None, enrichment_errors: list[str] | None = None) -> dict:
    r = dict(rec)
    authors = r.get("authors") or []
    r["authorCount"] = len(authors)
    r["authors"] = authors[: cfg.max_authors]
    r["authorsTruncated"] = len(authors) > cfg.max_authors
    if cfg.include_abstract:
        r["abstract"], r["abstractTruncated"] = truncate(r.get("abstract"), cfg.abstract_max)
    else:
        r["abstract"], r["abstractTruncated"] = None, False
    if r.get("citedByOpenAlex") is not None:
        r["citedBy"], r["citedBySource"] = r["citedByOpenAlex"], "openalex"
    elif r.get("citedByCrossref") is not None:
        r["citedBy"], r["citedBySource"] = r["citedByCrossref"], "crossref"
    else:
        r["citedBy"], r["citedBySource"] = None, None
    if not r.get("isOA"):
        r["pdfUrl"] = None  # never link a possibly paywalled PDF
    if r.get("doi") and not r.get("landingUrl"):
        r["landingUrl"] = f"https://doi.org/{r['doi']}"
    r["doiUrl"] = f"https://doi.org/{r['doi']}" if r.get("doi") else None
    r["pubmedUrl"] = f"https://pubmed.ncbi.nlm.nih.gov/{r['pmid']}/" if r.get("pmid") else None
    r["pmcUrl"] = f"https://pmc.ncbi.nlm.nih.gov/articles/{r['pmcid']}/" if r.get("pmcid") else None
    r["arxivUrl"] = f"https://arxiv.org/abs/{r['arxivId']}" if r.get("arxivId") else None
    r["openAlexUrl"] = f"https://openalex.org/{r['openAlexId']}" if r.get("openAlexId") else None
    r["bibtex"] = bibtex(r) if cfg.include_bibtex else None
    r.update({
        "kind": kind,
        "idType": target.kind if kind == "work" else "query",
        "inputRaw": target.raw or target.value,
        "query": query,
        "source": sources[0] if sources else None,
        "sources": sources,
        "sourceLabel": " + ".join(SOURCE_LABELS.get(s, s) for s in sources),
        "enrichmentErrors": enrichment_errors or [],
        "status": "ok",
        "errorClass": None, "httpStatus": None, "errorMessage": None, "hint": None,
    })
    return r


def failed_row(target: Target, f: Fetch | None, error_class: str | None = None, message: str | None = None,
               tried: list[str] | None = None) -> dict:
    cls = error_class or (f.error_class if f else None) or "error"
    return {
        "kind": "work" if target.kind != "query" else "search_hit",
        "idType": target.kind,
        "inputRaw": target.raw or target.value,
        "query": target.value if target.kind == "query" else None,
        "doi": target.value if target.kind == "doi" else None,
        "pmid": target.value if target.kind == "pmid" else None,
        "arxivId": target.value if target.kind == "arxiv" else None,
        "title": None,
        "status": "failed",
        "source": f.source if f else None,
        "sources": tried or ([f.source] if f and f.source else []),
        "errorClass": cls,
        "httpStatus": f.http_status if f else None,
        "attempts": f.attempts if f else 0,
        "errorMessage": message or (f.message if f else None),
        "hint": HINTS.get(cls),
    }


# ---------------------------------------------------------------- orchestration
@dataclass
class LookupResult:
    ok: bool
    items: list[dict] = field(default_factory=list)


async def _enrich_openalex(client, rec: dict, cfg: Config, errs: list[str], sources: list[str],
                           key: str | None = None) -> dict:
    if not key:
        if rec.get("doi"):
            key = f"https://doi.org/{rec['doi']}"
        elif rec.get("pmid"):
            key = f"pmid:{rec['pmid']}"
        else:
            return rec
    f = await openalex_get(client, key, cfg)
    if f.ok and f.records:
        sources.append("openalex")
        return merge(rec, f.records[0])
    errs.append(f"openalex: {f.error_class}")
    return rec


async def lookup_id(client, t: Target, cfg: Config) -> LookupResult:
    p = cfg.provider
    errs: list[str] = []
    sources: list[str] = []
    arx_from_doi = arxiv_from_doi(t.value) if t.kind == "doi" else None

    # ---------- arXiv id or arXiv DataCite DOI
    if t.kind == "arxiv" or (arx_from_doi and p in ("auto", "arxiv")):
        if p not in ("auto", "arxiv", "openalex"):
            return LookupResult(False, [failed_row(t, None, "unsupported_for_provider")])
        aid = t.value if t.kind == "arxiv" else arx_from_doi
        if p == "openalex":
            f = await openalex_get(client, f"https://doi.org/{arxiv_doi(aid)}", cfg)
            if not f.ok or not f.records:
                return LookupResult(False, [failed_row(t, f)])
            return LookupResult(True, [finalize(f.records[0], target=t, sources=["openalex"], cfg=cfg)])
        f = await arxiv_query(client, cfg, id_list=aid, n=1)
        if not f.ok:
            return LookupResult(False, [failed_row(t, f)])
        if not f.records:
            return LookupResult(False, [failed_row(t, f, "not_found", f"arXiv has no entry for {aid}")])
        rec = f.records[0]
        rec["doi"] = rec.get("journalDoi") or arxiv_doi(rec["arxivId"])
        sources.append("arxiv")
        if p == "auto" and cfg.enrich:
            # Enrich by the journal DOI if arXiv lists one, else by the arXiv DataCite DOI (exact matches only).
            rec = await _enrich_openalex(client, rec, cfg, errs, sources, key=f"https://doi.org/{rec['doi']}")
        return LookupResult(True, [finalize(rec, target=t, sources=sources, cfg=cfg, enrichment_errors=errs)])

    # ---------- PMID
    if t.kind == "pmid":
        if p in ("crossref", "arxiv"):
            return LookupResult(False, [failed_row(t, None, "unsupported_for_provider")])
        if p == "openalex":
            f = await openalex_get(client, f"pmid:{t.value}", cfg)
            if not f.ok or not f.records:
                return LookupResult(False, [failed_row(t, f)])
            return LookupResult(True, [finalize(f.records[0], target=t, sources=["openalex"], cfg=cfg)])
        f = await pubmed_fetch(client, [t.value], cfg)
        if not f.ok:
            return LookupResult(False, [failed_row(t, f)])
        rec = next((r for r in f.records if r.get("pmid") == t.value), None)
        if not rec:
            return LookupResult(False, [failed_row(t, f, "not_found", f"PubMed has no record for PMID {t.value}")])
        sources.append("pubmed")
        if p == "auto" and cfg.enrich:
            rec = await _enrich_openalex(client, rec, cfg, errs, sources, key=f"pmid:{t.value}")
        return LookupResult(True, [finalize(rec, target=t, sources=sources, cfg=cfg, enrichment_errors=errs)])

    # ---------- DOI
    if t.kind == "doi":
        if p == "arxiv":
            return LookupResult(False, [failed_row(t, None, "unsupported_for_provider")])
        if p == "openalex":
            f = await openalex_get(client, f"https://doi.org/{t.value}", cfg)
            if not f.ok or not f.records:
                return LookupResult(False, [failed_row(t, f)])
            return LookupResult(True, [finalize(f.records[0], target=t, sources=["openalex"], cfg=cfg)])
        if p == "pubmed":
            f = await pubmed_search(client, f"{t.value}[doi]", 1, cfg)
            if not f.ok:
                return LookupResult(False, [failed_row(t, f)])
            if not f.records:
                return LookupResult(False, [failed_row(t, f, "not_found", "PubMed has no record with this DOI")])
            return LookupResult(True, [finalize(f.records[0], target=t, sources=["pubmed"], cfg=cfg)])
        f = await crossref_doi(client, t.value, cfg)
        if f.ok and f.records:
            rec = f.records[0]
            sources.append("crossref")
            if p == "auto" and cfg.enrich:
                rec = await _enrich_openalex(client, rec, cfg, errs, sources)
            return LookupResult(True, [finalize(rec, target=t, sources=sources, cfg=cfg, enrichment_errors=errs)])
        if p == "crossref" or f.error_class not in ("not_found",):
            return LookupResult(False, [failed_row(t, f)])
        # auto fallback: DOI not registered with Crossref (e.g. DataCite) → OpenAlex
        f2 = await openalex_get(client, f"https://doi.org/{t.value}", cfg)
        if f2.ok and f2.records:
            return LookupResult(True, [finalize(f2.records[0], target=t, sources=["openalex"], cfg=cfg,
                                                enrichment_errors=["crossref: not_found"])])
        row = failed_row(t, f2 if f2.error_class != "not_found" else f, tried=["crossref", "openalex"])
        if f2.error_class == "not_found":
            row["errorMessage"] = "DOI not found in Crossref or OpenAlex"
        return LookupResult(False, [row])

    return LookupResult(False, [failed_row(t, None, "invalid_input")])


async def search_query(client, t: Target, cfg: Config, limit: int) -> LookupResult:
    p = cfg.provider
    n = max(1, limit)
    if p == "crossref":
        f = await crossref_search(client, t.value, n, cfg)
    elif p == "pubmed":
        f = await pubmed_search(client, t.value, n, cfg)
    elif p == "arxiv":
        f = await arxiv_query(client, cfg, search=t.value, n=n)
    else:  # auto | openalex
        f = await openalex_search(client, t.value, n, cfg)
    if not f.ok:
        return LookupResult(False, [failed_row(t, f)])
    recs = [r for r in f.records if r.get("title")][:n]
    if not recs:
        return LookupResult(False, [failed_row(t, f, "not_found", f"No {SOURCE_LABELS.get(f.source)} hits for this query")])
    items = []
    for r in recs:
        if f.source == "arxiv" and not r.get("doi"):
            r["doi"] = r.get("journalDoi") or arxiv_doi(r["arxivId"])
        items.append(finalize(r, target=t, sources=[f.source], cfg=cfg, kind="search_hit", query=t.value))
    return LookupResult(True, items)


async def lookup_target(client, t: Target, cfg: Config, search_limit: int | None = None) -> LookupResult:
    if t.kind == "query":
        return await search_query(client, t, cfg, search_limit or cfg.search_limit)
    return await lookup_id(client, t, cfg)
