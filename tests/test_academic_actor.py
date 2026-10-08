import asyncio
import json
import os

import httpx
import pytest

import client as C
import inputs as I
import report as R

HERE = os.path.dirname(__file__)
TXT = lambda n: open(os.path.join(HERE, "fixtures", n), encoding="utf-8").read()  # noqa: E731
FX = lambda n: json.loads(TXT(n))  # noqa: E731


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(C, "BACKOFF_BASE", 0)
    monkeypatch.setattr(C, "INTERVALS", {k: 0.0 for k in C.INTERVALS})


def run(c):
    return asyncio.run(c)


def client_for(routes, calls=None):
    def handler(req):
        if calls is not None:
            calls.append(req)
        url = str(req.url)
        for key, val in routes.items():
            if key in url:
                if callable(val):
                    return val(req)
                status, body = val
                if isinstance(body, str):
                    return httpx.Response(status, text=body)
                return httpx.Response(status, json=body)
        return httpx.Response(404, text="unrouted")
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def T(raw):
    t = I.detect_id(raw)
    assert t, raw
    return t


# ---------- id detection ----------
def test_detect_doi_variants():
    for raw in ["10.1038/nature14539", "https://doi.org/10.1038/nature14539", "doi:10.1038/nature14539"]:
        t = T(raw)
        assert t.kind == "doi" and t.value.lower() == "10.1038/nature14539"


def test_detect_arxiv_variants():
    for raw in ["1706.03762", "arXiv:1706.03762v5", "https://arxiv.org/abs/1706.03762"]:
        t = T(raw)
        assert t.kind == "arxiv" and t.value.startswith("1706.03762")
    assert T("hep-th/9901001").kind == "arxiv"


def test_detect_pmid_and_invalid():
    assert T("26017442").kind == "pmid"
    assert T("PMID: 26017442").value == "26017442"
    assert I.detect_id("hello world??") is None


def test_arxiv_doi_roundtrip():
    assert I.arxiv_from_doi("10.48550/arXiv.1706.03762") == "1706.03762"
    assert I.arxiv_doi("1706.03762").lower() == "10.48550/arxiv.1706.03762"


def test_collect_targets_dedupe_and_bad():
    good, bad = I.collect_targets(ids=["10.1038/nature14539", "https://doi.org/10.1038/NATURE14539", "zzz!!"],
                                  queries=["deep learning", "deep learning"])
    kinds = [t.kind for t in good]
    assert kinds.count("doi") == 1 and kinds.count("query") == 1
    assert bad == ["zzz!!"]


def test_split_payload_variants():
    g, _ = I.split_payload([{"doi": "10.1038/nature14539"}, {"pmid": "26017442"}, "1706.03762"])
    assert {t.kind for t in g} == {"doi", "pmid", "arxiv"}


# ---------- parsers ----------
def test_parse_crossref_nature():
    r = C.parse_crossref(FX("crossref_nature.json")["message"])
    assert r["doi"].lower() == "10.1038/nature14539"
    assert "Deep learning" in r["title"] and r["year"] == 2015
    assert any("LeCun" in a for a in r["authors"])


def test_parse_openalex_nature():
    r = C.parse_openalex(FX("openalex_nature.json"))
    assert r["openAlexId"] and r["citedByOpenAlex"] == 85363


def test_parse_pubmed_and_arxiv():
    recs = C.parse_pubmed_xml(TXT("pubmed_efetch_nature.xml"))
    assert recs[0]["pmid"] == "26017442"
    assert C.parse_pubmed_xml(TXT("pubmed_efetch_missing.xml")) == []
    a = C.parse_arxiv_atom(TXT("arxiv_attention.xml"))
    assert a[0]["arxivId"].startswith("1706.03762") and "Attention" in a[0]["title"]
    assert C.parse_arxiv_atom(TXT("arxiv_missing.xml")) == []


def test_inverted_index_abstract():
    assert C.inverted_index_to_text({"Hello": [0], "world": [1]}) == "Hello world"


# ---------- lookups ----------
def test_doi_auto_crossref_plus_openalex():
    calls = []

    async def go():
        async with client_for({"api.crossref.org": (200, FX("crossref_nature.json")),
                               "api.openalex.org": (200, FX("openalex_nature.json"))}, calls) as c:
            return await C.lookup_target(c, T("10.1038/nature14539"), C.Config())
    res = run(go())
    assert res.ok and res.items[0]["status"] == "ok"
    assert res.items[0]["sources"] == ["crossref", "openalex"]
    ua = calls[0].headers["user-agent"]
    assert "mailto:" in ua and "noreply" in ua


def test_doi_not_found_both_free():
    async def go():
        async with client_for({"api.crossref.org": (404, "Resource not found."),
                               "api.openalex.org": (404, "nf")}) as c:
            return await C.lookup_target(c, T("10.9999/does-not-exist-xyz"), C.Config())
    res = run(go())
    assert not res.ok and res.items[0]["errorClass"] == "not_found" and res.items[0]["status"] == "failed"


def test_doi_404_no_retry():
    calls = []

    async def go():
        async with client_for({"api.crossref.org": (404, "x"), "api.openalex.org": (404, "x")}, calls) as c:
            return await C.lookup_target(c, T("10.9999/x"), C.Config(max_retries=3))
    run(go())
    assert len(calls) == 2


def test_arxiv_lookup():
    async def go():
        async with client_for({"export.arxiv.org": (200, TXT("arxiv_attention.xml")),
                               "api.openalex.org": (404, "nf")}) as c:
            return await C.lookup_target(c, T("1706.03762"), C.Config())
    res = run(go())
    it = res.items[0]
    assert res.ok and it["arxivId"].startswith("1706.03762") and "arxiv" in it["sources"]


def test_arxiv_missing_free():
    async def go():
        async with client_for({"export.arxiv.org": (200, TXT("arxiv_missing.xml"))}) as c:
            return await C.lookup_target(c, T("2399.99999"), C.Config(enrich=False))
    res = run(go())
    assert not res.ok and res.items[0]["errorClass"] == "not_found"


def test_pmid_lookup_and_missing():
    async def ok():
        async with client_for({"efetch": (200, TXT("pubmed_efetch_nature.xml")),
                               "api.openalex.org": (200, FX("openalex_nature.json"))}) as c:
            return await C.lookup_target(c, T("26017442"), C.Config())
    res = run(ok())
    assert res.ok and res.items[0]["pmid"] == "26017442" and res.items[0]["sources"][0] == "pubmed"

    async def miss():
        async with client_for({"efetch": (200, TXT("pubmed_efetch_missing.xml"))}) as c:
            return await C.lookup_target(c, T("99999999"), C.Config())
    res = run(miss())
    assert not res.ok and res.items[0]["errorClass"] == "not_found"


def test_429_retry_then_ok():
    n = {"i": 0}

    def h(req):
        n["i"] += 1
        return httpx.Response(429, text="slow") if n["i"] == 1 else httpx.Response(200, json=FX("crossref_nature.json"))

    async def go():
        async with client_for({"api.crossref.org": h}) as c:
            return await C.lookup_target(c, T("10.1038/nature14539"), C.Config(enrich=False))
    res = run(go())
    assert res.ok and n["i"] == 2


def test_query_search_openalex():
    async def go():
        async with client_for({"api.openalex.org": (200, FX("openalex_search.json"))}) as c:
            return await C.lookup_target(c, I.Target("query", "deep learning", "deep learning"), C.Config(), 3)
    res = run(go())
    assert res.ok and 1 <= len(res.items) <= 3 and all(i["kind"] == "search_hit" for i in res.items)


def test_keys_sent_and_redacted():
    calls = []

    async def go():
        async with client_for({"api.openalex.org": (500, "boom secretKEY123")}, calls) as c:
            cfg = C.Config(provider="openalex", openalex_key="secretKEY123", max_retries=0)
            return await C.lookup_target(c, T("10.1038/nature14539"), cfg)
    res = run(go())
    assert "api_key=secretKEY123" in str(calls[0].url)
    assert "secretKEY123" not in json.dumps(res.items)


# ---------- report / charging ----------
def _ok_card():
    async def go():
        async with client_for({"api.crossref.org": (200, FX("crossref_nature.json")),
                               "api.openalex.org": (200, FX("openalex_nature.json"))}) as c:
            return await C.lookup_target(c, T("10.1038/nature14539"), C.Config())
    return run(go()).items[0]


def test_markdown_sections_and_chunks():
    it = _ok_card()
    md = R.card_markdown(it)
    for s in ["Identity", "Authors", "Abstract", "Bibliometrics", "Links", "Source"]:
        assert f"### {s}" in md, s
    chunks = R.build_chunks_jsonl([it]).strip().splitlines()
    assert chunks and all(json.loads(x).get("text") for x in chunks)
    rep = R.build_markdown([it], {"cardsOk": 1})
    assert "Deep learning" in rep


def test_charging_contract():
    import main as M
    assert M.EVENT_CARD == "work-card"
    ppe = json.load(open(os.path.join(HERE, "..", ".actor", "pay_per_event.json")))
    assert ppe["work-card"]["eventPriceUsd"] == 0.004 and ppe["work-card"]["isPrimaryEvent"]
    assert ppe["apify-actor-start"]["eventPriceUsd"] == 0.001


def test_input_schema_secrets_and_no_envvars():
    sch = json.load(open(os.path.join(HERE, "..", ".actor", "input_schema.json")))
    for k in ("openAlexApiKey", "ncbiApiKey"):
        assert sch["properties"][k]["isSecret"] is True
    act = json.load(open(os.path.join(HERE, "..", ".actor", "actor.json")))
    assert act["defaultRunOptions"]["memoryMbytes"] == 256 and not act.get("environmentVariables")


def test_build_config_placeholder_keys_and_mailto():
    import main as M
    cfg = M.build_config({"openAlexApiKey": "YOUR_KEY", "ncbiApiKey": "Bearer abc", "contactEmail": "bad"})
    assert cfg.openalex_key is None and cfg.ncbi_key == "abc" and cfg.mailto == C.DEFAULT_MAILTO
