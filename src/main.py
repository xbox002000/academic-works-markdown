"""Apify Actor: DOI / PMID / arXiv id (+ keyword search) → scholarly work Markdown RAG cards + JSON.

Public keyless APIs: Crossref, OpenAlex, PubMed E-utilities, arXiv export API.
Optional buyer keys (OpenAlex api_key, NCBI api_key) only raise quotas — never stored by the publisher.
Pay-per-event: `work-card` per successful card. Invalid / not found / upstream failures are free.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
import re
import resource
import time
from collections import Counter

import httpx
from apify import Actor

import client as C
from charging import Charger
from inputs import Target, collect_targets, split_payload
from report import build_chunks_jsonl, build_markdown, card_markdown

EVENT_CARD = "work-card"
PLACEHOLDER_KEYS = {"", "YOUR_KEY", "YOUR_API_KEY", "YOUR_TOKEN", "XXX", "TODO", "CHANGEME", "NONE", "NULL"}
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

for _name in ("httpx", "httpcore"):
    logging.getLogger(_name).setLevel(logging.WARNING)


def _int(inp: dict, key: str, default: int, lo: int, hi: int) -> int:
    v = inp.get(key)
    try:
        v = int(default if v is None or v == "" else v)
    except (TypeError, ValueError):
        v = default
    return max(lo, min(v, hi))


def _clean_key(raw) -> str | None:
    if not raw:
        return None
    k = str(raw).strip()
    for pre in ("bearer ", "token "):
        if k.lower().startswith(pre):
            k = k[len(pre):].strip()
    return None if k.upper() in PLACEHOLDER_KEYS else k


def build_config(inp: dict) -> C.Config:
    provider = str(inp.get("provider") or "auto").strip().lower()
    if provider not in C.PROVIDERS:
        raise ValueError(f"`provider` must be one of {'|'.join(C.PROVIDERS)}, got {provider!r}")
    mail = str(inp.get("contactEmail") or "").strip()
    return C.Config(
        provider=provider,
        mailto=mail if _EMAIL.match(mail) else C.DEFAULT_MAILTO,
        openalex_key=_clean_key(inp.get("openAlexApiKey")),
        ncbi_key=_clean_key(inp.get("ncbiApiKey")),
        include_abstract=bool(inp.get("includeAbstract", True)),
        include_bibtex=bool(inp.get("includeBibtex", True)),
        enrich=bool(inp.get("enrich", True)),
        abstract_max=_int(inp, "abstractMaxChars", 4000, 200, 50000),
        max_authors=_int(inp, "maxAuthors", 100, 1, 2000),
        search_limit=_int(inp, "searchLimit", 5, 1, 50),
        max_retries=_int(inp, "maxRetries", 3, 0, 6),
        timeout=float(_int(inp, "requestTimeoutSecs", 30, 5, 120)),
    )


async def load_from_dataset(dataset_id: str, field: str | None, limit: int | None) -> tuple[list[Target], list[str], list[str]]:
    items: list = []
    try:
        ds = await Actor.open_dataset(id=dataset_id, force_cloud=bool(os.getenv("APIFY_IS_AT_HOME")))
        async for item in ds.iterate_items():
            if isinstance(item, dict):
                items.append(item)
            if limit and len(items) >= limit * 2:
                break
    except Exception as e:  # noqa: BLE001
        return [], [], [f"Could not read dataset {dataset_id}: {e}"]
    good, bad = collect_targets(ids=items, field=field)
    return good, bad, ([] if good or bad else [f"No DOI / PMID / arXiv / query fields found in dataset {dataset_id}"])


async def load_from_kv(store_id: str, key: str, field: str | None) -> tuple[list[Target], list[str], list[str]]:
    try:
        store = await Actor.open_key_value_store(id=store_id, force_cloud=bool(os.getenv("APIFY_IS_AT_HOME")))
        rec = await store.get_value(key)
    except Exception as e:  # noqa: BLE001
        return [], [], [f"Could not read KV {store_id}/{key}: {e}"]
    if rec is None:
        return [], [], [f"KV record {key} not found in store {store_id}"]
    good, bad = split_payload(rec, field=field)
    return good, bad, ([] if good or bad else [f"No inputs found in KV record {key}"])


def _merge(targets: list[Target], extra: list[Target]) -> None:
    seen = {(t.kind, t.value.lower()) for t in targets}
    for t in extra:
        k = (t.kind, t.value.lower())
        if k not in seen:
            targets.append(t)
            seen.add(k)


async def main() -> None:
    async with Actor:
        t0 = time.time()
        inp = await Actor.get_input() or {}
        charger = Charger(EVENT_CARD)
        cfg = build_config(inp)
        include_chunks = bool(inp.get("includeRagChunks", True))
        max_items = _int(inp, "maxItems", 50, 0, 100000) or None
        conc = _int(inp, "maxConcurrency", 2, 1, 5)
        field = (str(inp.get("idField") or "").strip() or None)

        warnings: list[str] = []
        targets, bad = collect_targets(ids=inp.get("ids"), queries=inp.get("queries"), field=field)
        if inp.get("datasetId"):
            g, b, w = await load_from_dataset(str(inp["datasetId"]), field, max_items)
            _merge(targets, g)
            bad += b
            warnings += w
            Actor.log.info(f"Loaded {len(g)} target(s) from dataset {inp['datasetId']}")
        if inp.get("keyValueStoreId"):
            key = str(inp.get("keyValueRecordKey") or "INPUT_WORKS")
            g, b, w = await load_from_kv(str(inp["keyValueStoreId"]), key, field)
            _merge(targets, g)
            bad += b
            warnings += w
            Actor.log.info(f"Loaded {len(g)} target(s) from KV {inp['keyValueStoreId']}/{key}")
        for w in warnings:
            Actor.log.warning(w)

        if max_items is not None and len(targets) > max_items:
            Actor.log.warning(f"{len(targets)} targets; capped to maxItems={max_items}.")
            targets = targets[:max_items]
        if not targets and not bad:
            raise ValueError("Provide at least one DOI / PMID / arXiv id in `ids`, a keyword in `queries`, "
                             "or a datasetId / keyValueStoreId.")

        kinds = Counter(t.kind for t in targets)
        Actor.log.info(
            f"{len(targets)} target(s) {dict(kinds)} (+{len(bad)} invalid); provider={cfg.provider}; "
            f"enrich={cfg.enrich}; concurrency={conc}; "
            f"openAlexKey={'yes (hidden)' if cfg.openalex_key else 'no'}; ncbiKey={'yes (hidden)' if cfg.ncbi_key else 'no'}"
        )
        await Actor.set_status_message(f"Resolving {len(targets)} scholarly target(s) via {cfg.provider}…")

        rows: list[tuple[int, dict]] = []
        cls_counts: Counter[str] = Counter()
        src_counts: Counter[str] = Counter()
        type_counts: Counter[str] = Counter()
        state = {"done": 0, "ok": 0, "fail": 0, "skipped": 0, "inflight": 0}

        for i, raw in enumerate(bad):
            item = {
                "kind": "work", "idType": None, "inputRaw": raw, "title": None, "status": "failed",
                "source": None, "sources": [], "errorClass": "invalid_input", "httpStatus": None,
                "attempts": 0, "errorMessage": "Not recognised as a DOI, PMID or arXiv id.",
                "hint": C.HINTS["invalid_input"],
                "checkedAt": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            }
            rows.append((-100000 + i, item))
            cls_counts["invalid_input"] += 1
            state["fail"] += 1
            await charger.push_free(item)

        sem = asyncio.Semaphore(conc)
        lock = asyncio.Lock()
        card_budget = max_items

        async with httpx.AsyncClient(follow_redirects=True, limits=httpx.Limits(max_connections=conc + 4)) as client:

            async def one(idx: int, t: Target) -> None:
                nonlocal card_budget
                async with sem:
                    async with lock:
                        if charger.limit_reached or (card_budget is not None and card_budget <= 0):
                            state["skipped"] += 1
                            return
                        b = charger.budget()
                        if b is not None and b - state["inflight"] <= 0:
                            state["skipped"] += 1
                            charger.limit_reached = True
                            return
                        limit = cfg.search_limit if card_budget is None else max(1, min(cfg.search_limit, card_budget - state["inflight"]))
                        state["inflight"] += 1
                    try:
                        try:
                            res = await C.lookup_target(client, t, cfg, search_limit=limit)
                        except Exception as e:  # noqa: BLE001 - never crash the batch on one bad payload
                            res = C.LookupResult(False, [C.failed_row(t, None, "bad_response", C.redact(str(e)[:300], cfg.secrets))])
                    finally:
                        async with lock:
                            state["inflight"] -= 1

                    async with lock:
                        state["done"] += 1
                        items = list(res.items or [])
                        if res.ok and card_budget is not None:
                            items = items[:card_budget]
                        n_ok = 0
                        for j, item in enumerate(items):
                            item["checkedAt"] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
                            if item.get("status") == "ok":
                                item["markdown"] = card_markdown(item)
                                if await charger.push_and_charge([item], event=EVENT_CARD):
                                    n_ok += 1
                                    state["ok"] += 1
                                    if card_budget is not None:
                                        card_budget -= 1
                                    cls_counts["ok"] += 1
                                    for s in item.get("sources") or []:
                                        src_counts[s] += 1
                                    type_counts[item.get("idType") or "?"] += 1
                                    rows.append((idx * 1000 + j, item))
                                else:
                                    state["skipped"] += 1
                            else:
                                state["fail"] += 1
                                cls_counts[item.get("errorClass") or "error"] += 1
                                rows.append((idx * 1000 + j, item))
                                await charger.push_free(item)
                        label = t.value[:70]
                        Actor.log.info(
                            f"[{state['done']}/{len(targets)}] {t.kind}:{label}: "
                            + (f"OK ×{n_ok} via {items[0].get('sources') if items else []}" if res.ok
                               else f"FAILED {(items[0].get('errorClass') if items else 'error')} (free)")
                        )
                        await Actor.set_status_message(
                            f"{state['done']}/{len(targets)} targets — ok {state['ok']}, failed {state['fail']}")

            await asyncio.gather(*(one(i, t) for i, t in enumerate(targets)))

        rows.sort(key=lambda x: x[0])
        out_rows = [r for _, r in rows]
        peak_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        summary = {
            "provider": cfg.provider,
            "enrich": cfg.enrich,
            "openAlexKeyProvided": bool(cfg.openalex_key),
            "ncbiKeyProvided": bool(cfg.ncbi_key),
            "targetsPlanned": len(targets),
            "targetsByType": dict(kinds),
            "invalidInputs": len(bad),
            "cardsOk": state["ok"],
            "cardsFailed": state["fail"],
            "skipped": state["skipped"],
            "byClass": dict(cls_counts),
            "bySource": dict(src_counts),
            "byIdType": dict(type_counts),
            "includeAbstract": cfg.include_abstract,
            "includeBibtex": cfg.include_bibtex,
            "includeRagChunks": include_chunks,
            "charged": dict(charger.counts),
            "spendingLimitReached": charger.limit_reached,
            "durationSecs": round(time.time() - t0, 1),
            "peakMemoryMb": peak_mb,
            "finishedAt": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        }
        store = await Actor.open_key_value_store()
        await store.set_value("REPORT.md", build_markdown(out_rows, summary), content_type="text/markdown; charset=utf-8")
        chunk_count = 0
        if include_chunks:
            jsonl = build_chunks_jsonl(out_rows)
            chunk_count = jsonl.count("\n")
            await store.set_value("CHUNKS.jsonl", jsonl, content_type="application/x-ndjson; charset=utf-8")
        summary["ragChunks"] = chunk_count
        await store.set_value("SUMMARY", summary)
        await store.set_value("OUTPUT", {
            "summary": summary,
            "works": [{k: r.get(k) for k in ("idType", "inputRaw", "doi", "pmid", "arxivId", "title", "year",
                                              "venue", "citedBy", "isOA", "landingUrl", "source", "status",
                                              "errorClass")} for r in out_rows],
        })
        msg = (f"Done: {state['ok']} card(s) OK, {state['fail']} failed (free), {state['skipped']} skipped. "
               f"Charged: {charger.counts or 'nothing'}.")
        Actor.log.info(msg + f" Peak memory {peak_mb} MB, {summary['durationSecs']} s.")
        await Actor.set_status_message(msg, is_terminal=True)


if __name__ == "__main__":
    asyncio.run(main())
