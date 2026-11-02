"""Canonical tables from raw OpenAlex / PubMed / ClinicalTrials.gov records.

Design rules (from the blueprint):
* stable source IDs only - authors are NEVER merged because their display names match;
* every work needs a usable publication year, otherwise it is dropped and counted in the quality report;
* nothing is guessed: missing values stay missing.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

NCT_RE = re.compile(r"\bNCT\d{8}\b", re.IGNORECASE)

WORK_COLS = ["work_id", "title", "abstract", "publication_date", "publication_year", "doi", "pmid",
             "source", "retrieved_at", "mesh_terms", "pubmed_nct_ids"]
AUTHORSHIP_COLS = ["work_id", "author_id", "institution_id", "author_position", "publication_year"]
TRIAL_COLS = ["nct_id", "title", "conditions", "interventions", "study_start", "registration_date",
              "registration_year", "phase", "status"]


def short_id(url: Optional[str]) -> Optional[str]:
    """``https://openalex.org/W123`` -> ``W123`` (stable, compact key)."""
    if not url:
        return None
    return str(url).rstrip("/").rsplit("/", 1)[-1] or None


def normalize_pmid(value: Optional[str]) -> Optional[str]:
    """Accept a PMID URL or bare number; return digits only (or None)."""
    if not value:
        return None
    m = re.search(r"(\d+)\s*$", str(value))
    return m.group(1) if m else None


def normalize_doi(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", str(value).strip(), flags=re.IGNORECASE).lower() or None


def reconstruct_abstract(inverted: Optional[Dict[str, List[int]]]) -> str:
    """OpenAlex ships abstracts as {word: [positions]}; rebuild the plain text in reading order."""
    if not inverted:
        return ""
    positions: Dict[int, str] = {}
    for word, idxs in inverted.items():
        for i in idxs:
            positions[int(i)] = word
    return " ".join(positions[i] for i in sorted(positions))


def _empty(cols: List[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})


def normalize_openalex(records: Iterable[Dict[str, Any]], retrieved_at: Optional[str] = None) -> Dict[str, Any]:
    """Convert raw OpenAlex works into canonical tables plus a data-quality report.

    Returns a dict with DataFrames ``works, authorships, authors, institutions, topics, work_topics,
    citations`` and a plain dict ``quality``.
    """
    retrieved_at = retrieved_at or datetime.now(timezone.utc).isoformat()
    works, ships, authors, insts, topics, wtopics, cites = [], [], {}, {}, {}, [], []
    n_records = n_no_id = n_no_year = n_mismatch = n_dup = n_ship_no_author = 0
    seen = set()
    for rec in records:
        n_records += 1
        wid = short_id(rec.get("id"))
        if not wid:
            n_no_id += 1
            continue
        if wid in seen:
            n_dup += 1
            continue
        seen.add(wid)
        date = pd.to_datetime(rec.get("publication_date"), errors="coerce")
        year = None
        if pd.notna(date):
            year = int(date.year)
            if rec.get("publication_year") is not None and int(rec["publication_year"]) != year:
                n_mismatch += 1
        elif rec.get("publication_year") is not None:
            year = int(rec["publication_year"])
        if year is None:
            n_no_year += 1
            continue
        pmid = normalize_pmid((rec.get("ids") or {}).get("pmid"))
        works.append({"work_id": wid, "title": rec.get("title") or "",
                      "abstract": reconstruct_abstract(rec.get("abstract_inverted_index")),
                      "publication_date": date.date().isoformat() if pd.notna(date) else None,
                      "publication_year": year, "doi": normalize_doi(rec.get("doi")), "pmid": pmid,
                      "source": "openalex", "retrieved_at": retrieved_at, "mesh_terms": "", "pubmed_nct_ids": ""})
        seen_authors = set()
        for pos, au in enumerate(rec.get("authorships") or []):
            aid = short_id((au.get("author") or {}).get("id"))
            if not aid:
                n_ship_no_author += 1
                continue
            if aid in seen_authors:  # same author listed twice on one work -> one authorship
                continue
            seen_authors.add(aid)
            authors.setdefault(aid, (au.get("author") or {}).get("display_name") or "")
            inst_list = au.get("institutions") or []
            first_inst = short_id(inst_list[0].get("id")) if inst_list else None
            for inst in inst_list:
                iid = short_id(inst.get("id"))
                if iid:
                    insts.setdefault(iid, (inst.get("display_name") or "", inst.get("country_code")))
            ships.append({"work_id": wid, "author_id": aid, "institution_id": first_inst,
                          "author_position": au.get("author_position") or str(pos), "publication_year": year})
        for t in rec.get("topics") or []:
            tid = short_id(t.get("id"))
            if tid:
                topics.setdefault(tid, t.get("display_name") or "")
                wtopics.append({"work_id": wid, "topic_id": tid, "score": float(t.get("score") or 0.0)})
        for ref in rec.get("referenced_works") or []:
            rid = short_id(ref)
            if rid:
                cites.append({"work_id": wid, "cited_work_id": rid})

    works_df = pd.DataFrame(works, columns=WORK_COLS) if works else _empty(WORK_COLS)
    ships_df = pd.DataFrame(ships, columns=AUTHORSHIP_COLS) if ships else _empty(AUTHORSHIP_COLS)
    quality = {
        "n_records": n_records, "n_works": len(works_df), "dropped_missing_id": n_no_id,
        "dropped_duplicates": n_dup, "dropped_missing_year": n_no_year,
        "date_year_mismatch": n_mismatch, "authorships_missing_author_id": n_ship_no_author,
        "frac_works_with_abstract": float((works_df["abstract"].str.len() > 0).mean()) if len(works_df) else 0.0,
        "frac_works_with_topics": float(works_df["work_id"].isin({w["work_id"] for w in wtopics}).mean()) if len(works_df) else 0.0,
        "frac_works_with_pmid": float(works_df["pmid"].notna().mean()) if len(works_df) else 0.0,
    }
    return {
        "works": works_df, "authorships": ships_df,
        "authors": pd.DataFrame([{"author_id": k, "display_name": v} for k, v in authors.items()],
                                columns=["author_id", "display_name"]),
        "institutions": pd.DataFrame([{"institution_id": k, "display_name": v[0], "country_code": v[1]}
                                      for k, v in insts.items()],
                                     columns=["institution_id", "display_name", "country_code"]),
        "topics": pd.DataFrame([{"topic_id": k, "label": v, "source": "openalex"} for k, v in topics.items()],
                               columns=["topic_id", "label", "source"]),
        "work_topics": pd.DataFrame(wtopics, columns=["work_id", "topic_id", "score"]),
        "citations": pd.DataFrame(cites, columns=["work_id", "cited_work_id"]),
        "quality": quality,
    }


def merge_pubmed(works: pd.DataFrame, pubmed_records: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Add MeSH terms and ClinicalTrials.gov DataBank links; fill missing abstracts from PubMed.

    Matching is by PMID only (an identifier), never by title.
    """
    by_pmid = {r["pmid"]: r for r in pubmed_records}
    out = works.copy()
    if out.empty:
        return out
    mesh, ncts, abstracts = [], [], []
    for _, row in out.iterrows():
        pmid = row["pmid"] if isinstance(row["pmid"], str) and row["pmid"] else None
        rec = by_pmid.get(pmid) if pmid else None
        mesh.append("|".join(rec["mesh_terms"]) if rec else "")
        ncts.append("|".join(rec["nct_ids"]) if rec else "")
        abstracts.append(row["abstract"] if isinstance(row["abstract"], str) and row["abstract"] else (rec["abstract"] if rec else ""))
    out["mesh_terms"], out["pubmed_nct_ids"], out["abstract"] = mesh, ncts, abstracts
    return out


def normalize_trials(records: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """Trial table with a parsed registration year; rows without a valid NCT ID or date are dropped."""
    rows = []
    for r in records:
        nct = str(r.get("nct_id") or "").upper()
        reg = pd.to_datetime(r.get("registration_date"), errors="coerce")
        if not NCT_RE.fullmatch(nct) or pd.isna(reg):
            continue
        rows.append({**{k: r.get(k) for k in TRIAL_COLS if k != "registration_year"},
                     "nct_id": nct, "registration_date": reg.date().isoformat(), "registration_year": int(reg.year)})
    df = pd.DataFrame(rows, columns=TRIAL_COLS) if rows else _empty(TRIAL_COLS)
    return df.drop_duplicates("nct_id").reset_index(drop=True)
