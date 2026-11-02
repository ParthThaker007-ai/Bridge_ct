"""PubMed enrichment through NCBI E-utilities ``efetch``.

We enrich only PMIDs that OpenAlex already returned (title/abstract/MeSH/ClinicalTrials.gov DataBank links).
NCBI rate limits: 3 requests/s without an API key, 10/s with one (NCBI_API_KEY). Verify current guidance at
https://www.ncbi.nlm.nih.gov/books/NBK25501/. POST is used so long ID lists do not overflow the URL.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, Iterable, List

from ..utils import get_logger
from .http import CachedClient

log = get_logger()
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def _text(el: ET.Element | None) -> str:
    """All text inside an element, including nested tags such as <i>, <sub>."""
    return "".join(el.itertext()).strip() if el is not None else ""


def parse_pubmed_xml(xml_text: str) -> List[Dict[str, Any]]:
    """Parse an efetch ``PubmedArticleSet`` into flat dictionaries."""
    root = ET.fromstring(xml_text)
    out: List[Dict[str, Any]] = []
    for art in root.iter("PubmedArticle"):
        pmid = _text(art.find("MedlineCitation/PMID"))
        if not pmid:
            continue
        abstract = " ".join(_text(a) for a in art.findall(".//Abstract/AbstractText"))
        mesh = [_text(d) for d in art.findall(".//MeshHeadingList/MeshHeading/DescriptorName")]
        ncts: List[str] = []
        for bank in art.findall(".//DataBankList/DataBank"):
            if _text(bank.find("DataBankName")).lower() == "clinicaltrials.gov":
                ncts.extend(_text(a).upper() for a in bank.findall("AccessionNumberList/AccessionNumber"))
        out.append({"pmid": pmid, "title": _text(art.find("MedlineCitation/Article/ArticleTitle")),
                    "abstract": abstract, "mesh_terms": [m for m in mesh if m],
                    "nct_ids": sorted(set(n for n in ncts if n))})
    return out


def fetch_pubmed_records(client: CachedClient, pmids: Iterable[str], batch_size: int = 200) -> List[Dict[str, Any]]:
    """Fetch and parse PubMed records for the given PMIDs (deduplicated, order-stable)."""
    secret = {}
    if os.environ.get("NCBI_API_KEY"):
        secret["api_key"] = os.environ["NCBI_API_KEY"]
    if os.environ.get("NCBI_EMAIL"):
        secret["email"] = os.environ["NCBI_EMAIL"]
    unique = list(dict.fromkeys(str(p) for p in pmids if p))
    records: List[Dict[str, Any]] = []
    for i in range(0, len(unique), batch_size):
        batch = unique[i:i + batch_size]
        data = {"db": "pubmed", "id": ",".join(batch), "retmode": "xml", "rettype": "abstract"}
        xml_text = client.post_text(EFETCH_URL, data=data, secret_params=secret)
        records.extend(parse_pubmed_xml(xml_text))
        log.info("PubMed: %d/%d PMIDs processed", min(i + batch_size, len(unique)), len(unique))
    return records
