"""ClinicalTrials.gov API v2 ingestion (token pagination).

Docs: https://clinicaltrials.gov/data-api/about-api . ``pageToken`` continues a query; ``fields`` limits the
payload. Year filtering is done locally on the registration (first-submit) date so we do not depend on the
server's advanced-filter syntax.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..utils import get_logger
from .http import CachedClient

log = get_logger()
CT_URL = "https://clinicaltrials.gov/api/v2/studies"
FIELDS = "NCTId|BriefTitle|Condition|InterventionName|StartDate|StudyFirstSubmitDate|Phase|OverallStatus"


def parse_study(study: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one v2 study record (defensive: any module may be absent)."""
    ps = study.get("protocolSection", {})
    ident = ps.get("identificationModule", {})
    status = ps.get("statusModule", {})
    interventions = ps.get("armsInterventionsModule", {}).get("interventions", []) or []
    return {
        "nct_id": str(ident.get("nctId", "")).upper(),
        "title": ident.get("briefTitle", ""),
        "conditions": "|".join(ps.get("conditionsModule", {}).get("conditions", []) or []),
        "interventions": "|".join(i.get("name", "") for i in interventions if i.get("name")),
        "study_start": (status.get("startDateStruct") or {}).get("date"),
        "registration_date": status.get("studyFirstSubmitDate"),
        "phase": "|".join(ps.get("designModule", {}).get("phases", []) or []),
        "status": status.get("overallStatus"),
    }


def fetch_trials(client: CachedClient, condition: str, start_year: int, end_year: int,
                 max_records: Optional[int] = None, page_size: int = 1000) -> List[Dict[str, Any]]:
    """Fetch all trials for a condition, then keep those first registered in [start_year, end_year]."""
    trials: List[Dict[str, Any]] = []
    token: Optional[str] = None
    while True:
        params = {"query.cond": condition, "pageSize": page_size, "format": "json", "fields": FIELDS}
        if token:
            params["pageToken"] = token
        page = client.get_json(CT_URL, params=params)
        for study in page.get("studies", []):
            rec = parse_study(study)
            reg = rec["registration_date"]
            if rec["nct_id"] and reg and start_year <= int(str(reg)[:4]) <= end_year:
                trials.append(rec)
        log.info("ClinicalTrials.gov: %d trials kept", len(trials))
        if max_records is not None and len(trials) >= max_records:
            return trials[:max_records]
        token = page.get("nextPageToken")
        if not token:
            break
    return trials
