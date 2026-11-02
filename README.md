# BRIDGE-CT

Temporal collaboration-network analysis and Bayesian link prediction for biomedical research communities
(OpenAlex + PubMed + ClinicalTrials.gov). Implements the BRIDGE-CT blueprint end to end.

**Not clinical decision support.** Scores describe statistical association with future co-authorship only.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env            # add your e-mail (+ API keys if you have them)
pytest                          # 17 tests, ~1 minute
bridge-ct demo --data-dir /tmp/bct/data --artifacts-dir /tmp/bct/art   # OFFLINE synthetic check, ~2 min
BRIDGE_CT_ARTIFACTS=/tmp/bct/art BRIDGE_CT_DATA=/tmp/bct/data streamlit run src/bridge_ct/app/dashboard.py
```
Real data: edit `configs/project.yaml` (disease, years), then `bridge-ct fetch`, `bridge-ct normalize`
(read `artifacts/metrics/data_audit.json` first!), `bridge-ct run`. Full runbook: `docs/STEP_BY_STEP.md`.
Function-by-function explanation: `docs/CODE_GUIDE.md`.

## Status (honest)
* Verified: all unit/leakage/smoke tests, full offline synthetic run, every dashboard page (Streamlit AppTest).
* NOT verified here: live API calls (the build sandbox had no access to openalex.org, pubmed, clinicaltrials.gov).
  Parsers were tested on hand-written records only. Section "First live run checklist" in the runbook lists
  exactly what to check.
* Synthetic results prove the code path, not the science.
