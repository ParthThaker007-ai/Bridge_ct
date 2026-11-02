# BRIDGE-CT runbook (atomic steps)

## 0. Setup (10 min)
1. `python3 --version` must be >= 3.10. `python -m venv .venv && source .venv/bin/activate`.
2. `pip install -e ".[dev]"` (core deps only; Leiden is optional: `pip install -e ".[leiden]"`).
3. `cp .env.example .env`; set `OPENALEX_MAILTO` (polite pool), optional `OPENALEX_API_KEY`, `NCBI_EMAIL`, `NCBI_API_KEY`.
4. `pytest` -> everything green. If not, stop and fix before touching data.
5. `bridge-ct demo --data-dir /tmp/bct/data --artifacts-dir /tmp/bct/art` -> confirms the whole pipeline on synthetic data.

## 1. Phase 0 - feasibility (week 1 of the blueprint)
1. In `configs/project.yaml` choose ONE disease (`project.disease`) and write inclusion/exclusion rules in your notes first.
2. Pilot: set `sources.max_works: 3000`, run `bridge-ct fetch` then `bridge-ct normalize`.
3. Open `artifacts/metrics/data_audit.json`: check `per_year` counts, `frac_works_with_abstract`, `authorships_missing_author_id`,
   `date_year_mismatch`, `n_links`, `top10_institution_share`, `warnings`.
4. Open `artifacts/metrics/link_review_sample.csv`; manually verify ~100 links (fill `correct` 1/0). Compute precision
   per method with `bridge_ct.cleaning.links.link_precision`. If precision < 0.9 raise `links.min_confidence` or drop trial features.
5. Decide the window (`start_year`, `end_year`) so every cutoff year has enough works; freeze target = future co-authorship.
6. Freeze `evaluation.cutoff_first / val_first / test_first` BEFORE looking at test metrics.

## 2. Full data pull
1. Set `sources.max_works: null`; `bridge-ct fetch` (cached: re-running is free). Check `data/raw/manifest.json` (checksums, timestamps).
2. `bridge-ct normalize` -> `data/processed/*.parquet`, audit JSON.

## 3. Graphs, communities, features, models
1. `bridge-ct run` executes: pair table -> yearly partitions -> evolution -> candidates/features/labels ->
   splits -> tuning on validation -> test evaluation -> ablations -> prospective shortlist -> evidence -> simulation -> model card.
2. Read `artifacts/metrics/results.json`: `comparisons` hold paired AUPRC differences with CIs (H1, H2); `metrics_test.csv`
   has Brier/ECE (H3); `simulation/simulation_summary.csv` is H4 (graph-simulation only).
3. Negative or null results are results: report them. Do not pick the best time window afterwards.

## 4. Dashboard and deliverables
`streamlit run src/bridge_ct/app/dashboard.py` (set `BRIDGE_CT_ARTIFACTS`, `BRIDGE_CT_DATA`).
Deliverables map: code = `src/`; manifest = `data/raw/manifest.json`; snapshots = `artifacts/features/`;
SNA report = `snapshot_summary.csv`, `community_stability.json`, `evolution.parquet`; benchmark = `metrics_*.csv`,
`results.json`; simulation = `artifacts/simulation/`; example export = `artifacts/shortlist/shortlist.csv`; model card.

## First live run checklist (things I could not test offline)
1. OpenAlex: confirm the filter `title_and_abstract.search` and the `select` fields still work (docs.openalex.org);
   if an API key is now mandatory, put it in `.env`. A 4xx error message names the offending parameter.
2. PubMed efetch XML: print one parsed record (`parse_pubmed_xml`) and compare with the website.
3. ClinicalTrials.gov v2: print one `parse_study` output; if `fields` names changed, remove `fields` from `FIELDS` use
   (the parser works on full records too).
4. Runtime: a 100k-work corpus needs RAM for the pair table (~tens of millions of rows is fine at 16 GB; lower
   `max_authors_per_paper` if not) and minutes per snapshot for sampled betweenness.
5. Tune `candidates.*` only on validation results, never on test.

## Troubleshooting
* "split 'train' is empty": widen `start_year`, reduce `prediction_horizon_years`, or set `purge: false` (weaker protocol, state it).
* "single label class": candidates too few/positives absent; raise `top_authors_per_community`, `neighbour_communities`.
* ROC-AUC is NaN: the test period has one class; revisit target/window (never hidden by the code).
* Memory error in citations: reduce candidate cap `max_candidates_per_snapshot`.
