# Code guide: what each function does and which other options exist

## Config (`config.py`)
`load_config(path, **path_overrides)`: YAML -> validated pydantic `Config`. Other options: pass `None` for defaults;
`data_dir=`/`artifacts_dir=` relocate outputs. Validators reject impossible year layouts. Derived: `labeled_cutoffs`,
`last_labeled_cutoff = end_year - horizon`, `prospective_cutoff = end_year` (labels unknowable, never evaluated).

## Ingestion
* `CachedClient(cache_dir, min_interval, timeout, retries, user_agent)`: retries 429/5xx with back-off honouring `Retry-After`;
  `get_json(url, params, secret_params)`, `post_text(url, data, secret_params)`. Secrets are sent but never cached.
  Cache files are written atomically; `manifest` lists every network call with a payload checksum.
* `openalex.fetch_works(client, query, start_year, end_year, max_records, work_types, per_page)`: cursor pagination
  (`per-page` max 200). OpenAlex also offers `sample`, `group_by`, `sort`, other filters (`concepts`, `topics.id`,
  `institutions.country_code`) that you can add to `build_filter`.
* `pubmed.fetch_pubmed_records(client, pmids, batch_size)`: efetch XML by POST. Other E-utilities: `esearch` (find PMIDs),
  `elink` (related records), `esummary` (light metadata).
* `clinicaltrials.fetch_trials(...)`: v2 token pagination; other query params: `query.term`, `query.intr`, `filter.overallStatus`.

## Cleaning
* `normalize_openalex(records)` -> tables + quality report. Drops works without id/year, dedups, never merges authors by name.
* `reconstruct_abstract(inverted_index)`: rebuilds text from {word: positions}.
* `merge_pubmed`: matches by PMID only. `normalize_trials`: validates NCT format and registration date.
* `build_paper_trial_links`: identifier-based only (PubMed DataBank 1.0, NCT in abstract 0.95); unknown trials dropped and counted.
  Semantic title matching is deliberately NOT implemented (blueprint: would need manual audit); add it as `candidate` rows with
  `reviewed=False` if you ever need it.
* `data_audit`: coverage, ID completeness, institution/country concentration, warnings.

## Graph
* `build_pair_table(authorships, max_authors, weight_mode)`: one row per co-author pair per paper (`fractional` 1/(n-1) or `binary`).
  Note: with fractional weights, weighted degree equals the author's paper count, so `wdeg` and `papers_total` are collinear
  (regularisation handles it; the ablation groups still separate them conceptually).
* `snapshot_edges(pair_table, cutoff)`: aggregate year <= cutoff (weight, first_year, last_year, n_papers).
* `graph_from_edges`: deterministic insertion order -> reproducible Louvain.
* `detect_communities(G, algorithm, resolution, seed, min_size)`: Louvain via `nx.community.louvain_communities`
  (params: weight, resolution (higher = more, smaller communities), threshold, max_level, seed) or Leiden (optional).
  `community_stability` reports NMI across seeds.
* `track_evolution(members_by_year, min_size, min_jaccard, split_frac, merge_frac)`: lineage ids, events
  birth/continue/split/merge, age, growth, turnover.

## Features (`features.py`)
* `FeatureBuilder.partition / snapshot / build_features / label_candidates`. Only `label_candidates` looks forward.
* Candidate rule: active authors, different communities (size >= min), unlinked, from (a) semantic-neighbour communities' top authors or
  (b) two-hop pairs through a broker. Same universe for every model.
* Feature groups: collaboration, brokerage (sampled betweenness, cross-community tie ratio), temporal (deltas vs previous year),
  evolution (community age/growth/turnover/size), semantic (TF-IDF or topic cosine; TF-IDF fitted per cutoff on works <= cutoff),
  citation (cross citations between the two authors' works), activity (recent output, trial-linked papers).
  Node features are stored as `_min/_max` so a pair has one representation regardless of order.
* `prepare_matrix`: stateless log1p on heavy-tailed counts.

## Models
* `make_splits(cutoffs, horizon, val_first, test_first, purge)`: purges training cutoffs whose label horizon reaches the next phase.
* Baselines: preferential attachment, common neighbours, Adamic-Adar, Jaccard, semantic-only, random.
* `LogisticModel(C, class_weight, max_iter)`: median-impute -> scale -> sklearn LogisticRegression. `class_weight="balanced"`
  improves ranking but distorts probabilities; other sklearn options: `penalty`, `solver`, `l1_ratio`.
* `BayesianLogisticRegression(prior_scale, intercept_scale, max_iter, tol, n_quad)`: MAP by damped Newton, Laplace posterior,
  `predict_proba` (Gauss-Hermite), `predict_interval(level)`, `contributions`, `coef_table`. Alternative: full MCMC in PyMC (not required).

## Evaluation
`evaluate_scores(y, score, ks, n_boot, seed, is_probability, cutoffs)`: AUPRC, ROC-AUC (NaN if one class), pooled and per-cutoff
P@K, R@K, bootstrap 95% CI, Brier/ECE. Ties broken randomly (seeded). `paired_bootstrap_diff`: AUPRC(A)-AUPRC(B) on shared resamples.
Caveat: the row bootstrap ignores within-snapshot dependence (CIs optimistic).

## Simulation
`GraphSimulator(G, membership, max_nodes)`: giant component, connected community pairs, cross-edge share, independent-cascade
coverage for strategies baseline/model/random/degree/betweenness with shared seeds per replicate.
Dense graphs saturate coverage (see synthetic demo); lower `ic_probability` or use sparser real data.
