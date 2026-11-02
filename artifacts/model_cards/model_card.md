# BRIDGE-CT model card

**Task.** Rank author pairs from different collaboration communities by the probability of a first co-authorship
within 3 years. Disease scope: *type 2 diabetes*; data window 2006-2022.

**Model.** Bayesian logistic regression (Laplace approximation), prior scale 3.0,
standardised features computed from snapshots that use only records available at each cutoff.

**Evaluation.** Chronological split with label-horizon purging: {"train": [2009, 2010], "val": [2013, 2014, 2015], "refit": [2009, 2010, 2011, 2012, 2013], "test": [2016, 2017, 2018, 2019]}.
Test base rate 0.0010.

## Test metrics
                                     n  n_pos  base_rate   auprc  roc_auc  precision_at_10  precision_at_50   brier     ece
model                                                                                                                      
preferential_attachment         240000    233      0.001  0.0072   0.8379              0.0             0.00     NaN     NaN
common_neighbours               240000    233      0.001  0.0116   0.8921              0.0             0.02     NaN     NaN
adamic_adar                     240000    233      0.001  0.0133   0.8942              0.0             0.02     NaN     NaN
jaccard                         240000    233      0.001  0.0077   0.8766              0.0             0.02     NaN     NaN
semantic_only                   240000    233      0.001  0.0085   0.8480              0.0             0.00     NaN     NaN
random                          240000    233      0.001  0.0013   0.5103              0.0             0.02     NaN     NaN
logistic_balanced               240000    233      0.001  0.0197   0.9419              0.0             0.04  0.1167  0.2137
bayes_full                      240000    233      0.001  0.0317   0.9444              0.2             0.10  0.0010  0.0012
bayes_minus_collaboration       240000    233      0.001  0.0262   0.9304              0.1             0.12  0.0010  0.0015
bayes_minus_brokerage           240000    233      0.001  0.0332   0.9440              0.2             0.06  0.0010  0.0013
bayes_minus_temporal            240000    233      0.001  0.0332   0.9445              0.1             0.08  0.0010  0.0012
bayes_minus_evolution           240000    233      0.001  0.0295   0.9429              0.1             0.10  0.0010  0.0011
bayes_minus_semantic            240000    233      0.001  0.0295   0.9396              0.2             0.06  0.0010  0.0012
bayes_minus_citation            240000    233      0.001  0.0329   0.9443              0.2             0.10  0.0010  0.0012
bayes_minus_activity            240000    233      0.001  0.0226   0.9424              0.1             0.08  0.0010  0.0018
bayes_minus_temporal_evolution  240000    233      0.001  0.0296   0.9429              0.1             0.10  0.0010  0.0011
bayes_semantic_only             240000    233      0.001  0.0098   0.9034              0.0             0.00  0.0010  0.0024

## Paired comparisons (AUPRC difference, bootstrap)
- **H1_full_vs_best_baseline(adamic_adar)**: dAUPRC = +0.0184 (95% CI +0.0093 to +0.0362)
- **H1_full_vs_semantic_only_baseline**: dAUPRC = +0.0233 (95% CI +0.0142 to +0.0425)
- **H1_full_vs_semantic_only_model**: dAUPRC = +0.0219 (95% CI +0.0131 to +0.0414)
- **H1_full_vs_minus_collaboration**: dAUPRC = +0.0055 (95% CI -0.0022 to +0.0176)
- **H2_full_vs_minus_temporal_evolution**: dAUPRC = +0.0021 (95% CI -0.0020 to +0.0122)
- **H2_full_vs_minus_evolution**: dAUPRC = +0.0022 (95% CI -0.0012 to +0.0112)
- **H2_full_vs_minus_temporal**: dAUPRC = -0.0015 (95% CI -0.0046 to +0.0029)
- **full_vs_logistic_balanced**: dAUPRC = +0.0120 (95% CI +0.0057 to +0.0266)

## Data audit warnings
- none

## Intended use and limits
Research analysis of publicly recorded collaboration structure. NOT clinical decision support. Scores indicate
statistical association with future co-authorship, not scientific merit, clinical promise or causal benefit of a
collaboration. Candidate generation, "unobserved = negative" labels, database coverage bias (large institutions,
English-language, high-volume topics) and author-ID errors all limit validity. Bootstrap intervals resample
candidate rows and understate dependence within a snapshot.
