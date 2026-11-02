"""Typed configuration loading.

All tunable values live in ``configs/project.yaml``. They are validated with pydantic so a typo or an
impossible combination (e.g. test period before validation period) fails immediately with a clear
message instead of silently producing wrong science.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator


class ProjectCfg(BaseModel):
    name: str = "BRIDGE-CT"
    disease: str = "type 2 diabetes"
    start_year: int = 2006
    end_year: int = 2022
    prediction_horizon_years: int = Field(3, ge=1)
    random_seed: int = 42


class PathsCfg(BaseModel):
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"


class SourcesCfg(BaseModel):
    openalex: bool = True
    pubmed: bool = True
    clinicaltrials: bool = True
    max_works: Optional[int] = None
    work_types: List[str] = ["article", "review"]


class GraphCfg(BaseModel):
    collaboration_weight: Literal["fractional", "binary"] = "fractional"
    community_algorithm: Literal["louvain", "leiden"] = "louvain"
    community_resolution: float = Field(1.0, gt=0)
    max_authors_per_paper: int = Field(50, ge=2)
    min_community_size: int = Field(5, ge=2)
    betweenness_samples: Optional[int] = 200


class SemanticCfg(BaseModel):
    method: Literal["tfidf", "topics"] = "tfidf"
    tfidf_max_features: int = Field(5000, ge=10)


class CandidatesCfg(BaseModel):
    active_window_years: int = Field(3, ge=1)
    top_authors_per_community: int = Field(15, ge=1)
    neighbour_communities: int = Field(5, ge=1)
    min_semantic_similarity: float = 0.05
    two_hop_per_broker: int = Field(20, ge=0)
    max_candidates_per_snapshot: int = Field(60000, ge=100)


class LinksCfg(BaseModel):
    min_confidence: float = Field(0.9, ge=0, le=1)


class EvaluationCfg(BaseModel):
    split_type: Literal["chronological"] = "chronological"
    primary_metric: Literal["average_precision"] = "average_precision"
    top_k: List[int] = [10, 50]
    n_bootstrap: int = Field(300, ge=10)
    cutoff_first: int = 2009
    val_first: int = 2013
    test_first: int = 2016
    purge: bool = True
    logistic_C_grid: List[float] = [0.01, 0.1, 1.0, 10.0]
    prior_scale_grid: List[float] = [0.3, 1.0, 3.0]
    shortlist_size: int = Field(200, ge=1)


class SimulationCfg(BaseModel):
    k_edges: int = Field(20, ge=1)
    n_replicates: int = Field(30, ge=1)
    ic_probability: float = Field(0.1, gt=0, le=1)
    n_seed_nodes: int = Field(10, ge=1)
    n_cascades: int = Field(30, ge=1)
    max_nodes: int = Field(20000, ge=100)


class Config(BaseModel):
    project: ProjectCfg = ProjectCfg()
    paths: PathsCfg = PathsCfg()
    sources: SourcesCfg = SourcesCfg()
    graph: GraphCfg = GraphCfg()
    semantic: SemanticCfg = SemanticCfg()
    candidates: CandidatesCfg = CandidatesCfg()
    links: LinksCfg = LinksCfg()
    evaluation: EvaluationCfg = EvaluationCfg()
    simulation: SimulationCfg = SimulationCfg()

    @model_validator(mode="after")
    def _check_consistency(self) -> "Config":
        ev, pr = self.evaluation, self.project
        if pr.start_year >= pr.end_year:
            raise ValueError("project.start_year must be < project.end_year")
        if not (ev.cutoff_first < ev.val_first < ev.test_first):
            raise ValueError("need evaluation.cutoff_first < val_first < test_first")
        if ev.cutoff_first <= pr.start_year:
            raise ValueError("cutoff_first must be after start_year (need burn-in history)")
        if ev.test_first > self.last_labeled_cutoff:
            raise ValueError(
                f"test_first={ev.test_first} is after the last cutoff with a fully observable "
                f"horizon ({self.last_labeled_cutoff} = end_year - horizon)"
            )
        return self

    # ------------------------------------------------------------------ derived values
    @property
    def horizon(self) -> int:
        return self.project.prediction_horizon_years

    @property
    def last_labeled_cutoff(self) -> int:
        """Last cutoff whose full future horizon lies inside the data window."""
        return self.project.end_year - self.horizon

    @property
    def labeled_cutoffs(self) -> List[int]:
        return list(range(self.evaluation.cutoff_first, self.last_labeled_cutoff + 1))

    @property
    def prospective_cutoff(self) -> int:
        """Cutoff used for the real shortlist: labels are unknowable, so it is never used for evaluation."""
        return self.project.end_year

    @property
    def data_dir(self) -> Path:
        return Path(self.paths.data_dir)

    @property
    def artifacts_dir(self) -> Path:
        return Path(self.paths.artifacts_dir)


def load_config(path: str | Path | None = None, **path_overrides: str) -> Config:
    """Load and validate a YAML configuration.

    Parameters
    ----------
    path : YAML file. ``None`` returns the defaults (useful in tests).
    path_overrides : ``data_dir=...`` / ``artifacts_dir=...`` to relocate outputs (used by the demo).
    """
    load_dotenv()  # make OPENALEX_* / NCBI_* variables from .env visible to ingestion
    raw: dict = {}
    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    cfg = Config.model_validate(raw)
    if path_overrides:
        paths = cfg.paths.model_copy(update=path_overrides)
        cfg = cfg.model_copy(update={"paths": paths})
    return cfg
