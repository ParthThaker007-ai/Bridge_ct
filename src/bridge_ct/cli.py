"""Command line interface: ``bridge-ct <command> --config configs/project.yaml``."""
from __future__ import annotations

import argparse
import sys

from .config import load_config
from .pipeline import (load_tables, run_demo, stage_all_after_normalize, stage_fetch, stage_normalize)
from .utils import get_logger

log = get_logger()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bridge-ct", description="BRIDGE-CT research pipeline")
    ap.add_argument("command", choices=["fetch", "normalize", "run", "all", "demo"],
                    help="fetch: download raw data | normalize: clean+audit | run: graphs->models->shortlist "
                         "| all: fetch+normalize+run | demo: offline synthetic end-to-end check")
    ap.add_argument("--config", default="configs/project.yaml")
    ap.add_argument("--data-dir", default=None, help="override paths.data_dir")
    ap.add_argument("--artifacts-dir", default=None, help="override paths.artifacts_dir")
    a = ap.parse_args(argv)
    over = {k: v for k, v in (("data_dir", a.data_dir), ("artifacts_dir", a.artifacts_dir)) if v}
    cfg = load_config(a.config, **over)
    if a.command == "demo":
        out = run_demo(cfg)
        log.info("demo finished; open the dashboard with:\n  BRIDGE_CT_ARTIFACTS=%s BRIDGE_CT_DATA=%s streamlit run src/bridge_ct/app/dashboard.py",
                 cfg.artifacts_dir, cfg.data_dir)
        return 0
    if a.command in ("fetch", "all"):
        stage_fetch(cfg)
    if a.command in ("normalize", "all"):
        stage_normalize(cfg)
    if a.command in ("run", "all"):
        stage_all_after_normalize(cfg, load_tables(cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
