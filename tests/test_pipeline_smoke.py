import pandas as pd

from bridge_ct.pipeline import load_tables, save_tables, stage_all_after_normalize


def test_end_to_end_on_tiny_synthetic(tiny_cfg, tiny_tables, tmp_path):
    cfg = tiny_cfg.model_copy(update={"paths": tiny_cfg.paths.model_copy(
        update={"data_dir": str(tmp_path / "data"), "artifacts_dir": str(tmp_path / "art")})})
    save_tables(tiny_tables, cfg)
    out = stage_all_after_normalize(cfg, load_tables(cfg))
    art = tmp_path / "art"
    for f in ["metrics/results.json", "metrics/metrics_test.csv", "shortlist/shortlist.csv",
              "simulation/simulation_summary.csv", "model_cards/model_card.md"]:
        assert (art / f).exists(), f
    sl = pd.read_csv(art / "shortlist" / "shortlist.csv")
    assert sl["prob"].between(0, 1).all() and (sl["prob_lo"] <= sl["prob_hi"]).all()
    assert (sl["cutoff"] == cfg.prospective_cutoff).all()
    assert out["results"]["splits"]["test"]


def test_dashboard_pages_render(tiny_cfg, tiny_tables, tmp_path, monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    cfg = tiny_cfg.model_copy(update={"paths": tiny_cfg.paths.model_copy(
        update={"data_dir": str(tmp_path / "data"), "artifacts_dir": str(tmp_path / "art")})})
    save_tables(tiny_tables, cfg)
    stage_all_after_normalize(cfg, load_tables(cfg))
    monkeypatch.setenv("BRIDGE_CT_ARTIFACTS", str(tmp_path / "art"))
    monkeypatch.setenv("BRIDGE_CT_DATA", str(tmp_path / "data"))
    script = Path(__file__).resolve().parents[1] / "src" / "bridge_ct" / "app" / "dashboard.py"
    at = AppTest.from_file(str(script), default_timeout=120).run()
    assert not at.exception
    for page in at.sidebar.radio[0].options:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, (page, [str(e.value) for e in at.exception])