"""BRIDGE-CT dashboard: "mission control" for research-community bridges.

Run:  BRIDGE_CT_ARTIFACTS=artifacts BRIDGE_CT_DATA=data streamlit run src/bridge_ct/app/dashboard.py
Every sentence in the explanation panels comes from bridge_ct.explain (rule-based, traceable to metrics).
"""
from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# make `bridge_ct` importable on Streamlit Cloud without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bridge_ct.explain import (BASELINE_NAMES, _ci_verdict, describe_feature, explain_pair, explain_results,
                               overall_summary)

ART = Path(os.environ.get("BRIDGE_CT_ARTIFACTS", "artifacts"))
DATA = Path(os.environ.get("BRIDGE_CT_DATA", "data"))
esc = html.escape

PALETTE = {"bg": "#0b1020", "panel": "#121a33", "edge": "#26325a", "text": "#e8ecff", "muted": "#93a0cf",
           "teal": "#2de2c4", "violet": "#8b7bff", "amber": "#ffb454", "rose": "#ff6b8a", "green": "#52e08f"}
VERDICT_STYLE = {"supported": ("SUPPORTED", "#52e08f"), "not_supported": ("CONTRADICTED", "#ff6b8a"),
                 "inconclusive": ("INCONCLUSIVE", "#ffb454"), "warning": ("WARNING", "#ff6b8a"), "info": ("INSIGHT", "#8b7bff")}
MODEL_LABELS = {"bayes_full": "Bayesian model (all features)", "logistic_balanced": "Logistic (re-weighted)",
                "preferential_attachment": "Rule: popularity", "common_neighbours": "Rule: friends-of-friends",
                "adamic_adar": "Rule: weighted friends", "jaccard": "Rule: overlapping circles",
                "semantic_only": "Rule: similar topics", "random": "Random guess",
                "bayes_semantic_only": "Bayesian: topics only"}

st.set_page_config(page_title="BRIDGE-CT | Research Bridges", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', 'Segoe UI', sans-serif; }
.stApp { background: radial-gradient(1200px 600px at 8% -10%, #1b2a5c55 0%, transparent 60%),
                     radial-gradient(900px 500px at 100% 0%, #2de2c41f 0%, transparent 55%), #0b1020; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f1730 0%, #0b1020 100%); border-right: 1px solid #26325a; }
.hero { padding: 26px 30px; border-radius: 22px; border: 1px solid #26325a;
        background: linear-gradient(120deg, #151f45cc, #0f1a3acc 55%, #12332f99); position: relative; overflow: hidden; }
.hero:after { content: ""; position: absolute; right: -60px; top: -60px; width: 260px; height: 260px; border-radius: 50%;
        background: conic-gradient(from 90deg, #2de2c455, #8b7bff55, #ff6b8a44, #2de2c455); filter: blur(30px); }
.hero h1 { margin: 0; font-size: 2.1rem; letter-spacing: .5px; }
.hero p { margin: 6px 0 0 0; color: #93a0cf; max-width: 760px; }
.tag { display:inline-block; padding:2px 10px; border-radius:99px; border:1px solid #26325a; color:#93a0cf; font-size:.74rem; margin-right:6px; }
.kpi { border:1px solid #26325a; background:#121a33cc; border-radius:16px; padding:16px 18px; height:100%; }
.kpi .l { color:#93a0cf; font-size:.74rem; text-transform:uppercase; letter-spacing:1.4px; }
.kpi .v { font-family:'JetBrains Mono', monospace; font-size:1.9rem; font-weight:600; margin-top:2px; }
.kpi .s { color:#93a0cf; font-size:.8rem; margin-top:2px; }
.card { border:1px solid #26325a; border-left-width:5px; background:#121a33cc; border-radius:14px; padding:14px 18px; margin-bottom:12px; }
.card h4 { margin:0 0 4px 0; font-size:1rem; }
.card .hl { font-weight:600; margin-bottom:4px; }
.card .tx { color:#c9d2f5; font-size:.92rem; }
.badge { display:inline-block; font-size:.68rem; font-weight:700; letter-spacing:1.2px; padding:2px 9px; border-radius:99px; margin-left:8px; }
.sec { font-size:.78rem; text-transform:uppercase; letter-spacing:2px; color:#2de2c4; margin:22px 0 8px 0; }
.brand { font-weight:700; font-size:1.35rem; letter-spacing:1px; }
.brand span { color:#2de2c4; }
.note { color:#93a0cf; font-size:.82rem; }
.reason { border-radius:10px; padding:8px 12px; margin:6px 0; font-size:.9rem; border:1px solid #26325a; }
.stTabs [data-baseweb="tab"] { font-weight:600; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- data access
@st.cache_data(show_spinner=False)
def _csv(p: str) -> pd.DataFrame:
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def _parquet(p: str) -> pd.DataFrame:
    return pd.read_parquet(p)


@st.cache_data(show_spinner=False)
def _json(p: str):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def load(kind: str, rel: str):
    p = ART / rel
    if not p.exists():
        return None
    return {"csv": _csv, "parquet": _parquet, "json": _json}[kind](str(p))


def D() -> dict:
    return {"audit": load("json", "metrics/data_audit.json"), "results": load("json", "metrics/results.json"),
            "test": load("csv", "metrics/metrics_test.csv"), "calib": load("csv", "metrics/calibration_test.csv"),
            "coef": load("csv", "metrics/coefficients.csv"), "snap": load("csv", "metrics/snapshot_summary.csv"),
            "sl": load("parquet", "shortlist/shortlist.parquet"), "reports": load("json", "shortlist/evidence_reports.json"),
            "evo": load("parquet", "features/evolution.parquet"), "mem": load("parquet", "features/memberships.parquet"),
            "cedge": load("parquet", "features/community_edges.parquet"), "sim": load("csv", "simulation/simulation_summary.csv"),
            "simruns": load("csv", "simulation/simulation_runs.csv")}


# ----------------------------------------------------------------------------- UI helpers
def fig_style(fig: go.Figure, h: int = 420) -> go.Figure:
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=h,
                      font=dict(family="Space Grotesk, sans-serif", color=PALETTE["text"]), margin=dict(l=10, r=10, t=50, b=10))
    return fig


def kpi(label: str, value: str, sub: str = "") -> str:
    return f'<div class="kpi"><div class="l">{esc(label)}</div><div class="v">{esc(value)}</div><div class="s">{esc(sub)}</div></div>'


def finding_card(f: dict, plain: bool) -> None:
    label, color = VERDICT_STYLE.get(f["verdict"], ("INFO", "#8b7bff"))
    body = f["plain"] if plain else f["technical"]
    st.markdown(f'<div class="card" style="border-left-color:{color}"><h4>{esc(f["title"])}'
                f'<span class="badge" style="background:{color}22;color:{color};border:1px solid {color}66">{label}</span></h4>'
                f'<div class="hl">{esc(f["headline"])}</div><div class="tx">{esc(body)}</div></div>', unsafe_allow_html=True)
    with st.expander("Details and caveats"):
        st.markdown(f"**Technical:** {f['technical']}")
        st.markdown(f"**Caveat:** {f['caveat']}")


def section(title: str) -> None:
    st.markdown(f'<div class="sec">{esc(title)}</div>', unsafe_allow_html=True)


def missing(name: str) -> None:
    st.info(f"`{name}` not found in {ART}. Run `bridge-ct run` (or `bridge-ct demo`) first.")


# ----------------------------------------------------------------------------- pages
def page_brief(d: dict, plain: bool) -> None:
    a = d["audit"] or {}
    tags = "".join(f'<span class="tag">{esc(t)}</span>' for t in (
        ["SYNTHETIC DATA"] if a.get("synthetic") else []) + [f"{a.get('n_works', 0):,} works", f"{a.get('n_authors', 0):,} authors",
                                                              f"{a.get('n_trials', 0):,} trials"])
    st.markdown(f'<div class="hero"><h1>Where should research communities connect next?</h1>'
                f'<p>BRIDGE-CT reads public collaboration records, finds groups of researchers who are topically close but '
                f'structurally disconnected, and estimates which pairs are likely to start working together.</p>'
                f'<div style="margin-top:12px">{tags}</div></div>', unsafe_allow_html=True)
    if a.get("synthetic"):
        st.warning("Demo mode: this is synthetic data for verifying the software. Do not report these numbers as findings.")
    res, test = d["results"], d["test"]
    if res is None or test is None:
        missing("metrics/results.json")
        return
    row = test[test["model"] == "bayes_full"].iloc[0]
    base = res["base_rates"]["test"]
    p10 = row.get("precision_at_10_per_cutoff", row.get("precision_at_10", np.nan))
    best_name = res.get("best_baseline_on_val")
    best_ap = float(test[test["model"] == best_name]["auprc"].iloc[0]) if best_name in set(test["model"]) else np.nan
    c = st.columns(4)
    c[0].markdown(kpi("Model accuracy (AUPRC)", f"{row['auprc']:.3f}", f"random would score ≈ {base:.3f}"), unsafe_allow_html=True)
    c[1].markdown(kpi("Lift over random", f"{row['auprc'] / base:.1f}×", "on the held-out future period"), unsafe_allow_html=True)
    c[2].markdown(kpi("Top-10 hit rate", f"{p10 * 100:.0f}%", "of the top 10 pairs per year actually link"), unsafe_allow_html=True)
    c[3].markdown(kpi("Best simple rule", f"{best_ap:.3f}" if not np.isnan(best_ap) else "n/a", BASELINE_NAMES.get(best_name, "")[:34]), unsafe_allow_html=True)
    findings = explain_results(res, a, d["simruns"])
    section("Verdict board")
    st.markdown(f"**{esc(overall_summary(findings))}**")
    cols = st.columns(2)
    for i, f in enumerate([f for f in findings if f["key"] in ("H1", "H1b", "H2", "H3", "H4", "VALUE")]):
        with cols[i % 2]:
            finding_card(f, plain)
    extras = [f for f in findings if f["key"] in ("N", "DATA", "TRIAL")]
    if extras:
        section("Things to watch")
        for f in extras:
            finding_card(f, plain)
    with st.expander("Reading guide: what do these words mean?"):
        st.markdown("""
- **Bridge**: a pair of researchers from two different communities who have never co-authored but might.
- **AUPRC**: an accuracy score built for rare events. A random guesser scores about the share of true links; higher is better.
- **Lift**: how many times better than random guessing.
- **Calibration**: when the model says 10%, do about 10% of such pairs really link?
- **Credible interval**: the range the model considers plausible for a pair's probability.
- **Simulation**: a what-if on the network graph; it is not a medical or causal result.""")


def page_atlas(d: dict, plain: bool) -> None:
    mem, ce, sl, evo = d["mem"], d["cedge"], d["sl"], d["evo"]
    if mem is None or ce is None:
        return missing("features/memberships.parquet")
    years = sorted(mem["year"].unique())
    c1, c2, c3 = st.columns([1, 1, 1])
    year = c1.select_slider("Snapshot year", options=years, value=years[-1])
    topn = c2.slider("Predicted bridges to draw", 0, 100, 25)
    min_e = c3.slider("Hide existing ties below", 1, 20, 1)
    m = mem[mem["year"] == year]
    sizes = m.groupby("community_id").size()
    age = evo[evo["year"] == year].set_index("community_id")["age"] if evo is not None else pd.Series(dtype=float)
    e = ce[(ce["year"] == year) & ce["comm_a"].isin(sizes.index) & ce["comm_b"].isin(sizes.index) & (ce["n_edges"] >= min_e)]
    G = nx.Graph(); G.add_nodes_from(sizes.index)
    G.add_weighted_edges_from(e[["comm_a", "comm_b", "n_edges"]].itertuples(index=False))
    pos = nx.spring_layout(G, seed=42, weight="weight", k=1.6 / max(1, np.sqrt(len(G))))
    cx, cy = np.mean([p[0] for p in pos.values()]), np.mean([p[1] for p in pos.values()])
    fig = go.Figure()
    for a, b, w in G.edges(data="weight"):
        fig.add_trace(go.Scatter(x=[pos[a][0], pos[b][0]], y=[pos[a][1], pos[b][1]], mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(width=0.4 + min(5, np.sqrt(w) / 3), color="rgba(147,160,207,0.28)")))
    corr = pd.DataFrame()
    if sl is not None and len(sl) and int(sl["cutoff"].iloc[0]) == year and topn:
        s = sl.head(topn)
        corr = (s.assign(a=np.minimum(s["comm_u"], s["comm_v"]), b=np.maximum(s["comm_u"], s["comm_v"]))
                .groupby(["a", "b"]).agg(pairs=("prob", "size"), mean_prob=("prob", "mean")).reset_index())
        corr = corr[corr["a"].isin(pos) & corr["b"].isin(pos)]
        lo, hi = corr["mean_prob"].min(), corr["mean_prob"].max()
        for r in corr.itertuples(index=False):
            p0, p1 = np.array(pos[r.a]), np.array(pos[r.b])
            ctrl = np.array([cx, cy]) + ((p0 + p1) / 2 - np.array([cx, cy])) * 0.35
            t = np.linspace(0, 1, 28)[:, None]
            curve = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t ** 2 * p1
            col = px.colors.sample_colorscale(["#2de2c4", "#ffb454", "#ff6b8a"], [(r.mean_prob - lo) / (hi - lo + 1e-12)])[0]
            fig.add_trace(go.Scatter(x=curve[:, 0], y=curve[:, 1], mode="lines", showlegend=False,
                                     line=dict(width=1.5 + 2 * np.sqrt(r.pairs), color=col, dash="dot"),
                                     hovertemplate=f"Bridge {r.a} ↔ {r.b}<br>{r.pairs} candidate pair(s)<br>mean probability {r.mean_prob:.3f}<extra></extra>"))
    nodes = list(G.nodes())
    fig.add_trace(go.Scatter(x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text", showlegend=False,
                             text=[str(n) for n in nodes], textfont=dict(size=10, color="#0b1020"),
                             marker=dict(size=[16 + 3.2 * np.sqrt(sizes[n]) for n in nodes], line=dict(width=1.5, color="#e8ecff"),
                                         color=[float(age.get(n, 1)) for n in nodes], colorscale="Purp", colorbar=dict(title="Age (yrs)", thickness=10)),
                             hovertext=[f"Community {n}<br>{sizes[n]} authors<br>age {int(age.get(n, 1))} yr" for n in nodes], hoverinfo="text"))
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), title=f"Community atlas, {year}")
    st.plotly_chart(fig_style(fig, 640))
    st.caption("Circles are communities (size = authors, colour = age). Grey lines are existing cross-community collaborations. "
               "Dotted glowing arcs are predicted bridges (thicker = more candidate pairs; teal to rose = higher mean probability).")
    if len(corr):
        section("Bridge corridors")
        ex = {(r.comm_a, r.comm_b): r.n_edges for r in ce[ce["year"] == year].itertuples(index=False)}
        corr["existing_ties"] = [ex.get((r.a, r.b), 0) for r in corr.itertuples(index=False)]
        corr["gap"] = np.where(corr["existing_ties"] == 0, "no ties yet", "thin ties")
        st.dataframe(corr.sort_values("mean_prob", ascending=False).rename(
            columns={"a": "community A", "b": "community B", "pairs": "candidate pairs", "mean_prob": "mean probability"}), hide_index=True)
        if plain:
            st.markdown("<span class='note'>A corridor with 'no ties yet' joins two communities that have never co-authored but whose members look likely to start.</span>",
                        unsafe_allow_html=True)
    elif year != years[-1] or sl is None:
        st.info("Predicted bridges exist only for the most recent snapshot (the forecast year). Move the year slider to the last value.")


def page_timeline(d: dict, plain: bool) -> None:
    evo = d["evo"]
    if evo is None:
        return missing("features/evolution.parquet")
    ev = evo.groupby(["year", "event"]).size().unstack(fill_value=0)
    c = st.columns(4)
    for col, name in zip(c, ["birth", "continue", "split", "merge"]):
        col.markdown(kpi(name.title() + "s", str(int(ev[name].sum()) if name in ev else 0), "all years"), unsafe_allow_html=True)
    top = evo.groupby("lineage_id")["size"].max().nlargest(10).index
    df = evo.assign(lin=np.where(evo["lineage_id"].isin(top), "L" + evo["lineage_id"].astype(str), "other"))
    area = df.groupby(["year", "lin"])["size"].sum().reset_index()
    st.plotly_chart(fig_style(px.area(area, x="year", y="size", color="lin", title="Community lineages over time (authors)",
                                      color_discrete_sequence=px.colors.qualitative.Pastel), 420))
    if plain:
        st.caption("Each colour band is one community lineage followed across years. A band that thickens is a growing community; "
                   "bands that appear are births; two bands combining are merges.")
    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(fig_style(px.imshow(ev.T, aspect="auto", color_continuous_scale=["#0b1020", "#8b7bff", "#2de2c4"],
                                            title="Event heatmap (count per year)"), 330))
    with right:
        lin = st.selectbox("Inspect a lineage", sorted(evo["lineage_id"].unique()))
        sub = evo[evo["lineage_id"] == lin].sort_values("year")
        st.plotly_chart(fig_style(px.line(sub, x="year", y="size", markers=True, hover_data=["event", "age", "growth", "turnover"],
                                          title=f"Lineage {lin}", color_discrete_sequence=[PALETTE["teal"]]), 330))


def page_ledger(d: dict, plain: bool) -> None:
    sl, reports, res = d["sl"], d["reports"], d["results"]
    if sl is None:
        return missing("shortlist/shortlist.parquet")
    base = res["base_rates"]["test"] if res else None
    H = res.get("horizon_years") if res else None
    c1, c2, c3 = st.columns([1, 1, 1.4])
    pmin = c1.slider("Minimum probability", 0.0, float(sl["prob"].max()), 0.0, step=0.005)
    src = c2.multiselect("How it was found", sorted(sl["source"].unique()), default=sorted(sl["source"].unique()))
    q = c3.text_input("Search author name or id")
    f = sl[(sl["prob"] >= pmin) & sl["source"].isin(src)]
    if q:
        mask = f["name_u"].fillna("").str.contains(q, case=False) | f["name_v"].fillna("").str.contains(q, case=False) | \
               f["u"].str.contains(q, case=False) | f["v"].str.contains(q, case=False)
        f = f[mask]
    st.dataframe(f[["rank", "name_u", "name_v", "prob", "prob_lo", "prob_hi", "source", "common_neighbors", "semantic_author", "comm_u", "comm_v"]],
                 hide_index=True, column_config={"prob": st.column_config.ProgressColumn("Probability", min_value=0.0, max_value=float(sl["prob"].max()), format="%.3f"),
                                                 "semantic_author": st.column_config.NumberColumn("Topic sim.", format="%.2f")})
    if f.empty:
        return st.info("No pairs match the filters.")
    section("Pair dossier")
    pick = st.selectbox("Choose a pair", f["rank"].tolist(), format_func=lambda r: f"#{r}  {sl.loc[sl['rank'] == r, 'name_u'].iloc[0]}  ↔  {sl.loc[sl['rank'] == r, 'name_v'].iloc[0]}")
    row = sl[sl["rank"] == pick].iloc[0].to_dict()
    ex = explain_pair(row, base, H)
    left, right = st.columns([1, 1.25])
    with left:
        g = go.Figure(go.Indicator(mode="gauge+number", value=row["prob"] * 100, number=dict(suffix="%", valueformat=".1f"),
                                   gauge=dict(axis=dict(range=[0, max(10, float(sl["prob"].max()) * 120)]), bar=dict(color=PALETTE["teal"]),
                                              steps=[dict(range=[(row["prob_lo"]) * 100, (row["prob_hi"]) * 100], color="#8b7bff55")],
                                              threshold=dict(line=dict(color=PALETTE["amber"], width=3), value=(base or 0) * 100)),
                                   title=dict(text="Chance of first co-authorship")))
        st.plotly_chart(fig_style(g, 280))
        st.caption("Violet band = 90% credible interval. Amber line = average candidate.")
    with right:
        st.markdown(f"**{esc(ex['headline'])}**")
        for fct in ex["facts"]:
            st.markdown(f"- {esc(fct)}")
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**Why the model likes this pair**")
        for r in ex["reasons_for"] or [{"label": "No dominant positive factor", "meaning": "", "effect": 0}]:
            st.markdown(f'<div class="reason" style="border-left:4px solid {PALETTE["green"]}"><b>{esc(r["label"])}</b> ({r["effect"]:+.2f})<br>'
                        f'<span class="note">{esc(r["meaning"])}</span></div>', unsafe_allow_html=True)
    with r2:
        st.markdown("**What holds it back**")
        for r in ex["reasons_against"] or [{"label": "Nothing major", "meaning": "", "effect": 0}]:
            st.markdown(f'<div class="reason" style="border-left:4px solid {PALETTE["rose"]}"><b>{esc(r["label"])}</b> ({r["effect"]:+.2f})<br>'
                        f'<span class="note">{esc(r["meaning"])}</span></div>', unsafe_allow_html=True)
    st.caption(ex["caution"])
    rep = next((x for x in (reports or []) if x["rank"] == pick), None)
    if rep is None:
        st.info("Source-record evidence is generated for the top 25 pairs only.")
    elif rep["status"] != "ok":
        st.warning(rep["status"])
    else:
        with st.expander("Evidence: source records", expanded=True):
            st.markdown("**Shared topics:** " + (", ".join(rep["shared_topics"]) or "none found"))
            for aid, works in rep["supporting_works"].items():
                st.markdown(f"**Recent works of {esc(aid)}**")
                for w in works:
                    st.markdown(f"- ({w['year']}) [{w['title']}]({w['url']})")


def page_trials(d: dict, plain: bool) -> None:
    reps = d["reports"]
    if reps is None:
        return missing("shortlist/evidence_reports.json")
    rows = [dict(t, rank=r["rank"]) for r in reps for t in r.get("trial_evidence", [])]
    a = d["audit"] or {}
    c = st.columns(3)
    c[0].markdown(kpi("Paper-trial links", f"{a.get('n_links', 0):,}", "identifier-based only"), unsafe_allow_html=True)
    c[1].markdown(kpi("Via PubMed record", str(a.get("link_methods", {}).get("pubmed_databank", 0)), "confidence 1.00"), unsafe_allow_html=True)
    c[2].markdown(kpi("Via NCT ID in abstract", str(a.get("link_methods", {}).get("abstract_nct_id", 0)), "confidence 0.95"), unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(rows) if rows else pd.DataFrame({"info": ["No trial links for the top candidates"]}), hide_index=True)
    if plain:
        st.caption("A link appears only when a PubMed record or the abstract itself names the trial's NCT number. Nothing is guessed from similar wording.")


def page_sim(d: dict, plain: bool) -> None:
    s, runs = d["sim"], d["simruns"]
    if s is None:
        return missing("simulation/simulation_summary.csv")
    order = {"baseline": 0, "random": 1, "degree": 2, "betweenness": 3, "model": 4}
    s = s.assign(o=s["strategy"].map(order)).sort_values("o")
    fig = go.Figure(go.Bar(x=s["strategy"], y=s["coverage_delta"], error_y=dict(type="data", array=s["coverage_sd"]),
                           marker_color=[PALETTE["teal"] if x == "model" else PALETTE["violet"] for x in s["strategy"]]))
    fig.update_layout(title="Extra reach vs adding no links (independent-cascade diffusion)", yaxis_title="Δ coverage")
    st.plotly_chart(fig_style(fig, 380))
    st.dataframe(s.drop(columns="o").round(4), hide_index=True)
    f = [x for x in explain_results(d["results"] or {}, None, runs) if x["key"] == "H4"]
    if f:
        finding_card(f[0], plain)
    st.caption("Graph simulation under stated assumptions, not a causal or clinical effect.")


def page_lab(d: dict, plain: bool) -> None:
    test, res = d["test"], d["results"]
    if test is None or res is None:
        return missing("metrics/metrics_test.csv")
    t1, t2, t3, t4, t5 = st.tabs(["Leaderboard", "What matters", "Calibration", "Feature effects", "Head-to-head"])
    with t1:
        m = test.sort_values("auprc")
        lab = [MODEL_LABELS.get(x, x.replace("bayes_", "Bayes ").replace("_", " ")) for x in m["model"]]
        err_lo = (m["auprc"] - m.get("auprc_ci_lo", m["auprc"])).fillna(0)
        err_hi = (m.get("auprc_ci_hi", m["auprc"]) - m["auprc"]).fillna(0)
        fam = ["Bayesian" if x.startswith("bayes") else "Logistic" if x.startswith("logistic") else "Simple rule" for x in m["model"]]
        fig = go.Figure(go.Bar(y=lab, x=m["auprc"], orientation="h", marker_color=[{"Bayesian": PALETTE["teal"], "Logistic": PALETTE["violet"]}.get(f, PALETTE["amber"]) for f in fam],
                               error_x=dict(type="data", symmetric=False, array=err_hi, arrayminus=err_lo)))
        fig.add_vline(x=res["base_rates"]["test"], line_dash="dash", line_color=PALETTE["rose"], annotation_text="random guess")
        fig.update_layout(title="Test-period accuracy (AUPRC, higher is better; bars show 95% interval where computed)")
        st.plotly_chart(fig_style(fig, 560))
    with t2:
        full = float(test.loc[test["model"] == "bayes_full", "auprc"].iloc[0])
        ab = test[test["model"].str.startswith("bayes_minus_")].copy()
        ab["group"] = ab["model"].str.replace("bayes_minus_", "", regex=False)
        ab["drop"] = full - ab["auprc"]
        fig = px.bar(ab.sort_values("drop"), x="drop", y="group", orientation="h", color="drop", color_continuous_scale=["#ff6b8a", "#0b1020", "#2de2c4"],
                     color_continuous_midpoint=0, title="Accuracy lost when a feature group is removed")
        st.plotly_chart(fig_style(fig, 420))
        st.caption("Longer teal bar = that group matters more. Negative = the model did slightly better without it.")
    with t3:
        cal = d["calib"]
        if cal is not None:
            fig = px.line(cal, x="mean_pred", y="frac_pos", color="model", markers=True, title="Predicted vs observed frequency (closer to the dashed line is better)",
                          color_discrete_map={"bayes_full": PALETTE["teal"], "logistic_balanced": PALETTE["rose"]})
            mx = float(cal["mean_pred"].max())
            fig.add_shape(type="line", x0=0, y0=0, x1=mx, y1=mx, line=dict(dash="dash", color=PALETTE["muted"]))
            st.plotly_chart(fig_style(fig, 420))
    with t4:
        co = d["coef"]
        if co is not None:
            co = co.head(15).copy()
            co["label"] = [describe_feature(x)[0] for x in co["feature"]]
            co = co.sort_values("mean")
            fig = go.Figure(go.Scatter(x=co["mean"], y=co["label"], mode="markers", marker=dict(size=11, color=[PALETTE["green"] if v > 0 else PALETTE["rose"] for v in co["mean"]]),
                                       error_x=dict(type="data", symmetric=False, array=co["hi"] - co["mean"], arrayminus=co["mean"] - co["lo"])))
            fig.add_vline(x=0, line_color=PALETTE["muted"])
            fig.update_layout(title="Posterior effect of each feature (log-odds per 1 std; interval = 94%)")
            st.plotly_chart(fig_style(fig, 520))
            st.caption("Right of zero raises the chance of a new collaboration, left lowers it. An interval crossing zero means the data cannot tell.")
    with t5:
        rows = [{"comparison": k, "AUPRC gain": v["diff_auprc"], "CI low": v["ci_lo"], "CI high": v["ci_hi"], "verdict": _ci_verdict(v)} for k, v in res["comparisons"].items()]
        st.dataframe(pd.DataFrame(rows).round(4), hide_index=True)


def page_export(d: dict, plain: bool) -> None:
    for rel, label in (("shortlist/shortlist.csv", "Shortlist (CSV)"), ("metrics/results.json", "Results (JSON)"),
                       ("model_cards/model_card.md", "Model card"), ("metrics/data_audit.json", "Data audit (JSON)")):
        p = ART / rel
        if p.exists():
            st.download_button(f"Download {label}", p.read_bytes(), p.name, key=rel)


# ----------------------------------------------------------------------------- shell
PAGES = {"🧭  Mission brief": page_brief, "🌐  Community atlas": page_atlas, "⏳  Evolution timeline": page_timeline,
         "🔗  Bridge ledger": page_ledger, "🧪  Trial evidence": page_trials, "🎲  Simulation lab": page_sim,
         "📊  Model lab": page_lab, "📦  Export": page_export}
st.sidebar.markdown('<div class="brand">BRIDGE<span>·</span>CT</div><div class="note">Research-community bridge explorer</div>', unsafe_allow_html=True)
choice = st.sidebar.radio("Navigate", list(PAGES), label_visibility="collapsed")
plain_mode = st.sidebar.toggle("Plain-language explanations", value=True, help="Off shows the technical wording.")
st.sidebar.markdown("---")
st.sidebar.markdown('<div class="note">Research analysis only. Not clinical decision support; scores show statistical association '
                    'with future co-authorship, not clinical promise.</div>', unsafe_allow_html=True)
PAGES[choice](D(), plain_mode)