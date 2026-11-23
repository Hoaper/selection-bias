"""Breakdown of the selection-bias experiment (run selection_bias.py first).

1. Why a run is missing from the look-ahead panel: it ended before 2025, it is too short by 2025, or its
   output in 2025 is below 30 works.
2. MASE by group and model in the real-time scenario; paired tests on the runs the panel drops.
3. Institution-cluster bootstrap CI for the change in MASE between the scenarios.

Run: python selection_analysis.py   -> data/results/selection_groups.csv, selection_bootstrap.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import forecasting as F
from selection_bias import MIN_LAST, MIN_SPAN, OUT, segment_panel

seg = segment_panel()
info = seg[seg.active].groupby("inst_id").agg(start=("year", "min"), end=("year", "max"), survivor=("survivor", "first"))
last = seg[seg.year == F.LAST_YEAR].set_index("inst_id").works
info["works_2025"] = last.reindex(info.index)


def reason(r):
    if r.survivor:
        return "in panel"
    if r.end < F.LAST_YEAR:
        return "ended before 2025"
    if r.end - r.start + 1 < MIN_SPAN:
        return "too young in 2025"
    return "below 30 works in 2025"


info["group"] = info.apply(reason, axis=1)
assert (info[info.group == "below 30 works in 2025"].works_2025 < MIN_LAST).all()

ev = pd.read_csv(OUT / "selection_scored.csv")
ev = ev.join(info.group, on="inst_id")
rt = ev[ev.scenario == "realtime"]

models = ["naive", "drift", "arima", "mlp", "gbm", "hybrid"]
rows = []
for (ind, grp), g in rt[rt.model.isin(models)].groupby(["indicator", "group"]):
    w = g.pivot_table(index=["inst_id", "origin", "h"], columns="model", values="sae")
    row = {"indicator": ind, "group": grp, "points": len(w), "runs": w.index.get_level_values(0).nunique()}
    for m in models:
        row[f"MASE_{m}"] = w[m].mean()
    if grp != "in panel":
        for m in ("naive", "gbm"):
            row[f"p_hybrid_vs_{m}"] = wilcoxon(w.hybrid, w[m]).pvalue
            row[f"hybrid_better_{m}"] = (w.hybrid < w[m]).mean()
    rows.append(row)
groups = pd.DataFrame(rows)
groups.to_csv(OUT / "selection_groups.csv", index=False)

# all runs the panel drops, pooled
drop = rt[(rt.group != "in panel") & rt.model.isin(models)]
pooled = []
for ind, g in drop.groupby("indicator"):
    w = g.pivot_table(index=["inst_id", "origin", "h"], columns="model", values="sae")
    pooled.append({"indicator": ind, "points": len(w), "runs": w.index.get_level_values(0).nunique(),
                   **{f"MASE_{m}": w[m].mean() for m in models},
                   "p_hybrid_vs_naive": wilcoxon(w.hybrid, w.naive).pvalue,
                   "p_hybrid_vs_gbm": wilcoxon(w.hybrid, w.gbm).pvalue,
                   "hybrid_better_naive": (w.hybrid < w.naive).mean()})
pooled = pd.DataFrame(pooled)
pooled.to_csv(OUT / "selection_dropped.csv", index=False)

# bootstrap over original institutions: MASE(real-time) - MASE(look-ahead), and the H3/GBM ratio in each
rng = np.random.default_rng(42)
boot = []
for ind in F.INDICATORS:
    e = ev[(ev.indicator == ind) & ev.model.isin(["hybrid", "gbm"])]
    agg = e.groupby(["base_id", "scenario", "model"]).sae.agg(["sum", "count"]).unstack(["scenario", "model"]).fillna(0)
    ids = agg.index.to_numpy()
    idx = rng.integers(0, len(ids), size=(2000, len(ids)))

    def mase(sc, m, idx=idx):
        return agg[("sum", sc, m)].to_numpy()[idx].sum(1) / agg[("count", sc, m)].to_numpy()[idx].sum(1)

    full = {(sc, m): agg[("sum", sc, m)].sum() / agg[("count", sc, m)].sum()
            for sc in ("lookahead", "realtime") for m in ("hybrid", "gbm")}
    for m in ("hybrid", "gbm"):
        d = mase("realtime", m) - mase("lookahead", m)
        boot.append({"indicator": ind, "stat": f"MASE change, {m}", "value": full[("realtime", m)] - full[("lookahead", m)],
                     "rel_pct": 100 * (full[("realtime", m)] / full[("lookahead", m)] - 1),
                     "ci_lo": np.quantile(d, 0.025), "ci_hi": np.quantile(d, 0.975)})
    for sc in ("lookahead", "realtime"):
        r = mase(sc, "hybrid") / mase(sc, "gbm")
        boot.append({"indicator": ind, "stat": f"H3/GBM ratio, {sc}", "value": full[(sc, "hybrid")] / full[(sc, "gbm")],
                     "ci_lo": np.quantile(r, 0.025), "ci_hi": np.quantile(r, 0.975)})
    boot.append({"indicator": ind, "stat": "institutions", "value": len(ids)})
boot = pd.DataFrame(boot)
boot.to_csv(OUT / "selection_bootstrap.csv", index=False)

pd.set_option("display.width", 250)
print(info.group.value_counts().to_string(), "\n")
print(groups.round(4).to_string(index=False), "\n")
print(pooled.round(4).to_string(index=False), "\n")
print(boot.round(4).to_string(index=False))
