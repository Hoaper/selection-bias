"""Look-ahead vs real-time selection of institutions in the backtest.

The main panel (forecasting.build_panel) keeps institutions that are continuously active up to 2025 and have
at least 30 works in 2025, so every institution in the backtest is known to survive. Here the same models are
run on two universes built from the same raw data:

  look-ahead : the 202 panel institutions (selection uses information from 2025)
  real-time  : every continuous active run (>= 10 works per year) of every institution; a run is a forecast
               target at origin T whenever it is active at T, whatever happens to it later

A run's series continues up to three years after it ends, so that its later outcomes, including a collapse
of output, are scored. Training uses only active origin rows whose target is observed by the forecast origin.

Run: python selection_bias.py   -> data/results/selection_*.csv
"""
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import forecasting as F

OUT = F.DATA / "results"
OUT.mkdir(exist_ok=True)
EVAL_FROM = 2019
CAL_ORIGINS = range(2013, 2025)
MIN_WORKS, MIN_SPAN, MIN_LAST = 10, 6, 30


def segment_panel() -> pd.DataFrame:
    raw = pd.read_csv(F.DATA / "panel_raw.csv")
    raw = raw[(raw.year >= F.FIRST_YEAR) & (raw.year <= F.LAST_YEAR)]
    raw = raw.groupby(["inst_id", "name", "country", "year"], as_index=False)[["works", "oa_works"]].sum()
    rows = []
    for iid, g in raw.groupby("inst_id"):
        s = g.set_index("year").reindex(range(F.FIRST_YEAR, F.LAST_YEAR + 1))
        s[["works", "oa_works"]] = s[["works", "oa_works"]].fillna(0)
        active = (s.works >= MIN_WORKS).to_numpy()
        years = s.index.to_numpy()
        k, i = 0, 0
        while i < len(years):
            if not active[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(years) and active[j + 1]:
                j += 1
            start, end = years[i], years[j]
            # the panel criteria of build_panel, which use the 2025 value
            survivor = end == F.LAST_YEAR and end - start + 1 >= MIN_SPAN and s.loc[end, "works"] >= MIN_LAST
            seg = s.loc[start:min(end + 3, F.LAST_YEAR)].copy()
            seg["active"] = seg.index <= end
            seg["inst_id"] = f"{iid}#{k}"
            seg["base_id"] = iid
            seg["name"] = g["name"].iloc[0]
            seg["country"] = g["country"].iloc[0]
            seg["start"] = start
            seg["survivor"] = survivor
            rows.append(seg.reset_index().rename(columns={"index": "year"}))
            k += 1
            i = j + 1
    p = pd.concat(rows, ignore_index=True)
    p["oa_share"] = np.where(p.works > 0, p.oa_works / p.works.clip(lower=1), np.nan)
    return p


def add_context(p: pd.DataFrame) -> pd.DataFrame:
    wdi = pd.read_csv(F.DATA / "wdi.csv").sort_values(["country", "year"])
    grid = pd.MultiIndex.from_product([wdi.country.unique(), range(F.FIRST_YEAR, F.LAST_YEAR + 1)],
                                      names=["country", "year"]).to_frame(index=False)
    wdi = grid.merge(wdi, on=["country", "year"], how="left")
    vals = [c for c in wdi.columns if c not in ("country", "year")]
    wdi[vals] = wdi.groupby("country")[vals].ffill()
    p = p.merge(wdi, on=["country", "year"], how="left")
    tot = p[p.active].groupby(["country", "year"]).works.sum().rename("tot")
    p = p.join(tot, on=["country", "year"])
    p["country_share"] = p.works / p.tot.fillna(p.works).clip(lower=1)
    return p.drop(columns="tot").sort_values(["inst_id", "year"]).reset_index(drop=True)


def backtest(fp: pd.DataFrame, cache: pd.DataFrame, origins) -> pd.DataFrame:
    """forecasting.run_backtest restricted to active origin rows."""
    act = fp[["inst_id", "year", "active"]].rename(columns={"year": "origin"})
    frames = []
    for ind in F.INDICATORS:
        for h in F.HORIZONS:
            pairs = F.make_pairs(fp, ind, h, cache).merge(act, on=["inst_id", "origin"])
            pairs = pairs[pairs.active].drop(columns="active")
            for t in origins:
                if t + h > F.LAST_YEAR:
                    continue
                f = F.forecast_origin(pairs, t, h)
                if f is not None:
                    f["indicator"] = ind
                    frames.append(f)
    wide = pd.concat(frames, ignore_index=True)
    return wide.melt(id_vars=["indicator", "inst_id", "origin", "h", "z_origin", "z_target"],
                     value_vars=list(F.MODELS), var_name="model", value_name="z_hat")


def run(name: str, seg: pd.DataFrame):
    t0 = time.time()
    fp = F.add_features(add_context(seg))
    path = OUT / f"selection_arima_cache_{name}.csv"
    if not path.exists():
        F.compute_arima_cache(fp, range(2012, F.LAST_YEAR + 1)).to_csv(path, index=False)
    cache = pd.read_csv(path)  # read back from CSV, as in run_experiments.py
    fc = F.add_scale(backtest(fp, cache, CAL_ORIGINS), fp)
    fc = fc[fc.z_target.notna()]
    val = F.summarize(F.score(fc[(fc.origin >= 2015) & (fc.origin < EVAL_FROM)]))
    val_mase = val[val.model.isin(["hybrid_res", "hybrid_feat", "hybrid_ens"])].groupby("model").MASE.mean()
    chosen = val_mase.idxmin()
    fc = pd.concat([fc, fc[fc.model == chosen].assign(model="hybrid")], ignore_index=True)
    ev = F.score(fc[fc.origin >= EVAL_FROM])
    ev = ev.merge(fp[["inst_id", "year", "survivor", "base_id", "age"]].rename(columns={"year": "origin"}),
                  on=["inst_id", "origin"])
    ev["scenario"] = name
    print(name, f"{time.time() - t0:.0f}s", "chosen", chosen, val_mase.round(3).to_dict())
    return ev, fp, {"chosen": chosen, "validation_mase": val_mase.round(4).to_dict()}


if __name__ == "__main__":
    seg = segment_panel()
    ev_la, fp_la, info_la = run("lookahead", seg[seg.survivor])
    ev_rt, fp_rt, info_rt = run("realtime", seg)
    ev = pd.concat([ev_la, ev_rt], ignore_index=True)
    ev.to_csv(OUT / "selection_scored.csv", index=False)

    models = list(F.MODELS) + ["hybrid"]
    ev["group"] = np.where(ev.survivor, "survivor", "non-survivor")
    overall = F.summarize(ev, by=("scenario", "indicator", "model"))
    by_group = F.summarize(ev[ev.scenario == "realtime"], by=("indicator", "group", "model"))
    by_h = F.summarize(ev, by=("scenario", "indicator", "h", "model"))
    overall.to_csv(OUT / "selection_overall.csv", index=False)
    by_group.to_csv(OUT / "selection_by_group.csv", index=False)
    by_h.to_csv(OUT / "selection_by_h.csv", index=False)

    # paired tests of the proposed hybrid against every other model in each scenario
    tests = []
    for sc in ("lookahead", "realtime"):
        for ind in F.INDICATORS:
            d = ev[(ev.scenario == sc) & (ev.indicator == ind)].pivot_table(index=["inst_id", "origin", "h"],
                                                                             columns="model", values="sae")
            inst = d.groupby(level="inst_id").mean()
            for m in models:
                if m in ("hybrid", info_la["chosen"] if sc == "lookahead" else info_rt["chosen"]):
                    continue
                tests.append({"scenario": sc, "indicator": ind, "vs": m, "n": len(d),
                              "mase_ratio": d.hybrid.mean() / d[m].mean(),
                              "better_share": (d.hybrid < d[m]).mean(), "p_pairs": wilcoxon(d.hybrid, d[m]).pvalue,
                              "n_inst": len(inst), "p_inst": wilcoxon(inst.hybrid, inst[m]).pvalue})
    pd.DataFrame(tests).to_csv(OUT / "selection_tests.csv", index=False)

    # what the real-time universe adds
    pts = ev[(ev.model == "hybrid")].groupby(["scenario", "indicator"]).agg(
        points=("sae", "size"), runs=("inst_id", "nunique"), institutions=("base_id", "nunique"),
        non_survivor_points=("survivor", lambda s: int((~s).sum())))
    pts.to_csv(OUT / "selection_points.csv")
    rt = ev[(ev.scenario == "realtime") & (ev.model == "hybrid") & (ev.indicator == "works")]
    drop = rt.assign(fell=rt.y < MIN_WORKS).groupby("survivor").fell.mean()
    summary = {"lookahead": info_la, "realtime": info_rt,
               "segments": int(seg.inst_id.nunique()), "survivor_segments": int(seg[seg.survivor].inst_id.nunique()),
               "institutions_raw": int(seg.base_id.nunique()),
               "share_targets_below_min_works": {str(k): round(float(v), 4) for k, v in drop.items()}}
    (OUT / "selection_summary.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 200)
    print(pts.to_string(), "\n")
    print(overall[overall.model.isin(["naive", "arima", "mlp", "gbm", "hybrid"])].pivot_table(
        index=["indicator", "model"], columns="scenario", values=["MASE", "MdAPE"]).round(3).to_string(), "\n")
    print(by_group[by_group.model.isin(["naive", "arima", "gbm", "hybrid"])].pivot_table(
        index=["indicator", "model"], columns="group", values=["MASE", "n"]).round(3).to_string(), "\n")
    print(pd.DataFrame(tests).round(4).to_string(index=False))
    print(summary)
