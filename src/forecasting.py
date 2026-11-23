"""Core forecasting framework for university development indicators.

Indicators (per institution, per year):
  * works     - research output (number of indexed publications)
  * oa_share  - share of open-access publications

All models operate on a transformed scale z (log1p for counts, logit for
shares) and forecast h = 1..3 years ahead from an origin year T.

Models
  naive   - z_{T+h} = z_T
  drift   - random walk with drift estimated on the last 5 years
  arima   - per-series ARIMA(p,1,q) selected by AICc (local statistical model)
  gbm     - global LightGBM, direct multi-horizon, cross-indicator features
  mlp     - global multilayer perceptron with the same features
  combo   - equal-weight combination of arima and gbm
  hybrid_res  - ARIMA-GBM residual hybrid: a global GBM learns the
                correction to the local ARIMA forecast (Zhang-type, but global)
  hybrid_feat - ARIMA-informed global GBM: the local ARIMA forecast enters the
                global model as a feature (feature-level hybridisation)
  hybrid_ens  - statistical-ML hybrid ensemble: mean of gbm, mlp and hybrid_feat
The proposed "hybrid" is the hybrid design with the lowest validation MASE.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIRST_YEAR, LAST_YEAR = 2010, 2025  # 2026 is incomplete at collection time
HORIZONS = (1, 2, 3)
INDICATORS = ("works", "oa_share")
MODELS = ("naive", "drift", "arima", "mlp", "gbm", "combo", "hybrid_res", "hybrid_feat", "hybrid_ens")
ARIMA_ORDERS = [((0, 1, 0), "t"), ((1, 1, 0), "t"), ((0, 1, 1), "t"),
                ((1, 1, 0), "n"), ((1, 1, 1), "t")]
GBM_PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=15,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=42, verbose=-1)


# --------------------------------------------------------------------------- data
def to_z(indicator: str, y):
    y = np.asarray(y, dtype=float)
    if indicator == "works":
        return np.log1p(y)
    p = np.clip(y, 0.01, 0.99)
    return np.log(p / (1 - p))


def from_z(indicator: str, z):
    z = np.asarray(z, dtype=float)
    if indicator == "works":
        return np.expm1(z)
    return 1 / (1 + np.exp(-z))


def build_panel(min_works: int = 10, min_span: int = 6, min_last: int = 30) -> pd.DataFrame:
    """Balanced-from-start panel of active institutions with WDI context."""
    raw = pd.read_csv(DATA / "panel_raw.csv")
    raw = raw[(raw.year >= FIRST_YEAR) & (raw.year <= LAST_YEAR)]
    raw = raw.groupby(["inst_id", "name", "country", "year"], as_index=False)[["works", "oa_works"]].sum()

    rows = []
    for iid, g in raw.groupby("inst_id"):
        s = g.set_index("year").reindex(range(FIRST_YEAR, LAST_YEAR + 1))
        s["works"] = s["works"].fillna(0)
        s["oa_works"] = s["oa_works"].fillna(0)
        active = s["works"] >= min_works
        # first year from which the institution is continuously active up to LAST_YEAR
        start = None
        for yr in range(LAST_YEAR, FIRST_YEAR - 1, -1):
            if active.loc[yr]:
                start = yr
            else:
                break
        if start is None or LAST_YEAR - start + 1 < min_span or s.loc[LAST_YEAR, "works"] < min_last:
            continue
        s = s.loc[start:].copy()
        s["inst_id"] = iid
        s["name"] = g["name"].iloc[0]
        s["country"] = g["country"].iloc[0]
        s["start"] = start
        rows.append(s.reset_index().rename(columns={"index": "year"}))
    panel = pd.concat(rows, ignore_index=True)
    panel["oa_share"] = panel["oa_works"] / panel["works"].clip(lower=1)

    # disambiguate duplicated display names
    dup = panel.groupby("name")["inst_id"].nunique()
    for name in dup[dup > 1].index:
        ids = (panel[panel.name == name].groupby("inst_id")["works"].sum().sort_values(ascending=False).index)
        for k, iid in enumerate(ids[1:], start=2):
            panel.loc[panel.inst_id == iid, "name"] = f"{name} ({k})"

    # country context (World Bank WDI), forward-filled within country
    wdi = pd.read_csv(DATA / "wdi.csv").sort_values(["country", "year"])
    grid = pd.MultiIndex.from_product([wdi.country.unique(), range(FIRST_YEAR, LAST_YEAR + 1)],
                                      names=["country", "year"]).to_frame(index=False)
    wdi = grid.merge(wdi, on=["country", "year"], how="left")
    vals = [c for c in wdi.columns if c not in ("country", "year")]
    wdi[vals] = wdi.groupby("country")[vals].ffill()
    panel = panel.merge(wdi, on=["country", "year"], how="left")
    country_tot = panel.groupby(["country", "year"])["works"].transform("sum")
    panel["country_share"] = panel["works"] / country_tot.clip(lower=1)
    return panel.sort_values(["inst_id", "year"]).reset_index(drop=True)


# ----------------------------------------------------------------------- features
FEATURES = ["z_works", "g1_works", "g2_works", "g3_works", "g_mean3_works", "g_std5_works",
            "z_oa", "g1_oa", "g2_oa", "age", "log_country_share",
            "log_gdp_pc", "rd_exp", "tert_enrol", "researchers_pm"]
FEATURE_LABELS = {
    "z_works": "Output level (log)", "g1_works": "Output growth, t", "g2_works": "Output growth, t-1",
    "g3_works": "Output growth, t-2", "g_mean3_works": "Mean growth, 3 y", "g_std5_works": "Growth volatility, 5 y",
    "z_oa": "Open-access share (logit)", "g1_oa": "OA share change, t", "g2_oa": "OA share change, t-1",
    "age": "Years active", "log_country_share": "Share of national output", "log_gdp_pc": "GDP per capita (PPP, log)",
    "rd_exp": "R&D expenditure, % GDP", "tert_enrol": "Tertiary enrolment, %", "researchers_pm": "Researchers per million",
    "arima_delta": "ARIMA forecast change",
}


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    p["z_works"] = to_z("works", p["works"])
    p["z_oa"] = to_z("oa_share", p["oa_share"])
    g = p.groupby("inst_id")
    p["g1_works"] = g["z_works"].diff()
    p["g2_works"] = g["g1_works"].shift(1)
    p["g3_works"] = g["g1_works"].shift(2)
    p["g_mean3_works"] = g["g1_works"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    p["g_std5_works"] = g["g1_works"].transform(lambda s: s.rolling(5, min_periods=2).std())
    p["g1_oa"] = g["z_oa"].diff()
    p["g2_oa"] = g["g1_oa"].shift(1)
    p["age"] = p["year"] - p["start"]
    p["log_country_share"] = np.log(p["country_share"].clip(lower=1e-4))
    p["log_gdp_pc"] = np.log(p["gdp_pc_ppp"])
    p["rd_exp"] = p["rd_expenditure_pct_gdp"]
    p["tert_enrol"] = p["tertiary_enrolment_gross_pct"]
    p["researchers_pm"] = p["researchers_per_million"]
    return p


# ------------------------------------------------------------------- local models
def _aicc(res, n):
    k = len(res.params)
    return res.aic + (2 * k * (k + 1)) / max(n - k - 1, 1)


def arima_forecast(z: np.ndarray, hmax: int = 3) -> np.ndarray:
    """Auto-ARIMA (d=1) by AICc; drift fallback for very short series."""
    z = np.asarray(z, dtype=float)
    if len(z) < 5:
        return drift_forecast(z, hmax)
    best, best_score = None, np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order, trend in ARIMA_ORDERS:
            try:
                res = ARIMA(z, order=order, trend=trend).fit()
                score = _aicc(res, len(z))
                if np.isfinite(score) and score < best_score:
                    best, best_score = res, score
            except Exception:
                continue
    if best is None:
        return drift_forecast(z, hmax)
    return np.asarray(best.forecast(hmax))


def drift_forecast(z: np.ndarray, hmax: int = 3, window: int = 5) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    w = z[-(window + 1):]
    slope = (w[-1] - w[0]) / max(len(w) - 1, 1)
    return z[-1] + slope * np.arange(1, hmax + 1)


# ------------------------------------------------------------- forecast backtest
@dataclass
class Backtest:
    forecasts: pd.DataFrame   # one row per (indicator, model, inst, origin, h)
    arima_cache: pd.DataFrame


def compute_arima_cache(fp: pd.DataFrame, origins, hmax: int = 3) -> pd.DataFrame:
    """ARIMA forecasts for every series at every origin (needed by the hybrid)."""
    rows = []
    for ind in INDICATORS:
        col = "z_works" if ind == "works" else "z_oa"
        for iid, g in fp.groupby("inst_id"):
            g = g.set_index("year")[col]
            for t in origins:
                if t not in g.index or t - g.index.min() < 2:
                    continue
                hist = g.loc[:t].values
                fa = arima_forecast(hist, hmax)
                fd = drift_forecast(hist, hmax)
                for h in range(1, hmax + 1):
                    rows.append((ind, iid, t, h, fa[h - 1], fd[h - 1], hist[-1]))
    return pd.DataFrame(rows, columns=["indicator", "inst_id", "origin", "h", "arima", "drift", "z_origin"])


def make_pairs(fp: pd.DataFrame, ind: str, h: int, cache: pd.DataFrame) -> pd.DataFrame:
    col = "z_works" if ind == "works" else "z_oa"
    d = fp[["inst_id", "year"] + FEATURES].copy()
    tgt = fp[["inst_id", "year", col]].copy()
    tgt["year"] -= h
    d = d.merge(tgt.rename(columns={col: "z_target"}), on=["inst_id", "year"], how="left")
    c = cache[(cache.indicator == ind) & (cache.h == h)][["inst_id", "origin", "arima", "drift"]]
    d = d.merge(c.rename(columns={"origin": "year"}), on=["inst_id", "year"], how="left")
    d["z_origin"] = d[col]
    d["arima_delta"] = d["arima"] - d["z_origin"]
    return d.rename(columns={"year": "origin"})


def _fit_gbm(X, y):
    m = lgb.LGBMRegressor(**GBM_PARAMS)
    m.fit(X, y)
    return m


def _fit_mlp(X, y):
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-2, learning_rate_init=1e-3,
                                   max_iter=3000, early_stopping=True, random_state=42))
    m.fit(X, y)
    return m


def forecast_origin(pairs: pd.DataFrame, origin: int, h: int, return_models: bool = False):
    """Train global models on pairs whose target is observed by `origin`, predict at `origin`."""
    train = pairs[(pairs.origin + h <= origin) & pairs.z_target.notna() & pairs.g1_works.notna()]
    test = pairs[(pairs.origin == origin) & pairs.arima.notna()].copy()
    if test.empty or len(train) < 50:
        return (None, None) if return_models else None
    y = train.z_target - train.z_origin
    gbm = _fit_gbm(train[FEATURES], y)
    mlp = _fit_mlp(train[FEATURES], y)
    train_h = train[train.arima.notna()]
    hyb = _fit_gbm(train_h[FEATURES + ["arima_delta"]], train_h.z_target - train_h.arima)
    hyf = _fit_gbm(train_h[FEATURES + ["arima_delta"]], train_h.z_target - train_h.z_origin)

    out = test[["inst_id", "origin", "z_origin", "z_target"]].copy()
    out["naive"] = test.z_origin
    out["drift"] = test.drift
    out["arima"] = test.arima
    out["gbm"] = test.z_origin + gbm.predict(test[FEATURES])
    out["mlp"] = test.z_origin + mlp.predict(test[FEATURES])
    out["combo"] = 0.5 * (out["arima"] + out["gbm"])
    out["hybrid_res"] = test.arima + hyb.predict(test[FEATURES + ["arima_delta"]])
    out["hybrid_feat"] = test.z_origin + hyf.predict(test[FEATURES + ["arima_delta"]])
    out["hybrid_ens"] = (out["gbm"] + out["mlp"] + out["hybrid_feat"]) / 3
    out["h"] = h
    if return_models:
        return out, {"gbm": gbm, "mlp": mlp, "hybrid_res": hyb, "hybrid_feat": hyf, "train": train}
    return out


def run_backtest(fp: pd.DataFrame, origins, cache: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for ind in INDICATORS:
        for h in HORIZONS:
            pairs = make_pairs(fp, ind, h, cache)
            for t in origins:
                if t + h > LAST_YEAR:
                    continue
                f = forecast_origin(pairs, t, h)
                if f is not None:
                    f["indicator"] = ind
                    frames.append(f)
    wide = pd.concat(frames, ignore_index=True)
    long = wide.melt(id_vars=["indicator", "inst_id", "origin", "h", "z_origin", "z_target"],
                     value_vars=list(MODELS), var_name="model", value_name="z_hat")
    return long


# ------------------------------------------------------------------------ metrics
def add_scale(fc: pd.DataFrame, fp: pd.DataFrame) -> pd.DataFrame:
    """MASE scale: in-sample mean absolute one-step naive error up to the origin."""
    rows = []
    for ind in INDICATORS:
        for iid, g in fp.groupby("inst_id"):
            y = g.set_index("year")[ind]
            for t in fc.origin.unique():
                hist = y.loc[:t]
                if len(hist) >= 3:
                    rows.append((ind, iid, t, np.mean(np.abs(np.diff(hist.values)))))
    sc = pd.DataFrame(rows, columns=["indicator", "inst_id", "origin", "scale"])
    return fc.merge(sc, on=["indicator", "inst_id", "origin"], how="left")


def score(fc: pd.DataFrame) -> pd.DataFrame:
    d = fc.copy()
    d["y"] = [from_z(i, z) for i, z in zip(d.indicator, d.z_target)]
    d["y_hat"] = [from_z(i, z) for i, z in zip(d.indicator, d.z_hat)]
    d["y0"] = [from_z(i, z) for i, z in zip(d.indicator, d.z_origin)]
    d["err"] = d.y_hat - d.y
    d["ae"] = d.err.abs()
    d["ape"] = 100 * d.ae / d.y.abs().clip(lower=1e-9)
    floor = np.where(d.indicator == "works", 1.0, 0.01)  # 1 publication / 1 p.p. of OA share
    d["sae"] = d.ae / np.maximum(d.scale, floor)
    d["dir_ok"] = np.sign(d.y - d.y0) == np.sign(d.y_hat - d.y0)
    return d


def summarize(d: pd.DataFrame, by=("indicator", "model")) -> pd.DataFrame:
    g = d.groupby(list(by))
    return pd.DataFrame({
        "MAE": g.ae.mean(),
        "RMSE": g.err.apply(lambda e: np.sqrt(np.mean(e ** 2))),
        "MAPE": g.ape.mean(),
        "MdAPE": g.ape.median(),
        "MASE": g.sae.mean(),
        "DirAcc": 100 * g.dir_ok.mean(),
        "n": g.size(),
    }).reset_index()


# ---------------------------------------------------------- conformal intervals
def conformal_intervals(fc: pd.DataFrame, fp: pd.DataFrame, model: str = "hybrid",
                        alpha: float = 0.2, scaled: bool = True) -> pd.DataFrame:
    """Split-conformal intervals on the z-scale using only past residuals.

    scaled=True normalises residuals by each series' recent growth volatility
    (locally adaptive conformal prediction), so young, volatile institutions
    receive wider intervals than mature, stable ones.
    """
    d = fc[fc.model == model].copy()
    vol = fp[["inst_id", "year", "g_std5_works"]].rename(columns={"year": "origin", "g_std5_works": "vol"})
    d = d.merge(vol, on=["inst_id", "origin"], how="left")
    d["vol"] = d["vol"].fillna(d["vol"].median()).clip(lower=0.05)
    d["r"] = (d.z_target - d.z_hat).abs() / (d.vol if scaled else 1.0)
    lo, hi = [], []
    for _, row in d.iterrows():
        cal = d[(d.indicator == row.indicator) & (d.h == row.h) &
                (d.origin + d.h <= row.origin) & d.r.notna()].r.values
        if len(cal) < 20:
            lo.append(np.nan); hi.append(np.nan); continue
        n = len(cal)
        q = np.quantile(cal, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n))
        w = q * (row.vol if scaled else 1.0)
        lo.append(row.z_hat - w); hi.append(row.z_hat + w)
    d["z_lo"], d["z_hi"] = lo, hi
    return d


def interval_from_residuals(z_hat: float, vol: float, residuals: np.ndarray, alpha: float = 0.2):
    n = len(residuals)
    q = np.quantile(residuals, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n))
    w = q * max(vol, 0.05)
    return z_hat - w, z_hat + w
