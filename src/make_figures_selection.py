"""Figures for the selection-bias paper: figures/selection/*_ru.png (Russian labels) and *_en.png, 300 dpi."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import forecasting as F

RES = F.DATA / "results"
FIG = F.ROOT / "figures" / "selection"
FIG.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE, AQUA, GRAY1, GRAY2 = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#c9c8c1"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
    "axes.edgecolor": GRAY1, "axes.labelcolor": INK, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
    "legend.fontsize": 8, "lines.linewidth": 1.6, "savefig.dpi": 300, "savefig.bbox": "tight",
})
TEXT = {
    "ru": dict(models={"naive": "Наивный", "drift": "Дрейф", "combo": "ARIMA+GBM"},
               titles={"works": "Число публикаций", "oa_share": "Доля открытого доступа"},
               la="отбор по данным 2025 г.", rt="отбор на момент прогноза", mase="MASE (меньше – лучше)",
               cases={"I4210088031": "КГУСТА им. Н. Исанова (Кыргызстан)",
                      "I4405267534": "Ташкентский гос. медицинский ун-т (Узбекистан)"},
               thr="порог активности: 10 работ в год", ylog="Публикаций в год + 1 (лог. шкала)"),
    "en": dict(models={"naive": "Naive", "drift": "Drift", "combo": "ARIMA+GBM"},
               titles={"works": "Research output", "oa_share": "Open-access share"},
               la="selected with 2025 data", rt="selected at the forecast origin", mase="MASE (lower is better)",
               cases={"I4210088031": "Kyrgyz State Univ. of Construction, Transport and Architecture",
                      "I4405267534": "Tashkent State Medical University"},
               thr="activity threshold: 10 works a year", ylog="Works per year + 1 (log scale)"),
}
ORDER = ["naive", "drift", "arima", "mlp", "gbm", "combo", "hybrid_res", "hybrid_feat", "hybrid"]
SHORT = {"arima": "ARIMA", "mlp": "MLP", "gbm": "LightGBM", "hybrid_res": "H1", "hybrid_feat": "H2", "hybrid": "H3"}

ov = pd.read_csv(RES / "selection_overall.csv")
raw = pd.read_csv(F.DATA / "panel_raw.csv")

for lang, T in TEXT.items():
    names = {**SHORT, **T["models"]}
    # 1. MASE of every model under the two selection rules
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for ax, ind in zip(axes, ("works", "oa_share")):
        o = ov[ov.indicator == ind].pivot_table(index="model", columns="scenario", values="MASE").loc[ORDER]
        pos = np.arange(len(ORDER))
        ax.bar(pos - 0.2, o.lookahead, 0.4, color=GRAY2, label=T["la"])
        ax.bar(pos + 0.2, o.realtime, 0.4, color=BLUE, label=T["rt"])
        ax.set_xticks(pos)
        ax.set_xticklabels([names[m] for m in ORDER], rotation=45, ha="right")
        ax.set_ylabel(T["mase"])
        ax.set_title(T["titles"][ind])
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / f"fig1_mase_{lang}.png")
    plt.close(fig)

    # 2. the two runs that ended before 2025 and appear at the test origins
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    for (iid, name), col in zip(T["cases"].items(), (ORANGE, AQUA)):
        s = raw[(raw.inst_id == iid) & raw.year.between(2014, 2025)].groupby("year").works.sum()
        s = s.reindex(range(2014, 2026), fill_value=0)
        ax.plot(s.index, s.values + 1, color=col, marker="o", ms=3, label=name)
    ax.axhline(11, color=GRAY1, ls="--", lw=1)
    ax.text(2014.1, 13, T["thr"], color=MUTED, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel(T["ylog"])
    ax.set_xticks(range(2014, 2026, 2))
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / f"fig2_dropped_{lang}.png")
    plt.close(fig)
print("saved", sorted(p.name for p in FIG.glob("*.png")))
