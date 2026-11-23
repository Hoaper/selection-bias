# Look-ahead selection in backtests of university forecasts

Code for the paper *Does the choice of universities change how accurate their research forecasts look?* (M. Turbulyak, G. Soltan, Astana IT University), prepared for the Journal of Problems in Computer Science and Information Technologies (al-Farabi Kazakh National University).

Backtests of forecasting models are often run on institutions that still exist today, which uses information from the future. This project measures how much that choice changes the measured accuracy of models that forecast the research output and open-access share of universities.

## Data

The OpenAlex institutions API gives yearly counts of works and open-access works for every education institution with more than 300 indexed works in KZ, UZ, KG, TJ, TM, AZ, GE, AM and MN: 254 institutions, 2010–2025. World Bank WDI indicators supply the country context. The snapshot downloaded on 25 October 2026 is in `data/`. OpenAlex data are CC0, WDI data CC BY 4.0. `fetch_data.py` downloads fresh data, which will differ slightly because OpenAlex keeps indexing.

## Design

An institution is active in a year with at least 10 works; a continuous active period is a run (315 runs).

| Scenario | Who is forecast at origin T |
|---|---|
| look-ahead | the 202 runs that last to 2025, are at least six years long and have at least 30 works in 2025 (the panel of the original study) |
| real-time | every run that is active at T and has at least three active years, whatever happens to it later |

Both scenarios run the same nine methods (naive, drift, auto-ARIMA, global LightGBM and MLP, ARIMA+GBM average, hybrids H1–H3) in a rolling-origin backtest over origins 2019–2024, h = 1–3. The look-ahead scenario reproduces the results of the original study exactly.

## Usage

```bash
pip install -r requirements.txt
cd src
python selection_bias.py          # both scenarios, about 10 minutes -> data/results/selection_*.csv
python selection_analysis.py      # groups of dropped runs, paired tests, institution-level bootstrap
python make_figures_selection.py  # figures -> figures/selection/
```

## Layout

```
src/forecasting.py             data preparation, models, scoring
src/selection_bias.py          builds the runs and the two scenarios, runs the backtest
src/selection_analysis.py      breakdown of the difference between the scenarios
src/make_figures_selection.py  figures for the paper
src/fetch_data.py              OpenAlex and World Bank download
data/                          raw data snapshot (25 Oct 2026)
```

## Limitations

The real-time scenario still starts from institutions that had more than 300 OpenAlex works in October 2026, and OpenAlex re-indexes past years, so it is not completely free of future information.

## License

MIT, see `LICENSE`.
