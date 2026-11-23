"""Collect open data for the study.

1. OpenAlex: annual research indicators (works, open-access works, citations)
   for all education institutions in Central Asia, the South Caucasus and
   Mongolia with more than 300 indexed works.
2. World Bank WDI: country-level higher-education and R&D context indicators.

Output: data/raw/*.json and data/panel_raw.csv, data/wdi.csv
"""
import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

COUNTRIES = ["KZ", "UZ", "KG", "TJ", "TM", "AZ", "GE", "AM", "MN"]
OA_URL = "https://api.openalex.org/institutions"
FIELDS = "id,display_name,country_code,type,works_count,cited_by_count,counts_by_year,summary_stats"

WDI = {
    "SE.TER.ENRR": "tertiary_enrolment_gross_pct",
    "GB.XPD.RSDV.GD.ZS": "rd_expenditure_pct_gdp",
    "SP.POP.SCIE.RD.P6": "researchers_per_million",
    "NY.GDP.PCAP.PP.KD": "gdp_pc_ppp",
    "SE.XPD.TERT.ZS": "tertiary_share_of_edu_spending",
}


def fetch_institutions():
    out, page = [], 1
    while True:
        params = {
            "filter": f"country_code:{'|'.join(COUNTRIES)},type:education,works_count:>300",
            "per_page": 200,
            "page": page,
            "select": FIELDS,
        }
        r = requests.get(OA_URL, params=params, timeout=60)
        r.raise_for_status()
        d = r.json()
        out.extend(d["results"])
        if len(out) >= d["meta"]["count"] or not d["results"]:
            break
        page += 1
        time.sleep(1)
    (RAW / "openalex_institutions.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    rows = []
    for inst in out:
        for c in inst["counts_by_year"]:
            rows.append({
                "inst_id": inst["id"].rsplit("/", 1)[-1],
                "name": inst["display_name"],
                "country": inst["country_code"],
                "year": c["year"],
                "works": c["works_count"],
                "oa_works": c["oa_works_count"],
                "citations": c["cited_by_count"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data" / "panel_raw.csv", index=False)
    print(f"OpenAlex: {len(out)} institutions, {len(df)} institution-years")


def fetch_wdi():
    frames = []
    for code, name in WDI.items():
        url = f"https://api.worldbank.org/v2/country/{';'.join(COUNTRIES)}/indicator/{code}"
        r = requests.get(url, params={"format": "json", "per_page": 2000, "date": "2005:2025"}, timeout=60)
        r.raise_for_status()
        payload = r.json()
        (RAW / f"wdi_{code}.json").write_text(json.dumps(payload), encoding="utf-8")
        for rec in payload[1] or []:
            frames.append({"country": rec["country"]["id"], "year": int(rec["date"]),
                           "indicator": name, "value": rec["value"]})
    wdi = pd.DataFrame(frames).pivot_table(index=["country", "year"], columns="indicator", values="value").reset_index()
    wdi.to_csv(ROOT / "data" / "wdi.csv", index=False)
    print(f"WDI: {wdi.shape}")


if __name__ == "__main__":
    fetch_institutions()
    fetch_wdi()
