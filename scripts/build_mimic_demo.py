#!/usr/bin/env python
"""Build EHRAgent's MIMIC-III tables from the open-access MIMIC-III Demo v1.4."""

from __future__ import annotations

import random
import shutil
import sqlite3
import sys
from pathlib import Path

import pandas as pd

COLS = {
    "PATIENTS": ["subject_id", "gender", "dob", "dod"],
    "ADMISSIONS": [
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "admission_type",
        "admission_location",
        "discharge_location",
        "insurance",
        "language",
        "marital_status",
        "ethnicity",
    ],
    "CHARTEVENTS": [
        "subject_id",
        "hadm_id",
        "icustay_id",
        "itemid",
        "charttime",
        "valuenum",
        "valueuom",
    ],
    "D_ICD_DIAGNOSES": ["icd9_code", "short_title", "long_title"],
    "D_ICD_PROCEDURES": ["icd9_code", "short_title", "long_title"],
    "D_ITEMS": ["itemid", "label", "linksto"],
    "D_LABITEMS": ["itemid", "label"],
    "ICUSTAYS": [
        "subject_id",
        "hadm_id",
        "icustay_id",
        "first_careunit",
        "last_careunit",
        "first_wardid",
        "last_wardid",
        "intime",
        "outtime",
    ],
    "INPUTEVENTS_CV": ["subject_id", "hadm_id", "icustay_id", "charttime", "itemid", "amount"],
    "LABEVENTS": ["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom"],
    "MICROBIOLOGYEVENTS": ["subject_id", "hadm_id", "charttime", "spec_type_desc", "org_name"],
    "OUTPUTEVENTS": ["subject_id", "hadm_id", "icustay_id", "charttime", "itemid", "value"],
    "PRESCRIPTIONS": [
        "subject_id",
        "hadm_id",
        "startdate",
        "enddate",
        "drug",
        "dose_val_rx",
        "dose_unit_rx",
        "route",
    ],
    "TRANSFERS": [
        "subject_id",
        "hadm_id",
        "icustay_id",
        "eventtype",
        "curr_careunit",
        "curr_wardid",
        "intime",
        "outtime",
    ],
}
COST_RANGE = {
    "diagnoses_icd": (1.5, 10.0),
    "procedures_icd": (8.0, 15.0),
    "labevents": (5.5, 11.0),
    "prescriptions": (8.0, 14.0),
}


def read(src: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(src / f"{name}.csv", low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    return df


def lower(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].map(lambda v: v.lower() if isinstance(v, str) else v)
    return df


def finish(df: pd.DataFrame) -> pd.DataFrame:
    df = lower(df.reset_index(drop=True))
    df.insert(0, "row_id", range(len(df)))
    df.columns = [c.upper() for c in df.columns]
    return df


def build(src: Path, out: Path, minja_dir: Path, seed: int = 0) -> dict[str, pd.DataFrame]:
    t: dict[str, pd.DataFrame] = {}
    for name, cols in COLS.items():
        t[name] = read(src, name)[cols]
    t["TRANSFERS"] = t["TRANSFERS"].rename(
        columns={"curr_careunit": "careunit", "curr_wardid": "wardid"}
    )
    adm = t["ADMISSIONS"]
    dob = read(src, "PATIENTS").set_index("subject_id")["dob"]
    adm["age"] = (
        pd.to_datetime(adm["admittime"]).dt.year
        - pd.to_datetime(adm["subject_id"].map(dob)).dt.year
    ).clip(upper=89)
    admit = adm.set_index("hadm_id")["admittime"]
    for name in ("DIAGNOSES_ICD", "PROCEDURES_ICD"):
        d = read(src, name)[["subject_id", "hadm_id", "icd9_code"]]
        d["charttime"] = d["hadm_id"].map(admit)
        t[name] = d.dropna(subset=["charttime"])
    rng = random.Random(seed)
    cost = []
    for etype, table, time_col in (
        ("diagnoses_icd", "DIAGNOSES_ICD", "charttime"),
        ("procedures_icd", "PROCEDURES_ICD", "charttime"),
        ("labevents", "LABEVENTS", "charttime"),
        ("prescriptions", "PRESCRIPTIONS", "startdate"),
    ):
        lo, hi = COST_RANGE[etype]
        for i, r in enumerate(t[table].itertuples(index=False)):
            if pd.isna(r.hadm_id):
                continue
            cost.append(
                (
                    r.subject_id,
                    int(r.hadm_id),
                    etype,
                    i,
                    getattr(r, time_col),
                    round(rng.uniform(lo, hi), 2),
                )
            )
    t["COST"] = pd.DataFrame(
        cost, columns=["subject_id", "hadm_id", "event_type", "event_id", "chargetime", "cost"]
    )
    t = {k: finish(v) for k, v in t.items()}

    out.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(out / "mimic_iii.db")
    for name, df in t.items():
        df.to_csv(out / f"{name}.csv", index=False)
        df.to_sql(name, con, if_exists="replace", index=False)
    con.close()
    for q in ("patient_only.json", "valid_preprocessed.json", "valid.json", "train.json"):
        if (minja_dir / q).exists():
            shutil.copy(minja_dir / q, out / q)
    return t


def main(argv: list[str]) -> int:
    src, out = Path(argv[1]), Path(argv[2])
    minja = (
        Path(argv[3])
        if len(argv) > 3
        else Path("third_party/MINJA/EHR/ehragent/ehrsql-ehragent/mimic_iii")
    )
    t = build(src, out, minja)
    pat = t["PATIENTS"]
    busy = t["ADMISSIONS"].groupby("SUBJECT_ID").size().sort_values(ascending=False)
    print(f"{len(pat)} patients, {len(t['ADMISSIONS'])} admissions -> {out}")
    print(
        "patients with the most admissions (pick VICTIM and TARGET from these):",
        list(busy.index[:6]),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
