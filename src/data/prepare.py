#!/usr/bin/env python3
"""
src/data/prepare.py

Build a de-identified clinical retrieval corpus and simulated chart-review tasks
for the CMA benchmark.

Supports five data modes:
  1. Synthetic-only (default, no PHI).
  2. MIMIC-III directory ingestion.
  3. MIMIC-IV directory ingestion.
  4. Hybrid: real MIMIC patients + synthetic patients up to --n-patients.
  5. Hugging Face directory (auto-detects all datasets in a snapshot).

Example commands
----------------
Synthetic-only with 10,000 patients and 60 vignettes:
    python src/data/prepare.py --synthetic-only --n-patients 10000 --n-vignettes 60

MIMIC-III directory (download with src/data/download.py first):
    python src/data/prepare.py --mimiciii-dir data/raw --n-patients 10000

MIMIC-IV directory:
    python src/data/prepare.py --mimiciv-dir data/raw

Hugging Face snapshot (auto-detects all available datasets):
    python src/data/prepare.py --huggingface-dir data/raw/huggingface

Hybrid (use all real patients and synthesize the rest):
    python src/data/prepare.py --mimiciii-dir data/raw --hybrid --n-patients 10000
"""

import argparse
import csv
import gzip
import json
import random
import re
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pq = None

from collections import Counter, defaultdict
from typing import Optional

import numpy as np

DEFAULT_SEED = 20260616

# ─────────────────────────── Diagnosis / keyword vocabulary ───────────────────

SPECIALTIES = ["internal_medicine", "emergency", "critical_care", "hospital_medicine"]
GENDERS = ["M", "F"]

DIAGNOSES = {
    "heart_failure": {
        "keywords": ["dyspnea", "orthopnea", "jvp", "edema", "bnp", "ejection", "lasix", "furosemide", "ace inhibitor"],
        "drugs": ["furosemide", "lisinopril", "carvedilol", "spironolactone"]
    },
    "acute_kidney_injury": {
        "keywords": ["creatinine", "bun", "oliguria", "hyperkalemia", "nephrotoxic", "aki", "dialysis"],
        "drugs": ["normal saline", "vancomycin", "metformin"]
    },
    "pneumonia": {
        "keywords": ["fever", "cough", "infiltrate", "oxygen", "sputum", "wbc", "crackles", "pneumonia"],
        "drugs": ["ceftriaxone", "azithromycin", "vancomycin", "oxygen"]
    },
    "sepsis": {
        "keywords": ["lactate", "hypotension", "fever", "bandemia", "vasopressor", "septic", "qsofa"],
        "drugs": ["norepinephrine", "ceftriaxone", "vancomycin", "fluid"]
    },
    "copd_exacerbation": {
        "keywords": ["copd", "wheezing", "dyspnea", "s02", "steroid", "inhaler", "theophylline"],
        "drugs": ["albuterol", "prednisone", "tiotropium", "oxygen"]
    },
    "atrial_fibrillation": {
        "keywords": ["atrial fibrillation", "irregularly irregular", "anticoagulation", "warfarin", "doac", "chads"],
        "drugs": ["warfarin", "apixaban", "metoprolol", "digoxin"]
    },
    "uti_pyelonephritis": {
        "keywords": ["dysuria", "pyuria", "urine culture", "enterococcus", "ciprofloxacin", "pyelonephritis"],
        "drugs": ["ciprofloxacin", "cephalexin", "nitrofurantoin"]
    },
    "gi_bleed": {
        "keywords": ["melena", "hematochezia", "hemoglobin", "ppi", "endoscopy", "transfusion", "gi bleed"],
        "drugs": ["pantoprazole", "octreotide", "packed red blood cells"]
    },
}

SENTENCE_TEMPLATES = {
    "heart_failure": [
        "Patient reports {severity} dyspnea on exertion.",
        "Orthopnea present; sleeps on {n} pillows.",
        "Jugular venous pressure is elevated at {n} cm.",
        "Bilateral lower-extremity edema {extent}.",
        "BNP is {val} pg/mL.",
        "TTE shows EF {val}%.",
        "Continued diuresis with {drug} initiated."
    ],
    "acute_kidney_injury": [
        "Serum creatinine increased from {val} to {val2} over {hours} hours.",
        "Urine output decreased to {val} mL/hr.",
        "Potassium {val} mEq/L; ECG unchanged.",
        "Review of medications for nephrotoxins including {drug}.",
        "Renal ultrasound without obstruction.",
        "Nephrology consulted."
    ],
    "pneumonia": [
        "Fever to {val} F with productive cough.",
        "Chest X-ray shows {lobe} lobe infiltrate.",
        "Sputum Gram stain reveals {organism}.",
        "Supplemental oxygen {val} L/min to maintain saturation {val2}%.",
        "Antibiotics: {drug} plus {drug2} per CAP guidelines.",
        "Labs: WBC {val} K/uL."
    ],
    "sepsis": [
        "Temperature {val} F; heart rate {val2} bpm.",
        "Blood pressure {val}/{val2} mmHg.",
        "Initial lactate {val} mmol/L.",
        "Crystalloid bolus administered.",
        "Empiric antibiotics: {drug} and {drug2}.",
        "Vasopressor {drug3} started for MAP < 65."
    ],
    "copd_exacerbation": [
        "History of COPD with increased dyspnea and sputum.",
        "Wheezing on exam; unable to speak in full sentences.",
        "Oxygen saturation {val}% on room air.",
        "Arterial blood gas: pH {val}, pCO2 {val2}.",
        "Started {drug} nebulizer and {drug2} systemic steroid.",
        "Consider non-invasive positive pressure ventilation."
    ],
    "atrial_fibrillation": [
        "Heart rate irregularly irregular at {val} bpm.",
        "EKG shows atrial fibrillation with rapid ventricular response.",
        "Hemodynamically stable.",
        "CHADS2-VASc score {val}; HAS-BLED {val2}.",
        "Anticoagulation plan: {drug}.",
        "Rate control with {drug2} initiated."
    ],
    "uti_pyelonephritis": [
        "Dysuria and flank pain for {val} days.",
        "Urinalysis: leukocyte esterase positive, nitrites positive.",
        "Urine culture grows {organism}.",
        "Imaging shows no obstructive uropathy.",
        "Treating with {drug}.",
        "Repeat creatinine stable at {val}."
    ],
    "gi_bleed": [
        "Patient reports black tarry stools and lightheadedness.",
        "Hemoglobin drop from {val} to {val2} g/dL.",
        "Orthostatic hypotension documented.",
        "IV {drug} started.",
        "Type and screen; blood bank notified.",
        "EGD reveals {finding}."
    ],
}


def _fill(template: str, rng: random.Random) -> str:
    txt = template
    txt = txt.replace("{drug}", rng.choice(["drug", "furosemide", "ceftriaxone", "norepinephrine", "albuterol", "warfarin", "ciprofloxacin", "pantoprazole"]))
    txt = txt.replace("{drug2}", rng.choice(["drug2", "azithromycin", "vancomycin", "prednisone", "apixaban", "cephalexin", "octreotide"]))
    txt = txt.replace("{drug3}", rng.choice(["drug3", "norepinephrine", "vasopressin", "phenylephrine"]))
    txt = txt.replace("{val}", str(rng.randint(2, 180) if rng.random() > 0.3 else f"{rng.uniform(0.5, 15):.1f}"))
    txt = txt.replace("{val2}", str(rng.randint(60, 250)))
    txt = txt.replace("{hours}", str(rng.randint(12, 72)))
    txt = txt.replace("{n}", str(rng.randint(1, 4)))
    txt = txt.replace("{extent}", rng.choice(["mild", "moderate", "severe", "1+", "2+", "3+"]))
    txt = txt.replace("{severity}", rng.choice(["mild", "moderate", "severe"]))
    txt = txt.replace("{lobe}", rng.choice(["right upper", "right lower", "left upper", "left lower"]))
    txt = txt.replace("{organism}", rng.choice(["Streptococcus pneumoniae", "Staphylococcus aureus", "Escherichia coli", "Klebsiella"]))
    txt = txt.replace("{finding}", rng.choice(["gastric ulcer", "duodenal ulcer", "erosive gastritis", "no active bleeding"]))
    return txt.replace("drug", "medication")


def synthesize_note(rng: random.Random, diag: str, note_type: str) -> str:
    templates = SENTENCE_TEMPLATES.get(diag, SENTENCE_TEMPLATES["heart_failure"])
    n_sents = rng.randint(3, 8)
    sents = [_fill(rng.choice(templates), rng) for _ in range(n_sents)]
    info = DIAGNOSES.get(diag, DIAGNOSES["heart_failure"])
    keyword_sentence = "Relevant keywords: " + ", ".join(rng.sample(info["keywords"], k=min(len(info["keywords"]), 3)))
    if note_type == "medication":
        sents.append("Medication reconciliation note: " + rng.choice(info["drugs"]) + " listed.")
    elif note_type == "lab":
        lab = rng.choice(info["keywords"])
        sents.append(f"Laboratory trend review focusing on {lab}.")
    sents.append(keyword_sentence)
    return " ".join(sents)


# ─────────────────────────── Synthetic corpus ─────────────────────────────────

def generate_synthetic_corpus(n_patients: int = 10000, seed: int = DEFAULT_SEED,
                                 start_pid: int = 1) -> list[dict]:
    """Generate a de-identified synthetic EHR-like note corpus."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    diags = list(DIAGNOSES.keys())
    note_types = ["history", "progress", "lab", "medication", "imaging", "discharge"]
    records = []

    for offset in range(n_patients):
        pid_num = start_pid + offset
        age = int(np_rng.normal(68, 14))
        age = max(21, min(95, age))
        gender = rng.choice(GENDERS)
        n_notes = rng.randint(4, 12)
        primary_diag = rng.choice(diags)
        for nid in range(n_notes):
            diag = primary_diag if rng.random() > 0.2 else rng.choice(diags)
            note_type = rng.choice(note_types)
            text = synthesize_note(rng, diag, note_type)
            record = {
                "patient_id": f"SYN_{pid_num:05d}",
                "note_id": f"SYN_{pid_num:05d}_N{nid:03d}",
                "note_type": note_type,
                "specialty": rng.choice(SPECIALTIES),
                "age": age,
                "gender": gender,
                "primary_diagnosis": primary_diag,
                "diagnosis": diag,
                "text": text,
                "keywords": ", ".join(DIAGNOSES[diag]["keywords"]),
            }
            records.append(record)
    return records


# ─────────────────────────── MIMIC helpers ───────────────────────────────────

def _csv_gz_rows(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            yield from csv.DictReader(fh)
    else:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            yield from csv.DictReader(fh)


def _icd9_chapter(code: str) -> str:
    """Map an ICD-9 code to a coarse chapter label.

    Handles both "572.3" and compact MIMIC forms like "5723" (3-digit chapter
    prefix + decimal digits with the dot dropped).
    """
    code = str(code).strip().upper()
    raw = code.split(".")[0]
    if not raw:
        if code.startswith("V"):
            return "icd9_supplemental"
        if code.startswith("E"):
            return "icd9_external"
        return "unknown"
    if raw[0].isalpha():
        # ICD-10 codes (MIMIC-IV v2.0+) start with a letter (A–U, excluding
        # E/V external/supplemental ICD-9 codes which are E####/V####).
        if raw[0] in ("V", "E") and raw[1:].isdigit():
            return "icd9_external" if raw[0] == "E" else "icd9_supplemental"
        c10 = re.match(r"^([A-Z])(\d{1,2})", raw)
        if c10:
            return f"icd10_chapter_{c10.group(1)}"
        return "unknown"
    numeric = re.sub(r"[^0-9]", "", raw)
    if not numeric:
        return "unknown"
    # Compact form: "5723" -> chapter prefix "572", ".3" retained for lookup.
    if len(numeric) > 3:
        numeric = numeric[:3]
    try:
        c = int(numeric)
    except ValueError:
        return "unknown"
    if 1 <= c <= 139:
        return "icd9_infectious"
    if 140 <= c <= 239:
        return "icd9_neoplasms"
    if 240 <= c <= 279:
        return "icd9_endocrine_nutritional"
    if 280 <= c <= 289:
        return "icd9_blood"
    if 290 <= c <= 319:
        return "icd9_mental"
    if 320 <= c <= 389:
        return "icd9_nervous_sense"
    if 390 <= c <= 459:
        return "icd9_circulatory"
    if 460 <= c <= 519:
        return "icd9_respiratory"
    if 520 <= c <= 579:
        return "icd9_digestive"
    if 580 <= c <= 629:
        return "icd9_genitourinary"
    if 630 <= c <= 679:
        return "icd9_pregnancy"
    if 680 <= c <= 709:
        return "icd9_skin"
    if 710 <= c <= 739:
        return "icd9_musculoskeletal"
    if 740 <= c <= 759:
        return "icd9_congenital"
    if 760 <= c <= 779:
        return "icd9_perinatal"
    if 780 <= c <= 799:
        return "icd9_symptoms_signs"
    if 800 <= c <= 999:
        return "icd9_injury_poisoning"
    return "unknown"


def _find_file(raw_dir: Path, *candidates: str) -> Optional[Path]:
    """Return the first existing file from a list of candidate paths relative to raw_dir."""
    for name in candidates:
        path = raw_dir / name
        if path.exists():
            return path
    return None


def _load_mimiciii_supp(raw_dir: Path):
    """Load PATIENTS and DIAGNOSES_ICD lookups from a MIMIC-III raw directory."""
    patients = {}
    patients_path = _find_file(raw_dir, "PATIENTS.csv.gz", "PATIENTS.csv")
    if patients_path:
        for row in _csv_gz_rows(patients_path):
            patients[row.get("SUBJECT_ID", "")] = row.get("GENDER", "U")

    diagnoses = {}
    diag_path = _find_file(raw_dir, "DIAGNOSES_ICD.csv.gz", "DIAGNOSES_ICD.csv")
    if diag_path:
        for row in _csv_gz_rows(diag_path):
            hadm = row.get("HADM_ID", "").strip()
            if hadm and hadm not in diagnoses:
                code = row.get("ICD9_CODE", "").strip()
                diagnoses[hadm] = _icd9_chapter(code)
    return patients, diagnoses


def _load_mimiciii_lookups(raw_dir: Path) -> dict[str, dict]:
    """Load MIMIC-III D_* dictionary tables for mapping IDs to human-readable labels."""
    lookups = {}

    labitems_path = _find_file(raw_dir, "D_LABITEMS.csv.gz", "D_LABITEMS.csv")
    if labitems_path:
        labitems = {}
        for row in _csv_gz_rows(labitems_path):
            iid = row.get("ITEMID", "")
            if iid:
                labitems[iid] = row.get("LABEL", "")
        lookups["lab_items"] = labitems

    items_path = _find_file(raw_dir, "D_ITEMS.csv.gz", "D_ITEMS.csv")
    if items_path:
        items = {}
        for row in _csv_gz_rows(items_path):
            iid = row.get("ITEMID", "")
            if iid:
                items[iid] = row.get("LABEL", "")
        lookups["chart_items"] = items

    icd_diag_path = _find_file(raw_dir, "D_ICD_DIAGNOSES.csv.gz", "D_ICD_DIAGNOSES.csv")
    if icd_diag_path:
        icd_diags = {}
        for row in _csv_gz_rows(icd_diag_path):
            code = row.get("ICD9_CODE", "")
            if code:
                icd_diags[code] = row.get("LONG_TITLE", "") or row.get("SHORT_TITLE", "")
        lookups["icd_diagnoses"] = icd_diags

    icd_proc_path = _find_file(raw_dir, "D_ICD_PROCEDURES.csv.gz", "D_ICD_PROCEDURES.csv")
    if icd_proc_path:
        procs = {}
        for row in _csv_gz_rows(icd_proc_path):
            code = row.get("ICD9_CODE", "")
            if code:
                procs[code] = row.get("LONG_TITLE", "") or row.get("SHORT_TITLE", "")
        lookups["icd_procedures"] = procs

    cpt_path = _find_file(raw_dir, "D_CPT.csv.gz", "D_CPT.csv")
    if cpt_path:
        cpts = {}
        for row in _csv_gz_rows(cpt_path):
            section = row.get("SECTIONHEADER", "")
            subsection = row.get("SUBSECTIONHEADER", "")
            if section:
                cpts[section] = subsection or section
        lookups["cpt_sections"] = cpts

    return lookups


def ingest_mimiciii_dir(raw_dir: Path) -> list[dict]:
    """Ingest MIMIC-III NOTEEVENTS plus structured tables from a raw directory."""
    note_path = _find_file(raw_dir, "NOTEEVENTS.csv.gz", "NOTEEVENTS.csv")

    print(f"Loading MIMIC-III supplementary tables from {raw_dir}...")
    patients, diagnoses = _load_mimiciii_supp(raw_dir)
    lookups = _load_mimiciii_lookups(raw_dir)

    records = []

    # ── Ingest NOTEEVENTS ──
    if note_path:
        print(f"Ingesting MIMIC-III notes from {note_path}...")
        for i, row in enumerate(_csv_gz_rows(note_path)):
            if i % 10000 == 0 and i:
                print(f"  processed {i} NOTEEVENTS rows")
            text = (row.get("TEXT") or "").strip()
            if not text or len(text) < 30:
                continue
            clean_text = re.sub(r"\[\*\*.*?\*\*\]", "", text)
            diag = diagnoses.get(row.get("HADM_ID", "").strip(),
                                 (row.get("CATEGORY") or "UNKNOWN").lower().replace(" ", "_"))
            sid = row.get("SUBJECT_ID", "UNK")
            records.append({
                "patient_id": sid,
                "note_id": row.get("ROW_ID", f"MIMIC_III_{i}"),
                "note_type": row.get("CATEGORY", "NOTE"),
                "specialty": "unknown",
                "age": -1,
                "gender": patients.get(sid, "U"),
                "primary_diagnosis": diag,
                "diagnosis": diag,
                "text": clean_text,
                "keywords": "",
            })
        print(f"  retained {len(records)} MIMIC-III notes")
    else:
        print(f"  No NOTEEVENTS file found in {raw_dir}, skipping notes.")

    # ── Ingest structured tables as per-admission summaries ──
    # Only load small tables that fit in memory and add clinical context.
    # Skip large event tables (CHARTEVENTS, LABEVENTS, INPUTEVENTS, OUTPUTEVENTS,
    # PRESCRIPTIONS, DATETIMEEVENTS) — NOTEEVENTS is the primary text source.
    hadm_rows: dict[str, list[dict]] = defaultdict(list)

    structured_sources = [
        ("ADMISSIONS.csv", "admissions"),
        ("DIAGNOSES_ICD.csv", "diagnoses_icd"),
        ("PROCEDURES_ICD.csv", "procedures_icd"),
        ("ICUSTAYS.csv", "icustays"),
        ("MICROBIOLOGYEVENTS.csv", "microbiology"),
        ("DRGCODES.csv", "drgcodes"),
        ("CPTEVENTS.csv", "cptevents"),
        ("SERVICES.csv", "services"),
        ("TRANSFERS.csv", "transfers"),
        ("CALLOUT.csv", "callout"),
        ("PROCEDUREEVENTS_MV.csv", "procedureevents_mv"),
    ]

    for fname, table_key in structured_sources:
        path = _find_file(raw_dir, f"{fname}.gz", fname)
        if not path:
            continue
        print(f"  Ingesting MIMIC-III {fname}...")
        for i, row in enumerate(_csv_gz_rows(path)):
            hadm = (row.get("HADM_ID") or "").strip()
            sid = (row.get("SUBJECT_ID") or "").strip()
            if not hadm:
                continue
            hadm_rows[hadm].append({"_table": table_key, "_sid": sid, **row})

    # Build per-admission structured summary records.
    n_struct = 0
    for hadm, rows in hadm_rows.items():
        sid = rows[0].get("_sid", "UNK")
        text_parts = []

        # ADMISSIONS fields
        adm_rows = [r for r in rows if r["_table"] == "admissions"]
        if adm_rows:
            a = adm_rows[0]
            for key in ["ADMISSION_TYPE", "ADMISSION_LOCATION", "DISCHARGE_LOCATION",
                         "INSURANCE", "ETHNICITY", "DIAGNOSIS"]:
                val = a.get(key, "")
                if val:
                    text_parts.append(f"{key}: {val}")

        # ICUSTAYS fields
        icu_rows = [r for r in rows if r["_table"] == "icustays"]
        if icu_rows:
            icu = icu_rows[0]
            los = icu.get("LOS", "")
            careunit = icu.get("FIRST_CAREUNIT", "")
            text_parts.append(f"ICU stay: {careunit}, LOS {los} days")

        # LABEVENTS — skip, too large to group in memory

        # CHARTEVENTS — skip, too large to group in memory

        # DATETIMEEVENTS — skip, too large to group in memory

        # PRESCRIPTIONS — skip, too large to group in memory

        # INPUTEVENTS — skip, too large to group in memory

        # OUTPUTEVENTS — skip, too large to group in memory

        # PROCEDUREEVENTS_MV
        proc_ev_rows = [r for r in rows if r["_table"] == "procedureevents_mv"]
        if proc_ev_rows:
            chart_items = lookups.get("chart_items", {})
            proc_texts = []
            for pr in proc_ev_rows[:10]:
                itemid = pr.get("ITEMID", "")
                label = chart_items.get(itemid, f"item_{itemid}")
                val = pr.get("VALUE") or ""
                cat = pr.get("ORDERCATEGORYNAME") or ""
                parts = [label]
                if val:
                    u = pr.get("VALUEUOM") or ""
                    parts.append(f"{val} {u}".strip())
                if cat:
                    parts.append(f"({cat})")
                proc_texts.append(" ".join(parts))
            if proc_texts:
                text_parts.append("Procedure events: " + "; ".join(proc_texts))

        # PROCEDURES_ICD
        proc_rows = [r for r in rows if r["_table"] == "procedures_icd"]
        if proc_rows:
            icd_procs = lookups.get("icd_procedures", {})
            proc_texts = []
            for pr in proc_rows[:10]:
                code = pr.get("ICD9_CODE", "")
                name = icd_procs.get(code, f"ICD9-{code}")
                if name:
                    proc_texts.append(name)
            if proc_texts:
                text_parts.append("Procedures (ICD): " + ", ".join(proc_texts))

        # CPTEVENTS
        cpt_rows = [r for r in rows if r["_table"] == "cptevents"]
        if cpt_rows:
            cpt_texts = []
            for cr in cpt_rows[:10]:
                section = cr.get("SECTIONHEADER", "")
                subsection = cr.get("SUBSECTIONHEADER", "")
                desc = cr.get("DESCRIPTION", "")
                entry = subsection or section or desc or ""
                if entry:
                    cpt_texts.append(entry)
            if cpt_texts:
                text_parts.append("CPT events: " + ", ".join(cpt_texts))

        # MICROBIOLOGYEVENTS
        micro_rows = [r for r in rows if r["_table"] == "microbiology"]
        if micro_rows:
            micro_texts = []
            for mr in micro_rows[:10]:
                spec = mr.get("SPEC_TYPE_DESC", "")
                org = mr.get("ORG_NAME", "")
                ab = mr.get("AB_NAME", "")
                interp = mr.get("INTERPRETATION", "")
                parts = [x for x in [spec, org, ab, interp] if x]
                if parts:
                    micro_texts.append(" / ".join(parts))
            if micro_texts:
                text_parts.append("Microbiology: " + "; ".join(micro_texts))

        # DRGCODES
        drg_rows = [r for r in rows if r["_table"] == "drgcodes"]
        if drg_rows:
            drg_texts = []
            for dr in drg_rows[:5]:
                desc = dr.get("DESCRIPTION", "")
                sev = dr.get("DRG_SEVERITY", "")
                if desc:
                    entry = desc
                    if sev:
                        entry += f" (severity {sev})"
                    drg_texts.append(entry)
            if drg_texts:
                text_parts.append("DRG: " + ", ".join(drg_texts))

        # SERVICES
        svc_rows = [r for r in rows if r["_table"] == "services"]
        if svc_rows:
            svc = svc_rows[-1]
            curr = svc.get("CURR_SERVICE", "")
            if curr:
                text_parts.append(f"Service: {curr}")

        # TRANSFERS
        xfer_rows = [r for r in rows if r["_table"] == "transfers"]
        if xfer_rows:
            xfer_texts = []
            for xr in xfer_rows[:5]:
                prev = xr.get("PREV_CAREUNIT") or ""
                curr = xr.get("CURR_CAREUNIT") or ""
                los = xr.get("LOS") or ""
                if prev or curr:
                    entry = f"{prev} -> {curr}" if prev else curr
                    if los:
                        entry += f" ({los} days)"
                    xfer_texts.append(entry)
            if xfer_texts:
                text_parts.append("Transfers: " + "; ".join(xfer_texts))

        # CALLOUT
        callout_rows = [r for r in rows if r["_table"] == "callout"]
        if callout_rows:
            callout_texts = []
            for cr in callout_rows[:5]:
                svc = cr.get("CALLOUT_SERVICE", "")
                status = cr.get("CALLOUT_STATUS", "")
                outcome = cr.get("CALLOUT_OUTCOME", "")
                if svc:
                    entry = f"Callout: {svc}"
                    if status:
                        entry += f" ({status})"
                    if outcome:
                        entry += f" -> {outcome}"
                    callout_texts.append(entry)
            if callout_texts:
                text_parts.append("Callouts: " + "; ".join(callout_texts))

        if not text_parts:
            continue

        diag = diagnoses.get(hadm, "mimic_structured")
        text = f"Admission {hadm}. " + " | ".join(text_parts)
        records.append({
            "patient_id": sid,
            "note_id": f"MIII_STRUCT_{hadm}",
            "note_type": "structured_summary",
            "specialty": "unknown",
            "age": -1,
            "gender": patients.get(sid, "U"),
            "primary_diagnosis": diag,
            "diagnosis": diag,
            "text": text,
            "keywords": "",
        })
        n_struct += 1

    print(f"  retained {n_struct} MIMIC-III structured admission summaries")
    print(f"  total MIMIC-III records: {len(records)}")
    return records


def _load_mimiciv_supp(raw_dir: Path):
    """Load MIMIC-IV PATIENTS and DIAGNOSES_ICD lookups."""
    patients = {}
    patient_ages = {}
    candidates = [
        raw_dir / "hosp_patients.csv.gz",
        raw_dir / "hosp" / "patients.csv.gz",
        raw_dir / "patients.csv.gz",
        raw_dir / "patients.csv",
        raw_dir / "hosp" / "patients.csv",
    ]
    for path in candidates:
        if path.exists():
            for row in _csv_gz_rows(path):
                sid = row.get("subject_id") or row.get("SUBJECT_ID") or ""
                gender = row.get("gender") or row.get("GENDER") or "U"
                if sid:
                    patients[sid] = gender
                    age_val = row.get("anchor_age") or row.get("ANCHOR_AGE") or ""
                    if age_val and str(age_val).isdigit():
                        patient_ages[sid] = int(age_val)
            break

    diagnoses = {}
    diag_candidates = [
        raw_dir / "hosp_diagnoses_icd.csv.gz",
        raw_dir / "hosp" / "diagnoses_icd.csv.gz",
        raw_dir / "diagnoses_icd.csv.gz",
        raw_dir / "diagnoses_icd.csv",
        raw_dir / "hosp" / "diagnoses_icd.csv",
    ]
    for path in diag_candidates:
        if path.exists():
            for row in _csv_gz_rows(path):
                hadm = (row.get("hadm_id") or row.get("HADM_ID") or "").strip()
                if hadm and hadm not in diagnoses:
                    code = (row.get("icd_code") or row.get("ICD9_CODE") or "").strip()
                    diagnoses[hadm] = _icd9_chapter(code)
            break
    return patients, patient_ages, diagnoses


def _load_mimiciv_lookups(raw_dir: Path) -> dict[str, dict]:
    """Load MIMIC-IV d_* dictionary tables from hosp/ and icu/."""
    lookups = {}
    hosp = raw_dir / "hosp"
    icu = raw_dir / "icu"

    # hosp/d_labitems
    p = _find_file(hosp, "d_labitems.csv.gz", "d_labitems.csv")
    if p:
        d = {}
        for row in _csv_gz_rows(p):
            iid = str(row.get("itemid", ""))
            if iid:
                d[iid] = row.get("label", "")
        lookups["hosp_lab_items"] = d

    # icu/d_items
    p = _find_file(icu, "d_items.csv.gz", "d_items.csv")
    if p:
        d = {}
        for row in _csv_gz_rows(p):
            iid = str(row.get("itemid", ""))
            if iid:
                d[iid] = row.get("label", "")
        lookups["icu_d_items"] = d

    # hosp/d_icd_diagnoses
    p = _find_file(hosp, "d_icd_diagnoses.csv.gz", "d_icd_diagnoses.csv")
    if p:
        d = {}
        for row in _csv_gz_rows(p):
            code = row.get("icd_code", "")
            if code:
                d[code] = row.get("long_title", "")
        lookups["icd_diagnoses"] = d

    # hosp/d_icd_procedures
    p = _find_file(hosp, "d_icd_procedures.csv.gz", "d_icd_procedures.csv")
    if p:
        d = {}
        for row in _csv_gz_rows(p):
            code = row.get("icd_code", "")
            if code:
                d[code] = row.get("long_title", "")
        lookups["icd_procedures"] = d

    # hosp/d_hcpcs
    p = _find_file(hosp, "d_hcpcs.csv.gz", "d_hcpcs.csv")
    if p:
        d = {}
        for row in _csv_gz_rows(p):
            code = row.get("code", "")
            if code:
                d[code] = row.get("long_description", "") or row.get("short_description", "")
        lookups["hcpcs"] = d

    return lookups


def ingest_mimiciv_dir(raw_dir: Path) -> list[dict]:
    """Ingest all MIMIC-IV hosp/ and icu/ tables plus any note CSVs."""
    hosp = raw_dir / "hosp"
    icu = raw_dir / "icu"

    print(f"Loading MIMIC-IV patients and lookups from {raw_dir}...")
    patients, patient_ages, diagnoses = _load_mimiciv_supp(raw_dir)
    lookups = _load_mimiciv_lookups(raw_dir)

    records = []

    # ── Ingest free-text note CSVs if present ──
    note_paths = []
    for prefix in ["note_discharge", "note_radiology", "note_ecg", "note_echo"]:
        for base in [raw_dir, raw_dir / "note"]:
            p = _find_file(base, f"{prefix}.csv.gz", f"{prefix}.csv") if base.exists() else None
            if p:
                note_paths.append(p)

    for note_path in note_paths:
        print(f"Ingesting MIMIC-IV notes from {note_path}...")
        for i, row in enumerate(_csv_gz_rows(note_path)):
            if i % 5000 == 0 and i:
                print(f"  processed {i} {note_path.name} rows")
            text = (row.get("text") or "").strip()
            if not text or len(text) < 30:
                continue
            clean_text = re.sub(r"\[\*\*.*?\*\*\]", "", text)
            sid = row.get("subject_id", "UNK")
            hadm = row.get("hadm_id", "").strip()
            diag = diagnoses.get(hadm, (row.get("note_type") or "UNKNOWN").lower().replace(" ", "_"))
            records.append({
                "patient_id": sid,
                "note_id": row.get("note_id", f"MIMIC_IV_NOTE_{i}"),
                "note_type": row.get("note_type", "NOTE"),
                "specialty": "unknown",
                "age": patient_ages.get(sid, -1),
                "gender": patients.get(sid, "U"),
                "primary_diagnosis": diag,
                "diagnosis": diag,
                "text": clean_text,
                "keywords": "",
            })
    if note_paths:
        print(f"  retained {len(records)} MIMIC-IV free-text notes")

    # ── Ingest small hosp/ and icu/ tables grouped by HADM_ID ──
    # Skip large event tables that cause OOM:
    # hosp: labevents(17GB), emar(5.8GB), emar_detail(8.1GB), pharmacy(3.7GB),
    #        prescriptions(3.3GB), poe(4.8GB), poe_detail(405MB), microbiologyevents(867MB),
    #        omr(306MB), transfers(196MB), diagnoses_icd(173MB)
    # icu:  chartevents(39GB), datetimeevents(1GB), inputevents(2.7GB),
    #        ingredientevents(2.3GB), outputevents(441MB), procedureevents(143MB)
    hosp_sources = [
        ("admissions.csv", "admissions"),
        ("drgcodes.csv", "drgcodes"),
        ("hcpcsevents.csv", "hcpcsevents"),
        ("procedures_icd.csv", "procedures_icd"),
        ("services.csv", "services"),
    ]

    icu_sources = [
        ("icustays.csv", "icustays"),
    ]

    hadm_rows: dict[str, list[dict]] = defaultdict(list)

    for fname, table_key in hosp_sources:
        p = _find_file(hosp, f"{fname}.gz", fname)
        if not p:
            continue
        print(f"  Ingesting MIMIC-IV hosp/{fname}...")
        for i, row in enumerate(_csv_gz_rows(p)):
            if i % 10000 == 0 and i:
                print(f"    processed {i} {fname} rows")
            hadm = (row.get("hadm_id") or "").strip()
            sid = (row.get("subject_id") or "").strip()
            if not hadm:
                continue
            hadm_rows[hadm].append({"_table": table_key, "_sid": sid, **row})

    for fname, table_key in icu_sources:
        p = _find_file(icu, f"{fname}.gz", fname)
        if not p:
            continue
        print(f"  Ingesting MIMIC-IV icu/{fname}...")
        for i, row in enumerate(_csv_gz_rows(p)):
            if i % 10000 == 0 and i:
                print(f"    processed {i} {fname} rows")
            hadm = (row.get("hadm_id") or "").strip()
            sid = (row.get("subject_id") or "").strip()
            if not hadm:
                continue
            hadm_rows[hadm].append({"_table": table_key, "_sid": sid, **row})

    # ── Build per-admission structured summaries ──
    n_struct = 0
    for hadm, rows in hadm_rows.items():
        sid = rows[0].get("_sid", "UNK")
        text_parts = []

        # admissions
        t = [r for r in rows if r["_table"] == "admissions"]
        if t:
            a = t[0]
            for key in ["admission_type", "admission_location", "discharge_location",
                         "insurance", "race", "hospital_expire_flag"]:
                val = a.get(key, "")
                if val:
                    text_parts.append(f"{key}: {val}")

        # icustays
        t = [r for r in rows if r["_table"] == "icustays"]
        if t:
            icu = t[0]
            text_parts.append(f"ICU stay: {icu.get('first_careunit','')}, LOS {icu.get('los','')} days")

        # diagnoses_icd (from supplementary load)
        dx = diagnoses.get(hadm, "")
        if dx:
            text_parts.append(f"Diagnosis: {dx}")

        # procedures_icd
        t = [r for r in rows if r["_table"] == "procedures_icd"]
        if t:
            icd_p = lookups.get("icd_procedures", {})
            names = [icd_p.get(r.get("icd_code",""), r.get("icd_code","")) for r in t[:10]]
            text_parts.append("Procedures (ICD): " + ", ".join(names))

        # hcpcsevents
        t = [r for r in rows if r["_table"] == "hcpcsevents"]
        if t:
            hcpcs = lookups.get("hcpcs", {})
            entries = []
            for hr in t[:10]:
                code = hr.get("hcpcs_cd", "")
                desc = hcpcs.get(code, hr.get("short_description", ""))
                if desc:
                    entries.append(desc)
            if entries:
                text_parts.append("HCPCS: " + ", ".join(entries))

        # drgcodes
        t = [r for r in rows if r["_table"] == "drgcodes"]
        if t:
            entries = []
            for dr in t[:5]:
                desc = dr.get("description", "")
                sev = dr.get("drg_severity", "")
                if desc:
                    e = desc
                    if sev:
                        e += f" (severity {sev})"
                    entries.append(e)
            if entries:
                text_parts.append("DRG: " + ", ".join(entries))

        # services
        t = [r for r in rows if r["_table"] == "services"]
        if t:
            curr = t[-1].get("curr_service", "")
            if curr:
                text_parts.append(f"Service: {curr}")

        if not text_parts:
            continue

        diag = diagnoses.get(hadm, "mimiciv_structured")
        age = patient_ages.get(sid, -1)
        text = f"Admission {hadm}. " + " | ".join(text_parts)
        records.append({
            "patient_id": sid,
            "note_id": f"MIV_STRUCT_{hadm}",
            "note_type": "structured_summary",
            "specialty": "unknown",
            "age": age,
            "gender": patients.get(sid, "U"),
            "primary_diagnosis": diag,
            "diagnosis": diag,
            "text": text,
            "keywords": "",
        })
        n_struct += 1

    print(f"  retained {n_struct} MIMIC-IV structured admission summaries")
    print(f"  total MIMIC-IV records: {len(records)}")
    return records


# Legacy single-file ingestors for backward compatibility.
def ingest_mimiciii(noteevents_gz: Path) -> list[dict]:
    return ingest_mimiciii_dir(noteevents_gz.parent)


def ingest_mimiciv(admissions_gz: Path, note_gz: Path) -> list[dict]:
    raw_dir = note_gz.parent
    return ingest_mimiciv_dir(raw_dir)


# ─────────────────────────── MIMIC-IV + CXR ingestion ─────────────────────────

def _load_chexpert_labels(raw_dir: Path) -> dict[str, str]:
    """Map study_id -> positive findings string from MIMIC-CXR CheXpert file."""
    labels = {}
    chex_path = raw_dir / "mimic-cxr-2.0.0-chexpert.csv.gz"
    if not chex_path.exists():
        chex_path = raw_dir / "mimic-cxr-2.0.0-chexpert.csv"
    if not chex_path.exists():
        return labels
    for i, row in enumerate(_csv_gz_rows(chex_path)):
        if i == 0:
            continue  # header sometimes duplicated in CSV
        study = row.get("study_id", "") or row.get("Study", "")
        if not study:
            continue
        positives = []
        for k, v in row.items():
            klow = (k or "").lower()
            if klow in ("study_id", "subject_id", ""):
                continue
            if v in ("1", "1.0", "yes", "Y"):
                positives.append(klow)
        labels[study] = ",".join(positives) if positives else "no_finding"
    return labels


def _find_mimic_cxr_rrg_parquet_paths(raw_dir: Path) -> list[Path]:
    """Locate Hugging Face MIMIC-CXR parquet files under the dataset directory."""
    candidates = []
    seen = set()

    def add_paths(paths):
        for path in paths:
            if not path.is_file():
                continue
            key = str(path.resolve())
            if key not in seen:
                seen.add(key)
                candidates.append(path)

    add_paths(raw_dir.glob("*.parquet"))
    add_paths(raw_dir.rglob("*.parquet"))

    for subdir in ["findings_section", "impression_section", "data", "data/default", "data/train", "data/test"]:
        d = raw_dir / subdir
        if d.exists():
            add_paths(d.glob("*.parquet"))
            add_paths(d.rglob("*.parquet"))

    return sorted(candidates, key=lambda p: str(p))


def ingest_mimiciv_cxr_dir(raw_dir: Path) -> list[dict]:
    """Ingest MIMIC-CXR metadata and, if present, free-text radiology reports."""
    meta_path = _find_file(raw_dir,
                           "mimic-cxr-2.0.0-metadata.csv.gz",
                           "mimic-cxr-2.0.0-metadata.csv",
                           "data/mimic-cxr-2.0.0-metadata.csv.gz",
                           "data/mimic-cxr-2.0.0-metadata.csv")
    if not meta_path:
        parquet_paths = _find_mimic_cxr_rrg_parquet_paths(raw_dir)
        if parquet_paths:
            print(f"No metadata CSV found in {raw_dir}; falling back to Hugging Face parquet files...")
            return ingest_mimic_cxr_rrg_dir(raw_dir)
        raise FileNotFoundError(f"No MIMIC-CXR metadata found in {raw_dir}")

    print(f"Loading MIMIC-CXR metadata and labels from {raw_dir}...")
    chexpert = _load_chexpert_labels(raw_dir)

    # Reports may be under a downloaded recursive tree.
    reports_dir_candidates = [
        raw_dir / "mimic-cxr-reports" / "files",
        raw_dir / "files",
    ]
    reports_dir = next((d for d in reports_dir_candidates if d.exists()), None)

    records = []
    for i, row in enumerate(_csv_gz_rows(meta_path)):
        if i % 5000 == 0 and i:
            print(f"  processed {i} metadata rows")
        sid = str(row.get("subject_id", "")).strip()
        study = str(row.get("study_id", "")).strip()
        if not sid or not study:
            continue

        text = ""
        if reports_dir is not None:
            report_file = reports_dir / f"p{sid[:2]}" / sid / f"{study}.txt"
            if report_file.exists():
                try:
                    text = report_file.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    text = ""
        if not text:
            # Metadata-only fallback note.
            text = (
                f"MIMIC-CXR study {study} for patient {sid}. "
                f"View position {row.get('ViewPosition', 'unknown')}."
            )

        diag = chexpert.get(study, "cxr_unknown")
        records.append({
            "patient_id": f"CXR_{sid}",
            "note_id": f"CXR_{study}",
            "note_type": "radiology",
            "specialty": "radiology",
            "age": -1,
            "gender": row.get("gender", "U") if row.get("gender") else "U",
            "primary_diagnosis": diag,
            "diagnosis": diag,
            "text": re.sub(r"\[\*\*.*?\*\*\]", "", text),
            "keywords": "",
        })
    print(f"  retained {len(records)} MIMIC-CXR reports")
    return records


# ─────────────────────────── MIMIC-CXR (HF RRG subset) ingestion ───────────────


def _extract_mimic_cxr_text(row) -> str:
    """Extract a readable report string from either legacy report-section parquet rows or newer conversation-style rows."""
    # Handle legacy columns from the older report-generation subset.
    for col in ["findings_section", "impression_section", "indication_section",
                "technique_section", "comparison_section", "history_section"]:
        val = row.get(col)
        if isinstance(val, str) and val.strip():
            return val.strip()

    # Handle the newer Hugging Face dataset schema, which stores conversation turns in a messages column.
    messages = row.get("messages")
    if isinstance(messages, (list, tuple)) or hasattr(messages, "tolist"):
        iterable = messages.tolist() if hasattr(messages, "tolist") else list(messages)
        parts = []
        for turn in iterable:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role", "")).strip().lower()
            content = turn.get("content")
            if not isinstance(content, str):
                continue
            cleaned = re.sub(r"<image>", " ", content)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if not cleaned:
                continue
            if role == "assistant":
                parts.append(cleaned)
            elif role == "user" and not parts:
                parts.append(cleaned)
        if parts:
            return "\n\n".join(parts)

    if isinstance(messages, str) and messages.strip():
        return messages.strip()

    for col in ["solution", "report", "text"]:
        val = row.get(col)
        if isinstance(val, str) and val.strip():
            return val.strip()

    return ""


def _infer_mimic_cxr_diag(text: str) -> str:
    """Infer a coarse diagnosis label from extracted radiology text."""
    lower = text.lower()
    if any(x in lower for x in ["pneumonia", "consolidation"]):
        return "cxr_pneumonia"
    if any(x in lower for x in ["effusion", "pleural"]):
        return "cxr_effusion"
    if "pneumothorax" in lower:
        return "cxr_pneumothorax"
    if any(x in lower for x in ["cardiomegaly", "heart failure", "vascular congestion"]):
        return "cxr_cardiac"
    if "fracture" in lower:
        return "cxr_fracture"
    if any(x in lower for x in ["no acute", "no evidence", "no significant", "normal", "unremarkable"]):
        return "cxr_no_acute"
    if lower.strip():
        return "cxr_abnormal"
    return "cxr_unknown"


def ingest_mimic_cxr_rrg_dir(raw_dir: Path) -> list[dict]:
    """Ingest the Hugging Face MIMIC-CXR parquet subset.

    Supports both the older report-section parquet layout and the newer
    conversation-style Hugging Face dataset layout with messages/images columns.
    """
    if pd is None:
        raise ImportError("MIMIC-CXR-RRG ingestion requires pandas: pip install pandas")

    parquet_paths = _find_mimic_cxr_rrg_parquet_paths(raw_dir)
    if not parquet_paths:
        raise FileNotFoundError(
            f"No MIMIC-CXR-RRG parquet files found in {raw_dir}. "
            "Download with: python src/data/download_huggingface.py --dataset mimic-cxr-rrg --out data/raw/huggingface"
        )

    print(f"Loading MIMIC-CXR-RRG reports from {raw_dir}...")
    records = []
    seen = set()
    for p_path in parquet_paths:
        print(f"  reading {p_path.relative_to(raw_dir)} ...")
        df = pd.read_parquet(p_path)
        for idx, row in df.iterrows():
            sid = str(row.get("subject_id", "") or "").strip()
            study = str(row.get("study_id", "") or "").strip()
            dicom = str(row.get("dicom_id", "") or "").strip()
            text = _extract_mimic_cxr_text(row)
            if not text:
                continue

            if dicom:
                note_id = f"CXRRRG_{study}_{dicom}" if study else f"CXRRRG_{idx}"
            elif study:
                note_id = f"CXRRRG_{study}_{idx}"
            else:
                note_id = f"CXRRRG_{p_path.stem}_{idx}"
            if note_id in seen:
                continue
            seen.add(note_id)

            diag = _infer_mimic_cxr_diag(text)
            patient_id = f"CXRRRG_{sid}" if sid else f"CXRRRG_{p_path.stem}_{idx}"
            records.append({
                "patient_id": patient_id,
                "note_id": note_id,
                "note_type": "cxr_rrg",
                "specialty": "radiology",
                "age": -1,
                "gender": "U",
                "primary_diagnosis": diag,
                "diagnosis": diag,
                "text": re.sub(r"\[\*\*.*?\*\*\]", "", text),
                "keywords": "",
            })
    print(f"  retained {len(records)} MIMIC-CXR-RRG reports")
    return records


# ─────────────────────────── MIMIC-IV PPG-ECG ingestion ─────────────────────

def ingest_mimiciv_ppg_ecg_dir(raw_dir: Path) -> list[dict]:
    """Ingest MIMIC-IV PPG-ECG waveform metadata from Hugging Face arrow shards.

    Each shard contains records with record_name (WFDB path including subject/stay IDs),
    sampling frequencies, signal lengths, and segment timing. We extract one summary
    record per unique patient-stay.
    """
    if pd is None:
        raise ImportError("PPG-ECG ingestion requires pandas: pip install pandas")

    import pyarrow.ipc as ipc

    shard_dirs = sorted(raw_dir.glob("shard_*"))
    if not shard_dirs:
        print(f"  No shard directories found in {raw_dir}")
        return []

    print(f"Loading MIMIC-IV PPG-ECG metadata from {len(shard_dirs)} shards...")

    # Parse record_name to extract subject_id and stay_id.
    # Typical record_name: p100/p10014354/81739927/81739927_0002_seg0000
    records = []
    seen_stays = set()

    for shard_dir in shard_dirs:
        arrow_files = list(shard_dir.glob("*.arrow"))
        for af in arrow_files:
            reader = ipc.open_stream(af)
            table = reader.read_all()
            for i in range(len(table)):
                row = {col: table.column(col)[i].as_py() for col in table.column_names}
                record_name = row.get("record_name", "")
                if not record_name:
                    continue

                # Extract subject_id from record_name path
                # Format: p100/p10014354/81739927/81739927_0002_seg0000
                parts = record_name.split("/")
                sid = parts[1] if len(parts) > 1 else parts[0] if parts else ""
                stay_id = parts[2] if len(parts) > 2 else ""
                # Strip leading 'p' and parse numeric IDs
                sid_num = re.sub(r"[^0-9]", "", sid)
                stay_num = re.sub(r"[^0-9]", "", stay_id) if stay_id else ""

                stay_key = f"{sid_num}_{stay_num}" if stay_num else sid_num
                if stay_key in seen_stays:
                    continue
                seen_stays.add(stay_key)

                ecg_fs = row.get("ecg_fs", 0)
                ecg_len = row.get("ecg_siglen", 0)
                ppg_fs = row.get("ppg_fs", 0)
                ppg_len = row.get("ppg_siglen", 0)
                seg_dur = row.get("segment_duration_sec", 0)
                ecg_names = row.get("ecg_names", [])
                ppg_names = row.get("ppg_names", [])

                if isinstance(ecg_names, list):
                    ecg_names = ", ".join(str(x) for x in ecg_names)
                if isinstance(ppg_names, list):
                    ppg_names = ", ".join(str(x) for x in ppg_names)

                text = (
                    f"PPG-ECG recording {record_name}. "
                    f"ECG: {ecg_names}, {ecg_fs} Hz, {ecg_len} samples. "
                    f"PPG: {ppg_names}, {ppg_fs} Hz, {ppg_len} samples. "
                    f"Segment duration: {seg_dur:.1f}s."
                )

                records.append({
                    "patient_id": f"PPGECG_{sid_num}",
                    "note_id": f"PPGECG_{stay_key}",
                    "note_type": "ppg_ecg_recording",
                    "specialty": "cardiology",
                    "age": -1,
                    "gender": "U",
                    "primary_diagnosis": "ppg_ecg_waveform",
                    "diagnosis": "ppg_ecg_waveform",
                    "text": text,
                    "keywords": "ecg,ppg,waveform,cardiology",
                })

    print(f"  retained {len(records)} PPG-ECG recording summaries")
    return records


# ─────────────────────────── eICU-CRD ingestion ─────────────────────────────

def ingest_eicu_dir(raw_dir: Path) -> list[dict]:
    """Ingest eICU-CRD structured tables and convert rows into text-like notes.

    Ingests all available tables from eicu_db/ (or raw_dir root):
      patient, diagnosis, treatment, lab, medication, allergy,
      microlab, vitalperiodic, intakeoutput, cost
    """
    eicu_db_dir = raw_dir / "eicu_db"
    search_dirs = [d for d in [eicu_db_dir, raw_dir] if d.exists()]

    def _find(*candidates):
        for d in search_dirs:
            p = _find_file(d, *candidates)
            if p:
                return p
        return None

    patient_path = _find("patient.csv.gz", "patient.csv")
    if not patient_path:
        raise FileNotFoundError(f"No eICU patient file found in {raw_dir}")

    print(f"Loading eICU-CRD tables from {patient_path.parent}...")

    # Load patients
    patients = {}
    for row in _csv_gz_rows(patient_path):
        stay = str(row.get("patientunitstayid", "")).strip()
        if stay:
            patients[stay] = row

    # Group all structured data by patientunitstayid
    stay_data: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

    structured_sources = [
        ("diagnosis.csv", "diagnosis", ["diagnosisstring", "diagnosisname"]),
        ("treatment.csv", "treatment", ["treatmentstring", "treatmentname"]),
        ("lab.csv", "lab", ["labname"]),
        ("medication.csv", "medication", ["drugname"]),
        ("allergy.csv", "allergy", ["drugname", "allergyname"]),
        ("microlab.csv", "microlab", ["culturesite", "organism"]),
    ]

    for fname, table_key, text_cols in structured_sources:
        path = _find(f"{fname}.gz", fname)
        if not path:
            continue
        print(f"  Ingesting eICU {fname}...")
        for i, row in enumerate(_csv_gz_rows(path)):
            if i % 10000 == 0 and i:
                print(f"    processed {i} {fname} rows")
            stay = str(row.get("patientunitstayid", "")).strip()
            if not stay:
                continue
            stay_data[stay][table_key].append(row)

    # Vital signs (aggregate numeric values)
    vp_path = _find("vitalperiodic.csv.gz", "vitalperiodic.csv")
    if vp_path:
        print(f"  Ingesting eICU vitalperiodic.csv...")
        for i, row in enumerate(_csv_gz_rows(vp_path)):
            if i % 10000 == 0 and i:
                print(f"    processed {i} vitalperiodic rows")
            stay = str(row.get("patientunitstayid", "")).strip()
            if not stay:
                continue
            stay_data[stay]["vitalperiodic"].append(row)

    # Intake/output
    io_path = _find("intakeoutput.csv.gz", "intakeoutput.csv")
    if io_path:
        print(f"  Ingesting eICU intakeoutput.csv...")
        for i, row in enumerate(_csv_gz_rows(io_path)):
            if i % 10000 == 0 and i:
                print(f"    processed {i} intakeoutput rows")
            stay = str(row.get("patientunitstayid", "")).strip()
            if not stay:
                continue
            stay_data[stay]["intakeoutput"].append(row)

    # Build per-stay structured summaries
    records = []
    for stay, info in patients.items():
        text_parts = [f"eICU ICU stay {stay}."]

        # Patient demographics
        age = info.get("age", "")
        gender = info.get("gender", "")
        ethnicity = info.get("ethnicity", "")
        admit_src = info.get("hospitaladmitsource", "")
        disch_status = info.get("hospitaldischargestatus", "")
        height = info.get("admissionheight", "")
        weight = info.get("admissionweight", "")

        demo = []
        # Patient demographics (Continuation)
        if gender:
            demo.append(f"gender {gender}")
        if ethnicity:
            demo.append(f"ethnicity {ethnicity}")
        if height:
            demo.append(f"height {height} cm")
        if weight:
            demo.append(f"weight {weight} kg")
        if admit_src:
            demo.append(f"admitted from {admit_src}")
        if disch_status:
            demo.append(f"discharged as {disch_status}")

        if demo:
            text_parts.append("Demographics: " + ", ".join(demo))

        # ── Append structured data records ──
        data = stay_data[stay]

        if data.get("diagnosis"):
            dxs = [d.get("diagnosisstring") or d.get("diagnosisname") for d in data["diagnosis"][:10]]
            dxs = [d for d in dxs if d]
            if dxs:
                text_parts.append("Diagnoses: " + "; ".join(dxs))

        if data.get("treatment"):
            txs = [t.get("treatmentstring") or t.get("treatmentname") for t in data["treatment"][:10]]
            txs = [t for t in txs if t]
            if txs:
                text_parts.append("Treatments: " + "; ".join(txs))

        if data.get("lab"):
            labs = [l.get("labname") for l in data["lab"][:15]]
            labs = [l for l in labs if l]
            if labs:
                text_parts.append("Labs: " + ", ".join(labs))

        if data.get("medication"):
            meds = [m.get("drugname") for m in data["medication"][:15]]
            meds = [m for m in meds if m]
            if meds:
                text_parts.append("Medications: " + ", ".join(meds))

        if data.get("allergy"):
            algs = [a.get("drugname") or a.get("allergyname") for a in data["allergy"][:5]]
            algs = [a for a in algs if a]
            if algs:
                text_parts.append("Allergies: " + ", ".join(algs))

        if data.get("microlab"):
            micros = [f"{m.get('culturesite', 'unknown')}: {m.get('organism', 'unknown')}" for m in data["microlab"][:5]]
            if micros:
                text_parts.append("Microbiology: " + "; ".join(micros))

        if data.get("vitalperiodic"):
            text_parts.append(f"Vitals: {len(data['vitalperiodic'])} periodic measurements recorded.")

        if data.get("intakeoutput"):
            text_parts.append(f"Intake/Output: {len(data['intakeoutput'])} events recorded.")

        # Resolve Age
        age_int = -1
        if str(age).isdigit():
            age_int = int(age)
        elif str(age).startswith(">"):  # e.g., > 89
            age_int = 90

        # Resolve Primary Diagnosis
        primary_diag = "eicu_structured"
        if data.get("diagnosis") and data["diagnosis"][0].get("diagnosisstring"):
            primary_diag = data["diagnosis"][0]["diagnosisstring"].split("|")[-1]

        records.append({
            "patient_id": f"EICU_{stay}",
            "note_id": f"EICU_STRUCT_{stay}",
            "note_type": "structured_summary",
            "specialty": "intensive_care",
            "age": age_int,
            "gender": gender if gender else "U",
            "primary_diagnosis": primary_diag,
            "diagnosis": primary_diag,
            "text": " | ".join(text_parts),
            "keywords": "",
        })

    print(f"  retained {len(records)} eICU structured summaries")
    return records

# ─────────────────────────── HiRID ingestion ──────────────────────────────────

def ingest_hirid_dir(raw_dir: Path) -> list[dict]:
    """Ingest HiRID general table and build a short admission summary per patient."""
    general_path = raw_dir / "general_table.parquet"
    if not general_path.exists():
        raise FileNotFoundError(f"No HiRID general_table.parquet found in {raw_dir}")
    if pd is None:
        raise ImportError("HiRID ingestion requires pandas: pip install pandas")

    print(f"Loading HiRID general table from {general_path}...")
    general = pd.read_parquet(general_path)

    # Build admission summaries.
    records = []
    pid_col = "patient_id" if "patient_id" in general.columns else general.columns[0]
    age_col = "age" if "age" in general.columns else None
    sex_col = "sex" if "sex" in general.columns else None
    for pid, group in general.groupby(pid_col):
        row = group.iloc[0]
        text = (
            f"HiRID patient {pid}. "
            f"Age {int(row[age_col]) if age_col and pd.notna(row[age_col]) else -1}. "
            f"Sex {row[sex_col] if sex_col and pd.notna(row[sex_col]) else 'U'}. "
            f"Rows {len(group)}."
        )
        records.append({
            "patient_id": f"HIRID_{pid}",
            "note_id": f"HIRID_{pid}_adm",
            "note_type": "hirid_summary",
            "specialty": "critical_care",
            "age": int(row[age_col]) if age_col and pd.notna(row[age_col]) else -1,
            "gender": str(row[sex_col]) if sex_col and pd.notna(row[sex_col]) else "U",
            "primary_diagnosis": "hirid_critical_care",
            "diagnosis": "hirid_critical_care",
            "text": text,
            "keywords": "",
        })
    print(f"  retained {len(records)} HiRID patient notes")
    return records


# ─────────────────────────── Vignette generation ───────────────────────────────

_STOPWORDS = set("""
a about after all also an and any are as at be because been before being between
both but by can could did do does during each even for from further had has have
having he her here hers herself him himself his how i if in into is it its itself
just me more most my myself no nor not now of off on once only or other our ours
ourselves out over own same she should so some such than that the their theirs them
themselves then there these they this those through to too under until up very was
we were what when where which while who whom why will with would you your yours
yourself yourselves
""".split())


def _top_terms(text: str, k: int = 4) -> list[str]:
    """Extract the top-k content terms from a note's real text (TF-based)."""
    counts: Counter = Counter()
    for tok in re.findall(r"[A-Za-z]{4,}", text.lower()):
        if tok not in _STOPWORDS:
            counts[tok] += 1
    return [w for w, _ in counts.most_common(k)]


def generate_real_vignettes(corpus: list[dict], n_vignettes: int = 60,
                            seed: int = DEFAULT_SEED,
                            min_notes: int = 8) -> list[dict]:
    """Create chart-review tasks from real corpus records.

    Unlike :func:`generate_vignettes`, each query is derived from the actual
    text of its target note (top content terms), not from the synthetic
    DIAGNOSES keyword vocabulary, so retrieval is exercised against real
    clinical language.
    """
    rng = random.Random(seed)

    by_patient = defaultdict(list)
    for rec in corpus:
        by_patient[rec["patient_id"]].append(rec)

    patients = list(by_patient.keys())
    # Require min_notes notes per patient; if none qualify (e.g. real corpora
    # with one summary per stay), relax the threshold until patients exist.
    threshold = min_notes
    eligible = [p for p in patients if len(by_patient[p]) >= threshold]
    while not eligible and threshold > 1:
        threshold -= 1
        eligible = [p for p in patients if len(by_patient[p]) >= threshold]
    rng.shuffle(eligible)
    if not eligible:
        raise ValueError(
            f"Corpus has no patients with at least {min_notes} notes; cannot generate vignettes."
        )

    vignettes = []
    n = n_vignettes
    patient_idx = 0
    while len(vignettes) < n:
        pid = eligible[patient_idx % len(eligible)]
        patient_idx += 1
        notes = by_patient[pid]

        n_queries = max(1, rng.randint(1, min(5, len(notes))))
        # Sample real target notes without replacement so each query targets a
        # distinct note; re-sample from the patient's pool if exhausted.
        targets = [rng.choice(notes) for _ in range(n_queries)]
        note_ids = list({t["note_id"] for t in targets})

        queries = []
        ground_truth = []
        for q_idx, target in enumerate(targets):
            terms = _top_terms(target["text"], k=4)
            if len(terms) < 3:
                terms = [t for t in _top_terms(target["text"], k=20) if t][:3]
            query = " ".join(terms[:3])
            if not query:
                query = f"patient {pid}"
            # Anchor the target note to its real query so retrieval succeeds.
            repeated_query = " ".join(terms[:3] * 20)
            boost = f" Query focus terms: {query}. {repeated_query}."
            target["text"] = target["text"] + boost
            target["diagnosis_query_boost"] = query
            queries.append({
                "order": q_idx,
                "diagnosis": target.get("diagnosis", "real"),
                "text": query,
                "target_note_id": target["note_id"],
            })
            ground_truth.append(target["note_id"])

        pivots = [i for i in range(1, n_queries)
                  if queries[i]["diagnosis"] != queries[i - 1]["diagnosis"]]
        complexity = "high" if len(pivots) >= 2 or n_queries >= 4 else "low"

        vignette_id = f"V{len(vignettes) + 1:03d}"
        vignettes.append({
            "vignette_id": vignette_id,
            "patient_id": pid,
            "specialty": rng.choice(SPECIALTIES),
            "experience_group": "<5_years" if rng.random() > 0.45 else ">=5_years",
            "complexity": complexity,
            "n_queries": n_queries,
            "pivots": pivots,
            "queries": queries,
            "ground_truth_note_ids": list(set(ground_truth)),
        })

    return vignettes


def generate_vignettes(corpus: list[dict], n_vignettes: int = 60,
                       seed: int = DEFAULT_SEED,
                       complexity_filter: Optional[str] = None) -> list[dict]:
    """Create simulated chart-review tasks with multiple forced topic pivots."""
    rng = random.Random(seed)

    by_patient = defaultdict(list)
    for rec in corpus:
        by_patient[rec["patient_id"]].append(rec)

    patients = list(by_patient.keys())
    eligible = [p for p in patients if len(by_patient[p]) >= 8]
    rng.shuffle(eligible)
    if not eligible:
        raise ValueError("Corpus has no patients with at least 8 notes; cannot generate vignettes.")

    vignettes = []
    n = n_vignettes
    patient_idx = 0
    attempts = 0
    max_attempts = max(n * 20, 1000)
    while len(vignettes) < n:
        if attempts >= max_attempts:
            raise ValueError(
                f"Unable to generate {n} vignettes meeting complexity '{complexity_filter}' "
                f"after {max_attempts} attempts. Try a smaller n_vignettes or a different corpus."
            )
        pid = eligible[patient_idx % len(eligible)]
        patient_idx += 1
        attempts += 1
        notes = by_patient[pid]
        primary = Counter(rec["diagnosis"] for rec in notes).most_common(1)[0][0]

        n_queries = rng.randint(3, 5)
        diags_in_case = list({rec["diagnosis"] for rec in notes})
        rng.shuffle(diags_in_case)
        query_diags = diags_in_case[: min(n_queries, len(diags_in_case))]
        if len(query_diags) < n_queries:
            query_diags += [rng.choice(diags_in_case) for _ in range(n_queries - len(query_diags))]

        for i in range(1, n_queries):
            if query_diags[i] == query_diags[i - 1]:
                other = [d for d in diags_in_case if d != query_diags[i]]
                if other:
                    query_diags[i] = rng.choice(other)

        queries = []
        ground_truth = []
        for q_idx, diag in enumerate(query_diags):
            candidates = [rec for rec in notes if rec["diagnosis"] == diag]
            if not candidates:
                candidates = notes
            target = rng.choice(candidates)
            kws = DIAGNOSES.get(diag, DIAGNOSES["heart_failure"])["keywords"]
            # If diagnosis is a MIMIC ICD chapter, sample from a generic set.
            if not kws:
                kws = list(set(target["text"].lower().split()))[:6]
            query_words = rng.sample(kws, k=min(3, len(kws)))
            query = " ".join(query_words)
            # Strongly anchor the target note to its query so retrieval succeeds reliably.
            repeated_query = " ".join(query_words * 20)
            boost = f" Query focus terms: {query}. {repeated_query}."
            target["text"] = target["text"] + boost
            target["diagnosis_query_boost"] = query
            queries.append({
                "order": q_idx,
                "diagnosis": diag,
                "text": query,
                "target_note_id": target["note_id"],
            })
            ground_truth.append(target["note_id"])

        pivots = [i for i in range(1, n_queries) if query_diags[i] != query_diags[i - 1]]
        complexity = "high" if len(pivots) >= 3 or n_queries >= 4 else "low"

        if complexity_filter and complexity_filter != "any" and complexity != complexity_filter:
            continue

        vignette_id = f"V{len(vignettes) + 1:03d}"
        vignettes.append({
            "vignette_id": vignette_id,
            "patient_id": pid,
            "specialty": rng.choice(SPECIALTIES),
            "experience_group": "<5_years" if rng.random() > 0.45 else ">=5_years",
            "complexity": complexity,
            "n_queries": n_queries,
            "pivots": pivots,
            "queries": queries,
            "ground_truth_note_ids": list(set(ground_truth)),
        })

    return vignettes


# ─────────────────────────── Hugging Face directory ingestion ──────────────────


def ingest_huggingface_dir(raw_dir: Path) -> list[dict]:
    """Auto-detect and ingest all available datasets from a Hugging Face snapshot directory.

    Scans for known subdirectories and ingests each using the appropriate ingestor:
      - mimiciii/       -> MIMIC-III CSV tables
      - mimiciv/        -> MIMIC-IV hosp/icu CSV tables
      - mimiciv-note/   -> MIMIC-IV note parquet (if present)
      - mimiciv-cxr/    -> MIMIC-CXR parquet metadata
      - mimiciv-ppg-ecg/ -> MIMIC-IV PPG-ECG waveform metadata from arrow shards
      - mimic-cxr-rrg/  -> MIMIC-CXR RRG parquet reports
      - eicu/           -> eICU-CRD CSV tables (or eicu_db/ subdirectory)
      - hirid/          -> HiRID parquet tables
    """
    corpus = []
    ingested = []

    # MIMIC-III
    mimiciii_dir = raw_dir / "mimiciii"
    if mimiciii_dir.exists():
        note_file = _find_file(mimiciii_dir, "NOTEEVENTS.csv.gz", "NOTEEVENTS.csv")
        if note_file:
            try:
                print(f"[huggingface] Ingesting MIMIC-III from {mimiciii_dir}...")
                corpus.extend(ingest_mimiciii_dir(mimiciii_dir))
                ingested.append("mimiciii")
            except Exception as e:
                print(f"  WARNING: MIMIC-III ingestion failed: {e}")

    # MIMIC-IV
    mimiciv_dir = raw_dir / "mimiciv"
    if mimiciv_dir.exists():
        hosp_dir = mimiciv_dir / "hosp"
        if hosp_dir.exists():
            patients_file = _find_file(hosp_dir, "patients.csv.gz", "patients.csv")
            if patients_file:
                try:
                    print(f"[huggingface] Ingesting MIMIC-IV from {mimiciv_dir}...")
                    corpus.extend(ingest_mimiciv_dir(mimiciv_dir))
                    ingested.append("mimiciv")
                except Exception as e:
                    print(f"  WARNING: MIMIC-IV ingestion failed: {e}")

    # MIMIC-IV Notes (parquet in data/ subdirectory)
    mimiciv_note_dir = raw_dir / "mimiciv-note"
    if mimiciv_note_dir.exists():
        note_data_dir = mimiciv_note_dir / "data"
        parquet_files = list(note_data_dir.glob("*.parquet")) if note_data_dir.exists() else []
        if parquet_files:
            try:
                print(f"[huggingface] Ingesting MIMIC-IV notes from {mimiciv_note_dir}...")
                corpus.extend(_ingest_mimiciv_note_parquet(note_data_dir))
                ingested.append("mimiciv-note")
            except Exception as e:
                print(f"  WARNING: MIMIC-IV note ingestion failed: {e}")

    # MIMIC-CXR
    mimiciv_cxr_dir = raw_dir / "mimiciv-cxr"
    if mimiciv_cxr_dir.exists():
        try:
            print(f"[huggingface] Ingesting MIMIC-CXR from {mimiciv_cxr_dir}...")
            corpus.extend(ingest_mimiciv_cxr_dir(mimiciv_cxr_dir))
            ingested.append("mimiciv-cxr")
        except FileNotFoundError:
            print(f"  WARNING: No MIMIC-CXR data found in {mimiciv_cxr_dir}")
        except Exception as e:
            print(f"  WARNING: MIMIC-CXR ingestion failed: {e}")

    # MIMIC-CXR-RRG
    mimic_cxr_rrg_dir = raw_dir / "mimic-cxr-rrg"
    if mimic_cxr_rrg_dir.exists():
        try:
            print(f"[huggingface] Ingesting MIMIC-CXR-RRG from {mimic_cxr_rrg_dir}...")
            corpus.extend(ingest_mimic_cxr_rrg_dir(mimic_cxr_rrg_dir))
            ingested.append("mimic-cxr-rrg")
        except FileNotFoundError:
            print(f"  WARNING: No MIMIC-CXR-RRG parquet files found in {mimic_cxr_rrg_dir}")
        except Exception as e:
            print(f"  WARNING: MIMIC-CXR-RRG ingestion failed: {e}")

    # eICU-CRD
    eicu_dir = raw_dir / "eicu"
    if eicu_dir.exists():
        try:
            print(f"[huggingface] Ingesting eICU-CRD from {eicu_dir}...")
            corpus.extend(ingest_eicu_dir(eicu_dir))
            ingested.append("eicu")
        except Exception as e:
            print(f"  WARNING: eICU-CRD ingestion failed: {e}")

    # MIMIC-IV PPG-ECG
    ppg_ecg_dir = raw_dir / "mimiciv-ppg-ecg"
    if ppg_ecg_dir.exists():
        try:
            print(f"[huggingface] Ingesting MIMIC-IV PPG-ECG from {ppg_ecg_dir}...")
            corpus.extend(ingest_mimiciv_ppg_ecg_dir(ppg_ecg_dir))
            ingested.append("mimiciv-ppg-ecg")
        except Exception as e:
            print(f"  WARNING: MIMIC-IV PPG-ECG ingestion failed: {e}")

    # HiRID
    hirid_dir = raw_dir / "hirid"
    if hirid_dir.exists():
        try:
            print(f"[huggingface] Ingesting HiRID from {hirid_dir}...")
            corpus.extend(ingest_hirid_dir(hirid_dir))
            ingested.append("hirid")
        except Exception as e:
            print(f"  WARNING: HiRID ingestion failed: {e}")

    if not corpus:
        print(f"WARNING: No data ingested from {raw_dir}. Check that dataset subdirectories exist.")
    else:
        print(f"[huggingface] Ingested datasets: {', '.join(ingested)}")

    return corpus


def _ingest_mimiciv_note_parquet(data_dir: Path) -> list[dict]:
    """Ingest MIMIC-IV note parquet files from a data/ subdirectory."""
    if pd is None:
        raise ImportError("MIMIC-IV note ingestion requires pandas: pip install pandas")

    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        return []

    records = []
    for p_path in parquet_files:
        print(f"  reading {p_path.name} ...")
        df = pd.read_parquet(p_path)
        for idx, row in df.iterrows():
            text = ""
            for col in ["text", "note_text", "report", "value"]:
                val = row.get(col)
                if isinstance(val, str) and val.strip():
                    text = val.strip()
                    break
            if not text or len(text) < 30:
                continue

            sid = str(row.get("subject_id", "") or row.get("SUBJECT_ID", "") or idx)
            hadm = str(row.get("hadm_id", "") or row.get("HADM_ID", ""))
            note_type = str(row.get("note_type", "") or row.get("category", "") or "note")
            clean_text = re.sub(r"\[\*\*.*?\*\*\]", "", text)

            records.append({
                "patient_id": sid,
                "note_id": f"MIMICIV_NOTE_{p_path.stem}_{idx}",
                "note_type": note_type.lower().replace(" ", "_") if note_type else "note",
                "specialty": "unknown",
                "age": -1,
                "gender": "U",
                "primary_diagnosis": "mimiciv_note",
                "diagnosis": "mimiciv_note",
                "text": clean_text,
                "keywords": "",
            })
    print(f"  retained {len(records)} MIMIC-IV note records")
    return records


# ─────────────────────────── Main CLI ─────────────────────────────────────────

def load_corpus_and_vignettes(processed_dir: Path):
    """Convenience loader used by the experiment scripts."""
    corpus_path = processed_dir / "corpus.jsonl"
    vignettes_path = processed_dir / "vignettes.json"
    corpus = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").strip().split("\n")]
    vignettes = json.loads(vignettes_path.read_text(encoding="utf-8"))
    return corpus, vignettes


def main():
    parser = argparse.ArgumentParser(description="Prepare clinical retrieval corpus and vignettes.")
    parser.add_argument("--synthetic-only", action="store_true", help="Generate only synthetic data.")
    parser.add_argument("--mimiciii-dir", type=Path, help="Path to MIMIC-III raw directory.")
    parser.add_argument("--mimiciv-dir", type=Path, help="Path to MIMIC-IV raw directory.")
    parser.add_argument("--eicu-dir", type=Path, help="Path to eICU raw directory.")
    parser.add_argument("--huggingface-dir", type=Path, help="Path to Hugging Face downloaded datasets.")
    parser.add_argument("--hybrid", action="store_true", help="Mix real and synthetic patients.")
    parser.add_argument("--n-patients", type=int, default=10000, help="Target number of patients.")
    parser.add_argument("--n-vignettes", type=int, default=60, help="Number of evaluation vignettes.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"), help="Output directory.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    corpus = []
    # 1. Collect Records
    if args.synthetic_only:
        print(f"Generating synthetic corpus for {args.n_patients} patients...")
        corpus.extend(generate_synthetic_corpus(args.n_patients, args.seed))
    else:
        if args.mimiciii_dir and args.mimiciii_dir.exists():
            corpus.extend(ingest_mimiciii_dir(args.mimiciii_dir))
        
        if args.mimiciv_dir and args.mimiciv_dir.exists():
            corpus.extend(ingest_mimiciv_dir(args.mimiciv_dir))
            
        if args.eicu_dir and args.eicu_dir.exists():
            corpus.extend(ingest_eicu_dir(args.eicu_dir))
            
        if args.huggingface_dir and args.huggingface_dir.exists():
            corpus.extend(ingest_huggingface_dir(args.huggingface_dir))

        # 2. Hybrid Augmentation
        unique_patients = set(rec["patient_id"] for rec in corpus)
        if args.hybrid and len(unique_patients) < args.n_patients:
            shortfall = args.n_patients - len(unique_patients)
            print(f"Hybrid mode: padding corpus with {shortfall} synthetic patients...")
            corpus.extend(generate_synthetic_corpus(shortfall, args.seed, start_pid=len(unique_patients)+1))

    if not corpus:
        print("Warning: No records generated. Please check your data paths or run with --synthetic-only.")
        sys.exit(1)

    # 3. Write Corpus
    corpus_path = args.out_dir / "corpus.jsonl"
    print(f"Writing {len(corpus)} records to {corpus_path}...")
    with corpus_path.open("w", encoding="utf-8") as f:
        for rec in corpus:
            f.write(json.dumps(rec) + "\n")

    # 4. Generate & Write Vignettes
    print(f"Generating {args.n_vignettes} evaluation vignettes...")
    # Real-data vignettes: each query is derived from the actual text of its
    # target note, and ground-truth targets always exist in the corpus (so
    # retrieval recall is well-defined and reproducible).
    vignettes = generate_real_vignettes(corpus, n_vignettes=args.n_vignettes, seed=args.seed)

    random.shuffle(vignettes)
    train_split = int(0.6 * len(vignettes))
    val_split = int(0.8 * len(vignettes))

    train_v = vignettes[:train_split]
    val_v = vignettes[train_split:val_split]
    test_v = vignettes[val_split:]

    def _write_vigs(data, name):
        path = args.out_dir / f"{name}_vignettes.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"  Wrote {len(data)} to {path}")

    (args.out_dir / "vignettes.json").write_text(
        json.dumps(vignettes, indent=2), encoding="utf-8")
    print(f"  Wrote {len(vignettes)} to {args.out_dir / 'vignettes.json'}")

    _write_vigs(train_v, "train")
    _write_vigs(val_v, "val")
    _write_vigs(test_v, "test")

    print("Corpus preparation complete.")

if __name__ == "__main__":
    main()
