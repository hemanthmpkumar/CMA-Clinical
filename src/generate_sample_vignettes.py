import json
from pathlib import Path

TEMPLATES = [
    {
        "id": "v1",
        "title": "Acute on Chronic Heart Failure with Rising Creatinine",
        "text": "A 72-year-old with CHF presents with dyspnea; BP 110/70; creatinine rising from 1.1 to 2.3 over 48h. Review meds and determine cause."
    },
    {
        "id": "v2",
        "title": "Sepsis Concern After Urinary Tract Infection",
        "text": "A 65-year-old with fever and altered mental status; initial lactate 3.2; consider early sepsis bundle and vasopressor thresholds."
    }
]

def save(out_dir="./data"):
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    for t in TEMPLATES:
        (d / f"{t['id']}.json").write_text(json.dumps(t, indent=2))

if __name__ == '__main__':
    save()
