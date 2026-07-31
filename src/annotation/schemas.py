import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional


USEFULNESS_LABELS = {
    1: "not at all useful",
    2: "slightly useful",
    3: "somewhat useful",
    4: "moderately useful",
    5: "quite useful",
    6: "very useful",
    7: "extremely useful",
}

TRUST_LABELS = {
    1: "no trust",
    2: "very little trust",
    3: "little trust",
    4: "moderate trust",
    5: "significant trust",
    6: "high trust",
    7: "complete trust",
}

SAFETY_LEVELS = ["safe", "ambiguous", "unsafe"]


@dataclass
class QueryAnnotation:
    vignette_id: str
    condition: str
    query_index: int
    query_text: str
    retrieved_note_ids: list[str]
    usefulness: Optional[int] = None
    safety: Optional[str] = None
    safety_notes: str = ""
    annotator_id: str = ""
    annotation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class SessionAnnotation:
    vignette_id: str
    condition: str
    trust: Optional[int] = None
    workload_feedback: str = ""
    annotator_id: str = ""
    annotation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class AdjudicationRecord:
    vignette_id: str
    condition: str
    disputed_field: str
    control_value: str
    cma_value: str
    adjudicated_value: str
    adjudicator_id: str
    rationale: str = ""
    adjudication_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


def query_annotation_csv_columns():
    return [
        "annotation_id", "vignette_id", "condition", "query_index",
        "query_text", "retrieved_note_ids", "usefulness", "safety",
        "safety_notes", "annotator_id", "timestamp",
    ]


def session_annotation_csv_columns():
    return [
        "annotation_id", "vignette_id", "condition", "trust",
        "workload_feedback", "annotator_id", "timestamp",
    ]


def adjudication_csv_columns():
    return [
        "adjudication_id", "vignette_id", "condition", "disputed_field",
        "control_value", "cma_value", "adjudicated_value",
        "adjudicator_id", "rationale", "timestamp",
    ]
