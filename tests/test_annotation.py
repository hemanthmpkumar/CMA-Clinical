import csv
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.annotation.schemas import (
    QueryAnnotation,
    SessionAnnotation,
    AdjudicationRecord,
    query_annotation_csv_columns,
    session_annotation_csv_columns,
    adjudication_csv_columns,
)
from src.annotation.export import (
    load_annotations,
    merge_with_results,
    apply_adjudications,
    summary,
)
from src.analysis.analyze import human_annotation_analysis


class TestAnnotationSchemas(unittest.TestCase):
    def test_query_annotation(self):
        ann = QueryAnnotation(
            vignette_id="V001",
            condition="cma",
            query_index=0,
            query_text="heart failure",
            retrieved_note_ids=["n1", "n2"],
            usefulness=5,
            safety="safe",
            annotator_id="expert_001",
        )
        self.assertEqual(ann.vignette_id, "V001")
        self.assertEqual(ann.usefulness, 5)
        self.assertIsNotNone(ann.annotation_id)
        self.assertEqual(len(ann.annotation_id), 8)

    def test_session_annotation(self):
        ann = SessionAnnotation(
            vignette_id="V001",
            condition="control",
            trust=4,
        )
        self.assertEqual(ann.trust, 4)

    def test_adjudication_record(self):
        rec = AdjudicationRecord(
            vignette_id="V001",
            condition="adjudicated",
            disputed_field="accuracy",
            control_value="0",
            cma_value="1",
            adjudicated_value="1",
            adjudicator_id="adj_001",
        )
        self.assertEqual(rec.adjudicated_value, "1")

    def test_csv_column_lists(self):
        self.assertIn("usefulness", query_annotation_csv_columns())
        self.assertIn("trust", session_annotation_csv_columns())
        self.assertIn("adjudicated_value", adjudication_csv_columns())


class TestAnnotationExport(unittest.TestCase):
    def test_merge_with_results(self):
        results = pd.DataFrame({
            "vignette_id": ["V001", "V001", "V002", "V002"],
            "condition": ["control", "cma", "control", "cma"],
            "time_to_info": [120.0, 90.0, 200.0, 150.0],
            "condition_code": [0, 1, 0, 1],
        })
        session_ann = pd.DataFrame({
            "vignette_id": ["V001", "V001"],
            "condition": ["control", "cma"],
            "trust": [4, 6],
            "annotator_id": ["expert_001", "expert_001"],
        })
        merged = merge_with_results(results, {"session_annotations": session_ann})
        self.assertIn("trust", merged.columns)
        self.assertEqual(merged.loc[0, "trust"], 4)

    def test_apply_adjudications(self):
        results = pd.DataFrame({
            "vignette_id": ["V001", "V001"],
            "condition": ["control", "cma"],
            "accuracy": [0, 1],
        })
        adj = pd.DataFrame({
            "vignette_id": ["V001"],
            "disputed_field": ["accuracy"],
            "adjudicated_value": ["1"],
        })
        merged = apply_adjudications(results, adj)
        self.assertEqual(merged.loc[0, "accuracy"], 1.0)

    def test_summary(self):
        sa = pd.DataFrame({
            "vignette_id": ["V001", "V001", "V002", "V002"],
            "condition": ["control", "cma", "control", "cma"],
            "trust": [3, 5, 4, 6],
        })
        ann = {"session_annotations": sa}
        summ = summary(ann)
        self.assertIn("trust", summ)
        self.assertEqual(summ["trust"]["control_mean"], 3.5)
        self.assertEqual(summ["trust"]["cma_mean"], 5.5)

    def test_load_annotations(self):
        with tempfile.TemporaryDirectory() as td:
            sa_path = Path(td) / "session_annotations.csv"
            with sa_path.open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(session_annotation_csv_columns())
                w.writerow(["ann1", "V001", "control", 4, "", "expert_001", "2026-01-01"])
            ann = load_annotations(Path(td))
            self.assertIn("session_annotations", ann)
            self.assertEqual(len(ann["session_annotations"]), 1)


class TestHumanAnnotationAnalysis(unittest.TestCase):
    def test_analyze_trust(self):
        df = pd.DataFrame({
            "vignette_id": ["V001", "V001", "V002", "V002", "V003", "V003"],
            "condition": ["control", "gdt", "control", "gdt", "control", "gdt"],
            "trust": [3, 5, 4, 6, 2, 5],
        })
        result = human_annotation_analysis(df)
        self.assertIn("gdt_trust", result)
        self.assertIn("summary", result["gdt_trust"])
        self.assertIn("cohens_d", result["gdt_trust"])


if __name__ == "__main__":
    unittest.main()
