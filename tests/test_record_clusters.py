import unittest

from src.data.record_clusters import pack_record_clusters


class PackRecordClustersTests(unittest.TestCase):
    def test_groups_rows_and_builds_compact_summaries(self):
        rows = [
            {"cluster": "alpha", "note": "Patient reported fever and cough after travel."},
            {"cluster": "beta", "note": "Blood pressure remained elevated throughout admission."},
            {"cluster": "alpha", "note": "Chest x-ray showed mild infiltrates."},
            {"cluster": "alpha", "note": "Medication was adjusted following review."},
        ]

        summary = pack_record_clusters(rows, "cluster", "note", max_examples=2)

        self.assertEqual(len(summary), 2)
        self.assertEqual(summary[0]["group"], "alpha")
        self.assertEqual(summary[0]["count"], 3)
        self.assertIn("Patient reported fever and cough after travel.", summary[0]["signature"])
        self.assertIn("fever", summary[0]["keywords"])
        self.assertEqual(len(summary[0]["examples"]), 2)
        self.assertEqual(summary[1]["group"], "beta")
        self.assertEqual(summary[1]["count"], 1)


if __name__ == "__main__":
    unittest.main()
