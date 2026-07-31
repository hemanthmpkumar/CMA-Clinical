import unittest

import pandas as pd

from src.experiments.ablation import ABLATION_CONFIGS, ablation_summary


class TestAblationConfigs(unittest.TestCase):
    def test_three_variants(self):
        self.assertEqual(len(ABLATION_CONFIGS), 3)
        self.assertIn("full_cma", ABLATION_CONFIGS)
        self.assertIn("gsi_only", ABLATION_CONFIGS)
        self.assertIn("jepa_only", ABLATION_CONFIGS)

    def test_full_cma_defaults(self):
        cfg = ABLATION_CONFIGS["full_cma"]
        self.assertEqual(cfg["curvature_threshold"], 0.65)
        self.assertEqual(cfg["gate_discount"], 0.05)
        self.assertEqual(cfg["prefetch_weight"], 0.4)

    def test_gsi_only_disables_prefetch(self):
        cfg = ABLATION_CONFIGS["gsi_only"]
        self.assertEqual(cfg["prefetch_weight"], 0.0)
        self.assertEqual(cfg["curvature_threshold"], 0.65)

    def test_jepa_only_disables_gate(self):
        cfg = ABLATION_CONFIGS["jepa_only"]
        self.assertEqual(cfg["curvature_threshold"], float("inf"))
        self.assertEqual(cfg["prefetch_weight"], 0.4)

    def test_each_has_label(self):
        for key, cfg in ABLATION_CONFIGS.items():
            self.assertIn("label", cfg)
            self.assertIsInstance(cfg["label"], str)
            self.assertTrue(len(cfg["label"]) > 0)


class TestAblationSummary(unittest.TestCase):
    def setUp(self):
        variants = ["Full CMA", "GSI only", "JEPA only"]
        rows = []
        for v in variants:
            for i in range(4):
                rows.append({
                    "ablation_variant": v,
                    "condition": "control" if i % 2 == 0 else "cma",
                    "time_to_info": 100.0 + i * 10,
                    "accuracy": 0.5 + i * 0.1,
                    "cognitive_load": 50.0 + i * 5,
                    "latency_ms": 100.0 + i * 20,
                    "n_queries_issued": 3.0 + i * 0.5,
                })
        self.df = pd.DataFrame(rows)

    def test_summary_returns_two_dfs(self):
        summary_df, change_df = ablation_summary(self.df)
        self.assertIsInstance(summary_df, pd.DataFrame)
        self.assertIsInstance(change_df, pd.DataFrame)

    def test_summary_has_all_variants(self):
        summary_df, _ = ablation_summary(self.df)
        variants = summary_df["variant"].unique()
        self.assertEqual(len(variants), 3)

    def test_summary_has_both_conditions(self):
        summary_df, _ = ablation_summary(self.df)
        conditions = summary_df["condition"].unique()
        self.assertIn("control", conditions)
        self.assertIn("cma", conditions)

    def test_change_has_pct_column(self):
        _, change_df = ablation_summary(self.df)
        self.assertIn("pct_change", change_df.columns)

    def test_change_reports_all_metrics(self):
        _, change_df = ablation_summary(self.df)
        metrics = change_df["metric"].unique()
        for m in ["time_to_info", "accuracy", "cognitive_load", "latency_ms"]:
            self.assertIn(m, metrics)


if __name__ == "__main__":
    unittest.main()
