"""Unit tests for Stage 8: Environment Robustness Analysis."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from watkins_stage8.run_stage8_robustness import (
    analyze_species_location_confounding,
    build_environment_analysis_csv,
    perform_metadata_audit,
)


class WatkinsStage8RobustnessTests(unittest.TestCase):
    def test_perform_metadata_audit_flags_missing_variables(self) -> None:
        synthetic_meta = pd.DataFrame(
            {
                "recording_id": [1, 2, 3],
                "location_domain": ["loc1", "loc2", None],
                "location_names": ["Name 1", "Name 2", None],
                "location_coordinates": ['[{"lat": 1.0, "lon": 2.0}]', None, None],
                "observation_date": ["1960-01-01T00:00:00Z", None, "1980-01-01T00:00:00Z"],
                "species_code": ["SP1", "SP2", None],
                "sound_channel_source_code": ["11A", "41B", None],
                "source_sample_rate_hz": [10000, 20000, 10000],
                "signal_source_code": ["OTF", "N", None],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            audit_df = perform_metadata_audit(synthetic_meta, Path(tmp_dir))
            self.assertTrue((Path(tmp_dir) / "metadata_audit.csv").exists())

            # Check that critical missing oceanographic/sensor variables are identified
            missing_entries = audit_df[~audit_df["available_in_dataset"]]
            self.assertGreaterEqual(len(missing_entries), 5)
            self.assertIn("hydrophone_model_sensitivity", missing_entries["field_name"].values)
            self.assertIn("water_temp_salinity_ssp", missing_entries["field_name"].values)
            self.assertIn("verified_deployment_session_id", missing_entries["field_name"].values)

    def test_analyze_species_location_confounding_detects_single_location_species(self) -> None:
        synthetic_meta = pd.DataFrame(
            {
                "species_code": ["SP_A", "SP_A", "SP_B", "SP_B", "SP_C"],
                "location_domain": ["LOC_1", "LOC_1", "LOC_2", "LOC_3", "LOC_3"],
            }
        )
        confounding = analyze_species_location_confounding(synthetic_meta)

        self.assertEqual(confounding["total_species"], 3)
        self.assertEqual(confounding["species_single_location"], 2)  # SP_A (LOC_1), SP_C (LOC_3)
        self.assertEqual(confounding["species_multi_location"], 1)   # SP_B (LOC_2, LOC_3)
        self.assertAlmostEqual(confounding["species_single_location_pct"], 200.0 / 3.0)
        self.assertGreater(confounding["cramer_v"], 0.5)

    def test_build_environment_analysis_csv_schema(self) -> None:
        confounding = {
            "cramer_v": 0.93,
            "nmi": 0.77,
            "ari": 0.61,
            "species_single_location_pct": 26.4,
            "locations_single_species_pct": 50.0,
            "top_location_share_pct": 21.3,
            "top_5_locations_share_pct": 35.4,
        }
        cluster_results = {
            "cluster_location_nmi": 0.68,
            "cluster_location_ari": 0.15,
        }
        novelty_results = {
            "kruskal_wallis_h": 3500.0,
            "kruskal_wallis_p": 0.0,
        }
        pairwise = {
            "same_species_same_location": {"mean": 0.69},
            "same_species_different_location": {"mean": 0.51},
            "different_species_same_location": {"mean": 0.51},
            "different_species_different_location": {"mean": 0.32},
            "same_species_same_channel": {"mean": 0.67},
            "same_species_different_channel": {"mean": 0.48},
            "location_gap_mean": 0.18,
            "location_boost_mean": 0.19,
            "channel_gap_mean": 0.19,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_df = build_environment_analysis_csv(
                confounding, cluster_results, novelty_results, pairwise, Path(tmp_dir)
            )
            self.assertTrue((Path(tmp_dir) / "environment_analysis.csv").exists())
            self.assertIn("metric_group", csv_df.columns)
            self.assertIn("metric_name", csv_df.columns)
            self.assertIn("value", csv_df.columns)
            self.assertGreater(len(csv_df), 10)


if __name__ == "__main__":
    unittest.main()
