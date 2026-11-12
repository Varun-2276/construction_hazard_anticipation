import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from safety_pipeline import (  # noqa: E402
    engineer_movement_features,
    identify_zone_events,
    normalized_zone_to_pixels,
)


class PipelineUnitTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for frame, x, y in [(0, 10, 10), (1, 11, 11), (2, 12, 12), (4, 30, 30), (5, 31, 31)]:
            rows.append({
                "video_id": "video_test", "frame": frame, "timestamp": frame / 10,
                "track_id": 7, "class_id": 0, "class_name": "person", "confidence": 0.9,
                "x1": x - 2, "y1": y - 5, "x2": x + 2, "y2": y,
                "center_x": x, "center_y": y - 2.5, "bottom_x": x, "bottom_y": y,
            })
        self.people = pd.DataFrame(rows)

    def test_movement_summary_is_generated(self):
        people, summary = engineer_movement_features(self.people, 10)
        self.assertEqual(len(people), 5)
        self.assertEqual(len(summary), 1)
        self.assertEqual(int(summary.iloc[0]["track_id"]), 7)

    def test_zone_episode_is_grouped(self):
        people, _ = engineer_movement_features(self.people, 10)
        zone_rows, events = identify_zone_events(people, {"x1": 0, "y1": 0, "x2": 20, "y2": 20}, 10)
        self.assertEqual(len(zone_rows), 3)
        self.assertEqual(len(events), 1)
        self.assertEqual(events.iloc[0]["risk_level"], "Low")

    def test_zone_conversion(self):
        zone = normalized_zone_to_pixels(1920, 1080, 40, 40, 70, 80)
        self.assertEqual(zone, {"x1": 768, "y1": 432, "x2": 1344, "y2": 864})

    def test_invalid_zone_rejected(self):
        with self.assertRaises(ValueError):
            normalized_zone_to_pixels(1920, 1080, 70, 40, 40, 80)


if __name__ == "__main__":
    unittest.main()
