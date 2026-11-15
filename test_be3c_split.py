import unittest

from prepare_be3c_split import (
    _assign_connected_location_groups,
    _select_validation_groups,
)


class Be3cSplitTests(unittest.TestCase):
    def test_location_names_and_coordinates_keep_rows_together(self) -> None:
        records = [
            {
                "location_names": ["SE Sable Island"],
                "location_coordinates": [{"lat": 43, "lon": -59}],
            },
            {
                "location_names": ["SE Sable Islandx"],
                "location_coordinates": [{"lat": 43, "lon": -59}],
            },
            {
                "location_names": ["Gully"],
                "location_coordinates": [{"lat": 43, "lon": -58}],
            },
            {
                "location_names": ["Gully"],
                "location_coordinates": [{"lat": 43, "lon": -59}],
            },
            {
                "location_names": [],
                "location_coordinates": [{"lat": 39, "lon": -72}],
            },
        ]

        groups, unlocated = _assign_connected_location_groups(records)

        self.assertEqual(unlocated, 0)
        self.assertEqual(sorted(len(indices) for indices in groups.values()), [1, 4])

    def test_validation_selection_targets_fraction_without_splitting_groups(self) -> None:
        groups = {
            "large": list(range(506)),
            "sable": list(range(506, 746)),
            "browns": list(range(746, 918)),
            "dildo": list(range(918, 980)),
            "small": list(range(980, 1169)),
        }

        validation = _select_validation_groups(groups, 0.2)

        self.assertEqual(sum(len(groups[group]) for group in validation), 234)
        self.assertTrue(validation.issubset(groups))
        self.assertEqual(len(validation), 2)


if __name__ == "__main__":
    unittest.main()
