# BE3C location-held-out feasibility audit

- Species code: `BE3C` (long-finned pilot whale).
- Records: 1,169; location/date/sample-rate/channel metadata was audited. The two selected WAV previews were decoded successfully.
- Location name coverage: 931/1,169.
- Coordinate coverage: 1,167/1,169.
- Date coverage: 1,169/1,169.
- Sample-rate metadata coverage: 1,169/1,169.
- Sample-rate metadata values observed: 11; channel configurations: 12.
- `mid` values observed: {'whoi0039': 1104, 'whoi0051': 65}. The README documents `mid` as a species identifier, not recording-session metadata.
- Records with neither location names nor coordinates: 0; unlocated rows without a location grouping entity: 0. These remain in training.

## Split

- Train: 935 records.
- Validation: 234 records (20.0%).
- Location-connected groups: 11; validation groups: 2.
- Held-out location groups: W of Browns Bank (41,-66 | 43,-62): 172 records; Dildo Arm, Trinity Bay, Newfoundland, Canada (47,-53): 62 records.
- Group construction links rows sharing any normalized location name or rounded recorded coordinate.
- The split is deterministic and selects whole connected location groups nearest the requested 20% validation size; all source records remain represented exactly once in the CSV manifest.
- Dataset documentation describes `record_number` as the original Watkins recording identifier and `mid` as a species identifier. `mid` is not used as a session split key.

## Caveats

- Location coordinates are coarse (often rounded to whole degrees), and labels range from named places to broad regions. This is a preliminary geographic holdout, not proof of independent hydrophone/site conditions.
- No explicit recording-session field was found. Recording notes and `signal.position` may help investigate session relationships, but do not assume they define sessions without validating their semantics.
- Two WAV previews (up to 10 seconds each) and their spectrograms are provided for manual listening/visual review. The previous dataset audit fully decoded all audio and found 13 exact duplicates in the entire dataset; a BE3C-specific hash check found no duplicate payloads. This script decodes the selected previews but does not claim biological call identification.

## Files

- `be3c_location_split.csv`: one row per source record, assigned to train/validation.
- `be3c_location_groups.csv`: location groups and split assignment.
- `be3c_example_clips.csv`: metadata for representative clips.
- `be3c_examples/`: extracted train and validation WAV samples.
- `be3c_spectrograms.png`: representative log-frequency spectrograms.

Original Parquet shards were read only and were not modified.
