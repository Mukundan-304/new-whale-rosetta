"""Run the streaming Watkins Parquet dataset audit."""

import argparse
from pathlib import Path

from marine_data import analyze_dataset


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Inspect Parquet audio, species imbalance, and metadata quality."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_dir / "watkins-marine-mammal-full-cuts" / "data",
        help="Directory containing Parquet shards.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir,
        help="Directory for CSV reports, charts, and the strategy summary.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Rows per Arrow batch while scanning the dataset.",
    )
    args = parser.parse_args()

    result = analyze_dataset(args.data_dir, args.output_dir, args.batch_size)
    print(f"Scanned {result['record_count']:,} records across {result['quality']['shards']} shards.")
    print(f"Species label groups: {result['species_count']}")
    print(f"Inferred label field: {result['schema'].label or 'not detected'}")
    print(f"Inferred audio field: {result['schema'].audio or 'not detected'}")
    print(f"Reports and charts: {result['output_dir']}")
    for chart in result["charts"]:
        print(f"  {chart}")
    print("  species_distribution.csv")
    print("  dataset_quality_report.csv")
    print("  imbalance_strategy_report.md")


if __name__ == "__main__":
    main()
