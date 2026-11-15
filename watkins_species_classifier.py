"""Evaluate frozen audio features for location-held-out species identification."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from watkins_embedding_baseline import (
    OUTPUT_DIR,
    _aggregate_recordings,
    _read_metadata,
)


def _classification_metrics(
    y_true: list[str],
    y_predicted: list[str],
    train_labels: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(y_true) != len(y_predicted):
        raise ValueError("True and predicted label counts must match")
    if not y_true:
        raise ValueError("Validation labels must not be empty")

    train_counts = Counter(train_labels)
    train_species = set(train_counts)
    validation_counts = Counter(y_true)
    validation_species = set(validation_counts)
    known_mask = [label in train_species for label in y_true]
    known_true = [label for label, known in zip(y_true, known_mask) if known]
    known_predicted = [
        prediction
        for prediction, known in zip(y_predicted, known_mask)
        if known
    ]
    labels = sorted(train_species | validation_species)
    per_species = []
    for label in labels:
        true_positive = sum(
            actual == label and predicted == label
            for actual, predicted in zip(y_true, y_predicted)
        )
        false_positive = sum(
            actual != label and predicted == label
            for actual, predicted in zip(y_true, y_predicted)
        )
        support = validation_counts[label]
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_species.append(
            {
                "species_label": label,
                "train_record_count": train_counts[label],
                "validation_record_count": support,
                "seen_in_training": label in train_species,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    if known_true:
        evaluated_species = sorted(set(known_true))
        per_species_by_label = {
            row["species_label"]: row for row in per_species
        }
        macro_precision = float(
            np.mean(
                [
                    per_species_by_label[label]["precision"]
                    for label in evaluated_species
                ]
            )
        )
        macro_recall = float(
            np.mean(
                [
                    per_species_by_label[label]["recall"]
                    for label in evaluated_species
                ]
            )
        )
        macro_f1 = float(
            np.mean(
                [
                    per_species_by_label[label]["f1"]
                    for label in evaluated_species
                ]
            )
        )
        known_accuracy = accuracy_score(known_true, known_predicted)
        known_balanced_accuracy = macro_recall
    else:
        macro_precision = macro_recall = macro_f1 = ""
        known_accuracy = known_balanced_accuracy = ""

    summary = {
        "train_record_count": len(train_labels),
        "train_species_count": len(train_species),
        "validation_record_count": len(y_true),
        "validation_species_count": len(validation_species),
        "validation_species_seen_in_train": len(
            validation_species & train_species
        ),
        "validation_unseen_species_count": len(
            validation_species - train_species
        ),
        "validation_records_seen_in_train_species": len(known_true),
        "validation_records_from_unseen_species": len(y_true) - len(known_true),
        "validation_species_coverage": len(known_true) / len(y_true),
        "validation_accuracy_all_labeled": accuracy_score(y_true, y_predicted),
        "validation_accuracy_seen_species": known_accuracy,
        "validation_balanced_accuracy_seen_species": known_balanced_accuracy,
        "validation_macro_precision_seen_species": macro_precision,
        "validation_macro_recall_seen_species": macro_recall,
        "validation_macro_f1_seen_species": macro_f1,
    }
    return summary, per_species


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_species_classification(
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    metadata_path = output_dir / "watkins_embedding_chunks.csv"
    features_path = output_dir / "watkins_embedding_features.npz"
    if not metadata_path.is_file() or not features_path.is_file():
        raise FileNotFoundError(
            "Feature files are missing; run watkins_embedding_baseline.py first"
        )
    with np.load(features_path) as stored:
        clap_chunks = stored["clap_embeddings"]
        full_band_chunks = stored["full_band_features"]
    record_rows, clap_features, full_band_features = _aggregate_recordings(
        _read_metadata(metadata_path), clap_chunks, full_band_chunks
    )
    train_indices = [
        index
        for index, row in enumerate(record_rows)
        if row["split"] == "train"
        and not row["species_label"].startswith("[UNLABELED]")
    ]
    validation_indices = [
        index
        for index, row in enumerate(record_rows)
        if row["split"] == "validation"
        and not row["species_label"].startswith("[UNLABELED]")
    ]
    if not train_indices or not validation_indices:
        raise ValueError("Need labeled records in both train and validation splits")

    y_train = [record_rows[index]["species_label"] for index in train_indices]
    y_validation = [
        record_rows[index]["species_label"] for index in validation_indices
    ]
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    species_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for representation, features in (
        ("clap", clap_features),
        ("full_band", full_band_features),
    ):
        scaler = StandardScaler()
        train_features = scaler.fit_transform(features[train_indices])
        validation_features = scaler.transform(features[validation_indices])
        majority_label = Counter(y_train).most_common(1)[0][0]
        models: list[tuple[str, np.ndarray]] = [
            ("majority", np.full(len(validation_indices), majority_label))
        ]
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=29,
        )
        classifier.fit(train_features, y_train)
        models.append(
            (
                "balanced_logistic_regression",
                classifier.predict(validation_features),
            )
        )

        for model_name, predictions_array in models:
            predictions = [str(label) for label in predictions_array]
            summary, per_species = _classification_metrics(
                y_validation, predictions, y_train
            )
            summaries.append(
                {
                    "representation": representation,
                    "model": model_name,
                    **summary,
                    "unlabeled_train_records_excluded": sum(
                        row["split"] == "train"
                        and row["species_label"].startswith("[UNLABELED]")
                        for row in record_rows
                    ),
                    "unlabeled_validation_records_excluded": sum(
                        row["split"] == "validation"
                        and row["species_label"].startswith("[UNLABELED]")
                        for row in record_rows
                    ),
                }
            )
            species_rows.extend(
                {
                    "representation": representation,
                    "model": model_name,
                    **row,
                }
                for row in per_species
            )
            prediction_rows.extend(
                {
                    "representation": representation,
                    "model": model_name,
                    "record_number": record_rows[index]["record_number"],
                    "location_group": record_rows[index]["location_group"],
                    "true_species": y_validation[position],
                    "predicted_species": predictions[position],
                    "correct": y_validation[position] == predictions[position],
                    "species_seen_in_training": (
                        y_validation[position] in set(y_train)
                    ),
                }
                for position, index in enumerate(validation_indices)
            )

    summary_path = output_dir / "watkins_species_classification_summary.csv"
    species_path = output_dir / "watkins_species_classification_per_species.csv"
    predictions_path = output_dir / "watkins_species_classification_predictions.csv"
    _write_csv(summary_path, summaries)
    _write_csv(species_path, species_rows)
    _write_csv(predictions_path, prediction_rows)

    report_path = output_dir / "watkins_species_classification_report.md"
    report_lines = [
        "# Watkins species-identification baseline",
        "",
        "This is a supervised benchmark using existing species labels and frozen audio features. "
        "It does not train the audio encoder.",
        "",
        "- All chunks from each recording were averaged into one recording-level feature vector.",
        "- The existing location-held-out training and validation split was preserved.",
        "- Feature scaling and classifiers were fitted on training records only.",
        "- The balanced logistic regression gives each training species equal total class weight.",
        "- The majority-class model is a simple reference baseline.",
        "- Unlabeled records were excluded. Validation species absent from training are counted "
        "as unseen and cannot be identified by these classifiers.",
        "- Validation accuracy across all labeled records and accuracy on seen species are "
        "reported separately; macro metrics are calculated over species represented in both "
        "training and validation.",
        "- This is one held-out split, not yet a final estimate of real-world performance.",
        "",
        "## Validation results",
        "",
        "| Features | Model | Validation accuracy (all labeled) | Accuracy (seen species) | Macro F1 (seen species) | Species coverage | Unseen validation species |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        accuracy = row["validation_accuracy_all_labeled"]
        seen_accuracy = row["validation_accuracy_seen_species"]
        macro_f1 = row["validation_macro_f1_seen_species"]
        report_lines.append(
            f"| {row['representation']} | {row['model']} | "
            f"{accuracy:.3f} | "
            f"{seen_accuracy if isinstance(seen_accuracy, str) else f'{seen_accuracy:.3f}'} | "
            f"{macro_f1 if isinstance(macro_f1, str) else f'{macro_f1:.3f}'} | "
            f"{row['validation_species_coverage']:.1%} | "
            f"{row['validation_unseen_species_count']} |"
        )
    report_lines.extend(
        [
            "",
            "Per-species results and validation predictions are in "
            "`watkins_species_classification_per_species.csv` and "
            "`watkins_species_classification_predictions.csv`.",
            "",
        ]
    )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return {
        "summary_path": summary_path,
        "per_species_path": species_path,
        "predictions_path": predictions_path,
        "report_path": report_path,
        "summaries": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen audio features for species identification."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    result = run_species_classification(args.output_dir)
    print(f"Species classification completed. Report: {result['report_path']}")
    for row in result["summaries"]:
        print(
            f"{row['representation']} / {row['model']}: "
            f"accuracy={row['validation_accuracy_all_labeled']:.3f}, "
            f"macro-F1-seen={row['validation_macro_f1_seen_species']}"
        )


if __name__ == "__main__":
    main()
