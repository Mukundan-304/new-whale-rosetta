"""Compare species classifiers with repeated nested location-group CV."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from watkins_embedding_baseline import (
    OUTPUT_DIR,
    _aggregate_recordings,
    _read_metadata,
)


OUTER_FOLDS = 5
INNER_FOLDS = 3
REPEAT_SEEDS = (29, 47)
REGULARIZATION_VALUES = (0.1, 1.0, 10.0)


def _score(
    y_true: list[str],
    y_predicted: list[str],
    train_labels: set[str],
) -> dict[str, Any]:
    if len(y_true) != len(y_predicted):
        raise ValueError("True and predicted labels must have equal lengths")
    if not y_true:
        raise ValueError("Cannot score empty validation labels")

    validation_counts = Counter(y_true)
    seen_mask = [label in train_labels for label in y_true]
    seen_true = [label for label, seen in zip(y_true, seen_mask) if seen]
    seen_predicted = [
        prediction
        for prediction, seen in zip(y_predicted, seen_mask)
        if seen
    ]
    evaluated_labels = sorted(set(seen_true))
    per_label: dict[str, tuple[float, float, float]] = {}
    for label in evaluated_labels:
        true_positive = sum(
            actual == label and predicted == label
            for actual, predicted in zip(seen_true, seen_predicted)
        )
        false_positive = sum(
            actual != label and predicted == label
            for actual, predicted in zip(seen_true, seen_predicted)
        )
        support = sum(actual == label for actual in seen_true)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = true_positive / support
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_label[label] = (precision, recall, f1)

    return {
        "validation_record_count": len(y_true),
        "validation_species_count": len(validation_counts),
        "seen_record_count": len(seen_true),
        "unseen_species_record_count": len(y_true) - len(seen_true),
        "species_coverage": len(seen_true) / len(y_true),
        "accuracy_all_records": accuracy_score(y_true, y_predicted),
        "accuracy_seen_species": (
            accuracy_score(seen_true, seen_predicted) if seen_true else ""
        ),
        "macro_precision_seen_species": (
            float(np.mean([values[0] for values in per_label.values()]))
            if per_label
            else ""
        ),
        "macro_recall_seen_species": (
            float(np.mean([values[1] for values in per_label.values()]))
            if per_label
            else ""
        ),
        "macro_f1_seen_species": (
            float(np.mean([values[2] for values in per_label.values()]))
            if per_label
            else ""
        ),
    }


def _candidates() -> list[dict[str, Any]]:
    return [
        *(
            {
                "model": "logistic_regression",
                "class_weight": "balanced",
                "C": regularization,
            }
            for regularization in REGULARIZATION_VALUES
        ),
        {
            "model": "logistic_regression",
            "class_weight": None,
            "C": 1.0,
        },
    ]


def _fit_candidate(
    parameters: dict[str, Any],
    train_features: np.ndarray,
    train_labels: list[str],
) -> Any:
    if parameters["model"] == "logistic_regression":
        estimator = LogisticRegression(
            C=parameters["C"],
            class_weight=parameters["class_weight"],
            max_iter=1000,
            random_state=29,
        )
    else:
        raise ValueError(f"Unknown classifier: {parameters['model']}")

    pipeline = make_pipeline(StandardScaler(), estimator)
    pipeline.fit(train_features, train_labels)
    return pipeline


def _make_splits(
    labels: list[str],
    groups: list[str],
    split_count: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(
        n_splits=split_count,
        shuffle=True,
        random_state=seed,
    )
    placeholder_features = np.zeros((len(labels), 1), dtype=np.float32)
    return list(splitter.split(placeholder_features, labels, groups))


def _select_candidate(
    features: np.ndarray,
    labels: list[str],
    groups: list[str],
    seed: int,
) -> tuple[dict[str, Any], float]:
    inner_splits = _make_splits(labels, groups, INNER_FOLDS, seed)
    candidate_scores: list[tuple[float, int, dict[str, Any]]] = []
    for candidate_index, parameters in enumerate(_candidates()):
        fold_scores = []
        for train_indices, validation_indices in inner_splits:
            train_labels = [labels[index] for index in train_indices]
            validation_labels = [
                labels[index] for index in validation_indices
            ]
            estimator = _fit_candidate(
                parameters,
                features[train_indices],
                train_labels,
            )
            predictions = estimator.predict(features[validation_indices])
            score = _score(
                validation_labels,
                [str(value) for value in predictions],
                set(train_labels),
            )
            if score["macro_f1_seen_species"] != "":
                fold_scores.append(score["macro_f1_seen_species"])
        if not fold_scores:
            raise ValueError(
                f"No inner fold could score candidate {parameters}"
            )
        candidate_scores.append(
            (float(np.mean(fold_scores)), -candidate_index, parameters)
        )
    best_score, _, best_parameters = max(candidate_scores)
    return best_parameters, best_score


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_grouped_species_cv(
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
    eligible_indices = [
        index
        for index, row in enumerate(record_rows)
        if not row["species_label"].startswith("[UNLABELED]")
    ]
    labels = [
        record_rows[index]["species_label"] for index in eligible_indices
    ]
    groups = [
        record_rows[index]["location_group"] for index in eligible_indices
    ]
    unique_groups = set(groups)
    if len(unique_groups) < OUTER_FOLDS:
        raise ValueError(
            f"Need at least {OUTER_FOLDS} location groups; found {len(unique_groups)}"
        )

    representations = {
        "clap": clap_features[eligible_indices],
        "full_band": full_band_features[eligible_indices],
    }
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    selection_counts: Counter[tuple[str, str, str, float]] = Counter()
    repeats: list[dict[str, Any]] = []

    for representation, features in representations.items():
        for repeat_index, seed in enumerate(REPEAT_SEEDS, start=1):
            outer_splits = _make_splits(
                labels, groups, OUTER_FOLDS, seed
            )
            repeat_fold_scores: list[dict[str, Any]] = []
            for fold_index, (train_indices, test_indices) in enumerate(
                outer_splits, start=1
            ):
                train_labels = [labels[index] for index in train_indices]
                test_labels = [labels[index] for index in test_indices]
                train_groups = {groups[index] for index in train_indices}
                test_groups = {groups[index] for index in test_indices}
                if train_groups & test_groups:
                    raise RuntimeError("Location group leaked across a fold")

                selected, inner_score = _select_candidate(
                    features[train_indices],
                    train_labels,
                    [groups[index] for index in train_indices],
                    seed + fold_index,
                )
                selected_model = _fit_candidate(
                    selected, features[train_indices], train_labels
                )
                selected_predictions = [
                    str(value)
                    for value in selected_model.predict(features[test_indices])
                ]

                majority_label = Counter(train_labels).most_common(1)[0][0]
                majority_predictions = [majority_label] * len(test_indices)
                fixed_baseline = _fit_candidate(
                    {
                        "model": "logistic_regression",
                        "class_weight": "balanced",
                        "C": 1.0,
                    },
                    features[train_indices],
                    train_labels,
                )
                baseline_predictions = [
                    str(value)
                    for value in fixed_baseline.predict(features[test_indices])
                ]
                model_predictions = (
                    ("majority", majority_predictions),
                    ("fixed_balanced_logistic_c1", baseline_predictions),
                    ("nested_selected", selected_predictions),
                )
                selected_key = (
                    representation,
                    selected["model"],
                    selected["class_weight"] or "none",
                    float(selected["C"]),
                )
                selection_counts[selected_key] += 1
                print(
                    f"{representation}: repeat {repeat_index}, fold "
                    f"{fold_index}/{OUTER_FOLDS} selected "
                    f"{selected['model']} "
                    f"(weight={selected['class_weight'] or 'none'}, "
                    f"C={selected['C']:g})",
                    flush=True,
                )

                for model_name, predictions in model_predictions:
                    metrics = _score(
                        test_labels,
                        predictions,
                        set(train_labels),
                    )
                    row = {
                        "representation": representation,
                        "repeat": repeat_index,
                        "seed": seed,
                        "fold": fold_index,
                        "model": model_name,
                        "selected_algorithm": selected["model"],
                        "selected_class_weight": (
                            selected["class_weight"] or "none"
                        ),
                        "selected_C": selected["C"],
                        "inner_macro_f1_seen_species": inner_score,
                        "train_location_groups": len(train_groups),
                        "validation_location_groups": len(test_groups),
                        **metrics,
                    }
                    fold_rows.append(row)
                    repeat_fold_scores.append(row)
                    if model_name == "nested_selected":
                        for position, eligible_index in enumerate(test_indices):
                            record_index = eligible_indices[eligible_index]
                            record = record_rows[record_index]
                            prediction_rows.append(
                                {
                                    "representation": representation,
                                    "repeat": repeat_index,
                                    "seed": seed,
                                    "fold": fold_index,
                                    "record_number": record["record_number"],
                                    "species_label": record["species_label"],
                                    "location_group": record["location_group"],
                                    "predicted_species": predictions[position],
                                    "correct": (
                                        test_labels[position]
                                        == predictions[position]
                                    ),
                                    "species_seen_in_training": (
                                        test_labels[position] in set(train_labels)
                                    ),
                                    "selected_algorithm": selected["model"],
                                    "selected_class_weight": (
                                        selected["class_weight"] or "none"
                                    ),
                                    "selected_C": selected["C"],
                                }
                            )

            for model_name in (
                "majority",
                "fixed_balanced_logistic_c1",
                "nested_selected",
            ):
                model_rows = [
                    row
                    for row in repeat_fold_scores
                    if row["model"] == model_name
                ]
                repeats.append(
                    {
                        "representation": representation,
                        "repeat": repeat_index,
                        "seed": seed,
                        "model": model_name,
                        "mean_fold_accuracy_all_records": float(
                            np.mean(
                                [
                                    row["accuracy_all_records"]
                                    for row in model_rows
                                ]
                            )
                        ),
                        "std_fold_accuracy_all_records": float(
                            np.std(
                                [
                                    row["accuracy_all_records"]
                                    for row in model_rows
                                ],
                                ddof=1,
                            )
                        ),
                        "mean_fold_accuracy_seen_species": float(
                            np.mean(
                                [
                                    row["accuracy_seen_species"]
                                    for row in model_rows
                                ]
                            )
                        ),
                        "mean_fold_macro_f1_seen_species": float(
                            np.mean(
                                [
                                    row["macro_f1_seen_species"]
                                    for row in model_rows
                                ]
                            )
                        ),
                        "mean_fold_species_coverage": float(
                            np.mean(
                                [row["species_coverage"] for row in model_rows]
                            )
                        ),
                    }
                )

    grouped_summary: list[dict[str, Any]] = []
    summary_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in repeats:
        summary_groups[(row["representation"], row["model"])].append(row)
    for (representation, model_name), rows in sorted(summary_groups.items()):
        grouped_summary.append(
            {
                "representation": representation,
                "model": model_name,
                "repeat_count": len(rows),
                "mean_accuracy_all_records": float(
                    np.mean(
                        [row["mean_fold_accuracy_all_records"] for row in rows]
                    )
                ),
                "std_accuracy_all_records_across_repeats": float(
                    np.std(
                        [row["mean_fold_accuracy_all_records"] for row in rows],
                        ddof=1,
                    )
                ),
                "mean_accuracy_seen_species": float(
                    np.mean(
                        [row["mean_fold_accuracy_seen_species"] for row in rows]
                    )
                ),
                "mean_macro_f1_seen_species": float(
                    np.mean(
                        [row["mean_fold_macro_f1_seen_species"] for row in rows]
                    )
                ),
                "mean_species_coverage": float(
                    np.mean(
                        [row["mean_fold_species_coverage"] for row in rows]
                    )
                ),
            }
        )

    confusion_counts: Counter[tuple[str, str, str]] = Counter()
    species_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in prediction_rows:
        key = (row["representation"], row["species_label"])
        counts = species_counts[key]
        counts["validation_records"] += 1
        counts["correct_predictions"] += int(row["correct"])
        counts["seen_records"] += int(row["species_seen_in_training"])
        counts["seen_correct"] += int(
            row["correct"] and row["species_seen_in_training"]
        )
        if row["species_seen_in_training"] and not row["correct"]:
            confusion_counts[
                (
                    row["representation"],
                    row["species_label"],
                    row["predicted_species"],
                )
            ] += 1
    per_species_rows = [
        {
            "representation": representation,
            "species_label": species,
            **counts,
            "seen_species_accuracy": (
                counts["seen_correct"] / counts["seen_records"]
                if counts["seen_records"]
                else ""
            ),
        }
        for (representation, species), counts in sorted(species_counts.items())
    ]
    confusion_rows = [
        {
            "representation": representation,
            "true_species": true_species,
            "predicted_species": predicted_species,
            "error_count_across_repeats": count,
        }
        for (representation, true_species, predicted_species), count
        in confusion_counts.most_common()
    ]
    selection_rows = [
        {
            "representation": key[0],
            "selected_algorithm": key[1],
            "selected_class_weight": key[2],
            "selected_C": key[3],
            "outer_fold_selection_count": count,
        }
        for key, count in sorted(selection_counts.items())
    ]

    output_paths = {
        "folds_path": output_dir / "watkins_species_cv_folds.csv",
        "summary_path": output_dir / "watkins_species_cv_summary.csv",
        "predictions_path": output_dir / "watkins_species_cv_predictions.csv",
        "per_species_path": output_dir / "watkins_species_cv_per_species.csv",
        "confusions_path": output_dir / "watkins_species_cv_confusions.csv",
        "selection_path": output_dir / "watkins_species_cv_selections.csv",
    }
    _write_csv(output_paths["folds_path"], fold_rows)
    _write_csv(output_paths["summary_path"], grouped_summary)
    _write_csv(output_paths["predictions_path"], prediction_rows)
    _write_csv(output_paths["per_species_path"], per_species_rows)
    _write_csv(output_paths["confusions_path"], confusion_rows)
    _write_csv(output_paths["selection_path"], selection_rows)

    all_species_groups: dict[str, set[str]] = defaultdict(set)
    for label, group in zip(labels, groups):
        all_species_groups[label].add(group)
    single_group_species = sorted(
        species for species, species_groups in all_species_groups.items()
        if len(species_groups) == 1
    )
    report_path = output_dir / "watkins_species_cv_report.md"
    report_lines = [
        "# Watkins species identification: repeated location-grouped evaluation",
        "",
        f"- Labeled recordings evaluated: {len(labels):,}.",
        f"- Species labels: {len(all_species_groups)} across {len(unique_groups)} location groups.",
        f"- Repeated outer evaluation: {OUTER_FOLDS} folds × {len(REPEAT_SEEDS)} repeats.",
        f"- Hyperparameters selected by {INNER_FOLDS}-fold grouped cross-validation inside each outer training set.",
        "- All chunks were averaged per original recording. Scaling and model fitting were confined to training folds.",
        "- The primary comparison metric is macro-F1 among species represented in each training fold. Accuracy and the fraction of validation recordings from seen species are also reported.",
        f"- {len(single_group_species)} species occur in only one location group and cannot be tested as known species when that group is held out.",
        "- Accuracy and macro-F1 in the table are unweighted means across outer folds; the reported repeat SD describes variation between the two repeat-level means, not a confidence interval.",
        "",
        "## Results",
        "",
        "| Features | Evaluation | Mean fold accuracy, all records | Repeat SD | Mean fold accuracy, seen species | Mean fold macro-F1, seen species | Mean seen-species coverage |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in grouped_summary:
        report_lines.append(
            f"| {row['representation']} | {row['model']} | "
            f"{row['mean_accuracy_all_records']:.3f} | "
            f"{row['std_accuracy_all_records_across_repeats']:.3f} | "
            f"{row['mean_accuracy_seen_species']:.3f} | "
            f"{row['mean_macro_f1_seen_species']:.3f} | "
            f"{row['mean_species_coverage']:.1%} |"
        )
    report_lines.extend(
        [
            "",
            "The nested-selected row is an estimate of the full grouped model-selection procedure, not the score of a single fixed final model. The fixed balanced logistic row is directly comparable to the earlier one-split baseline.",
            "",
            "## Limitations",
            "",
            "- Location groups are a geographic proxy, not verified recording sessions.",
            "- One-location species are not learnable under a location-held-out design; see the per-species results for their coverage.",
            "- Even nested group cross-validation is an internal estimate. Independent recording-site or session metadata would provide a stronger test.",
            "",
            "Detailed folds, predictions, per-species scores, confusion pairs, and selected hyperparameters are in the accompanying CSV files.",
            "",
        ]
    )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return {
        **output_paths,
        "report_path": report_path,
        "summary": grouped_summary,
        "fold_count": len(REPEAT_SEEDS) * OUTER_FOLDS,
        "single_group_species": single_group_species,
    }


def run_fixed_weighting_comparison(
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
    eligible_indices = [
        index
        for index, row in enumerate(record_rows)
        if not row["species_label"].startswith("[UNLABELED]")
    ]
    labels = [
        record_rows[index]["species_label"] for index in eligible_indices
    ]
    groups = [
        record_rows[index]["location_group"] for index in eligible_indices
    ]
    fold_rows: list[dict[str, Any]] = []

    for representation, features in (
        ("clap", clap_features[eligible_indices]),
        ("full_band", full_band_features[eligible_indices]),
    ):
        for repeat_index, seed in enumerate(REPEAT_SEEDS, start=1):
            splits = _make_splits(labels, groups, OUTER_FOLDS, seed)
            for fold_index, (train_indices, test_indices) in enumerate(
                splits, start=1
            ):
                train_labels = [labels[index] for index in train_indices]
                test_labels = [labels[index] for index in test_indices]
                train_groups = {groups[index] for index in train_indices}
                test_groups = {groups[index] for index in test_indices}
                if train_groups & test_groups:
                    raise RuntimeError("Location group leaked across a fold")
                for weight_name, class_weight in (
                    ("balanced", "balanced"),
                    ("unweighted", None),
                ):
                    estimator = _fit_candidate(
                        {
                            "model": "logistic_regression",
                            "class_weight": class_weight,
                            "C": 1.0,
                        },
                        features[train_indices],
                        train_labels,
                    )
                    predictions = [
                        str(value)
                        for value in estimator.predict(features[test_indices])
                    ]
                    fold_rows.append(
                        {
                            "representation": representation,
                            "repeat": repeat_index,
                            "seed": seed,
                            "fold": fold_index,
                            "class_weight": weight_name,
                            "train_location_groups": len(train_groups),
                            "validation_location_groups": len(test_groups),
                            **_score(
                                test_labels,
                                predictions,
                                set(train_labels),
                            ),
                        }
                    )

    summary_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in fold_rows:
        grouped[(row["representation"], row["class_weight"])].append(row)
    for (representation, class_weight), rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "representation": representation,
                "class_weight": class_weight,
                "fold_count": len(rows),
                "mean_accuracy_all_records": float(
                    np.mean([row["accuracy_all_records"] for row in rows])
                ),
                "std_accuracy_all_records": float(
                    np.std(
                        [row["accuracy_all_records"] for row in rows],
                        ddof=1,
                    )
                ),
                "mean_accuracy_seen_species": float(
                    np.mean([row["accuracy_seen_species"] for row in rows])
                ),
                "mean_macro_f1_seen_species": float(
                    np.mean([row["macro_f1_seen_species"] for row in rows])
                ),
                "mean_species_coverage": float(
                    np.mean([row["species_coverage"] for row in rows])
                ),
            }
        )

    paired: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in fold_rows:
        paired[(row["representation"], row["repeat"], row["fold"])][
            row["class_weight"]
        ] = row
    delta_rows: list[dict[str, Any]] = []
    for representation in ("clap", "full_band"):
        fold_deltas = [
            (
                pair["unweighted"]["accuracy_all_records"]
                - pair["balanced"]["accuracy_all_records"],
                pair["unweighted"]["macro_f1_seen_species"]
                - pair["balanced"]["macro_f1_seen_species"],
            )
            for (rep, _, _), pair in paired.items()
            if rep == representation
        ]
        delta_rows.append(
            {
                "representation": representation,
                "unweighted_minus_balanced_accuracy_mean": float(
                    np.mean([delta[0] for delta in fold_deltas])
                ),
                "unweighted_minus_balanced_accuracy_fold_sd": float(
                    np.std([delta[0] for delta in fold_deltas], ddof=1)
                ),
                "unweighted_minus_balanced_macro_f1_mean": float(
                    np.mean([delta[1] for delta in fold_deltas])
                ),
                "unweighted_minus_balanced_macro_f1_fold_sd": float(
                    np.std([delta[1] for delta in fold_deltas], ddof=1)
                ),
            }
        )

    folds_path = output_dir / "watkins_species_cv_weighting_folds.csv"
    summary_path = output_dir / "watkins_species_cv_weighting_summary.csv"
    deltas_path = output_dir / "watkins_species_cv_weighting_deltas.csv"
    _write_csv(folds_path, fold_rows)
    _write_csv(summary_path, summary_rows)
    _write_csv(deltas_path, delta_rows)
    report_path = output_dir / "watkins_species_cv_weighting_report.md"
    report_lines = [
        "# Species-class weighting comparison",
        "",
        f"Balanced and unweighted logistic regression (C=1) were compared on the same {OUTER_FOLDS}-fold grouped splits across {len(REPEAT_SEEDS)} repeats. All scaling and fitting used training groups only. Means below average fold-level metrics; Fold SD is the standard deviation across the 10 outer folds, not a confidence interval.",
        "",
        "| Features | Class weighting | Mean fold accuracy, all records | Fold SD | Mean fold accuracy, seen species | Mean fold macro-F1, seen species | Mean coverage |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        report_lines.append(
            f"| {row['representation']} | {row['class_weight']} | "
            f"{row['mean_accuracy_all_records']:.3f} | "
            f"{row['std_accuracy_all_records']:.3f} | "
            f"{row['mean_accuracy_seen_species']:.3f} | "
            f"{row['mean_macro_f1_seen_species']:.3f} | "
            f"{row['mean_species_coverage']:.1%} |"
        )
    report_lines.extend(
        [
            "",
            "## Paired unweighted-minus-balanced difference",
            "",
            "| Features | Accuracy difference (mean ± fold SD) | Macro-F1 difference (mean ± fold SD) |",
            "|---|---:|---:|",
        ]
    )
    for row in delta_rows:
        report_lines.append(
            f"| {row['representation']} | "
            f"{row['unweighted_minus_balanced_accuracy_mean']:+.3f} ± "
            f"{row['unweighted_minus_balanced_accuracy_fold_sd']:.3f} | "
            f"{row['unweighted_minus_balanced_macro_f1_mean']:+.3f} ± "
            f"{row['unweighted_minus_balanced_macro_f1_fold_sd']:.3f} |"
        )
    report_lines.extend(
        [
            "",
            "Fold-level measurements are in `watkins_species_cv_weighting_folds.csv`; paired differences are in `watkins_species_cv_weighting_deltas.csv`.",
            "",
        ]
    )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return {
        "folds_path": folds_path,
        "summary_path": summary_path,
        "deltas_path": deltas_path,
        "report_path": report_path,
        "summary": summary_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate species classifiers using nested location-grouped CV."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--weighting-comparison-only",
        action="store_true",
        help="Compare balanced vs unweighted logistic regression on the same grouped folds.",
    )
    args = parser.parse_args()
    if args.weighting_comparison_only:
        result = run_fixed_weighting_comparison(args.output_dir)
        print(f"Class-weighting comparison completed: {result['report_path']}")
        for row in result["summary"]:
            print(
                f"{row['representation']} / {row['class_weight']}: "
                f"accuracy={row['mean_accuracy_all_records']:.3f} "
                f"+/- {row['std_accuracy_all_records']:.3f}, "
                f"macro-F1={row['mean_macro_f1_seen_species']:.3f}"
            )
        return
    result = run_grouped_species_cv(args.output_dir)
    print(f"Grouped species evaluation completed: {result['report_path']}")
    for row in result["summary"]:
        print(
            f"{row['representation']} / {row['model']}: "
            f"accuracy={row['mean_accuracy_all_records']:.3f} "
            f"+/- {row['std_accuracy_all_records_across_repeats']:.3f}, "
            f"macro-F1-seen={row['mean_macro_f1_seen_species']:.3f}, "
            f"coverage={row['mean_species_coverage']:.1%}"
        )


if __name__ == "__main__":
    main()
