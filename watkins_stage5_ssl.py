"""Waveform-level contrastive learning with a frozen CLAP audio encoder."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from preprocess_watkins_audio import PreprocessingConfig
from watkins_pretrained_embeddings import (
    MODEL_SAMPLE_RATE_HZ,
    _embed_batch,
    _iter_waveforms,
    _load_encoder,
    _read_manifest,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "watkins_stage5_config.json"
OUTPUT_DIR = PROJECT_DIR / "watkins_stage5"
LOGGER = logging.getLogger("watkins_stage5")


class ProjectionHead(nn.Module):
    def __init__(
        self, input_dim: int = 512, hidden_dim: int = 512, output_dim: int = 128
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def nt_xent_loss(
    first: torch.Tensor, second: torch.Tensor, temperature: float
) -> torch.Tensor:
    if first.ndim != 2 or second.shape != first.shape:
        raise ValueError("Both views must have the same two-dimensional shape")
    if first.shape[0] < 2:
        raise ValueError("NT-Xent requires at least two source samples per batch")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    batch_size = first.shape[0]
    vectors = F.normalize(torch.cat((first, second), dim=0).float(), dim=1)
    logits = vectors @ vectors.T / temperature
    logits.fill_diagonal_(float("-inf"))
    targets = torch.cat(
        (
            torch.arange(batch_size, 2 * batch_size, device=first.device),
            torch.arange(batch_size, device=first.device),
        )
    )
    return F.cross_entropy(logits, targets)


def augment_waveforms(
    waveforms: torch.Tensor,
    generator: torch.Generator,
    sample_rate: int,
    max_shift_ms: float,
    gain_db: float,
    min_noise_snr_db: float,
    max_noise_snr_db: float,
) -> torch.Tensor:
    if waveforms.ndim != 2:
        raise ValueError("waveforms must have shape (batch, samples)")
    if not torch.isfinite(waveforms).all():
        raise ValueError("waveforms contain non-finite samples")

    views = waveforms.clone()
    max_shift = round(sample_rate * max_shift_ms / 1000)
    if max_shift:
        shifts = torch.randint(
            -max_shift,
            max_shift + 1,
            (views.shape[0],),
            generator=generator,
        )
        for index, shift_value in enumerate(shifts.tolist()):
            if shift_value > 0:
                views[index, shift_value:] = waveforms[index, :-shift_value]
                views[index, :shift_value] = 0
            elif shift_value < 0:
                views[index, :shift_value] = waveforms[index, -shift_value:]
                views[index, shift_value:] = 0

    if gain_db:
        gain = torch.empty(views.shape[0]).uniform_(
            -gain_db, gain_db, generator=generator
        )
        views *= torch.pow(10.0, gain / 20.0).unsqueeze(1)

    if max_noise_snr_db > 0:
        snr = torch.empty(views.shape[0]).uniform_(
            min_noise_snr_db, max_noise_snr_db, generator=generator
        )
        rms = views.square().mean(dim=1).sqrt().clamp_min(1e-8)
        noise_std = rms / torch.pow(10.0, snr / 20.0)
        noise = torch.randn(
            views.shape, generator=generator, dtype=views.dtype
        )
        views += noise * noise_std.unsqueeze(1)
    return views


def _read_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    required = {
        "model_id",
        "random_seed",
        "batch_size",
        "epochs",
        "early_stopping_patience",
        "learning_rate",
        "weight_decay",
        "temperature",
        "projection_hidden_dim",
        "projection_dim",
        "shuffle_buffer_size",
        "mixed_precision",
        "augmentations",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"{path} lacks config keys: {sorted(missing)}")
    if int(config["batch_size"]) < 2:
        raise ValueError("batch_size must be at least 2 for NT-Xent")
    if int(config["epochs"]) < 1:
        raise ValueError("epochs must be positive")
    return config


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _shuffled_waveforms(
    rows: list[dict[str, str]],
    preprocessing: PreprocessingConfig,
    buffer_size: int,
    seed: int,
) -> Iterator[np.ndarray]:
    rng = random.Random(seed)
    buffer: list[np.ndarray] = []
    for _, _, waveform in _iter_waveforms(rows, preprocessing):
        buffer.append(waveform)
        if len(buffer) >= buffer_size:
            index = rng.randrange(len(buffer))
            yield buffer.pop(index)
    while buffer:
        index = rng.randrange(len(buffer))
        yield buffer.pop(index)


def _waveform_batches(
    rows: list[dict[str, str]],
    preprocessing: PreprocessingConfig,
    batch_size: int,
    shuffle_buffer_size: int | None,
    seed: int,
) -> Iterator[list[np.ndarray]]:
    source: Iterator[np.ndarray]
    if shuffle_buffer_size is None:
        source = (
            waveform
            for _, _, waveform in _iter_waveforms(rows, preprocessing)
        )
    else:
        source = _shuffled_waveforms(
            rows, preprocessing, shuffle_buffer_size, seed
        )
    batch: list[np.ndarray] = []
    for waveform in source:
        batch.append(waveform)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _encode_views(
    model: Any,
    processor: Any,
    device: torch.device,
    waveforms: torch.Tensor,
    mixed_precision: bool,
) -> torch.Tensor:
    inputs = processor(
        audio=[item for item in waveforms.numpy()],
        sampling_rate=MODEL_SAMPLE_RATE_HZ,
        return_tensors="pt",
    )
    model_inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    with torch.no_grad():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda" and mixed_precision,
        ):
            output = model.get_audio_features(**model_inputs)
    features = output.pooler_output
    if features is None:
        raise RuntimeError("CLAP did not return pooled audio embeddings")
    features = F.normalize(features.float(), dim=-1)
    if not torch.isfinite(features).all():
        raise RuntimeError("CLAP produced non-finite features")
    return features


def _pair_views(
    waveforms: list[np.ndarray],
    seed: int,
    augmentations: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    samples = torch.from_numpy(np.stack(waveforms).astype(np.float32, copy=False))
    generator_one = torch.Generator().manual_seed(seed)
    generator_two = torch.Generator().manual_seed(seed + 1)
    arguments = {
        "sample_rate": MODEL_SAMPLE_RATE_HZ,
        "max_shift_ms": float(augmentations["max_time_shift_ms"]),
        "gain_db": float(augmentations["gain_db"]),
        "min_noise_snr_db": float(augmentations["min_noise_snr_db"]),
        "max_noise_snr_db": float(augmentations["max_noise_snr_db"]),
    }
    first = augment_waveforms(samples, generator=generator_one, **arguments)
    second = augment_waveforms(samples, generator=generator_two, **arguments)
    return first, second


def _loss_for_batch(
    waveforms: list[np.ndarray],
    model: Any,
    processor: Any,
    projector: ProjectionHead,
    device: torch.device,
    mixed_precision: bool,
    temperature: float,
    augmentations: dict[str, float],
    seed: int,
    train: bool,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
) -> float:
    first, second = _pair_views(waveforms, seed, augmentations)
    combined = torch.cat((first, second), dim=0)
    features = _encode_views(model, processor, device, combined, mixed_precision)
    batch_size = len(waveforms)

    if train:
        assert optimizer is not None
        optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda" and mixed_precision,
    ):
        projected = projector(features)
    first_projection, second_projection = projected.split(batch_size, dim=0)
    loss = nt_xent_loss(first_projection, second_projection, temperature)
    if train:
        assert optimizer is not None and scaler is not None
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    return float(loss.detach().item())


def _atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _checkpoint(
    projector: ProjectionHead,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    epoch: int,
    best_loss: float,
    best_epoch: int,
    patience_count: int,
    elapsed_training_seconds: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "projector": projector.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best_validation_loss": best_loss,
        "best_epoch": best_epoch,
        "patience_count": patience_count,
        "elapsed_training_seconds": elapsed_training_seconds,
        "config": config,
    }


def _run_epoch(
    split: str,
    rows: list[dict[str, str]],
    preprocessing: PreprocessingConfig,
    model: Any,
    processor: Any,
    projector: ProjectionHead,
    device: torch.device,
    config: dict[str, Any],
    epoch: int,
    optimizer: torch.optim.Optimizer | None,
    scaler: Any | None,
    max_batches: int | None = None,
) -> tuple[float, int, float]:
    training = split == "train"
    projector.train(training)
    batches = _waveform_batches(
        rows,
        preprocessing,
        int(config["batch_size"]),
        int(config["shuffle_buffer_size"]) if training else None,
        int(config["random_seed"]) + epoch,
    )
    weighted_loss = 0.0
    samples_seen = 0
    started = time.perf_counter()
    for batch_index, waveforms in enumerate(batches):
        if max_batches is not None and batch_index >= max_batches:
            break
        loss = _loss_for_batch(
            waveforms,
            model,
            processor,
            projector,
            device,
            bool(config["mixed_precision"]),
            float(config["temperature"]),
            config["augmentations"],
            int(config["random_seed"])
            + epoch * 1_000_003
            + batch_index * 2,
            training,
            optimizer,
            scaler,
        )
        weighted_loss += loss * len(waveforms)
        samples_seen += len(waveforms)
        if training and (batch_index + 1) % 100 == 0:
            LOGGER.info(
                "epoch=%d train_batch=%d samples=%d loss=%.5f gpu_peak_mib=%.1f",
                epoch,
                batch_index + 1,
                samples_seen,
                loss,
                torch.cuda.max_memory_allocated(device) / 1024**2
                if device.type == "cuda"
                else 0.0,
            )
    if not samples_seen:
        raise RuntimeError(f"No {split} batches were processed")
    return weighted_loss / samples_seen, samples_seen, time.perf_counter() - started


def _append_training_log(path: Path, row: dict[str, Any]) -> None:
    fields = [
        "epoch",
        "train_loss",
        "validation_loss",
        "train_samples",
        "validation_samples",
        "train_seconds",
        "validation_seconds",
        "cuda_peak_allocated_mib",
        "best_epoch",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train(
    config: dict[str, Any],
    output_dir: Path,
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
) -> tuple[ProjectionHead, dict[str, Any], Any, Any, torch.device]:
    seed = int(config["random_seed"])
    _seed_everything(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    for name in ("train", "validation"):
        (output_dir / name).mkdir(exist_ok=True)

    train_rows = _read_manifest(PROJECT_DIR / "watkins_splits" / "train.csv", "train")
    validation_rows = _read_manifest(
        PROJECT_DIR / "watkins_splits" / "validation.csv", "validation"
    )
    preprocessing = PreprocessingConfig.load(
        PROJECT_DIR / "watkins_preprocessing_config.json"
    )
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor, device = _load_encoder(config["model_id"], device_name)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    projector = ProjectionHead(
        input_dim=512,
        hidden_dim=int(config["projection_hidden_dim"]),
        output_dim=int(config["projection_dim"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        projector.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and bool(config["mixed_precision"])
    )
    best_path = output_dir / "checkpoints" / "best.pt"
    latest_path = output_dir / "checkpoints" / "latest.pt"
    log_path = output_dir / "training_log.csv"
    if max_train_batches is not None or max_validation_batches is not None:
        latest_path = output_dir / "checkpoints" / "smoke_latest.pt"
        best_path = output_dir / "checkpoints" / "smoke_best.pt"
        log_path = output_dir / "smoke_training_log.csv"

    start_epoch = 1
    best_loss = math.inf
    best_epoch = 0
    patience_count = 0
    elapsed_training_seconds = 0.0
    if max_train_batches is None and latest_path.is_file():
        state = torch.load(latest_path, map_location=device, weights_only=False)
        if state["config"] != config:
            raise ValueError(
                "Checkpoint config does not match the current training config"
            )
        projector.load_state_dict(state["projector"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        best_loss = float(state["best_validation_loss"])
        best_epoch = int(state["best_epoch"])
        patience_count = int(state["patience_count"])
        elapsed_training_seconds = float(state["elapsed_training_seconds"])
        LOGGER.info("Resuming after epoch %d", start_epoch - 1)

    if start_epoch <= int(config["epochs"]):
        for epoch in range(start_epoch, int(config["epochs"]) + 1):
            epoch_started = time.perf_counter()
            train_loss, train_samples, train_seconds = _run_epoch(
                "train",
                train_rows,
                preprocessing,
                model,
                processor,
                projector,
                device,
                config,
                epoch,
                optimizer,
                scaler,
                max_train_batches,
            )
            validation_loss, validation_samples, validation_seconds = _run_epoch(
                "validation",
                validation_rows,
                preprocessing,
                model,
                processor,
                projector,
                device,
                config,
                epoch,
                None,
                None,
                max_validation_batches,
            )
            elapsed_training_seconds += time.perf_counter() - epoch_started
            improved = validation_loss < best_loss
            if improved:
                best_loss = validation_loss
                best_epoch = epoch
                patience_count = 0
                _atomic_torch_save(
                    _checkpoint(
                        projector,
                        optimizer,
                        scaler,
                        epoch,
                        best_loss,
                        best_epoch,
                        patience_count,
                        elapsed_training_seconds,
                        config,
                    ),
                    best_path,
                )
            else:
                patience_count += 1

            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2
            else:
                peak_memory = 0.0
            _append_training_log(
                log_path,
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.8f}",
                    "validation_loss": f"{validation_loss:.8f}",
                    "train_samples": train_samples,
                    "validation_samples": validation_samples,
                    "train_seconds": f"{train_seconds:.2f}",
                    "validation_seconds": f"{validation_seconds:.2f}",
                    "cuda_peak_allocated_mib": f"{peak_memory:.2f}",
                    "best_epoch": best_epoch,
                },
            )
            _atomic_torch_save(
                _checkpoint(
                    projector,
                    optimizer,
                    scaler,
                    epoch,
                    best_loss,
                    best_epoch,
                    patience_count,
                    elapsed_training_seconds,
                    config,
                ),
                latest_path,
            )
            LOGGER.info(
                "epoch=%d/%d train_loss=%.5f validation_loss=%.5f "
                "train_samples=%d validation_samples=%d elapsed=%.1fs "
                "epoch_wall=%.1fs best_epoch=%d peak_gpu_mib=%.1f",
                epoch,
                int(config["epochs"]),
                train_loss,
                validation_loss,
                train_samples,
                validation_samples,
                elapsed_training_seconds,
                time.perf_counter() - epoch_started,
                best_epoch,
                peak_memory,
            )
            if (
                max_train_batches is None
                and patience_count >= int(config["early_stopping_patience"])
            ):
                LOGGER.info("Early stopping after %d non-improving epochs", patience_count)
                break

    if not best_path.is_file():
        raise RuntimeError("Training did not produce a best checkpoint")
    best_state = torch.load(best_path, map_location=device, weights_only=False)
    projector.load_state_dict(best_state["projector"])
    projector.eval()
    summary = {
        "device": str(device),
        "encoder_model_id": config["model_id"],
        "encoder_trainable_parameters": 0,
        "projection_trainable_parameters": sum(
            parameter.numel()
            for parameter in projector.parameters()
            if parameter.requires_grad
        ),
        "best_epoch": best_state["best_epoch"],
        "best_validation_loss": best_state["best_validation_loss"],
        "elapsed_training_seconds": best_state["elapsed_training_seconds"],
        "train_samples_per_epoch": len(train_rows),
        "validation_samples_per_epoch": len(validation_rows),
        "configured_epochs": config["epochs"],
        "cuda_peak_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        ),
        "checkpoint": str(best_path),
        "smoke_only": max_train_batches is not None,
    }
    if max_train_batches is None and log_path.is_file():
        with log_path.open(encoding="utf-8", newline="") as stream:
            summary["train_samples_processed"] = sum(
                int(row["train_samples"]) for row in csv.DictReader(stream)
            )
    else:
        summary["train_samples_processed"] = 0
    return projector, summary, model, processor, device


def _generate_embeddings(
    split: str,
    projector: ProjectionHead,
    model: Any,
    processor: Any,
    device: torch.device,
    preprocessing: PreprocessingConfig,
    output_dir: Path,
    batch_size: int,
    mixed_precision: bool,
) -> dict[str, Any]:
    rows = _read_manifest(PROJECT_DIR / "watkins_splits" / f"{split}.csv", split)
    stage4_metadata = PROJECT_DIR / "watkins_stage4" / split / "metadata.csv"
    stage5_metadata = output_dir / split / "metadata.csv"
    with stage4_metadata.open(encoding="utf-8-sig", newline="") as stream:
        stage4_rows = list(csv.DictReader(stream))
    if [row["segment_id"] for row in rows] != [
        row["segment_id"] for row in stage4_rows
    ]:
        raise ValueError(f"{split} manifest IDs differ from Stage 4 metadata")

    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    destination = split_dir / "embeddings.npy"
    temporary = split_dir / "embeddings.tmp.npy"
    embeddings = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), int(projector.layers[-1].out_features)),
    )
    projector.eval()
    batch: list[tuple[int, np.ndarray]] = []
    position = 0
    written = np.zeros(len(rows), dtype=bool)

    def consume(batch_rows: list[tuple[int, np.ndarray]]) -> None:
        nonlocal position
        indices = [index for index, _ in batch_rows]
        waveforms = [waveform for _, waveform in batch_rows]
        if written[indices].any():
            raise RuntimeError(f"{split} iterator yielded duplicate manifest indexes")
        baseline = torch.from_numpy(
            _embed_batch(
                model, processor, device, waveforms, mixed_precision
            )
        ).to(device)
        with torch.no_grad():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda" and mixed_precision,
            ):
                projected = projector(baseline)
        projected = F.normalize(projected.float(), dim=-1).cpu().numpy()
        embeddings[indices] = projected
        written[indices] = True
        position += len(waveforms)

    for manifest_index, _, waveform in _iter_waveforms(rows, preprocessing):
        batch.append((manifest_index, waveform))
        if len(batch) == batch_size:
            consume(batch)
            batch = []
    if batch:
        consume(batch)
    embeddings.flush()
    del embeddings
    if position != len(rows) or not written.all():
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"{split}: wrote {position} embeddings for {len(rows)} segments; "
            f"{int(written.sum())} manifest rows were filled"
        )
    os.replace(temporary, destination)
    shutil.copyfile(stage4_metadata, stage5_metadata)
    saved = np.load(destination, mmap_mode="r")
    if not np.isfinite(saved).all():
        raise RuntimeError(f"Non-finite {split} Stage 5 embeddings")
    return {
        "split": split,
        "segments": len(rows),
        "embedding_dim": int(saved.shape[1]),
        "all_finite": True,
        "all_l2_normalized": bool(
            np.allclose(np.linalg.norm(saved, axis=1), 1.0, atol=1e-4)
        ),
        "embeddings_file": str(destination),
        "metadata_file": str(stage5_metadata),
    }


def run_smoke_test(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    _seed_everything(int(config["random_seed"]))
    rows = _read_manifest(PROJECT_DIR / "watkins_splits" / "train.csv", "train")
    preprocessing = PreprocessingConfig.load(
        PROJECT_DIR / "watkins_preprocessing_config.json"
    )
    model, processor, device = _load_encoder(
        config["model_id"], "cuda" if torch.cuda.is_available() else "cpu"
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    projector = ProjectionHead(
        hidden_dim=int(config["projection_hidden_dim"]),
        output_dim=int(config["projection_dim"]),
    ).to(device)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    batch = next(
        _waveform_batches(
            rows[: int(config["batch_size"])],
            preprocessing,
            int(config["batch_size"]),
            None,
            int(config["random_seed"]),
        )
    )
    parameters_before = [
        parameter.detach().clone() for parameter in projector.parameters()
    ]
    loss = _loss_for_batch(
        batch,
        model,
        processor,
        projector,
        device,
        bool(config["mixed_precision"]),
        float(config["temperature"]),
        config["augmentations"],
        int(config["random_seed"]),
        True,
        optimizer,
        scaler,
    )
    updated = any(
        not torch.equal(before, after)
        for before, after in zip(parameters_before, projector.parameters())
    )
    if not math.isfinite(loss) or not updated:
        raise RuntimeError(
            f"Smoke test failed: loss={loss}, projector_updated={updated}"
        )
    summary = {
        "status": "passed",
        "device": str(device),
        "batch_samples": len(batch),
        "view_count": 2 * len(batch),
        "contrastive_loss": loss,
        "projector_parameters_updated": updated,
        "encoder_frozen": all(not p.requires_grad for p in model.parameters()),
        "encoder_output_dim": 512,
        "projection_output_dim": int(config["projection_dim"]),
        "cuda_peak_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        ),
    }
    smoke_dir = output_dir / "smoke_test"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    (smoke_dir / "smoke_test_result.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "stage5.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a frozen-CLAP waveform-level SimCLR projection."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    config = _read_config(args.config)
    output_dir = args.output_dir.resolve()
    _setup_logging(output_dir)

    if args.smoke_test:
        LOGGER.info("Running one-batch waveform SSL smoke test")
        print(json.dumps(run_smoke_test(config, output_dir), indent=2))
        return

    run_started = time.perf_counter()
    projector, training_summary, model, processor, device = train(config, output_dir)
    (output_dir / "training_summary.json").write_text(
        json.dumps(training_summary, indent=2), encoding="utf-8"
    )
    projector.to(device)
    preprocessing = PreprocessingConfig.load(
        PROJECT_DIR / "watkins_preprocessing_config.json"
    )
    embedding_summaries = [
        _generate_embeddings(
            split,
            projector,
            model,
            processor,
            device,
            preprocessing,
            output_dir,
            int(config["batch_size"]),
            bool(config["mixed_precision"]),
        )
        for split in ("train", "validation", "test")
    ]
    result = {
        "status": "complete",
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_run_seconds": time.perf_counter() - run_started,
        "training": training_summary,
        "embeddings": embedding_summaries,
    }
    (output_dir / "generation_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    from watkins_stage5_evaluate import evaluate_and_report

    report_result = evaluate_and_report(output_dir, config)
    result["evaluation"] = report_result
    (output_dir / "generation_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    LOGGER.info(
        "Stage 5 complete: best validation loss %.5f, checkpoint %s",
        training_summary["best_validation_loss"],
        training_summary["checkpoint"],
    )


if __name__ == "__main__":
    main()
