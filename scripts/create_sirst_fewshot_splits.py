#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainSample:
    stem: str
    image_path_ref: str
    box_count: int
    total_box_area: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic 15-shot and 25-shot SIRST train folds."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/sirst"),
        help="Prepared SIRST dataset root.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2402,
        help="Deterministic seed used to assign train images to folds.",
    )
    return parser.parse_args()


def load_train_samples(dataset_dir: Path, repo_root: Path) -> list[TrainSample]:
    image_dir = dataset_dir / "images" / "train"
    label_dir = dataset_dir / "labels" / "train"
    if not image_dir.is_dir():
        raise SystemExit(f"Missing train image directory: {image_dir}")
    if not label_dir.is_dir():
        raise SystemExit(f"Missing train label directory: {label_dir}")

    samples: list[TrainSample] = []
    for image_path in sorted(image_dir.glob("*.png")):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise SystemExit(f"Missing label file for {image_path.name}: {label_path}")

        lines = [line.strip() for line in label_path.read_text().splitlines() if line.strip()]
        if not lines:
            raise SystemExit(f"Expected at least one target in {label_path}")

        total_box_area = 0.0
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                raise SystemExit(f"Invalid YOLO label in {label_path}: {line}")
            total_box_area += float(parts[3]) * float(parts[4])

        image_path_ref = repo_relative_or_absolute(image_path.resolve(), repo_root.resolve())
        samples.append(
            TrainSample(
                stem=image_path.stem,
                image_path_ref=image_path_ref,
                box_count=len(lines),
                total_box_area=total_box_area,
            )
        )

    return samples


def repo_relative_or_absolute(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def build_folds(samples: list[TrainSample], fold_size: int, seed: int) -> list[list[TrainSample]]:
    buckets: dict[int, list[TrainSample]] = defaultdict(list)
    for sample in samples:
        buckets[sample.box_count].append(sample)

    rng = random.Random(seed + fold_size)
    folds: list[list[TrainSample]] = [[] for _ in range(3)]
    fold_box_totals = [0 for _ in range(3)]
    fold_area_totals = [0.0 for _ in range(3)]

    for box_count in sorted(buckets.keys(), reverse=True):
        bucket = list(buckets[box_count])
        rng.shuffle(bucket)
        for sample in bucket:
            eligible = [i for i in range(3) if len(folds[i]) < fold_size]
            if not eligible:
                break

            target_fold = min(
                eligible,
                key=lambda i: (fold_box_totals[i], fold_area_totals[i], len(folds[i]), i),
            )
            folds[target_fold].append(sample)
            fold_box_totals[target_fold] += sample.box_count
            fold_area_totals[target_fold] += sample.total_box_area

        if all(len(fold) == fold_size for fold in folds):
            break

    if any(len(fold) != fold_size for fold in folds):
        sizes = ", ".join(str(len(fold)) for fold in folds)
        raise SystemExit(f"Unable to build 3 folds of size {fold_size}. Current sizes: {sizes}")

    return folds


def materialize_fold(dataset_dir: Path, split_name: str, samples: list[TrainSample], repo_root: Path) -> None:
    image_dir = dataset_dir / "images" / split_name
    label_dir = dataset_dir / "labels" / split_name
    if image_dir.exists():
        shutil.rmtree(image_dir)
    if label_dir.exists():
        shutil.rmtree(label_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        image_src = repo_root / sample.image_path_ref
        label_src = dataset_dir / "labels" / "train" / f"{sample.stem}.txt"
        shutil.copy2(image_src, image_dir / image_src.name)
        shutil.copy2(label_src, label_dir / label_src.name)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    dataset_dir = (repo_root / args.dataset).resolve() if not args.dataset.is_absolute() else args.dataset.resolve()

    samples = load_train_samples(dataset_dir, repo_root)
    if len(samples) < 75:
        raise SystemExit(f"Expected at least 75 train images, found {len(samples)}")

    for shot in (15, 25):
        folds = build_folds(samples, fold_size=shot, seed=args.seed)
        for fold_idx, fold_samples in enumerate(folds, start=1):
            split_name = f"fs{shot}_fold{fold_idx}"
            materialize_fold(dataset_dir, split_name, fold_samples, repo_root)


if __name__ == "__main__":
    main()
