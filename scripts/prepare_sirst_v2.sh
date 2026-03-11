#!/usr/bin/env bash
set -euo pipefail

# Prepare SIRST-V2 into a YOLOv7-compatible layout.
# Defaults to the V1-compatible 427-image protocol inside SIRST-V2 and writes it to datasets/sirst-v2.

ZIP_PATH="sirst-v2.zip"
OUT_DIR="datasets/sirst-v2"
TRAIN_FILE="splits/train_v1.txt"
VAL_FILE="splits/val_v1.txt"
TEST_FILE="splits/test_v1.txt"
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zip)
      ZIP_PATH="$2"
      shift 2
      ;;
    --out)
      OUT_DIR="$2"
      shift 2
      ;;
    --train-file)
      TRAIN_FILE="$2"
      shift 2
      ;;
    --val-file)
      VAL_FILE="$2"
      shift 2
      ;;
    --test-file)
      TEST_FILE="$2"
      shift 2
      ;;
    --split-family)
      case "$2" in
        v1)
          TRAIN_FILE="splits/train_v1.txt"
          VAL_FILE="splits/val_v1.txt"
          TEST_FILE="splits/test_v1.txt"
          ;;
        full)
          TRAIN_FILE="splits/train_full.txt"
          VAL_FILE="splits/val_full.txt"
          TEST_FILE="splits/test_full.txt"
          ;;
        *)
          echo "Unsupported split family: $2" >&2
          echo "Supported values: v1, full" >&2
          exit 1
          ;;
      esac
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      sed -n '1,80p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

[[ -f "$ZIP_PATH" ]] || exit 1
[[ ! -d "$OUT_DIR" || "$FORCE" -eq 1 ]] || exit 1

if [[ "$FORCE" -eq 1 ]]; then
  rm -rf "$OUT_DIR"
fi

echo "[1/2] Building YOLO labels and split folders from $ZIP_PATH ..."
ZIP_PATH="$ZIP_PATH" OUT_DIR="$OUT_DIR" TRAIN_FILE="$TRAIN_FILE" VAL_FILE="$VAL_FILE" TEST_FILE="$TEST_FILE" python - <<'PY'
import io
import os
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


zip_path = Path(os.environ["ZIP_PATH"])
out_dir = Path(os.environ["OUT_DIR"])
split_files = {
    "train": os.environ["TRAIN_FILE"],
    "val": os.environ["VAL_FILE"],
    "test": os.environ["TEST_FILE"],
}

with zipfile.ZipFile(zip_path) as archive:
    names = set(archive.namelist())
    required = set(split_files.values())
    missing = sorted(name for name in required if name not in names)
    if missing:
        raise SystemExit(f"Missing split files in {zip_path}: {missing}")

    for split in split_files:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    totals = {}
    object_totals = {}
    empty_totals = {}

    for split, split_file in split_files.items():
        stems = [
            line.strip()
            for line in archive.read(split_file).decode("utf-8").splitlines()
            if line.strip()
        ]
        totals[split] = len(stems)
        object_totals[split] = 0
        empty_totals[split] = 0

        for stem in stems:
            image_name = f"mixed/{stem}.png"
            bbox_name = f"annotations/bboxes/{stem}.xml"
            if image_name not in names:
                raise SystemExit(f"Missing image in archive: {image_name}")
            if bbox_name not in names:
                raise SystemExit(f"Missing bbox XML in archive: {bbox_name}")

            dst_img = out_dir / "images" / split / f"{stem}.png"
            with archive.open(image_name) as src, dst_img.open("wb") as dst:
                shutil.copyfileobj(src, dst)

            root = ET.fromstring(archive.read(bbox_name))
            size = root.find("size")
            if size is None:
                raise SystemExit(f"Missing <size> in {bbox_name}")

            width = int(size.findtext("width", default="0"))
            height = int(size.findtext("height", default="0"))
            if width <= 0 or height <= 0:
                raise SystemExit(f"Invalid image size in {bbox_name}: {(width, height)}")

            lines = []
            for obj in root.findall("object"):
                box = obj.find("bndbox")
                if box is None:
                    continue

                xmin = float(box.findtext("xmin", default="0"))
                ymin = float(box.findtext("ymin", default="0"))
                xmax = float(box.findtext("xmax", default="0"))
                ymax = float(box.findtext("ymax", default="0"))

                # SIRST-V2 boxes behave like inclusive pixel bounds.
                x0 = xmin
                y0 = ymin
                x1 = xmax + 1.0
                y1 = ymax + 1.0

                bw = clamp((x1 - x0) / width)
                bh = clamp((y1 - y0) / height)
                if bw <= 0.0 or bh <= 0.0:
                    continue

                x_center = clamp(((x0 + x1) / 2.0) / width)
                y_center = clamp(((y0 + y1) / 2.0) / height)
                lines.append(f"0 {x_center:.8f} {y_center:.8f} {bw:.8f} {bh:.8f}")

            if lines:
                object_totals[split] += len(lines)
            else:
                empty_totals[split] += 1

            (out_dir / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))

print("Prepared SIRST-V2 dataset:")
for split in ("train", "val", "test"):
    print(
        f"  {split}: {totals[split]} images, "
        f"{object_totals[split]} boxes, {empty_totals[split]} empty labels"
    )
PY

echo "[2/2] Done."
echo "Dataset ready at: $OUT_DIR"
