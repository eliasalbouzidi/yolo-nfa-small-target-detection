#!/usr/bin/env bash
set -euo pipefail

# Prepare SIRST zip release into YOLOv7 compatible layout.


ZIP_PATH="sirst.zip"
SPLIT_DIR="idx_427"
OUT_DIR="datasets/sirst"
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zip)
      ZIP_PATH="$2"
      shift 2
      ;;
    --split)
      SPLIT_DIR="$2"
      shift 2
      ;;
    --out)
      OUT_DIR="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      sed -n '1,40p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "Missing zip file: $ZIP_PATH" >&2
  exit 1
fi

if [[ -d "$OUT_DIR" && "$FORCE" -ne 1 ]]; then
  echo "Output directory already exists: $OUT_DIR" >&2
  echo "Use --force to overwrite." >&2
  exit 1
fi

if [[ "$FORCE" -eq 1 ]]; then
  rm -rf "$OUT_DIR"
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

echo "[1/4] Unpacking outer SIRST archive..."
unzip -q -o "$ZIP_PATH" -d "$WORK_DIR/sirst_root"

if [[ ! -f "$WORK_DIR/sirst_root/images.zip" || ! -f "$WORK_DIR/sirst_root/masks.zip" ]]; then
  echo "Expected images.zip and masks.zip inside $ZIP_PATH" >&2
  exit 1
fi

if [[ ! -d "$WORK_DIR/sirst_root/$SPLIT_DIR" ]]; then
  echo "Split folder not found: $SPLIT_DIR" >&2
  echo "Available split folders:" >&2
  find "$WORK_DIR/sirst_root" -maxdepth 1 -type d -name 'idx_*' -printf '  %f\n' >&2
  exit 1
fi

echo "[2/4] Unpacking images and masks..."
unzip -q -o "$WORK_DIR/sirst_root/images.zip" -d "$WORK_DIR/raw"
unzip -q -o "$WORK_DIR/sirst_root/masks.zip" -d "$WORK_DIR/raw"

echo "[3/4] Building YOLO labels and split folders..."
SIRST_ROOT="$WORK_DIR/sirst_root" RAW_ROOT="$WORK_DIR/raw" SPLIT_DIR="$SPLIT_DIR" OUT_DIR="$OUT_DIR" python - <<'PY'
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


sirst_root = Path(os.environ["SIRST_ROOT"])
raw_root = Path(os.environ["RAW_ROOT"])
split_dir = os.environ["SPLIT_DIR"]
out_dir = Path(os.environ["OUT_DIR"])

images_root = raw_root / "images"
masks_root = raw_root / "masks"
split_root = sirst_root / split_dir

if not images_root.is_dir():
    raise SystemExit(f"Missing extracted images folder: {images_root}")
if not masks_root.is_dir():
    raise SystemExit(f"Missing extracted masks folder: {masks_root}")

for split in ("train", "val", "test"):
    (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

totals = {}
for split in ("train", "val", "test"):
    split_file = split_root / f"{split}.txt"
    if not split_file.is_file():
        raise SystemExit(f"Missing split file: {split_file}")

    stems = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    totals[split] = len(stems)

    for stem in stems:
        src_img = images_root / f"{stem}.png"
        if not src_img.exists():
            raise SystemExit(f"Missing image for split '{split}': {src_img}")

        dst_img = out_dir / "images" / split / src_img.name
        shutil.copy2(src_img, dst_img)

        xml_path = masks_root / f"{stem}.xml"
        if not xml_path.exists():
            raise SystemExit(f"Missing annotation xml for {stem}: {xml_path}")

        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        if size is None:
            raise SystemExit(f"Annotation missing <size>: {xml_path}")
        w = float(size.findtext("width", "0"))
        h = float(size.findtext("height", "0"))
        if w <= 0 or h <= 0:
            raise SystemExit(f"Invalid size in {xml_path}")

        lines = []
        for obj in root.findall("object"):
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            xmin = float(bnd.findtext("xmin", "0"))
            ymin = float(bnd.findtext("ymin", "0"))
            xmax = float(bnd.findtext("xmax", "0"))
            ymax = float(bnd.findtext("ymax", "0"))

            # Convert VOC box to normalized YOLO xywh.
            x_center = clamp(((xmin + xmax) / 2.0) / w)
            y_center = clamp(((ymin + ymax) / 2.0) / h)
            bw = clamp((xmax - xmin) / w)
            bh = clamp((ymax - ymin) / h)
            if bw <= 0.0 or bh <= 0.0:
                continue
            lines.append(f"0 {x_center:.8f} {y_center:.8f} {bw:.8f} {bh:.8f}")

        (out_dir / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))

print("Prepared SIRST dataset:")
for split in ("train", "val", "test"):
    print(f"  {split}: {totals[split]} images")
PY

echo "[4/4] Done."
echo "Dataset ready at: $OUT_DIR"
echo "Use with: --data data/sirst.yaml"
