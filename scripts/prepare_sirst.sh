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

[[ -f "$ZIP_PATH" ]] || exit 1
[[ ! -d "$OUT_DIR" || "$FORCE" -eq 1 ]] || exit 1

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
from pathlib import Path

import numpy as np
from PIL import Image


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

        with Image.open(src_img) as image:
            w, h = image.size
        if w <= 0 or h <= 0:
            raise SystemExit(f"Invalid image size for {src_img}")

        mask_path = masks_root / f"{stem}_pixels0.png"
        if not mask_path.exists():
            raise SystemExit(f"Missing mask png for {stem}: {mask_path}")

        lines = []
        mask = np.array(Image.open(mask_path).convert("L")) > 0
        visited = np.zeros_like(mask, dtype=bool)
        ys, xs = np.where(mask)
        components = []

        # The V1 masks preserve separate targets as disconnected foreground blobs.
        # Convert each 8-connected component into its own YOLO box.
        for y, x in zip(ys, xs):
            if visited[y, x]:
                continue

            stack = [(int(y), int(x))]
            visited[y, x] = True
            comp_ys = []
            comp_xs = []

            while stack:
                cy, cx = stack.pop()
                comp_ys.append(cy)
                comp_xs.append(cx)

                for ny in range(max(0, cy - 1), min(mask.shape[0], cy + 2)):
                    for nx in range(max(0, cx - 1), min(mask.shape[1], cx + 2)):
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

            components.append(
                (
                    float(min(comp_xs)),
                    float(min(comp_ys)),
                    float(max(comp_xs) + 1),
                    float(max(comp_ys) + 1),
                )
            )

        components.sort(key=lambda box: (box[1], box[0], box[3], box[2]))
        for x0, y0, x1, y1 in components:
            x_center = clamp(((x0 + x1) / 2.0) / w)
            y_center = clamp(((y0 + y1) / 2.0) / h)
            bw = clamp((x1 - x0) / w)
            bh = clamp((y1 - y0) / h)
            if bw > 0.0 and bh > 0.0:
                lines.append(f"0 {x_center:.8f} {y_center:.8f} {bw:.8f} {bh:.8f}")

        (out_dir / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))

print("Prepared SIRST dataset:")
for split in ("train", "val", "test"):
    print(f"  {split}: {totals[split]} images")
PY

python scripts/create_sirst_fewshot_splits.py --dataset "$OUT_DIR"
find "$OUT_DIR/labels" -maxdepth 1 -type f \( -name 'fs15_fold*.cache' -o -name 'fs25_fold*.cache' \) -delete
find "$OUT_DIR/labels" -maxdepth 2 -type f -name '*.cache' -delete
# Remove ALL .cache files anywhere under datasets/ to prevent stale caches
find datasets/ -type f -name '*.cache' -delete 2>/dev/null || true
echo "[4/4] Done."
echo "Dataset ready at: $OUT_DIR"
echo "Use with: --data data/datasets/sirst.yaml"
