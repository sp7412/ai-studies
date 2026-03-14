"""
restructure_afo.py
==================
Restructures the AFO Kaggle download into the train/val/test layout
expected by AFODataset in the notebook.

Source layout (what Kaggle gives you):
  ~/data/afo/
    PART_1/PART_1/
      images/          ← jpg files
      1category/       ← YOLO labels, 1-class scheme  (skip)
      2categories/     ← YOLO labels, 2-class scheme  (skip)
      6categories/     ← YOLO labels, 6-class scheme  ← USE THIS
    PART_2/PART_2/
      images/
      (no label subdirs — images only)
    PART_3/PART_3/
      images/
      (no label subdirs — images only)

Target layout (what AFODataset expects):
  ~/data/afo_clean/
    images/
      train/   val/   test/
    labels/
      train/   val/   test/

Split: 70% train / 15% val / 15% test  (reproducible, seed=42)

Classes (6categories scheme):
  0: human
  1: surfer
  2: kayak
  3: boat
  4: buoy
  5: sailboat

Usage:
  python3 restructure_afo.py
  python3 restructure_afo.py --src ~/data/afo --dst ~/data/afo_clean
"""

import argparse
import random
import shutil
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_SRC = Path.home() / "data" / "afo"
DEFAULT_DST = Path.home() / "data" / "afo_clean"
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 1 - TRAIN - VAL = 0.15
SEED        = 42
LABEL_SUBDIR = "6categories"   # the annotation scheme we use

AFO_CLASSES = ["human", "surfer", "kayak", "boat", "buoy", "sailboat"]


def main(src: Path, dst: Path):
    random.seed(SEED)

    # ── 1. Collect all images ─────────────────────────────────────────────────
    all_images = sorted(src.rglob("images/*.jpg"))
    print(f"Found {len(all_images)} images across all PART dirs")

    # ── 2. For each image, find its 6categories label (if exists) ─────────────
    paired, unpaired = [], []
    for img_path in all_images:
        # label lives in sibling 6categories/ dir with same stem
        part_root  = img_path.parent.parent          # e.g. PART_1/PART_1/
        label_path = part_root / LABEL_SUBDIR / (img_path.stem + ".txt")
        if label_path.exists():
            paired.append((img_path, label_path))
        else:
            unpaired.append(img_path)

    print(f"  {len(paired)} images have 6-class labels")
    print(f"  {len(unpaired)} images have no 6-class label (skipping)")

    if not paired:
        print("\nERROR: No paired images found. Check that LABEL_SUBDIR is correct.")
        print(f"  Looking for: {LABEL_SUBDIR}/")
        print("  Available subdirs in PART_1:")
        for d in sorted((src / "PART_1" / "PART_1").iterdir()):
            print(f"    {d.name}/")
        return

    # ── 3. Shuffle and split ──────────────────────────────────────────────────
    random.shuffle(paired)
    n       = len(paired)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        "train": paired[:n_train],
        "val":   paired[n_train : n_train + n_val],
        "test":  paired[n_train + n_val :],
    }
    for split, items in splits.items():
        print(f"  {split:5}: {len(items)} samples")

    # ── 4. Create output dirs ─────────────────────────────────────────────────
    for split in splits:
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── 5. Copy files ─────────────────────────────────────────────────────────
    total = sum(len(v) for v in splits.values())
    done  = 0
    for split, items in splits.items():
        for img_src, lbl_src in items:
            shutil.copy2(img_src, dst / "images" / split / img_src.name)
            shutil.copy2(lbl_src, dst / "labels" / split / lbl_src.name)
            done += 1
            if done % 200 == 0 or done == total:
                print(f"  Copied {done}/{total} ...", end="\r")

    print(f"\nDone. Dataset written to: {dst}")

    # ── 6. Sanity check ───────────────────────────────────────────────────────
    print("\nSanity check:")
    for split in ["train", "val", "test"]:
        imgs = list((dst / "images" / split).glob("*.jpg"))
        lbls = list((dst / "labels" / split).glob("*.txt"))
        print(f"  {split:5}: {len(imgs)} images, {len(lbls)} labels", end="")
        if len(imgs) != len(lbls):
            print("  ← MISMATCH!", end="")
        print()

    # ── 7. Write classes.txt ──────────────────────────────────────────────────
    (dst / "classes.txt").write_text("\n".join(AFO_CLASSES) + "\n")
    print(f"\nClasses: {AFO_CLASSES}")
    print("All done ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restructure AFO Kaggle download")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC,
                        help=f"Path to raw AFO download (default: {DEFAULT_SRC})")
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST,
                        help=f"Path to write cleaned dataset (default: {DEFAULT_DST})")
    args = parser.parse_args()
    main(args.src, args.dst)
