"""
Data Augmentation Script
========================
Expands the brain tumor dataset from ~253 images to ~2,500+ images
using a battery of augmentations before RCNN training.

Usage:
    python src/augment_dataset.py

Output:
    dataset_augmented/yes/   — augmented tumor images
    dataset_augmented/no/    — augmented normal images
"""

import os
import cv2
import numpy as np
from pathlib import Path

DATASET_PATH    = "dataset"
OUTPUT_PATH     = "dataset_augmented"
AUGS_PER_IMAGE  = 10        # 10 augmented copies + 1 original = 11x expansion

# ──────────────────────────────────────────────────────────────────────────────
#  Individual augmentation functions
# ──────────────────────────────────────────────────────────────────────────────

def _flip_h(img):
    return cv2.flip(img, 1)

def _flip_v(img):
    return cv2.flip(img, 0)

def _rot90(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

def _rot270(img):
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

def _brightness_up(img):
    return np.clip(img.astype(np.int32) + 35, 0, 255).astype(np.uint8)

def _brightness_down(img):
    return np.clip(img.astype(np.int32) - 35, 0, 255).astype(np.uint8)

def _gaussian_noise(img):
    noise = np.random.normal(0, 12, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def _rotate15(img):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), 15, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def _rotate_neg15(img):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), -15, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def _zoom_in(img):
    h, w = img.shape[:2]
    margin_y, margin_x = h // 10, w // 10
    cropped = img[margin_y : h - margin_y, margin_x : w - margin_x]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

AUGMENTATIONS = [
    _flip_h,
    _flip_v,
    _rot90,
    _rot270,
    _brightness_up,
    _brightness_down,
    _gaussian_noise,
    _rotate15,
    _rotate_neg15,
    _zoom_in,
]

assert len(AUGMENTATIONS) == AUGS_PER_IMAGE, \
    f"AUGS_PER_IMAGE={AUGS_PER_IMAGE} but {len(AUGMENTATIONS)} functions defined"


# ──────────────────────────────────────────────────────────────────────────────
#  Main augmentation loop
# ──────────────────────────────────────────────────────────────────────────────

def augment_class(cls: str) -> None:
    src_dir = Path(DATASET_PATH) / cls
    dst_dir = Path(OUTPUT_PATH) / cls
    dst_dir.mkdir(parents=True, exist_ok=True)

    exts   = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [p for p in src_dir.iterdir() if p.suffix.lower() in exts]

    if not images:
        print(f"  [!] No images found in {src_dir}")
        return

    written = 0
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [!] Skipping unreadable file: {img_path.name}")
            continue

        # ── Original copy ──────────────────────────────────────────────────
        out_path = dst_dir / img_path.name
        cv2.imwrite(str(out_path), img)
        written += 1

        # ── Augmented copies ───────────────────────────────────────────────
        for aug_idx, aug_fn in enumerate(AUGMENTATIONS):
            aug_img  = aug_fn(img.copy())
            aug_name = f"{img_path.stem}_a{aug_idx:02d}{img_path.suffix}"
            cv2.imwrite(str(dst_dir / aug_name), aug_img)
            written += 1

    print(f"  '{cls}' class : {len(images)} originals → {written} images total")


def main() -> None:
    print(f"Source dataset : {DATASET_PATH}/")
    print(f"Output dataset : {OUTPUT_PATH}/")
    print(f"Augmentations  : {AUGS_PER_IMAGE} per image (×{AUGS_PER_IMAGE + 1} expansion)")
    print()

    for cls in ("yes", "no"):
        print(f"Processing '{cls}'...")
        augment_class(cls)

    # Print final counts
    print()
    for cls in ("yes", "no"):
        n = len(list((Path(OUTPUT_PATH) / cls).iterdir()))
        print(f"  dataset_augmented/{cls}/ : {n} images")

    print("\nDone. Run `python src/train_model.py` next.")


if __name__ == "__main__":
    np.random.seed(42)
    main()
