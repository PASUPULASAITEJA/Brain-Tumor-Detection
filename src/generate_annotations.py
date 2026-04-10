"""
Pseudo Bounding-Box Annotation Generator
=========================================
Runs Grad-CAM over the trained EfficientNetB0 classifier on every
tumour-positive image and converts the activation map into a COCO-format
bounding box.  The resulting annotations.json is used to train Faster RCNN.

Usage:
    python src/generate_annotations.py

Prerequisites:
    • model/efficientnet_classifier.h5  (from train_model.py)

Output:
    annotations/annotations.json
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input as _eff_preprocess

class EfficientNetPreprocess(tf.keras.layers.Layer):
    def call(self, x):
        return _eff_preprocess(x)
    def get_config(self):
        return super().get_config()
from PIL import Image
from pathlib import Path

# Make sibling imports work when called from project root
sys.path.insert(0, os.path.dirname(__file__))
from gradcam import get_gradcam, heatmap_to_bbox

# ──────────────────────────────────────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────────────────────────────────────

IMG_SIZE         = 224
MODEL_PATH       = "model/efficientnet_classifier.h5"
DATASET_PATH     = "dataset_augmented" if os.path.exists("dataset_augmented") else "dataset"
ANNOTATIONS_PATH = "annotations/annotations.json"

# Only annotate predictions above this threshold (confident positives)
PRED_THRESHOLD   = 0.50
# Grad-CAM activation threshold for bbox extraction
GRADCAM_THRESH   = 0.40

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def generate_annotations() -> None:
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at '{MODEL_PATH}'.")
        print("        Run `python src/train_model.py` first.")
        sys.exit(1)

    os.makedirs("annotations", exist_ok=True)

    print(f"Loading model : {MODEL_PATH}")
    model = tf.keras.models.load_model(
        MODEL_PATH, compile=False,
        custom_objects={"EfficientNetPreprocess": EfficientNetPreprocess}
    )

    coco = {
        "images":      [],
        "annotations": [],
        "categories":  [
            {"id": 0, "name": "background"},
            {"id": 1, "name": "tumor"},
        ],
    }

    img_id  = 0
    ann_id  = 0
    skipped = 0

    tumor_dir = Path(DATASET_PATH) / "yes"
    paths     = sorted(p for p in tumor_dir.iterdir()
                       if p.suffix.lower() in IMAGE_EXTS)

    print(f"Processing {len(paths)} tumour images from {tumor_dir} …")

    for img_path in paths:
        # ── Load & preprocess ──────────────────────────────────────────────
        try:
            pil_img  = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [!] Cannot open {img_path.name}: {e}")
            skipped += 1
            continue

        img_rs   = pil_img.resize((IMG_SIZE, IMG_SIZE))
        arr      = np.array(img_rs, dtype=np.float32)
        arr_b    = np.expand_dims(arr, 0)          # (1, 224, 224, 3)

        # ── Classifier confidence ──────────────────────────────────────────
        pred = float(model.predict(arr_b, verbose=0)[0][0])
        if pred < PRED_THRESHOLD:
            skipped += 1
            continue                               # low-confidence → skip

        # ── Grad-CAM → bbox ────────────────────────────────────────────────
        try:
            heatmap = get_gradcam(model, arr_b)
        except Exception as e:
            print(f"  [!] GradCAM failed for {img_path.name}: {e}")
            skipped += 1
            continue

        bbox_224 = heatmap_to_bbox(heatmap, (IMG_SIZE, IMG_SIZE),
                                   threshold=GRADCAM_THRESH)
        if bbox_224 is None:
            skipped += 1
            continue

        # ── Scale bbox to original image dimensions ────────────────────────
        orig_w, orig_h = pil_img.size
        sx = orig_w / IMG_SIZE
        sy = orig_h / IMG_SIZE

        x1, y1, x2, y2 = bbox_224
        bx   = int(x1 * sx)
        by   = int(y1 * sy)
        bw_  = int((x2 - x1) * sx)
        bh_  = int((y2 - y1) * sy)

        if bw_ < 10 or bh_ < 10:
            skipped += 1
            continue

        coco["images"].append({
            "id":        img_id,
            "file_name": str(img_path.resolve()),
            "width":     orig_w,
            "height":    orig_h,
        })
        coco["annotations"].append({
            "id":          ann_id,
            "image_id":    img_id,
            "category_id": 1,
            "bbox":        [bx, by, bw_, bh_],   # COCO: [x, y, w, h]
            "area":        bw_ * bh_,
            "iscrowd":     0,
            "score":       round(pred, 4),        # store classifier confidence
        })

        img_id += 1
        ann_id += 1

    # ── Add negative (no-tumour) images — no bounding boxes ──────────────────
    neg_dir   = Path(DATASET_PATH) / "no"
    neg_paths = sorted(p for p in neg_dir.iterdir()
                       if p.suffix.lower() in IMAGE_EXTS)
    # Cap negatives to match annotated positives so dataset stays balanced
    neg_paths = neg_paths[:ann_id]

    print(f"Adding {len(neg_paths)} negative (no-tumour) images …")
    for img_path in neg_paths:
        try:
            pil_img       = Image.open(img_path).convert("RGB")
            orig_w, orig_h = pil_img.size
        except Exception as e:
            print(f"  [!] Cannot open {img_path.name}: {e}")
            continue
        # No annotation entry → RCNN will treat as background image
        coco["images"].append({
            "id":        img_id,
            "file_name": str(img_path.resolve()),
            "width":     orig_w,
            "height":    orig_h,
        })
        img_id += 1

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(ANNOTATIONS_PATH, "w") as f:
        json.dump(coco, f, indent=2)

    total_images = ann_id + len(neg_paths)
    print(f"\n{'─'*50}")
    print(f"Tumour (positive) : {ann_id} images with boxes")
    print(f"Normal (negative) : {len(neg_paths)} images (no boxes)")
    print(f"Total             : {total_images}")
    print(f"Skipped           : {skipped} (low confidence or no clear region)")
    print(f"Saved             : {ANNOTATIONS_PATH}")
    print(f"\nNext step : python src/train_rcnn.py")


if __name__ == "__main__":
    generate_annotations()
